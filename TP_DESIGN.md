# TP-2 执行设计（2× RTX 5060 Ti, Qwen3.8-27B NVFP4）

## 0. 目标
在 WSL2 双 5060 Ti 上跑通 27B NVFP4 的 TP-2 decode（C=1..8），
MTP0 ≥28 tok/s，MTP3 ≥70 tok/s。TP=1 路径行为不变。

## 1. 切分表（27B dense, hidden=5120, 48 layers）

| 权重 | 全尺寸 | 切分方式 | 每 GPU 尺寸 | 通信 |
|---|---|---|---|---|
| QKV (attn) | [16384,5120] | 列并行（切 N） | [8192,5120] | 无（head 并行） |
| GDN qkv+z | [14336,5120] | 列并行（切 N） | [7168,5120] | 无（head 并行） |
| o_proj (attn) | [5120,6144] | 行并行（切 K） | [5120,3072] | all-reduce（sum） |
| GDN output | [5120,6144] | 行并行（切 K） | [5120,3072] | all-reduce（sum） |
| gate_up (MLP) | [34816,5120] | 列并行（切 N） | [17408,5120] | 无（silu_mul 本地） |
| down (MLP) | [5120,17408] | 行并行（切 K） | [5120,8704] | all-reduce（sum） |
| lm_head | [248320,5120] | 列并行（切 N） | [124160,5120] | all-gather（concat） |
| token_embedding | [248320,5120] | 复制 | 同全尺寸 | 无 |
| RMSNorm/bias | [5120] 等 | 复制 | 同全尺寸 | 无 |

每层 2 个 all-reduce 点（mixer output + MLP output）× 48 层 = 96 次/forward。
all-reduce 实现：host-staging（pinned buffer + CPU 加法），P2P 不可用（WSL2）。

## 2. 执行流（decode, T=1..8）

```
for each GPU in {0, 1}:
    x = embedding(ids)          // 复制，两 GPU 相同
for layer in 0..47:
    for each GPU in {0, 1}:
        x = rmsnorm(x)
        if full_attention:
            qkv = linear(x, QKV_shard)       // [8192, T] 列并行
            q, k, v, gate = split(qkv)       // 本地 head 切分
            attn_out = causal_softmax_attention(q, k, v, gate)  // head 并行
            out = linear(attn_out, o_proj_shard)  // [5120, T] 行并行（部分和）
        else:  // GDN
            qkvz = linear(x, GDN_shard)      // [7168, T] 列并行
            gdn_out = gated_delta_net(qkvz)  // head 并行
            out = linear(gdn_out, GDN_out_shard)  // [5120, T] 行并行（部分和）
    allreduce(out)                        // sum → 两 GPU 相同
    for each GPU in {0, 1}:
        x = rmsnorm(x)
        gu = linear(x, gate_up_shard)     // [17408, T] 列并行
        act = silu_mul(gu[:8704], gu[8704:])  // [8704, T] 本地
        delta = linear(act, down_shard)   // [5120, T] 行并行（部分和）
    allreduce(delta)                      // sum → 两 GPU 相同
    x += delta                            // residual（两 GPU 相同）
x = rmsnorm(x, final_norm)                // 复制
for each GPU in {0, 1}:
    logits_half = linear(x, lm_head_shard)  // [124160, T]
allgather(logits_half) → logits           // [248320, T]
sample(logits)                            // 单 GPU 采样
```

## 3. 关键设计决策

### 3.1 融合 Op 分解
27B dense 的 FFN 用融合 Op linear_swiglu + linear_add（dispatch 硬编码 full shape）。
TP-2 不注册新的融合 kernel，而是**分解**为：
- linear_swiglu → linear(gate_up_shard) + silu_mul（列并行，无需通信）
- linear_add → linear(down_shard) + residual_add + all-reduce（行并行）
切分逻辑在执行层（TextContext），Op 层保持已验证的半尺寸 linear。

### 3.2 KV/State 池
- Attention KV：head 并行，每 GPU 存自己的 head 的 KV（本地池，无通信）
- GDN state：head 并行，每 GPU 存自己的 head 的 state（本地池，无通信）
- 每 GPU 的 KV 容量 = 全容量 / 2（按 head 数比例）

### 3.3 lm_head vocab 切
- 每 GPU 算 [124160, T] logits（BF16 linear，通用 GEMM 无 shape 限制）
- all-gather（concat 沿 vocab 维）→ [248320, T]
- 单 GPU 采样（GPU 0）

### 3.4 同步策略
- 每层 2 个 all-reduce 点，每个 all-reduce 前需同步两 GPU 的 stream
- host-staging all-reduce 内部做 D2H（隐式等待 compute 完成）
- 无需显式 cudaStreamSynchronize（D2H memcpy 已阻塞到 source 就绪）
- 实际上需要：在 all-reduce 前 cudaStreamSynchronize 两 stream，确保 compute 完成

### 3.5 TP=1 不变
- 所有 TP 代码 gated on shard_count > 1
- TP=1 时 Parameters/TextContext/Program 行为与改造前字节一致

## 4. 实现阶段

### 4.1 权重切分工具（host 端）
- 输入：完整 NVFP4/BF16 权重 payload
- 输出：两个 shard payload（列切分=连续字节切片，行切分=strided+scale 重映射）
- 复用 Phase 2 oracle 的 slice_nvfp4_rows/slice_nvfp4_cols 逻辑
- BF16 权重：简单行/列切片（无 scale 平面）

### 4.2 双 Parameters 构造
- 从 Model 的完整权重切分 → 两个 Parameters（各持半尺寸权重）
- 每个 Parameters 的 Weight 指向对应 GPU 的设备内存
- 复制权重（norm/embedding）两 GPU 各持一份

### 4.3 TP 执行协调器
- 新类 TPTwoTextContext：持两个 TextContext + DevicePair
- run_layers_tp()：lockstep 逐层，每层两 GPU 各跑本地计算，all-reduce 点同步
- 修改 ordinary_decode_batch 入口：TP-2 时走 TPTwoTextContext

### 4.4 KV/State 池切分
- 每 GPU 独立 KV 池（容量 = 全容量 / 2）
- 每 GPU 独立 GDN state 池
- Program 持两个 KV store + 两个 state store

### 4.5 lm_head + 采样
- 每 GPU 算半 vocab logits → all-gather → 单 GPU 采样
- 采样结果广播到两 GPU（或只在 GPU 0 采样，结果通过 host 传递）

### 4.6 Program/Engine 集成
- Program 持 DevicePair + 双 Parameters + 双 KV/state 池
- Engine 启动时检测 TP 配置（--tp 2）
- TP=1 时走原路径（不变）

## 5. 验证
- 每步 oracle：TP-2 输出 vs TP-1 输出（同输入）
- 单 token forward 数值对照 Phase 2 oracle
- perplexity 评测（小 corpus）
- 用户测试回合

## 6. 风险
- R1: all-reduce 延迟（host-staging ~10-20μs/次 × 96 次 = ~1-2ms/forward）
  → 在 37.8 tok/s 天花板下可接受（每 token ~26ms）
- R2: 双 GPU 同步开销（stream sync × 96 次）
  → 用 cudaEvent 替代 stream sync 减少开销
- R3: KV 池容量减半 → 上下文长度受限
  → 27B 16GB 卡，KV 容量本身有限，TP-2 后每卡 8GB 给 KV
- R4: 融合 Op 分解的性能损失（linear + silu_mul vs 融合 linear_swiglu）
  → 正确性优先，性能调优在 Phase 7
