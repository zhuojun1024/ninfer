# PLAN: GGUF 权重的 TP-2 支持

## 1. 目标与验收

1. **GGUF 工件能在 TP-2 上物化与运行**（本机 2×RTX 5060 Ti）。
2. **把误导性的 `cudaMalloc OOM` 换成明确诊断**：当某个分片确实无法表示时，启动即报出
   "GGUF 块量化权重不支持该张量并行切分" 并指名对象与原因。
3. 不回归：单卡 GGUF 路径、NVFP4/FP8 的 TP-2 路径（既有 TP-2 测试套件）全部保持通过。

## 2. 现状（侦察结论）

TP-2 的切分是**布局无关**的：`src/core/tp/` 与 `src/runtime/engine/` 中 0 处 GGUF 引用，
全部按 `WeightSplitKind`（`ColumnParallel`/`RowParallel`/`GatherRows`/`GatherCols`/`Replicated`）工作。
缺口集中在三个"布局感知"的位置：

| 位置 | 现状 | 后果 |
|---|---|---|
| `tp_materialize.cpp: shard_geometry()` | 只对 RowSplit/NVFP4/FP8 有分支，其余按 **4 字节/元素** | GGUF ~0.44 B/元素 → 分片预算放大约 9× → `cudaMalloc` OOM（**实测**：`ninfer_qwen3_5_tp2_load_test --artifact <GSQ-RCO GGUF>` EXIT=1） |
| `weight_splitter.cpp: split_weight()/gather_rows_impl()/gather_cols_impl()` | 契约只覆盖 NVFP4/FP8/RowSplit/BF16 | 修好上面后 GGUF 会掉到 `throw "unsupported qtype"` |
| `tp_shard_views.cpp` | 用"元素"偏移算分片区域 | GGUF 单 code plane、行主序，元素↔字节需按 `code_bytes_per_row` 映射（现路径可复用，需核对） |

**关键可行点**：`weight_geometry()` 已支持 `GgufBlocks`（本次移植新增，`code_bytes_per_row = (k/B)*b`，
`scale_offset = code_bytes` 即**单一 code plane、无独立 scale 平面**）。因此：

- `ColumnParallel`（切 N 行）：分片载荷就是父对象 code plane 的一段连续字节 → **纯字节拷贝**。
- `RowParallel`（切 K 列）：GGUF 每行是 `k/B` 个自洽 block，按 **block 粒度**切列 = 每行取 [b0,b1) 个 block
  的连续字节 → **纯字节拷贝**，条件是 `k/2 % B == 0`（B ∈ {32,256}）且切点落在 block 边界。
- `GatherRows`/`GatherCols`：按 part 取行/列的一半，同理为字节拷贝。

## 3. 唯一的结构性障碍：`gdn/output` 的 `input_columns` 置换

recipe 里 `ssm_out`(= `gdn/output`) 的**存储 K 列是 tiled value-head 顺序**，其 Use 带 `input_columns`
辅助（列 c 乘 grouped 激活的元素 `input_columns[c]`）——这正是移植时保留它的原因（移动列会跨 block 重量化）。

而 TP-2 的头切分让每个 shard 的激活是 **grouped 半边**（自己的 24 个 value head）。`tiled 前一半`
与 `grouped 前一半` 对应的是**不同的 head 集合**（`tiled = (h%3)*16 + h//3`，两侧都不是对方的连续半边），
且 head = 128 元素 < 256 元素的 i-quant block → **任何 block 级切列都无法让权重半边与激活半边对齐**。

结论：这一个对象需要**执行侧**处理，即"在 `gdn/output` 之前让两个 shard 都拿到完整的 value 输出"。
可行的最小改动是复用**既有**的 in-kernel all-reduce（无需新集合原语）：shard 把自己的输出写进
**全宽**缓冲（范围外填 0）→ all-reduce 求和 → 两卡都得到完整 grouped 激活 → 再按现有
"K 切分 + 投影后 all-reduce" 模式计算（完整激活 + 各自 K 半边 = 部分和，仍需原有归约）。
代价：每个 GDN 层多一次 [6144,T] 归约（与既有的 [5120,T] 投影后归约同量级），该投影 2× FLOPs（占比极小）。

## 4. 分层实施

- L1 `shard_geometry` 加 `GgufBlocks` 分支（委托 `weight_geometry`）→ 预算正确，OOM 消失。
- L2 明确诊断：新增共享校验（布局 × 切分种类 × block 对齐），在**预算阶段**（cudaMalloc 之前）
  与**切分阶段**都在同一处报错，消息指名对象与原因。
- L3 `split_weight`：GGUF 的 `ColumnParallel`（行切片）与 `RowParallel`（block 对齐列切片）。
- L4 `gather_rows_impl`/`gather_cols_impl`：GGUF 分支（按 part 取半、拼接）。
- L5 模型侧 `build_tp_split_spec`：GGUF 感知——只选可表示的 kind；对带 `input_columns` 的对象在
  执行侧方案落地前**明确拒绝**（而不是静默 `Replicated`，那会算错）。
- L6 执行侧：`gdn/output` 的输入全宽化 + 复用 all-reduce（L5 的拒绝随之解除）。
- L7 测试：切分字节级 oracle 单测（从 block 几何独立重算期望字节）；诊断触发测试；
  `tp2_load/forward` 在 GGUF 工件上通过；既有 TP-2 套件不回归；最后双卡端到端。

## 5. 进度

- [x] 侦察（分支点 `bfb57e72`；本机 2×RTX 5060 Ti）
- [x] L1 + L2（预算与诊断）→ commit `c3267d14`
- [x] L3 + L4（切分与 gather）→ commit `c3267d14`
- [x] L5（spec GGUF 感知）→ commit `c3267d14`
- [x] L6（执行侧 input_columns 对象 + GGUF GDN 输入投影 + 分离父对象布局）见第 7 节
- [x] L7（验证）见第 6 节：GGUF 的 `tp2_load`/`tp2_forward` 通过，既有 TP-2 套件不回归，双卡端到端与单卡一致
- [x] 提交 `1a4a6a77`（`feat(tp): execute GGUF block checkpoints on the TP-2 pair`），并 ff-only 合入
      `feat/windows-native-port`（合并后 HEAD 复跑 `linear_gguf` / `tp2_load` / `tp2_forward` 通过）
- [x] 后续修复：GGUF 上 `--lm-head-draft` / `--spec dflash2` 的 head profile（见第 7 节末），已提交

## 6. 实测结果（本轮）

- 修复前：`ninfer_qwen3_5_tp2_load_test --artifact <GSQ-RCO GGUF>` → `cudaMalloc OOM`（分片预算按 4 B/元素膨胀）。
- 修复后：同一命令 → **物化成功**（`device_objects=830`，**7.70 GiB/卡**，两卡完全空闲时实测），随后在
  `tp_shard_views` 抛出精确诊断：
  `tensor parallel split: text/layers/0/gdn/output gathers its input columns, which a
  column-parallel TP-2 split cannot represent`。
- 独立复核：用工件报告的每对象 shape/format 在 Python 里按 `tp_split_spec.cpp` 的规则重算，
  每卡需求 ~8 GiB（Replicated 5.46 + RowParallel 1.84 + GatherRows 1.08 + 单卡组件 0.69），与实测 7.70 GiB 一致。
- 新增单测 `ninfer_tp_weight_splitter_test` 通过：Q8_0(32)/IQ4_XS(256)/Q4_K(256) 的 N 切分、K 切分
  （逐块、且两半能逐字节还原父对象）、按 part 的行/列 gather，以及三种诊断触发；期望值全部由
  `gguf_block_shape()` 独立重算。
- 回归：`ninfer_linear_tp2_split_{nvfp4,fp8_head,grouped_head}_test` 全部通过（NVFP4/FP8 TP-2 未受影响）。

### L7 验证（本轮）

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| GGUF 双卡物化 | `ninfer_qwen3_5_tp2_load_test --artifact Qwen3.8-27B-GSQ-RCO-IQ3_S` | 通过（64 层、gate [8704,5120]、head vocabulary-parallel、两卡 7.70 GiB） |
| GGUF 双卡前向 | `ninfer_qwen3_5_tp2_forward_test --artifact 同上` | 通过（11 探针分片一致、32 步解码、batched 与 sequential 一致；三种分块 argmax 相同、top5 gap 0.125） |
| NVFP4 双卡物化/前向 | 同上两条 + `qwen3_8_27b_nvfp4.ninfer` | 通过，且 128+172 / 64+236 仍标注 `(bit-exact)` |
| 算子/切分回归 | `ninfer_tp_weight_splitter_test`、`ninfer_linear_gguf_test`、`ninfer_embedding_test`、三个 `ninfer_linear_tp2_split_*` | 全部通过 |
| 双卡端到端（含 CUDA Graph） | `ninfer-serve --devices 0,1 <GGUF>` + `/v1/chat/completions` | 物化 10.9 GiB、`warmup complete`（图捕获成功）、短提示 57 token 与单卡 `ninfer.exe --device 0` 生成文本**完全一致** |
| 长提示（469 token）对齐 | 同上，`--prefill-chunk 256/1024` | 双卡两种分块都给出 `…Passage repeated. Question: … in capillary`；单卡 `--prefill-chunk 1024` 与之一致，单卡 `--prefill-chunk 256` 在第 9 个 token 处近平分叉（同一张卡上分块本身就会翻转近似平局） |

结论：GGUF 工件已在 TP-2 上物化、跑通前向与端到端服务，且与单卡路线在相同（乃至不同）分块下给出相同的贪心文本。
未跑：`tp2_sessions` / `tp2_dflash_*`（需要 dflash2 工件，且走的是本次改动未触及的引擎路线；spec 推导已由上述两个工件的物化用例覆盖）。

## 7. L6 设计定案（本轮实现）

结论：**不需要**让两个 shard 各自拥有激活；改为让 `gdn/output` 的 K 切分与 activation 的域解耦。

1. `input_columns` 的语义本来就是"第 c 列乘激活的第 input_columns[c] 个元素"——它索引的是**激活**，
   不是权重自己的 K。因此把"权重的 K"与"激活行宽"分开：op/bridge 接受 `source_k >= k`。
2. `gdn/output` 仍是连续 K 半切分；`input_columns`（`auxiliary/000000`，INT32 [6144]，**一个对象被 48 层共用**）
   在 `tp_shard_views` 里按同一半**开窗**（父对象整体驻留，只切视图，+12 KiB/卡）。
3. 执行侧两阶段：`run_layers_tp2` 让两卡各自把本卡 24 个 value head 写进全宽 [6144,T] 缓冲的本卡半边
   （其余清零），一次与既有模式相同的 allreduce 合并成完整 grouped 激活，再各自用本卡 K 半边投影得部分和，
   再由既有 delta allreduce 收尾。每 GDN 层多一次 [6144,T] allreduce（T=1024 时 12 MiB，阈值内）。

### 顺带发现并修掉的第二个缺口：分离的 gate/up
GGUF 工件的 `mlp/gate`(IQ2_XS) 与 `mlp/up`(IQ2_XXS) 格式不同 → 是**两个对象**，不是融合父对象，
所以 `n == 2*intermediate` 规则不匹配，落到 `Replicated`。而 `down` 是 RowParallel →
`ffn_delta` 会把 gate 自己的两半当成 gate/up 做 silu_mul（**形状合法、数值错误**）。
修复：spec 为分离的 gate/up 选 `ColumnParallel`（正好对上 down 的 K 半），并让 `ffn_delta` 支持 `p->up`。

### 第三个缺口：GGUF 工件的融合输入投影并不总是存在于一个对象里
同一工件的不同层，`gdn/{query,key,value,z}` 可能是**一个** [16384,5120] 父对象的四段（层 0），也可能是
`[q|k|v]` 一段 + `z` 单独对象（层 1 等），因为 recipe 只把**连续且块格式相同**的逻辑块合并成一个对象。
attention 同理：层 27 是 `[q|k]` 融合 + 独立 gate/value，其余层是四个独立对象。
修复（`tp_split_spec.cpp`）：把注意力输入投影从写死的 `n == 2*attn_q + 2*attn_k` 改成“按名字读出的连续块运行段”
（`run_parts` 辅助：名字决定段、宽度决定行数、必须正好等于对象行数 → `GatherRows`），并新增 GDN 的
`[q|k|v]`（GatherRows 三段）与单独 `z`（ColumnParallel）两条规则。
同时 `ops::gdn_input_proj` 的 GGUF 形式原来把 q/k/v/z 行数写死成全模型 profile（10240/6144），
头切分后被拒；改为由 parts 反推通道 profile（q==k、v==z==3q），并新增“单个连续 [q|k|v|z,T] 目标”重载
（GGUF 积只写连续目标，分片的两半本来就是同一父对象的行段）。

### 关于“分块 prefill 逐位一致”这一判据
`tp2_forward` 的 128+172 / 64+236 两个分块要求与单块 prefill **逐位一致**，NVFP4 路线满足，GGUF 不满足
（top5 gap 0.125、argmax 相同、max_logit_diff ≈0.4）。定位过程（临时探针，均已移除）：

1. 逐层摘要：同一 prompt、同一零状态，T=300 与 T=128 的**层输入**完全相同，**层 0 输出**开始就不同 →
   差异来自层内某个吃 GGUF 权重的算子（层 0 的第一个就是 GDN 输入投影）。
2. 算子级复现（`test_gguf.cpp` 临时用例，真实形状 rows=5120+3072、k=5120、IQ4_XS、两个 product 写同一目标）：
   `128 vs 300 → 140/1M 个元素不同，worst=16`；而小形状（rows=256、k=1536）零差异。
3. 机理：`third_party/ggml-quants` 的整数 MMQ 用 **stream-K** 把 K 维按 grid（行 tile × 列 tile）分给多个 block
   再 fixup 求和；网格随列数变化 → 同一列的浮点求和顺序随“本次调用的列数”变化。小形状下每个 tile 恰好一个
   block（无部分和）故零差异。关闭 stream-K / 固定 tile 宽度的临时探针都未能消除（说明不是 tile 选择而是
   分块本身），但算子级复现已足以证明这是**量化路线的固有性质**，属于既有 GGUF 移植（单卡路线同样如此），
   不是 L6 执行侧引入的。
4. 因此 `test_tp2_forward.cpp` 按路线收紧/放宽：吃 ggml 块权重（`GgufProjectionWeights`）的工件，分块判据
   改为“argmax 相同 + top5 gap ≤ 1.0”（仍能抓住 KV/GDN 状态丢失这类整数量级错误），其余路线维持逐位一致。

### 第四处缺口：`--lm-head-draft` 的半表 proposal head（用户配方报错，已修）
用户配方 `--devices 0,1 --spec dflash2 --lm-head-draft --vision --max-context 262144` 启动时报
`FATAL server failed during startup | linear_topk: unsupported head profile`（规划期 27 s 处）。
根因：词表切分把 131072 行的 reduced proposal head 分成每卡 65536 行，而 `linear_topk` 的 `resolve_profile`
只对 **Q4_G64_FP16** 认半表（`Q4OptimizedHalf`），GGUF 只认整表 131072 与整词表 248320 → 半表落到兜底 throw；
索引入口（带 row→global id map）同样只放行 131072。修复：新增 `is_reduced_rows()`，GGUF profile 同时接受整表
与半表（同一个 profile，行数取自权重）。
验证：`ninfer_linear_topk_test` 新增 GGUF（q8_0）整表/半表两种几何的排序用例（独立 FP64 oracle、id map、
CUDA Graph 重放、guard）与一条 head profile 接受性表（整词表/整表/半表接受，四分之一表与 5120 行拒绝）→ 通过；
用户配方服务启动成功（KV 262144 fp8、vision、`dflash2 accepted 7/14 (50%)`、同一贪心请求两次输出一致）；
新 `ninfer-serve.exe` 已部署到 `C:/ninfer/`。

### 顺带修掉的陈旧断言与它掩盖的越界（`test_tp2_load.cpp`）
`tp2_load --lm-head-draft --spec dflash2` 在 GGUF 上原先报 `shard 0 DFlash2 proposal head is not
replicated`：该断言停留在“切分之前”的设计（proposal head 整表常驻 shard 0、selector 词表 codebook 也在
shard 0）。实际设计（`load.cpp:137-143`、`draft.cpp:398-408`、`tp_split_spec.cpp:97-108`）是：reduced head
由两条投机路线各自切成两半（MTP 在 `proposal_argmax` 合并、masked draft 各排一半再用 `merge_topk_candidates`
合并），selector 与整词表 codebook 移到 shard 1。断言放行后又暴露第二个问题：测试直接读分片未物化的权重
（空 view）→ 空指针解引用 → 进程 0xC0000005（不是抛错）。已把断言改成“shard 0 持有本卡半表、且不持 codebook；
shard 1 不持 draft、持 [248320,256] codebook”。两个工件的四种组合（gguf/nvfp4 × dflash2/plain，另加 gguf mtp）
全部通过。

## 8. 剩余工作（原 L6 描述，保留作历史）

这是**唯一**阻塞 GGUF 上 TP-2 的结构性原因，需要执行侧改动（不是切分器问题）：

1. **执行侧**：在 GDN 混流器的输出投影之前，让两个 shard 都拿到**完整**的 grouped value 输出。
   复用**既有** in-kernel all-reduce（无需新集合原语）：各 shard 把自己的输出写进**全宽**缓冲
   （范围外填 0）→ all-reduce 求和 → 两卡得到完整激活。随后的 "K 切分权重 + 投影后 all-reduce"
   沿用现有模式不变（完整激活 + 各自 tiled K 半边 = 部分和，仍需原有归约）。
   代价：每个 GDN 层多一次 [6144,T] 归约（与既有 [5120,T] 投影后归约同量级），该投影 2× FLOPs（占比极小）。
   涉及 `src/runtime/engine/tp2_generation_core.cpp`（含 CUDA Graph 捕获与 in-kernel allreduce 的
   详细语义）与模型侧 `execution/text.cpp` 的 TP-2 分支。
2. **辅助对象随权重一起切**：`input_columns`（INT32 [K]，6144 项）目前因 `shape.size() < 2` 落到
   `Replicated`，而分片后的权重 `k = 3072`，`prepare.cpp` 会因 `columns.numel() != weight.k` 报错。
   需要让该辅助对象跟随其权重做同样的 K 切分（`tp_shard_views.cpp` 的 RowParallel 分支目前要求 2-D
   矩阵，需要为 1-D 辅助增加处理），并让 `shard_views` 的拒绝条件在"执行侧方案落地且辅助已切分"后解除。
3. 验证：`tp2_load/forward` 在 GGUF 工件上通过；双卡 `ninfer-serve --devices 0,1` 端到端生成与单卡
   逐位/数值对齐；既有 TP-2 套件不回归。
## 9. DFlash2 组件替换（同一份 draft 的量化策略）

用户问“能否把 `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` 的 DFlash2 换成
`qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer` 的”。结论：**两份 draft 是同一份权重**——91 个绑定的 shape
全同，33 个 bf16 张量逐字节相同；差别只在量化策略：GSQ 工件按 `qwen3_8_27b_gguf` 的 `_companions` 把 draft
投影存成 `q8_g32_fp16`、selector codebook 留 `bf16`（dflash2 共 2123.6 MiB），官方 recipe 用
`q4_g64_fp16`（含两个 codebook，共 977.2 MiB）。

`tools/convert/graft_components.py` 不能直接移植（要求 object id 与 format/bytes 完全一致，这里 id、格式、
长度都不同）。改走“修 recipe + 重转”，过程中修掉 `bfb57e72`（GGUF block 转换那次提交）引入的三个缺陷：

1. `tools/convert/__main__.py` 里 `SourceInputs` 定义了两次，后一个（旧实现）遮蔽前一个（按后缀分派），
   `--source gguf=` 被当 safetensors 打开 → GGUF 转换在 CLI 层就失败。
2. `gguf_blocks.text_sources` 调用不存在的 `gguf.readdirect_source`（3 处），实际 API 是 `read_direct`。
3. `gguf_blocks._companions` 把 DFlash2 硬编码成 Q8，与它自己的 docstring（“DFlash2 as the official
   recipe”）及官方 recipe 的 Q4 政策不符 → 抽出 `official_recipes.assign_dflash_formats`，两处共用。

产物 `D:\LLM\Qwen3.8-27B-GSQ-RCO-IQ3_S-dflash2q4.ninfer`（12.85 GiB，旧件 13.99 GiB）：

- dflash2：91/91 绑定与 donor **逐字节相同**，66 个对象与共享结构与 donor 一致；
- text/vision/mtp 与旧工件**逐字节相同**（963/441/16 个绑定）；
- `proposal/head` 由 `gguf_q4_k` 变为 `q4_g64_fp16`（当前 `--proposal` 的官方策略，与 donor 相同）。

TP-2 实测（`--max-context 262144 --kv-dtype fp8 --spec dflash2 --draft-tokens 7 --lm-head-draft --vision`，
贪心）：显存 卡0 12640→11664 MiB、卡1 11520→11332 MiB；接受率 paris/256tok 19.1%→15.6%
（decode 58.4→53.5 tok/s）、散文/381tok 10.5%→10.1%（43.7→44.1 tok/s）。结论：可以换，换来约 1 GiB
显存，代价是该 thinking-heavy 工作负载上接受率小幅下降。

转换器三处修复 + 文档更新仍未提交（工作树：`__main__.py`、`gguf_blocks.py`、`official_recipes.py`、
`docs/gguf.md`、`docs/weight-conversion.md`）。
## 10. MTP + GGUF 长 prompt 失败：`gguf matrix product launch: invalid argument`

用户报告：`--spec mtp` 下长 prompt 请求 500，日志 `C:\ninfer\serve-win.log`。复现（原始 GGUF 工件，
`--spec mtp --draft-tokens 3 --lm-head-draft --max-context 262144`）：短 prompt 成功，长 prompt
（≈7200 token，多个 1024 分块）必失败。失败的算子来自 SSE 错误体：

```
rows=5120, k=10240, columns=1024, J=128, I=128, threads=256, shared=57856,
row_tiles=40, column_tiles=8, blocks=320, stream_k=true, fixup=false,
kernel_static_shared=0, kernel_max_dynamic_shared=101376, kernel_max_threads=256,
kernel_regs=255, kernel_local=72, kernel_ptx=120, kernel_binary=120,
device_default_block_shared=49152, device_optin_shared=101376, device_sm_count=36
```

即 MTP 的 `mtp/input_projection`（`[5120, 10240]`，Q6_K）在 1024 token 预填分块上的 MMQ 启动。

排除项（均实测）：48 KiB 以上的 opt-in 已生效（把属性设为设备上限 101376 仍失败）；stream-k 与
tiled 两种网格都失败；`--no-cuda-graph` 与 `CUDA_MODULE_LOADING=EAGER` 都无效；启动前无残留错误
（`cudaGetLastError()` 干净）；失败后重设属性并重试同一 tile 仍失败；让 tile 用 J<=64（`shared=48384`
<= 48 KiB）则同一请求成功。结论：本机上 (Q6_K, J=128) 这个实例的 57856 B 动态共享内存启动被驱动拒绝，
且文档化的所有上限都满足，原因未定位。

处理：`select_matrix_launch` 增加 `max_J` 上界；`launch_matrix` 返回启动状态而不是抛出；
`matrix_product_impl` 在 `cudaErrorInvalidValue` 时退到更窄的列 tile（最多到 J=16），全部被拒才报错并
带上最宽尝试的几何信息；`matrix_fixup_bytes_impl` 改为对所有候选 tile 取最大值，保证工作区够用。
启动被拒是纯 host 侧、不会执行任何东西，因此回退安全；若失败与 tile 无关，回退也会同样失败而报错。

验证：MTP 长 prompt（7214 token）成功，decode 87.4 tok/s，`mtp accepted 40/69 (58.0%)`；dflash2
长 prompt 成功（68.8 tok/s，接受率 26.8%）；`ninfer_linear_gguf_test` 全 15 种量化（含 130 列的
宽 batch，覆盖 J=128 tile）通过。
