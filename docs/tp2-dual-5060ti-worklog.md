# TP-2 adaptation work log (2x RTX 5060 Ti, Qwen3.8-27B NVFP4)

Archived chronological record of the TP-2 adaptation: the working `PLAN.md` kept verbatim, from the
initial plan through the final round. Sections 0-5 are the original planning document (goals,
hardware, design decisions, phase plan, risk register, collaboration protocol); section 6 onward is
the round-by-round record, including the delivered benchmark/quantization/context work and the open
MTP-consistency item. Line numbers and round numbers are not sequential: early rounds were logged on
demand, one round number was reused, and later entries were appended under the header that was
current when they were written.

The body was reconstructed from this session's transcript while the plan file was being cleaned up,
so it is the record as it stood at delivery. It is a historical log, not an active authority:
current state, recommended configuration, and measurements live in
[Dual RTX 5060 Ti TP-2 tuning](tp2-dual-5060ti.md), and remaining work is tracked in `PLAN.md` at
the repository root.

Evidence artifacts referenced by the record live outside the repository: the WSL build tree
(`/home/zhuojun/ninfer`), logs under `/home/zhuojun/prof/`, and the driving scripts under
`tools/tp_bootstrap/`.

---

# NInfer TP-2 适配计划（2× RTX 5060 Ti，Qwen3.8-27B NVFP4）

> 本计划是跨上下文压缩的持久记忆。每完成一步更新第 6 节进度。
> 创建：2026-07（会话日期以 git log 为准）。状态：**规划完成，待 Phase 0**。

---

## 0. 目标与完成条件

**目标**：让 `ninfer-serve` 以张量并行 TP-2 在 2× RTX 5060 Ti（各 16GB / 448 GB/s / PCIe x8 5.0）
上运行 Qwen3.8-27B NVFP4 官方工件（23.7GB，含 text/mtp/dflash2，启动时禁用 vision）。

**完成条件**（全部满足才算完成）：
1. TP=1 路径与改造前行为一致（回归不破坏单卡路径）；
2. 所有切分 Op 通过独立 oracle 测试（对照完整 Op，FP32/FP64 参考）；
3. 端到端：perplexity 评测数值合理，用户人工生成质量确认正常；
4. 性能下限：MTP0 decode ≥ 28 tok/s（37.8 roofline 的 75%），MTP3 ≥ 70 tok/s；
5. 稳定性：C=1/2/4 多请求冒烟无 OOM、无挂起；有条件则跑 75 请求 corpus。

**明确不做**（本范围内）：MoE 35B 的 TP、流水线并行、DP、continuous batching 扩展、
Windows 原生移植（走 WSL2）、新架构支持。

---

## 1. 硬件与环境（已实测）

| 项 | 值 |
|---|---|
| GPU0 | RTX 5060 Ti 16GB，448 GB/s（128-bit GDDR7 1750MHz），当前被 llama-server 占用 |
| GPU1 | Tesla T10 16GB（sm_61，跑不了 sm_120a，空闲，可作会话模型迁移目标） |
| GPU2 | RTX 5060 Ti 16GB，448 GB/s，当前被 llama-server 占用 |
| 卡间互联 | 仅 PCIe x8 5.0（≈31.5 GB/s/方向），无 NVLink |
| 宿主 | Windows；NInfer 要求 Linux → **WSL2**（Phase 0 确认） |
| 宿主内存压力 | llama-server 占 15.7GB；编译用 `-j 4` |
| 工件 | 23.7GB，**单卡装不下** → 本机无 TP-1 对照基线 |

性能 roofline（decode，带宽瓶颈）：896 GB/s ÷ 23.7GB ≈ **37.8 tok/s**（MTP0 上限）。
5090 实测效率 69–94% roofline → 双卡预期 MTP0 26–34 tok/s，MTP3 70–125，DFlash2 90–180。

---

## 2. 关键设计决策（含理由）

1. **固定 TP=2，只支持 dense 27B**。无 MoE 切分、无 PP/DP。启动选项形如
   `--devices <name1>,<name2>`（按 GPU 名选择，不用索引——T10 插在中间）。
2. **加载时切分，不改工件格式**。converter 不动；materialization 阶段读完整张量 →
   切半 → 分别上传。复用官方 23.7GB 工件，用户无需重新转换。
3. **切分模式**（27B dense 的每层结构：norm → mixer → norm → MLP）：
   - QKV 投影 / gate+up（linear_swiglu）：**列并行**（切输出维，输入复制）
   - o_proj / down_proj（linear_add）：**行并行**（切输入维，输出 all-reduce）
   - softmax attention：按 head 切；KV 池每 GPU 只存本地 KV heads；attention 内部无通信
   - GDN 线性注意力：按 head 切；state [H,D,D] 按 head 切；chunked prefill 与 decode
     replay 均 head 并行，内部无通信
   - gdn_input_proj / gdn_gating_proj：按 head 切
   - lm_head：按 vocab 切，logits all-gather，单 GPU 采样
   - RMSNorm / bias / 小权重：复制（体积可忽略），作用于 all-reduce 后的完整 hidden
   - MTP / DFlash2 草稿层：同一套模式；selector 作用于 all-reduce 后的 hidden，天然安全
4. **通信：自研 P2P all-reduce，不引入 NCCL**。2-GPU all-reduce = 3 次 D2D
   cudaMemcpyPeer + 一次 add（~80KB/次，~15µs）。每层 2 个 all-reduce 点
   （mixer 输出后、MLP 输出后）× 48 层 = 96 次/forward ≈ 1.4ms，可接受。
   **回退方案**：若 WSL2 挡 P2P，改 host-staging（D2H→H2D 经 pinned 内存），
   延迟 ×3–5 但功能不变。Phase 0 实测决定走哪条。
5. **CUDA Graph 延后**。Phase 1–4 不用 graph（先正确性）；Phase 5 引入双 graph
   （每 GPU 一个）+ 跨设备 memcpy 节点（若可捕获），否则 all-reduce 留在 graph 外。
6. **执行模型：按层 lockstep**。batch 复制（TP 不切 batch）；两 GPU 并行执行同一
   Op 序列（独立 stream），层尾 all-reduce 即同步屏障。Engine worker 单线程驱动两卡。
7. **物理层按 shard 实例化**：arena / paged KV / state store / round buffers /
   workspace 各持 2 份实例；Program 并行驱动两个 shard。这是最大的重构面。
8. **TP=1 路径零改动**：所有新代码以 shard 数 gate，单卡走原路径（回归保护）。
9. **Frontend / Gateway / Engine 请求平面不改**：它们设备无关（待 Phase 3 验证该假设，
   若发现隐藏的单设备依赖再局部处理）。

---

## 3. 阶段计划

### Phase 0：环境与基线（1–2 天，需 GPU 窗口）
- [ ] 确认 WSL2 可用（`wsl --list`；不可用则触发风险 R2 的备选）
- [ ] WSL2 内装工具链：CUDA 13.1、CMake ≥3.28、Ninja、FFmpeg dev、libcurl ≥7.85、Python 3.11
- [ ] WSL2 内构建 NInfer TP=1（`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`）
- [ ] GPU 枚举：按名识别两张 5060 Ti（T10 在中间，索引不可靠）
- [ ] **P2P 实测**：小程序测两卡间 cudaMemcpyPeer 带宽 → 决定通信方案（R1）
- [ ] 单卡效率基线：用 Op 微基准（tools/bench 或临时 harness）测 5060 Ti 上
      linear/attention 内核效率，对照 5090 roofline 比例，校准性能预期
- [ ] 工件拷入 WSL2 文件系统（23.7GB，勿从 /mnt/d 直接加载）
- **交付**：基线数据表（P2P 带宽、内核效率、WSL2 可用性结论）

### Phase 1：设备层与通信（~1 周，基本不需 GPU）
- [ ] core：DevicePair（2× DeviceContext）、P2P enable
- [ ] core：all-reduce 原语（P2P 版 + host-staging 回退版，编译期/运行期切换）
- [ ] core：ShardedTensor 视图（Tensor + shard 索引 + 本地几何）
- [ ] core：双 arena、双 stream/event 管理
- [ ] 单测：all-reduce 数值正确（随机张量求和对照 CPU）
- **门禁**：编译通过 + TP=1 回归通过 + all-reduce 单测通过

### Phase 2：切分 Op（2–3 周，部分需 GPU 窗口）
每个 Op：实现切分入口 + oracle 测试（完整 Op vs 切分 Op + all-reduce，独立参考）。
- [x] linear（**NVFP4 半尺寸 5 shape**：列并行 3 + 行并行 2）：oracle PASS（T=1/8/32，FP64 独立参考）；其余 qtype 不在 27B NVFP4 工件范围，跳过
- [x] linear_swiglu：分解为半尺寸 linear(gate_up [17408,5120]) + silu_mul（列并行，输出行天然分区，无需通信）；切分逻辑在执行层（Phase 3）
- [x] linear_add：分解为半尺寸 linear(down [5120,8704]) + residual_add + all-reduce（行并行，all-reduce 已在 linear oracle 验证）；切分逻辑在执行层（Phase 3）
- [x] attn_input_proj：列并行 QKV [16384,5120]→[8192,5120]（半尺寸 linear 已验证）；head 并行无需改 kernel
- [x] softmax_attention：head 并行 + 本地 KV（kernel 按 head 索引，无需改；KV 池每 GPU 本地，Phase 3 执行层）
- [x] gdn_input_proj：列并行 [14336,5120]→[7168,5120]（半尺寸 linear 已验证）；head 并行
- [x] gdn_gating_proj：按 head 切（head 并行，无需改 kernel）
- [x] gated_delta_net：head 并行 + 本地 state（kernel 按 head 索引，无需改；state 池每 GPU 本地，Phase 3 执行层）
- [x] kv_cache append：每 GPU 本地池（Phase 3 执行层/Program）
- [x] lm_head：vocab 切 [248320,5120]→[124160,5120]×2 + all-gather + 单 GPU 采样（BF16，Phase 3 执行层）
- [x] sparse_moe：**跳过**（27B 是 dense）
- **门禁**：编译通过 + 全部 Op oracle 通过 + TP=1 回归通过 → **PASS**（半尺寸 linear oracle + tp_device_pair 全过；TP=1 路径不变）

### Phase 3：加载与模型集成（1–2 周）
- [ ] Materialization：加载时切分（完整读 → 切半 → 分别上传）
- [ ] Parameters：切分几何
- [ ] Planning：双设备 VRAM 预算、双 KV 容量解析（`--kv-capacity auto` 按每卡余量）
- [ ] Program：双 State/KV store、双 round buffers、按层 lockstep 执行 + all-reduce 点
- [ ] 验证 Frontend/Gateway/Engine 设备无关假设
- **门禁**：TP-2 加载工件成功 + 单 token forward 跑通（数值对照 Phase 2 oracle）

### Phase 4：端到端正确性（~1 周，GPU 窗口）
- [ ] 单请求生成冒烟（greedy，短输出，人工可读性检查）
- [ ] perplexity 评测（项目 perplexity 工具，小 corpus）
- [ ] C=2 / C=4 多请求冒烟
- **门禁**：用户测试回合 1 通过（或问题已修复）

### Phase 5：CUDA Graph（~1 周）
- [ ] 双 graph 捕获（每 GPU 一个，exact-B decode）
- [ ] 跨设备 memcpy 入图（若可捕获）；否则 all-reduce 留在图外
- [ ] 性能对比：graph on vs off
- **门禁**：graph 路径输出与非 graph 路径一致 + 性能不回退

### Phase 6：投机解码（1–2 周，GPU 窗口）
- [ ] MTP under TP（草稿层 TP + 验证批 TP）
- [ ] DFlash2 under TP（selector / proposal head / 验证）
- **门禁**：用户测试回合 2 通过

### Phase 7：调优与交付（~1 周）
- [ ] prefill chunk / KV dtype / 并发调优（以实测为准）
- [ ] 75 请求 corpus 基准（tools/bench），记录 makespan 与 decode tok/s
- [ ] 文档：README 与 docs 增加 TP-2 章节（选项、限制、性能）
- **门禁**：用户测试回合 3（最终）通过 → 目标完成

---

## 4. 风险登记

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | WSL2 虚拟化层挡 P2P | 中 | 高（all-reduce 退化为 host-staging，延迟 ×3–5） | Phase 0 实测；host-staging 回退已入设计 |
| R2 | WSL2 不可用 | 低 | 高（无 Linux 环境） | 备选：Linux VM + GPU 直通；或立项 Windows 移植（超出本范围） |
| R3 | 物理层重构量超预期 | 中 | 中（进度） | TP=1 路径 gate 隔离；每 Phase 编译门禁；小步提交 |
| R4 | 跨设备 CUDA Graph 不可捕获 | 中 | 低（all-reduce 图外，~1.4ms/forward） | Phase 5 已有回退 |
| R5 | VRAM 紧张（~10GB 权重/卡 + KV + workspace） | 中 | 中 | 禁 vision；必要时禁 DFlash2；降 prefill chunk；FP8 KV |
| R6 | T10 干扰设备枚举 | 低 | 低 | 按 GPU 名选择设备 |
| R7 | llama.cpp 占满双卡 | 确定 | 高（阻塞一切 GPU 测试） | GPU 窗口协议（见第 5 节）；或用户将会话模型迁到 T10 |
| R8 | 无 TP-1 对照基线（23.7GB 单卡装不下） | 确定 | 中（正确性验证链变弱） | 逐 Op oracle + perplexity + 用户人工判断三重兜底 |

---

## 5. 协作协议

- **GPU 窗口**：需要 GPU 测试时我宣布"需要 GPU 窗口"，用户停 llama-server，
  我测试，用户重启。**llama.cpp 的启停只有用户做**，我绝不触碰。
- **编译**：WSL2 内由我执行，用户无需参与；编译失败由我循环修复。
- **测试回合**：用户测试 → 发日志 → 我修复 → 再交用户测，预计 2–4 个回合。
- **进度**：每完成一步更新第 6 节；上下文压缩/会话恢复后先重读本文件。

---

## 6. 进度

### Round 1（环境侦察 + 关键决策）
- [x] WSL2 可用：Ubuntu-24.04，WSL 2.7.10，内核 6.18，免密
- [x] WSL 工具链：cmake 3.28.3 / ninja 1.11.1 / gcc 13.3 / git / python3.12；**CUDA 12.8**（/usr/local/cuda，非 13.1，CMake 无版本下限，可用）
- [x] sm_120a 编译验证通过（nvcc 12.8，-arch=sm_120a 内核可运行）
- [x] WSL 内设备枚举：实测顺序 **GPU0=5060 Ti，GPU1=5060 Ti，GPU2=T10（干扰项）**（Round 1 初判顺序有误，Round 2 用 `pick_devices()` 按名确认）→ 设备选择必须按名/按白名单，TP 设备对 = (0, 1)
- [x] **P2P 实测：0↔2 双向 canAccessPeer=0（WSL2 虚拟化层拦截）** → all-reduce 主路径定为 **host-staging**（pinned 内存 + CPU 端加法，双 stream 并行 D2H/H2D）；P2P 路径保留为运行期探测的可选优化
- [x] FFmpeg dev / libcurl 已装；pkg-config 缺失（本轮安装）
- [x] pkg-config 安装；FFmpeg dev + libcurl 8.5 安装（WSL 内）
- [x] 工件：用户提供 `D:\LLM\qwen3_8_27b_nvfp4.ninfer`（23,719,715,076 字节，与 manifest 一致）
- [x] 仓库拷入 WSL `/home/zhuojun/ninfer`（63MB，排除 .git/models）
- [x] **CMake 坑 1**：CMake 默认选 `/usr/bin/nvcc`（12.0）→ 必须显式 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`（已写入 build_wsl.sh）
- [x] **CMake 坑 2**：nvcc 12.8 对 sm_120a 的 static device link 报 shared memory 超限（`0xc000 max`）→ **安装 CUDA 13.1**（与 5090 验证工具链一致），13.1 下 429/429 全量构建通过
- [x] **CMake 坑 3（关键）**：WSL2 驱动 591.59 上**动态 `libcudart.so.13` 在 CUDA 初始化时段错误**（`cudaGetDeviceCount`→`pthread_once`→WSL `libcuda`→`libnvidia-ptxjitcompiler` SIGSEGV；A/B 对照证明仅动态链接崩、静态链接过）→ 新增 CMake 选项 `NINFER_STATIC_CUDART`（默认 OFF 保持产品行为，WSL 构建 ON），5 个目标改链 `CUDA::cudart_static`
- [x] **设备映射修正**：WSL 实际顺序为 **0=5060 Ti，1=5060 Ti，2=T10**（Round 1 记录的 0/1/2 顺序有误）；`pick_devices()` 按名匹配，TP 对 = (0,1)
- [x] **Phase 1 完成**：DevicePair + all-reduce（P2P 探测 + host-staging 回退）+ ShardedTensor/shard_view/column_shard/row_shard + 单测 `ninfer_tp_device_pair_test` PASS（20480B/16B/1MiB 数值正确，move 语义验证，`p2p_available=0` 确认走 host-staging）
- [x] 全量构建（产品+测试，672/672）通过；产品 CLI 静态 cudart 确认
- [x] **Phase 2 linear（NVFP4）oracle PASS**：`ninfer_linear_tp2_split_nvfp4_test` 覆盖 5 个半尺寸 shape（[7168,5120]/[8192,5120]/[17408,5120] 列并行 + [5120,3072]/[5120,8704] 行并行），T=1/8/32 全部对照 FP64 独立参考通过
- [x] **生产切分器 `ninfer::tp::split_weight`**（`src/core/tp/weight_splitter.{h,cpp}`）：NVFP4 列切分（行块连续字节切片）+ 行切分（code 按行 strided + scale 按 tile 重映射），几何由 (n,k) 确定性推导（scale_offset=align_up(n*k/2,256)）。oracle 已重构为直接调用生产切分器 → **PASS**（验证生产代码路径）
- [x] **Phase 2 完成**：融合 Op 分解为半尺寸 linear + 本地逐元素 Op（linear_swiglu→linear+silu_mul 列并行；linear_add→linear+residual_add+all-reduce 行并行），切分逻辑在执行层；attention/GDN head 并行无需改 kernel
- 待办：Phase 3 双 Parameters 构造 + TP 执行协调器（lockstep + all-reduce）+ KV/State 池切分 + lm_head vocab 切 + Program/Engine 集成

### Round 3（工件实测修正 + Phase 3 加载/执行骨架）
- [x] **工件实测修正**（dump 工具直读 23.7GB 冷目录）：attn/GDN 输入投影是 **FP8_E4M3FN_ROW_BF16 RowScale**（非 NVFP4），且为**多 part 拼接**对象（attn: q6144+k1024+gate6144+v1024；gdn: q2048+k2048+v6144+z6144）。盲 N/2 行切会错配 head → 改为 **per-part 行 gather**
- [x] **`gather_weight_rows`**（`src/core/tp/weight_splitter.{h,cpp}`）：对融合父对象按 part 独立切半并拼接，支持 NVFP4 / FP8 RowScale / BF16 三种几何；`WeightSplitKind::GatherRows` + `TPObjectSplit.parts` 接入 materializer；`shard_geometry` 补 FP8 RowScale 分支。core 编译通过
- [x] **注册半尺寸 FP8 shape**：`n8192_k5120`（gdn qkv+z shard）、`n7168_k5120`（attn qkv+gate shard），接入 fp8_shapes.h / fp8_dispatch.cpp / sources.cmake。ops 编译通过（Phase 2 已验证同几何）
- [x] **执行层重构**：`TextContext` 抽出 `mixer_layer` / `mlp_layer`（私有）+ `single_layer`（合成）+ 公开 `tp_mixer_layer` / `tp_mlp_layer` 表面；`run_layers` 改为对 `single_layer` 的循环（TP=1 行为不变）。all-reduce 插入点 = 每层 mixer 后 + FFN 后两处
- [x] 全量构建验证（Round 4 修复后通过）
- 待办：TP 执行协调器 + 双 KV/State 池 + lm_head vocab 切 + Program/Engine 集成（`--devices` 选项）
### Round 4（FFN-only 切分设计 + 双 shard Model 加载路径）
- [x] **架构决策：FFN-only 切分（首个里程碑）**。实测发现 attn_input_proj / gdn_input_proj 两个融合 Op 在 wrapper 层硬编码全模型行数（attn: q6144/kv1024/rows14336；gdn: q2048/k2048/v6144/z6144，投影校验 + conv + 递归 state 全部写死）。head 切分需改 Op（高风险）→ attention 与 GDN mixer 全量复制到两卡（Op 零改动，KV/state 池全量），仅 FFN 切分（gate_up 列并行 [17408,5120]→[8704,5120]x2；down 行并行 [5120,17408]→[5120,8704]x2）+ lm_head 列切（vocab 切，logits all-gather）。每层 1 次 all-reduce（FFN 后；mixer 输出在两卡已完整）。显存 ~10.3GB/卡（text-only），MTP0 roofline ~52 tok/s > 28 目标。attention/GDN head 切分留作后续优化
- [x] **tp_split_spec.cpp 重写为 FFN-only**：gate_up GatherRows（2 parts）、down RowParallel、output_head ColumnParallel、其余 Replicated（含全部 mixer 权重）。删除 conv GatherCols 分支（conv 现复制）
- [x] **tp_shard_views.{h,cpp}**（新）：从 PendingWeight.reference.binding.parts + Directory 对象 shape + spec 直接构造 shard 视图（不做全量单卡 materialize——23GB 单卡装不下）；part 重指向 shard backing 的 device parent，按 split kind 半化。接入 loading_sources.cmake
- [x] **双 shard Model 加载路径**：Model 增加 make_shard_model 友元工厂；load.h 声明 materialize_model_tp2（LoadPlan 友元）；load.cpp 实现：plan_load 后 build_tp_split_spec → tp::materialize_tp2（双卡上传）→ 双 shard_views → 双 make_shard_model
- [x] **链接修复**：gather_weight_rows/gather_weight_cols 原在匿名 namespace 内（tp_materialize.cpp 不可见）→ 改名 *_impl + namespace 作用域公开 wrapper
- [x] 全量构建验证（含 ffn_delta/tp_mlp_delta + tp2 load 测试，exit 0）
- [x] **all-reduce 正确性修正（关键）**：mixer 复制 ⇒ FFN 输入残差 x 在两卡**相同**，若对完整残差做 all-reduce 会把 x 计两次。正确做法 = 仅对 **FFN delta**（行并行 down 输出，residual add 之前）做 all-reduce，求和后加回共享残差一次。新增 `ffn_delta`（ffn.{h,cpp}，写 delta 不做 residual add）+ `TextContext::tp_mlp_delta`（post-norm + ffn_delta）；每层 1 次 all-reduce（FFN delta）
- [x] **TP-2 加载冒烟测试** `ninfer_qwen3_5_tp2_load_test`（tests/models/qwen3_5/test_tp2_load.cpp）：plan_load → materialize_model_tp2 → 双 execution::Parameters + shard shape 校验（gate[8704,5120]/down[5120,8704]/head[124160,5120]/mixer 复制）。实测：host 侧全路径通过，设备上传 OOM（两卡被 llama.cpp 占 92%，free 0.9GB；设计需 ~7GB/卡，llama.cpp 停掉后 fit）
- 待办：TP 执行驱动（lockstep 双 TextContext + 每层 FFN delta all-reduce + lm_head 列切 all-gather 采样）→ 单 token forward 冒烟 → Program/Engine 集成（--devices 选项）→ MTP/dflash2 组件切分

### Round 5（TP 执行驱动 + 单 token forward 冒烟）
- [x] **TP 执行驱动 `TextContext::forward_tp2`**（text.{h,cpp}，1168 行起）：bind lambda 在双 shard 上绑定复制 mixer 状态（position 0、envelope{1,1}、batch=1/width=1、GDN state slot 0、KV table row 0，全部 Scoped* RAII 守卫）→ 双卡 embedding 同一 token → `run_layers_tp2`（lockstep mixer + 每层 FFN delta all-reduce + residual_add）→ 双卡 final rmsnorm → 列并行 lm_head 各算半 vocab partial（shard0=[0,V/2)、shard1=[V/2,V)）→ 对各自 buffer 的异侧半区 cudaMemsetAsync 清零 → `pair.allreduce` 求和得全 vocab logits（两卡相同）
- [x] **`forward_tp2_token`**：forward_tp2 后 `ops::argmax`（valid_rows=vocab）+ D2H + stream sync，返回 token id
- [x] **单 token forward 冒烟测试** `ninfer_qwen3_5_tp2_forward_test`（tests/models/qwen3_5/test_tp2_forward.cpp + tests.cmake 注册，SKIP_RETURN_CODE 77）：materialize_model_tp2 → 双 shard 最小执行状态（每卡一个极小 PagedKVCache：full-attn 层数、capacity 64、1 个 execution-table row；GDN state pool 1 slot 全零；1GiB workspace）→ 双 TextContext 卡片 → 双向 `forward_tp2_token` → 断言两 shard argmax 一致（复制 mixer + all-reduce delta + gather logits 的 shard 一致性不变量）
- [x] 全量构建 GREEN（-j 4，CUDA 13.1，NINFER_STATIC_CUDART=ON）
- 待办：**GPU 窗口**——用户停 llama-server 后跑 `ninfer_qwen3_5_tp2_forward_test --artifact /mnt/d/LLM/qwen3_8_27b_nvfp4.ninfer` 验证单 token forward 数值正确性 → Program/Engine 集成（--devices 选项，TP prefill/decode 变体）→ MTP/dflash2 组件切分

### Round 6（OOM 根因修复：per-shard arena 容量）
- [x] **OOM 根因**：`materialize_tp2` 用**全模型** `plan.device_capacity_bytes`（~23.7GB）初始化每个 shard 的 `DeviceArena`，而 `DeviceArena(capacity)` 构造即 `cudaMalloc(capacity)` → 每卡多占 ~13GB，16GB 卡必然 OOM（即使 llama.cpp 已停、两卡空闲）。修复：新增 `object_shard_bytes`（按 spec 解析每对象 shard 字节：Replicated 计全量，GatherRows/GatherCols 计 parts 半量之和，ColumnParallel 计 n/2，RowParallel 计 k/2，均经 `shard_geometry` 解析几何）+ `shard_capacity_bytes`（全 plan 求和 + 每对象 256B 对齐余量），`tp_init` 改传 per-shard 容量（估算 ~10.5GB/卡 + 测试 1GiB workspace ≈ 11.5GB < 16GB）。构建 GREEN
- 待办：用户停 llama-server 后重跑单 token forward 冒烟（验证容量修复 + forward 数值正确性）→ Program/Engine 集成 → MTP/dflash2 组件切分

### Round 7（OOM 第二根因：当前设备语义 + forward 驱动设备绑定）
- [x] **第二根因（用户观察：GPU2 冲高 80% 回落、GPU0 不动）**：`DeviceArena(capacity)` 的 `cudaMalloc` 落在**当前设备**上；`materialize_tp2` 连续两次 `tp_init` 未切换设备（DevicePair 构造最后绑定 device1）→ 两个 shard 的权重 arena 全部分配到 GPU2。修复：每次 `tp_init` 前 `deviceN.bind_to_current_thread()`
- [x] **forward 驱动设备绑定**（text.{h,cpp}）：`run_layers_tp2` 每次切到 peer 卡操作前绑定（mixer/delta alloc/mlp_delta/residual_add 各 6 处）；`forward_tp2` 重写——原 bind lambda 的 RAII 守卫在 lambda 返回时即销毁（绑定未生效），现提升为函数作用域（make_bind 返回 BindState + 16 个 Scoped* 守卫存活到 forward 结束）；移除首尾 `work_.reset()`（会覆盖调用方 logits 缓冲）；每步操作前绑定对应设备；`forward_tp2_token` 在 argmax 前绑定本卡（forward_tp2 返回后当前设备停在 peer 卡）
- [x] **测试 build_shard_state**：每卡 3 个 arena（kv/state/workspace）创建前 `device.bind_to_current_thread()`
- [x] 构建 GREEN
- 待办：用户停 llama-server 后重跑单 token forward 冒烟（预期两卡各 ~11.5GB，输出 argmax_shard0==argmax_shard1）→ Program/Engine 集成 → MTP/dflash2 组件切分

### Round 8（切分器补 FP8 RowScale 行切：output_head）
- [x] **新报错 `weight splitter: unsupported qtype` 根因**（dump7 实测）：`text/output_head` 是 **fp8_e4m3fn_row_bf16[248320][5120]**（非 NVFP4），spec 标 ColumnParallel（vocab 行切半）走 `split_weight`，而 split_weight 只支持 NVFP4/BF16。down 是 NVFP4（Phase 2 已验证）、gate_up 走 GatherRows，唯一漏的是 output_head 的 FP8 行切
- [x] **`slice_fp8_rows`**（weight_splitter.cpp）：FP8 RowScale 行切 = code 平面（n×k 字节，每元素 1B）+ scale 平面（n×2 字节，每行 1 BF16）两段连续拷贝；shard payload 按 `shard_geometry` RowScale 布局对齐（code 后 256B 对齐再放 scale），与上传几何一致。`split_weight` 增加 FP8_E4M3FN_ROW_BF16 分支（仅 ColumnParallel，n 须为偶数；248320/2=124160 满足）
- [x] **lm_head 半尺寸 FP8 shape 注册**（主动排查，避免下一个 `fp8 linear: unsupported shape`）：`forward_tp2` 的列并行 lm_head 用 [124160,5120] FP8 权重走 `fp8_dispatch`，而 dispatch 表只有全尺寸 kFp8N248320K5120。新增 `shapes/n124160_k5120.cu`（照 n14336_k5120.cu 的通用 chunk launcher：GEMV@T=1 + SIMT≤11 + A8，Geometry=Fp8Geometry<124160,5120>）+ fp8_shapes.h 声明 + dispatch 表 + sources.cmake。NVFP4 侧 FFN 半尺寸 shape（kNvfp4N17408K5120 gate/up、kNvfp4N5120K8704 down）Phase 2 已注册，mixer 复制走全尺寸 shape，无需新增
- [x] 构建 GREEN
- 待办：用户停 llama-server 后重跑单 token forward 冒烟 → Program/Engine 集成 → MTP/dflash2 组件切分

### Round 9（混合精度 FFN：FP8 列切 + 半尺寸 shape + ffn_delta 解耦融合 swiglu）
- [x] **新报错 `FP8 row-scale supports row splits only` 根因**（dump8/9/10 实测）：text 组件 `num_hidden_layers=64`（非 48），主模型 64 层里前 56 层 FFN 是 NVFP4、后 8 层（56-63）FFN 是 **FP8 RowScale**（混合精度转换）；dflash2 是独立 5 层 draft。spec 按**形状**匹配（`n==hidden && k==intermediate`→RowParallel），把 8 个 FP8 down [5120,17408] 也标成 RowParallel，而 split_weight 的 FP8 分支只允许 ColumnParallel → 抛错。设计本意是所有 64 层 FFN 都切分，需补 FP8 列切
- [x] **`slice_fp8_cols`**（weight_splitter.cpp）：FP8 RowScale 列切 = code 平面逐行 strided 拷贝（k 字节/行，取 [col_begin, col_begin+col_count)）+ scale 平面整拷（n×2 字节，每行 1 BF16 与列无关）；shard payload 按 `shard_geometry` RowScale 布局对齐。`split_weight` FP8 分支扩展：ColumnParallel→`slice_fp8_rows`（output_head）、RowParallel→`slice_fp8_cols`（FFN down）
- [x] **FP8 半尺寸 shape 注册**（避免下一轮 `fp8 linear: unsupported shape`）：`shapes/n17408_k5120.cu`（gate/up shard，Geometry=Fp8Geometry<17408,5120>）、`shapes/n5120_k8704.cu`（down shard，Fp8Geometry<5120,8704>），照 n14336/n5120_k17408 的通用 chunk launcher；fp8_shapes.h 声明 + dispatch 表 + sources.cmake。NVFP4 侧对应 shape Phase 2 已注册
- [x] **`ffn_delta` 解耦融合 swiglu**（更深的阻塞点，主动排查）：融合 `linear_swiglu` 的 FP8/NVFP4 内核都**硬编码全尺寸几何** `N34816K5120`（decode.cu 校验 `weight.n != 34816` 即抛错），shard gate_up [17408,5120] 会在第 0 层就挂。改成 shape 泛化的 `linear`(gate_up) + `silu_mul`（与 `ffn` 的 mtp 分支同构）；`ffn_delta` 仅被 TP-2 的 `tp_mlp_delta` 调用，TP=1 路径（`ffn`）不变，字节一致性保持
- [x] 构建 GREEN
- 待办：用户停 llama-server 后重跑单 token forward 冒烟（预期两卡各 ~11.5GB，argmax_shard0==argmax_shard1）→ Program/Engine 集成 → MTP/dflash2 组件切分

### Round 10（shard_views 向量对象：矩阵形状检查误伤 1-D 权重）
- [x] **新报错 `shard view: weight object is not a matrix` 根因**（tp_shard_views.cpp:24）：`shard_views` 对**每个** pending weight 的 part 无条件调用 `object_shape`（要求 `shape.size()==2`），但 norms/biases 是 1-D 向量（`input_norm`/`post_attention_norm` [5120]、`final_norm` [5120]、`a_log`/`dt_bias` [48]、`gdn/norm` [128] 等）。这些对象是 Replicated，其分支只拷贝 `begin/end`、用 `reference.shape`，根本用不到 `n/k`——矩阵检查对它们是多余的，却先于 kind 分发执行了
- [x] **修复**：把 `object_shape` 调用从 part 循环顶部挪到非 Replicated 分支内（`else` 块首行）。Replicated 分支（norms/biases/mixers/embedding，含 1-D 向量）不再查询矩阵形状；只有 GatherRows/RowParallel/ColumnParallel（全是 2-D 矩阵）才查。`view.parts.push_back` 保持在 part 循环层级（Replicated 与切分对象都要登记 region）
- [x] 构建 GREEN
- 待办：用户停 llama-server 后重跑单 token forward 冒烟 → Program/Engine 集成 → MTP/dflash2 组件切分

### Round 11（shard_views GatherRows 映射：up 块越界）
- [x] **新报错 `text/layers/0/mlp/gate: projection region exceeds its parent or logical matrix` 根因**（weight_input.cpp:26）：报错名是 gate，实际越界的是同对象的 **up** part。`shard_views` GatherRows 映射 `row_begin = cum + shard*(block/2)` 错误——gather 布局里 shard payload 按 part 顺序拼接各 part 的"本 shard 半行"（gather 时已用 `rb = row_begin + shard*(row_count/2)` 选好行），两 shard 块结构相同，块在 shard 行空间的偏移就是 `cum`（前面 part 半行之和），与 shard 无关。多加 `shard*(block/2)` 使 shard 1 的 up 块 row_begin=17408，超出 shard 父 [17408,5120] 的元素域
- [x] **修复**：`row_begin = cum`（tp_shard_views.cpp）。gate 块 cum=0、up 块 cum=8704，两 shard 一致，region 合法
- [x] 构建 GREEN
- 待办：用户停 llama-server 后重跑单 token forward 冒烟 → Program/Engine 集成 → MTP/dflash2 组件切分

### Round 12（稳态 OOM：workspace 1GiB 过大 + 权重 arena 实测 14.06GiB）
- [x] **现象**：两卡稳态 ~14.2GB 且任务管理器无波动，跑约 11 个 probe 后 `cudaMalloc failed: out of memory`。稳态过高顶破 16GB 上限，forward 的临时分配（logits 等）把余量挤爆，不是某次大分配尖峰
- [x] **探针 dump11 实测每 shard 权重 arena**（tools/tp_bootstrap/dump11.cpp，复刻 object_shard_bytes/shard_geometry 逻辑，speculative=None）：device_objects=659；Replicated 531 个 = 9358 MiB（mixer 投影 + token_embedding + norms，FFN-only 切分下不可再降）；GatherRows 64 个 = 3357 MiB（gate_up 半切）；RowParallel 64 个 = 1678 MiB（down 半切）；**合计 14.06 GiB**。dflash2/mtp 未进 plan（speculative=None 时 config.draft/mtp 不置位，config.cpp:321/333 确认）
- [x] **根因**：稳态 = 权重 arena 14.06GiB + workspace 1GiB + GDN state ~83MiB + KV ~3MiB ≈ 15.15GiB > 16GB 卡实际可用（扣驱动/context ~15GiB）。单 token forward 峰值 workspace 仅 ~600KB（logits [248320,1] BF16 = 496KB + 若干 [N,1] 张量），1GiB 严重过大
- [x] **修复**：test_tp2_forward.cpp workspace 1GiB → 256MiB（单 token 冒烟绰绰有余），并打印 kv/gdn/workspace 三 arena 的 MiB 数便于下次诊断。预期稳态 ≈ 14.06 + 0.25 + 0.08 + 0.003 ≈ 14.4GiB，留出 ~0.6GiB 余量
- [x] 构建 GREEN（/tmp/full_build18.log）
- 待办：用户停 llama-server 后重跑单 token forward 冒烟（预期两卡 ~14.4GB，11 probe 全过）→ 若仍 OOM 则需进一步压缩（候选：mixer 切分 / embedding 半切，高风险，最后手段）→ Program/Engine 集成 → MTP/dflash2 组件切分

### Round 13（shard_geometry 漏更新 padded_columns：RowParallel down 校验失败）
- [x] **新报错 `text/layers/0 verify columns=1: nvfp4 linear: invalid NVFP4 weight` 根因**（nvfp4_format.cpp:57/64）：`shard_geometry`（tp_materialize.cpp）做 `WeightGeometry g = full;` 后只更新 shape/elements/planes，**漏更新 `padded_columns`**（和 FP8 RowScale 的 `group_size`）。RowParallel down [5120,17408]→[5120,8704] 的 shard 父 padded_columns 仍是 17408，`native_weight`（weight_view.cpp:270）据此设 `padded_shape[1]=17408`，而 `k=8704` → 校验 `padded_shape[1] != k` 抛错。Gate_up [17408,5120] 因 k 不变而通过（所以报错在 down 而非 gate_up，但错误信息带的是整个 layer 的 verify 上下文）
- [x] **修复**：`shard_geometry` 里 `g.padded_columns = k`；FP8 RowScale 额外 `g.group_size = k`（与 weight_geometry 的 RowScale 分支一致，weight_view.cpp:131）。对 NVFP4/FP8/BF16 的 ColumnParallel/GatherRows（k 不变）和 RowParallel（k 减半）都正确
- [x] 测试加了 layer-0 gate_up/down 的 Weight 字段诊断打印（[diag] 行），若再报错可定位具体字段
- [x] 构建 GREEN（/tmp/full_build18.log）
- 待办：用户停 llama-server 后重跑单 token forward 冒烟（预期两卡 ~14.4GB，11 probe 全过）→ Program/Engine 集成 → MTP/dflash2 组件切分

### Round 14（GDN output 投影被误判为 RowParallel：复制混合器读半张权重 OOB）
- [x] **新报错 `fp8_linear_add_decode.cu:29: cudaErrorIllegalAddress` 根因**：崩溃在 **GDN output 投影**（`linear_add`，几何 N5120K17408），且**没有任何 `[tp2]` 标签打印**（说明故障在 local mixer 内部、首个 sync 之前）。`build_tp_split_spec`（tp_split_spec.cpp）用形状规则 `n == hidden && k == intermediate` 判定 FFN down 为 RowParallel，但 **GDN output 投影形状也是 `[hidden, value_width]`，当 `value_width == intermediate`（本模型 17408）时与 FFN down 完全同形** → 被误判为 RowParallel，shard 权重被切成 `[5120,8704]`；而 GDN 混合器是**复制**的（`text.cpp:1059` 的 `linear_add` 按完整 `[5120,17408]` 读权重）→ 越界 → 非法地址。attention output 是 `[h, query_width]`（≠17408），故 GDN output 是唯一另一个 `[hidden, intermediate]` 权重
- [x] **修复**：RowParallel 判定改为 `n == hidden && k == intermediate && has_name("mlp/down")`；同形但名字是 `gdn/output` 的 GDN output 投影保持 **Replicated**。FFN down 绑定名是 `mlp/down`（load/text.cpp:29），GDN output 是 `gdn/output`（load/text.cpp:53），名字区分可靠
- [x] 临时插桩（待删）：`run_layers_tp2`（text.h）layer-0 四段 sync 标签（local/peer mixer、local/peer ffn）+ `ffn_delta`（ffn.cpp）三段 sync 标签（gate_up linear / silu_mul / down linear），用于把延迟 fault 定位到具体 kernel；冒烟通过后移除
- [x] 构建 GREEN（/tmp/full_build18.log）
- 待办：用户停 llama-server 后重跑单 token forward 冒烟（预期两卡 ~14.4GB，11 probe 全过）→ 删插桩 → Program/Engine 集成 → MTP/dflash2 组件切分

### Round 15（peer 侧用了 card0 的权重指针：跨设备访问 OOB）
- [x] **新报错 `text.h:309 (peer mixer sync): cudaErrorIllegalAddress` 根因**：`run_layers_tp2` 里 `block = parameters_.text.layers[layer]` 是 **card0** 的 BlockParameters，peer 调用 `peer.tp_mixer_layer(block, x_peer, ...)` 用的是 **card0 的权重指针**（dev0 显存）在 dev1 的 stream 上跑 → WSL2 无 P2P，跨设备指针访问 → 非法地址。证据：`[tp2] local mixer`（card0 权重 on card0 stream）干净打印，`[tp2] peer mixer`（card0 权重 on card1 stream）fault。`forward_tp2` 同样问题：peer 的 embedding/rmsnorm/lm_head 全用 card0 的 `*embed_`/`*final_norm_`/`*lm_head_`
- [x] **修复**：`run_layers_tp2` 加 `const auto& block_peer = peer.parameters_.text.layers[layer];`，peer 的 `tp_mixer_layer`/`tp_mlp_delta` 改用 `block_peer`；`forward_tp2` 的 peer embedding/rmsnorm/lm_head 改用 `*peer.embed_`/`*peer.final_norm_`/`*peer.lm_head_`（lm_head 是 ColumnParallel，card1 用上半 vocab 行）。每个 shard 的 Op 必须用自己 shard 的权重指针
- [x] 构建 GREEN（/tmp/full_build18.log）
- 待办：用户停 llama-server 后重跑单 token forward 冒烟（预期两卡 ~14.4GB，11 probe 全过）→ 删插桩 → Program/Engine 集成 → MTP/dflash2 组件切分

### Round 16（prompt attention kernel 的 smem opt-in 只设在当前设备：peer 卡 launch 被拒）
- [x] **新报错 `causal_cache/prompt.cu:59: cudaErrorInvalidValue` 根因**：layer 0（GDN）两卡全过后，崩在第一个 full-attention 层（layer 3）的 **peer** mixer。smem 诊断：5060 Ti `smemPerBlock=49152`（48KB 默认）、`smemPerBlockOptin=101376`（99KB opt-in）。prompt kernel 需 **98304B（96KB）动态 smem**，超默认 48KB、在 opt-in 99KB 内 → 必须 opt-in。但 `prompt.cu` 用 `static const cudaError_t attr_bf16 = cudaFuncSetAttribute(...)`，static 只在**进程内首次**调用时执行，而 `cudaFuncSetAttribute` 配置的是**当前设备**的 kernel 副本 → 首次是 local（dev0）mixer，dev1 的副本没设 opt-in → 仍 48KB 默认 → peer 的 96KB launch 被拒 `cudaErrorInvalidValue`。GDN kernel（state_passing/output）是**每次调用**设属性（非 static），所以两卡都过——prompt 的 `static const` 是异类
- [x] **修复**：`prompt.cu` 的 `attr_bf16`/`attr_i8` 去掉 `static`，改为每次 launch 都 `cudaFuncSetAttribute`（幂等，per-call 开销可忽略），匹配 GDN 的 per-call 模式。这样无论当前设备是 dev0 还是 dev1，其 kernel 副本都会被设 opt-in
- [x] `prompt.cu` 加入 build_phase4.sh 同步列表
- [x] 构建 GREEN（/tmp/full_build18.log）
- [x] 用户停 llama-server 后重跑：layer 0（GDN）+ 全部 64 层 FFN 两卡全过（128 个 `down linear`），崩点推进到 lm_head 投影

### Round 17（lm_head 与 token_embedding 权重绑定：ColumnParallel 假设错误）
- [x] **新报错 `linear: expected [K,T] x [N,K] -> [N,T]` 根因**：64 层 FFN 全过后，崩在 final rmsnorm 之后的 lm_head 投影。`dump12` 探针证实 `text/output_head` 与 `text/token_embedding` 是**同一物理对象 obj1**（weight-tied，elements=1271398400=248320×5120）。obj1 是 resource 对象（非 2-D TensorObject）→ `build_tp_split_spec` 跳过它 → 无 split 条目 → `shard_views` 默认 **Replicated**。于是每个 shard 的 lm_head 权重是**完整 [248320, 5120]**，但 `forward_tp2` 按 ColumnParallel 处理（期望 n=124160）→ `out.ne[0](124160) != w.n(248320)` → 精确命中该 throw
- [x] **修复**：lm_head 绑定到复制的 embedding，每 shard 持完整 [vocab, hidden] 权重；hidden_out 两卡一致（复制 mixer + all-reduce 后的 FFN delta），故每 shard 独立计算**全词表** logits，两卡精确一致，**无需 all-reduce**。`forward_tp2` 的 lm_head 段改为 `project(hidden_out, *lm_head_, logits, ...)` / `project(hidden_out_peer, *peer.lm_head_, logits_peer, ...)`，删除 zero-pad + all-reduce。全 head FP8 shape n248320_k5120 已注册（TP=1 同款）
- [x] 构建 GREEN（/tmp/full_build18.log）
- [x] 用户重跑：forward **端到端跑通**（lm_head 修复生效，不再崩），但 11-probe 校验分歧：`input_token=151643` 时 `argmax_shard0=198`、`argmax_shard1=151643`（shard1 回到输入 token）

### Round 18（两卡 logits 分歧定位：加 per-layer x 比较诊断 + 修诊断 race）
- [x] **结构分析**：TP-2 残差流 `x` 本身**不做 all-reduce**，只有 FFN delta 做。归纳法：只要 (a) 每层复制 mixer 两卡逐位相同、(b) all-reduce 正确，`x` 就保持逐位一致。已核对 `DevicePair::allreduce`（host-staging）实现无误（两卡都拿到 a+b）
- [x] **诊断 v1 结果**（`run_layers_tp2` 每层 mixer 后 / residual_add 后比较两卡 x 最大绝对差，仅 layer 0-3）：layer 0-2（GDN）post-mixer+post-residual **全 0**（GDN mixer 两卡逐位一致、FFN all-reduce 正确）；**layer 3（第一个 full-attention）post-mixer=0 但 post-residual=0.265625（at 3456）**，且数据相关（probe1=151643 出现、probe2=0 不出现）
- [x] **诊断 v1 的可靠性隐患（已修）**：`cmp_x` 用同步 `cudaMemcpy`（走 legacy 默认流），但 `residual_add` enqueue 在 `ctx_.stream`（`cudaStreamNonBlocking`，device.cu:64）。**non-blocking 流与 legacy 默认流无隐式同步** → post-residual 读取可能与 residual_add 竞争，读到进行中的混合新旧值。layer 3 的 0.265625 很可能是**诊断假象**（数据相关=时序相关，正是 race 特征）。逻辑上 post-mixer x 相同 → h 相同 → 两卡各算不同局部 delta（正常）→ all-reduce 后两卡拿到相同的和 → residual_add 后 x 必相同，故可靠的 post-residual 必为 0
- [x] **诊断 v2**：`cmp_x` 读取前先 `cudaStreamSynchronize` 两流（消除 race）；打印**全部 64 层**（layer 0-3 恒打印，其余仅非零时打印）。post-mixer 读取本就可靠（sync0/sync1 已同步两流），post-residual 现在也可靠
- [x] `Tensor::bytes()` 核对正确（delta [5120,1] BF16 = 10240B，all-reduce 覆盖全 buffer），排除"只覆盖部分"假设
- [x] 构建 GREEN（/tmp/full_build18.log）
- [x] **诊断 v2 结果（全 0）+ 复查发现 v2 自身 bug**：用户重跑后 layer 0-3 的 post-mixer/post-residual **全 0**，但 argmax 仍分歧（198 vs 151643）。复查 `cmp_x` v2 发现**跨设备 memcpy bug**：`peer.ctx_.bind_to_current_thread()` 之后当前设备一直是 device1，随后 `cudaMemcpy(a, x.data, D2H)` 读的是 device0 的 `x`——同步 cudaMemcpy 按**当前设备**解析设备指针，WSL2 无 P2P → `a` 读到跨设备垃圾值，"全 0"不可信。v1 的 memcpy 设备绑定是对的（先绑 device0 拷 x、再绑 device1 拷 x_peer），只是 post-residual 没同步流（有 race）
- [x] **逻辑矛盾**：若 x 在 layer 63 后真逐位一致，则 final rmsnorm（确定）→ lm_head（两卡全权重、同输入）→ logits 必逐位一致 → argmax 必相同。实际 argmax 不同 → 要么 x 没逐位一致（v2 bug 假 0），要么 final 路径有 bug
- [x] **诊断 v3**：修 `cmp_x` 设备绑定（每个 memcpy 前绑对应设备，消除跨设备读）；并在 `forward_tp2` final 路径加 `cmp_final`（rmsnorm 后比较 `hidden_out`、lm_head 后比较 `logits`，打印 `[tp2-final] hidden_out/logits max_abs_diff=… at …`）。一次重跑即可区分"分歧在某层"（某 `[tp2-diff]` 非零）还是"分歧在 final 路径"（`[tp2-final]` 非零而所有 `[tp2-diff]`=0）
- [x] 构建 GREEN（/tmp/full_build18.log）
- [x] **诊断 v3 结果（决定性）**：用户重跑后**所有 64 层 `[tp2-diff]` = 0**、**`[tp2-final] hidden_out = 0`、`logits = 0`**（两次 forward 都是）→ **单次 forward 内两卡 logits 逐位一致，TP-2 前向计算正确**；但 `argmax_shard0=198`、`argmax_shard1=151643` 仍不同
- [x] **argmax 确定性核对**：tiled-atomic 路径用 `argmax_better`（value 降序、index 升序）原子竞争，是全序 → 固定点唯一、与原子解析顺序无关 → logits 相同则 argmax 必相同，排除 argmax 非确定
- [x] **根因 = 测试 bug（GDN 状态未重置）**：测试对每个 probe 跑两次独立 forward（card0-local、card1-local），但 GDN 状态池只在 build 时清零**一次**——第一次 forward 就地更新 GDN 状态，第二次 forward 从**脏状态**开始 → 两次 forward 的 logits 不同（L1≠L2）→ argmax(L1)≠argmax(L2)。每次 forward 内部两卡始终一致（诊断已证），脏状态只让两次 forward 之间不同。151643=输入 token 正符合"脏状态让 mixer 输出变小、残差被 embedding 主导、lm_head 映回输入 token"的特征。测试注释声称 fresh state 但实际没重置
- [x] **修复**：`ShardState` 增加 `state_backing`（DeviceSpan）；probe 循环里每次 forward 前 `reset_state()` 重新清零两卡 GDN 状态（bind 对应设备 + cudaMemset）。KV 无需重置（position 0 时 bind 状态 kv_table_rows=0，attention 只 attend 自身，k/v 从 x 重算）
- [x] 构建 GREEN（/tmp/full_build18.log）
- [x] **用户重跑通过（决定性）**：11 个 probe 两卡 argmax 全部一致（151643→198/198、0→49276/49276、…、248319→177801/177801），176 条 `[tp2-diff]` 全 0、44 条 `[tp2-final]` 全 0，结尾 `TP-2 single-token forward passed: all 11 probes shard-consistent` → **TP-2 单 token forward 冒烟里程碑达成**
- [x] **插桩全部清除**：ffn.cpp sync lambda+调用+iostream；text.h run_layers_tp2 的 sync0-3/cmp_x/dbg+`<cmath>`/`<cuda_bf16.h>`；text.cpp cmp_final+两处调用；test smem 打印块+gate_up/down/output_head [diag] 块（pick_devices 的 cudaDeviceProp 是合法设备选择，保留）。构建 GREEN
- [x] **架构判断（关键）**：Engine 的 Program 深度绑定单卡（CUDA Graph decode、KV、context cache、MTP 全单卡机制），而 TP-2 的 host-staging all-reduce 与 exact-batch CUDA Graph **根本不兼容** → 完整 Engine `--devices` 集成面巨大且需旁路 CUDA Graph。近期更有价值且是前置的是**有状态 TP-2 decode**（验证多步 GDN/KV 演进下两卡仍逐位一致）
- [x] **`forward_tp2` 泛化为有状态**：接受 `position` 参数（默认 0，现有单 token 冒烟不变）。`make_bind` 把 cache/rope position 设为 `position`、envelope 设为 `{position+1, position+1}`（镜像 Program ordinary decode 约定 decode.cpp:307-331），KV row 0 + GDN state slot 0 不变。复制 mixer 会把当前 token 的 k/v 写进各自 paged cache、原地更新 GDN state（`tp_mixer_layer`=`mixer_layer`，与单卡 `single_layer` 同路径）→ 状态跨步累积
- [x] **`forward_tp2_token` 内部双卡 argmax 比较**：对两卡 logits 都 argmax，不一致即 `throw std::logic_error`（把 shard 分歧变成采样边界硬失败）。decode 每步只调一次 `card0.forward_tp2_token`（lockstep 驱动两卡 + 同时更新两卡状态），无需在两次 forward 间清零（区别于单 token 冒烟的独立 forward）
- [x] **测试加 32 步自回归 decode**：`ShardState` 加 `kv_lease`（持有物化页所有权）；`publish_kv_row` 对每卡 materialize_one 一页 + `publish_repeated(row0, page, 1, device->stream)`（H2D 拷贝排到 device stream，因 attention kernel 在 non-blocking stream 上不与 legacy stream 隐式同步）+ `cudaMemsetAsync` 清零 GDN state 排到 device stream；decode 循环 position 0..31 每步 `card0.forward_tp2_token(card1, pair, token, step)`，token 用上一步采样结果
- [x] 构建 GREEN（/tmp/full_build18.log）
- 待办：Program/Engine 集成（--devices，旁路 CUDA Graph 走有状态 TP-2 decode）→ MTP/dflash2 组件切分 → 性能对比基线
- [x] **Round 21 测试 PASSED**：11 probe 全部恢复 Round 19 真实值（198/49276/5328/14/220/381/264/1196/98142/149022/177801，两卡逐位一致）；32 步有状态自回归 decode 每步两卡 argmax 一致；结尾 `passed`。Round 20 的两个回归 bug（allreduce 跨 stream race + materialize_one 空 reservation）确认修复
- [x] **decode 重复模式核查 = 非 bug**：step 14–23 ≈ step 0–9 的循环是模型对"单 token 提示、无上下文"的真实退化行为。证据：(a) page size=64 tokens，单 page 容纳全部 32 步，无需增长；(b) attention 走与单卡生产 decode 相同的已验证 `causal_softmax_attention` op，每步写 cache_position 列、读 0..position 列；(c) **铁证** step 10 输入 2782→输出 5877，step 24 输入 2782→输出 12796——同输入不同历史长度→不同输出，证明 GDN 循环状态 + GQA KV 历史都在累积并被读取。两卡每步逐位一致才是 TP-2 正确性判据，已满足
- [x] **TP-2 有状态自回归 decode 里程碑达成**：lockstep 双 TextContext + 每层 FFN 后 all-reduce（跨 stream 显式同步）+ 复制 lm_head 独立全词表 argmax + 有状态 paged KV/GDN 累积，全部正确
- [x] **Round 20 首跑 FAIL 分析（两个独立 bug）**：(a) 11 probe argmax 全 0（Round 19 是真实值 198/49276/…）；(b) decode 段 `materialize_one` 抛 `Paged KV single-page materialization exceeds reservation`
- [x] **根因 (a) = 删插桩删掉了承重的 sync**：`DevicePair` 拥有**自己的** `DeviceContext a_/b_`（构造时新建，各有独立 non-blocking stream），`allreduce` 把 D2H 拷贝排到 `a_.stream`/`b_.stream`；而 FFN kernel 跑在 TextContext 的 `ctx_.stream` 上。**两条独立 non-blocking stream 无隐式同步** → D2H 在 down-linear 完成前读 delta → 陈旧/垃圾 delta → 错误 logits。Round 19 通过是因为 ffn.cpp 的 `sync("down linear")`（`cudaStreamSynchronize(ctx_.stream)`）承重——保证 delta 完成后才调 allreduce；Round 20 删插桩时当纯诊断删了，race 暴露。两卡读到相同陈旧数据（allreduce 求和后一致）→ argmax 仍相同不 throw，但值错（全 0）。这也解释 Round 19 日志 8000+ 行（11 probe × 2 forward × 64 层 × 3 条 `[ffn_delta]` sync 打印）
- [x] **修复 (a)**：`run_layers_tp2` 在 `pair.allreduce` 前显式 `cudaStreamSynchronize(ctx_.stream)` + `cudaStreamSynchronize(peer.ctx_.stream)`（bind 对应设备）。这是 TP-2 协议的正确位置——跨 stream exchange 前生产者必须完成。host-staging allreduce 本身已是 per-layer barrier（内部 sync 两条 pair stream），所以不增加额外串行化
- [x] **根因 (b) = `materialize_one` 要求非零 reservation**：`make_empty_reservation()` 返回 0 页 reservation，`materialize_one` 检查 `reservation.pages_ == 0` 即抛。Program 路径先 `reserve(n)` 再 materialize
- [x] **修复 (b)**：测试 `publish_kv_row` 先 `pool.reserve(1)`（返回 `std::optional<DeviceKVPageReservation>`，move-only——`const` optional 上 `std::move(*reserved)` 退化拷贝被删，须去 const）再 `materialize_one`
- [x] 构建 GREEN（/tmp/full_build18.log）
### Round 22（Program/Engine 集成：TP2GenerationCore + --devices CLI）
- [x] **新增 `TP2GenerationCore`**（src/runtime/engine/tp2_generation_core.{h,cpp}）：Engine Core 的第三个 variant 分支，拥有两个独立 DeviceContext（shard_a_/shard_b_）+ DevicePair + 各自独立的 paged KV / GDN state / workspace arena / TextContext。submit() 创建 OutputSession + GenerationBudget，wait() 驱动有状态 prefill（逐 token forward_tp2）+ 自回归 decode（forward_tp2 + ops::sample + OutputSession preview_model/commit_preview），两卡 lockstep、每层 FFN 后 all-reduce（run_layers_tp2 内置跨 stream 同步）。lm_head 复制 → 每 shard 独立全词表 logits → 采样只用 shard_a_（两卡 logits 逐位一致，已证）。memory_summary/runtime_stats/reset_memory_peaks/is_available 全部实现
- [x] **Engine::Impl 重构**：Core variant 加第四分支 unique_ptr<TP2Core>；Impl ctor 按 device_b>=0 分支——TP-2 时不构造单卡 device/ModelInstance，而是 make_unique<TP2Core> 并从其 frontend() 拷贝 Frontend 到 impl_->frontend_；单卡路径不变。所有前端路由方法（prepare/prepare_tokens/tokenize_text/count_tokens/prompt_capabilities/media_cache_summary）改走 impl_->frontend_ + impl_->capacity（TP-2 时 active==nullptr，capacity=options.max_context）。析构器只在 active!=nullptr 时同步 device
- [x] **EngineOptions.device_b**（include/ninfer/types.h）：默认 -1（单卡），>=0 启用 TP-2。normalize_engine_options 加 TP-2 分支（concurrency=1、pending=1、kv_capacity=explicit(max_context)、speculative={}、vision=false、cuda_graph=false、cache disabled）
- [x] **--devices A,B CLI**（src/serve/serve_options.{h,cpp}）：ServeOptions.device_b 默认 -1；--devices 解析逗号分隔两设备索引；generation_service.cpp 把 device_b 传入 EngineOptions
- [x] CMake：src/runtime/CMakeLists.txt 加 tp2_generation_core.cpp
- [x] 构建 GREEN（/tmp/full_build_tp2.log）
- 待办：用户停 llama-server 后用 --devices 0,2 启动 ninfer-serve 实测（预期两卡各 ~14.4GB，能正常对话）→ MTP/dflash2 组件切分 → 性能对比基线
### Round 23（TP-2 启动 OOM：workspace 512MiB 过大 + KV 384MiB）
- [x] **现象**：用户停 llama-server 后 `--devices 0,2 --max-context 8192` 启动，两卡显存平稳加载到 ~14.4GiB（权重 arena）后，某次 cudaMalloc 失败 → 进程退出、显存归零（1m52s 后 FATAL）
- [x] **根因**：TP2GenerationCore 每 shard 比已验证的单 token 测试多占两块——workspace 512MiB（测试 256MiB，实测单 token forward 峰值仅 ~600KB）+ KV cache 384MiB（8192 上下文 = 128 页 × 3MiB/页；测试只测 64 token = 3MiB）。合计 14.06+0.51+0.08+0.38 ≈ 15.03GiB > 16GB 卡实际可用（~15GiB）→ KV arena 的 cudaMalloc 失败
- [x] **修复**：kWorkspaceBytes 512MiB → 128MiB（峰值 600KB，200 倍余量）。每 shard ≈ 14.8GiB，留 ~0.2GiB 余量
- [x] 构建 GREEN（/tmp/full_build_tp2.log）
- 待办：用户重跑 --devices 0,2 实测 → 若仍 OOM 则压缩 KV（--max-context 4096 = 192MiB）或 mixer 切分（高风险，最后手段）→ MTP/dflash2 组件切分 → 性能对比基线
### Round 25（TP-2 启动失败：tokenizer_config.json 空输入）
- [x] **现象**：OOM 修复后用户把工件移到 /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer 重跑，29.7s 权重加载完成后 FATAL：malformed tokenizer_config.json: parse error ... attempting to parse an empty input
- [x] **排除工件损坏**：新路径字节数 23719715076 与原 /mnt/d 路径完全一致；权重正常加载 → 工件完整
- [x] **根因（TP-2 资源悬垂指针）**：FrontendResources 的 6 个字段是 std::string_view，指向 load plan 的 host 资源字节（binder host_object → MaterializationPlan.host_objects）。单卡路径 artifact::materialize 把 plan 的 host 字节 move 进每个 Model 的 backing_（materializer.cpp:169），地址不变 → string_view 有效。TP-2 路径 tp::materialize_tp2 只迭代 plan.device_objects 构建两个 shard backing（纯 device arena），host 字节留在 plan 里；materialize_model_tp2 返回时 plan（data）析构 → host 字节释放 → shard Model 的 resources_ string_view 悬垂 → make_frontend 解析 tokenizer_config.json 读到空。单卡路径与 Round 21 测试从未构建 Frontend，故从未暴露
- [x] **修复**：FrontendResources 六个 string_view 字段改为 std::string（资源.h），Model 自持资源字节，与架构"immutable Model data owns resources"一致。bind_resources 的 string_view 赋值隐式转 string；make_shard_model 按值拷贝 → 深拷贝。单卡路径同样受益（不再依赖 backing 地址稳定性）。无 src 内对真实 FrontendResources 的聚合初始化；test_frontend.cpp 用的是同名局部 struct（std::string 字段），不受影响
- [x] 构建 GREEN（job pwsh-5，285/285 目标，二进制 build/apps/ninfer-serve；仅 text.cpp:1193 两条既有 narrowing 警告，与本次改动无关）
- 待办：构建 GREEN 后交用户重跑 --devices 0,2 --max-context 8192（模型路径 /home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer）→ 预期启动成功进入 serve → 用户发日志继续调试
### Round 26（TP-2 warmup 失败：Paged KV materialization exceeds reserved capacity）
- [x] **现象**：tokenizer 修复后 engine ready（30.6s），但 warmup 1.59ms 后 FATAL：Paged KV materialization exceeds reserved capacity；日志 capacity 行显示 pages 0/0
- [x] **根因（两个 bug 叠加）**：(a) 直接原因——execute() 每请求调 pool.materialize(reservation, 128, shard.kv_pages)，但 materialize 检查 target_page_count > destination.capacity()（paged_kv_cache.cpp:298），shard.kv_pages 是空 vector（capacity 0）→ 128 > 0 抛错；调用方必须先 reserve(128)。(b) 潜在跨请求泄漏——即使先 reserve，每请求 reserve(128) 使 reserved_pages_ 累加、materialize 把 128 页变成常驻 lease 存于 shard.kv_pages（跨请求保留）→ 第 2 个请求 available_pages()=0 → reserve 返回 nullopt → "reservation failed"。根本问题：KV 页是固定物理分配，本应启动时一次建立、跨请求原地复用（max_concurrency=1、block table 映射恒定），不该每请求重建
- [x] **修复**：把 KV 页物化从 execute() 移到 build_shard()（启动时一次）：reserve(128) → kv_pages.reserve(128) → materialize → 发布 execution row 0（kv_row lease 常驻于 Shard）。execute() 每请求只 cudaMemsetAsync 清零 GDN state。KV 页原地复用：prefill 逐 token 覆写 position 0..N-1 的 K/V，frontier 之外的陈旧数据不会被读（attention 只读 ≤ 当前位置）。forward_tp2 固定绑 kv_table_rows=0（text.cpp:1199），与启动时发布的 row 0 一致
- [x] 构建 GREEN（job pwsh-6，二进制 build/apps/ninfer-serve；仅 text.cpp:1193 两条既有 narrowing 警告，与本次改动无关）
- 待办：交用户重跑 --devices 0,2 --max-context 8192 → 预期 warmup 通过、serve 就绪 → 用户发日志继续调试
### Round 27（TP-2 warmup 失败：cudaErrorSymbolNotFound @ scalar.cu:10）— 诊断中
- [x] **现象**：KV 修复后 engine ready，warmup 在第一个 forward_tp2 的 ops::set_i32_scalar 处 FATAL：scalar.cu:10 CUDA_CHECK(cudaGetLastError()) failed: cudaErrorSymbolNotFound: named symbol not found → Aborted (core dumped)
- [x] **排除链接层问题**：cuobjdump --dump-sass 显示 serve 与测试二进制各有 3191 个 SASS 函数，均含 _ZN6ninfer3ops21set_i32_scalar_kernelEPii；readelf 两者均有 .nv_fatbin/.nvFatBinSegment/__nv_module_id 段；nm 显示宿主 stub __device_stub__...set_i32_scalar_kernel 与 ninfer::ops::set_i32_scalar_kernel 均为 T（已定义）；__sti____cudaRegisterAllv 注册符号各 587 个，完全一致；两二进制时间戳同为 10:02（同一次构建，测试二进制非陈旧）。设备代码与注册均在，非缺 SASS
- [x] **排除 build_shard 首 kernel 假设**：build_shard 的 pool.materialize/tables.publish 只用 cudaMemset/cudaMemcpy（paged_kv_cache.cpp:541/831），不 launch 任何 kernel；publish 只是 host_shadow→device 的 cudaMemcpyAsync
- [x] **关键对照**：Round 21 测试二进制 ninfer_qwen3_5_tp2_forward_test 确实跑了两卡 forward_tp2_token（card0→card1、card1→card0，11 个 probe token）且通过 → 证明该二进制结构下 device 2 的 kernel launch 能工作。测试不走 TP2GenerationCore（自有 build_shard_state，用普通 DeviceArena 做 KV，不上传 14GiB 权重、不 materialize），故 Round 26 对 tp2_generation_core.cpp 的改动不在测试代码路径内
- [x] **serve 与测试的真实差异**：serve 的 materialize_model_tp2（14GiB 权重上传到 device 2）+ build_shard 在首次 forward 前于 device 2 运行；cudaGetLastError 粘滞 → 真正失败的可能是 device 2 上更早的某次 launch，错误在 forward_tp2 第一个 CUDA_CHECK 才浮出。代码无 CUDA_MODULE_LOADING/cudaSetDeviceFlags 设置（默认 LAZY）
- [x] **决定性实验（用户已跑，test.log）**：测试二进制**通过**——11 个 probe 全部 shard 一致 + 32 步自回归 decode（含 host-staging allreduce）shard 一致 → TP-2 CUDA 代码本身正确，排除全局性 RDC 链接问题
- [x] **根因（WSL2 GPU 枚举漂移 + 硬编码 --devices 0,2）**：test.log 第 4 行 `devices=0,1`——测试的 pick_devices() 自动选同名 GPU 对，跑测试那一刻 WSL2 枚举为 0=5060Ti,1=5060Ti,2=T10；而崩溃时 serve 用 --devices 0,2，那一刻 device 2 是 **T10（sm_61/7.5）**。sm_120a 的 fatbin 在 T10 上无镜像 → shard_b（device 2）第一次 kernel launch（forward_tp2 第 6 个 set_i32_scalar）报 cudaErrorSymbolNotFound，与崩溃点完全吻合。WSL2 每次 VM 重启会重新枚举 GPU，索引不稳定
- [x] **修复**：TP2GenerationCore 构造时加 validate_tp2_devices() 守卫（tp2_generation_core.cpp）——两卡必须都是 compute capability 12.0 且 props.name 相同，否则抛 invalid_argument 并提示 'nvidia-smi -L' 查真实索引。把 WSL2 枚举漂移从深层 cryptic 崩溃变成启动期清晰报错
- [x] 构建 GREEN（job pwsh-8，285/285；serve 二进制 12:16:32 含守卫字符串）
- [x] **用户确认**：索引问题坐实——按提示改 `--devices 0,1` 后守卫通过、engine ready（28.6s），warmup 进入新错误 → 进入 Round 28
### Round 28（TP-2 warmup 失败：output session already has a preview）
- [x] **现象**：--devices 0,1 后 engine ready，warmup 3.8s 后 FATAL：output session already has a preview（output_session.cpp:414/542/607 三处 preview_ready 守卫之一）
- [x] **根因（TP2GenerationCore decode 循环双 preview）**：decode 每轮先调 `preview_model(step, ...)`——它内部已处理终止（stop token → StopToken、预算耗尽 → limit_reason）并置 `preview_ready=true`；随后 `publish_preview(decision.finished())` 在 finished 时又调 `preview_terminal(...)` → 撞上 preview_ready 守卫必抛。单 GPU 引擎（engine_core.h:1113-1226）的正确模式是 preview_model 后直接 commit_preview，preview_terminal 仅用于取消路径（preview_model 尚未调用时）。warmup 短回复一旦命中 stop token 或预算耗尽即触发
- [x] **修复**：decode 循环 `publish_preview(decision.finished())` → `publish_preview(false)`（preview_model 已建立含终止状态的 preview，直接 commit）。取消路径（299/335 行）的 preview_terminal 调用不受影响——那些路径 preview_model 尚未调用
- [x] 构建 GREEN（job pwsh-9）
- [x] **用户确认**：warmup 通过（3.7s），serve 启动期 FATAL：loaded artifact model name must not be empty → 进入 Round 29
### Round 29（serve 启动失败：artifact model name 为空）
- [x] **现象**：warmup 通过后 `server failed during startup | loaded artifact model name must not be empty`（serve_options.cpp:381 resolve_public_model_id）；engine ready 日志的 model name 与 formats 字段均为空、weights 0 B
- [x] **根因（TP-2 分支从未填充 LoadSummary）**：engine.cpp 的 TP-2 分支（device_b>=0）直接构造 TP2GenerationCore，但 Impl::load（LoadSummary）保持默认构造——单卡路径经 construct_model 填充 LoadSummary（model_instance.cpp:207-226），TP-2 路径没有等价步骤 → model_name/weight_formats/host_to_device_bytes 全空 → resolve_public_model_id 抛错。TP-2 core 自身持有 shard model（info().name、storage_stats()、weight_data()），数据都在，只是没暴露
- [x] **修复**：TP2GenerationCore 新增 load_summary()（从 shard_a_.model 取 architecture/model_name/weight_formats/storage_stats，与单卡路径同源同构）+ 构造器记录 load_seconds_；engine.cpp TP-2 分支 `load = tp2_core->load_summary()`。附带修复 engine ready 日志的空字段与 weights 0 B
- [x] 构建 GREEN（job pwsh-10，serve 二进制 12:55:47）
- [x] **用户确认**：serve 启动成功（listening on 8088），但首个真实请求 404（浏览器 GET / curl 无 body，POST-only 路由）→ 用正确 POST 请求后进入 Round 30
### Round 30（真实请求失败：std::bad_alloc）
- [x] **现象**：正确 POST（thinking xhigh、max_tokens 64）→ req#1 进入 execute 跑完 prefill（5.0s，prefill 1 tok），随后 FATAL：std::bad_alloc（HTTP 500 internal_error）
- [x] **排除主机内存**：bad_alloc 来自 DeviceArena::alloc_bytes 容量耗尽（arena.cu:215 `if (end > cap_) throw std::bad_alloc()`），不是主机 malloc 失败；workspace 是 DeviceArena（GPU）
- [x] **定位（ws_b 工作区跨 forward 泄漏）**：execute 的 prefill/decode 循环只建 `ws_a.scope()`（tp2_generation_core.cpp:311/351），**从未建 ws_b.scope()** → shard_b 的 workspace 每次 forward 的中间激活（attention/GDN/FFN 的 [N,1] 张量 + lm_head GEMM scratch）只增不减，跨所有 forward 累积，最终撑爆 128 MiB arena → bad_alloc。warmup（prompt "hi"≈2 token）只泄漏 ~2 次 forward 未爆；真实请求 xhigh 长 prompt 泄漏足够多次 forward 后爆
- [x] **为何测试未暴露**：测试二进制走 forward_tp2_token（text.cpp:1269 自带 `work_.scope()` 包裹整个 forward，每 token 重置 workspace）；serve 路径直接调 forward_tp2（无 scope）→ ws_b 泄漏。测试只跑 position-0 probe + 32 步 decode，且走 token 路径，故从未触发
- [x] **修复**：prefill 与 decode 循环各加 `auto scope_b = ws_b.scope();`（与既有 ws_a scope 对称）——每次 forward/decode 轮次结束两个 shard 的 workspace 都回到起点。scope 析构顺序：scope_b 先析构（后声明），无依赖问题
- [x] 构建 GREEN（job pwsh-11，serve 二进制 14:19:46）
- [x] **用户确认成功**：同一 POST 请求完整返回 JSON——prompt 58 / output 64（finish_reason length，命中 max_tokens）、reasoning_content 正常（63 reasoning tokens，thinking xhigh 解析正确）、TTFT 510 us、total 6.3s；decode 稳态 10.0 tok/s（首窗口 2.6 tok/s 含 CUDA 懒加载）。**TP-2 端到端链路（prefill + 多步 decode + host-staging allreduce + 采样 + thinking 输出解析）首次全通**
### Round 31（功能验证：多请求 / 流式 / stop 终止 / 性能基线）
- [x] 验证 1 ✅：req#1–#5 连续 5 请求全部成功；req#1/req#3 同 prompt（57/32）重复执行输出正常——跨请求状态重置（KV 覆写 + GDN 清零）正确，无 shard 分歧
- [x] 验证 2 ✅：req#4/req#5 流式——SSE 分片完整（reasoning delta → content delta → 末 chunk 带 finish_reason+timings → [DONE]）
- [x] 验证 3 ✅：req#5「1+1等于几」→ finish_reason stop（output 43，答完 "2" 命中 stop token）
- [ ] 补充验证：长 prompt（~2000 token）——覆盖长序列 attention（SmallT 多 split 路径，短 prompt 从未触发）+ 持续 prefill（ws_b scope 修复的压力测试）
- [x] **性能基线（nsys kernel 级剖析，job 见 tools/tp_bootstrap/analyze5.py）**：
  - nsys profile 请求（`"1+1等于几？只回答数字"`，max_tokens=24）实测 **completion_tokens=24**（usage 确认，analyze5 的 /24 分母正确）：span 1212.5ms/24 = **50.5 ms/token ≈ 19.8 tok/s**（GPU 空闲、无 CUPTI 之外的干扰）。CUPTI 有 ~5-10% 开销，真实略高（~21-22）
  - **⚠ 用户真实运行（Round 33 后）3 条请求 total 反推仅 ~10.5 tok/s（95 ms/token）**，与 profile 的 50.5ms/token 差 1.9×。最可能原因：**llama.cpp 仍在 5060 Ti 上运行、抢占 GPU**（用户自述"llama.cpp在占用"；nvidia-smi 显示两卡各 15GB 占用且无 ninfer compute-app）。profile 跑在 GPU 空闲时，用户测试跑在 llama.cpp 活跃时 → 不可直接对比。需停掉 llama.cpp 后复测才能分离 allreduce 优化的真实增益
  - **每 token 时间分解（per-shard，两 shard 并行）**：
    - GEMM/attention/GDN 计算 **33.8 ms（67%）**——访存受限、已近最优（lm_head 实测 439 GB/s ≈ 5060 Ti 峰值带宽，GEMV 效率 ~75-90%）
    - **allreduce 屏障 13.9 ms（28%）**——WSL2 无 P2P 的 host-staging 每层往返（2 compute-stream sync + 4 D2H/H2D + CPU 加法）。standalone 基准 90us/层，真实管线 217us/层（多出的 ~127us 是 compute-stream sync 等 FFN kernel 排空 + H2D 阻塞 residual）
    - kernel 启动开销 **2.8 ms（5%）**——16918 个小间隙（均值 4us）
  - **结论**：GEMM 已近带宽下限，优化空间在 allreduce 屏障（WSL2 特有税）。裸金属 P2P 下 allreduce 会从 13.9ms 降到 ~1.5ms → 可达 ~35 tok/s。**WSL2 TP-2 5060 Ti 的现实上限 ≈ 22 tok/s，28 tok/s 目标大概率是裸金属/P2P 目标，WSL2 下难达成**
  - **可选优化（ROI 排序）**：① 减 allreduce sync 数（D2H/H2D 直接发在 compute stream 上，6 sync→2 sync，省 ~3-5ms/token → +2-3 tok/s）；② CUDA graph 消启动开销（但 host allreduce 无法入图，收益有限 ~2.8ms）；③ 接受现状（~20 tok/s 已近 WSL2 上限）
- [x] 用户决策：**方案 3**（先做 allreduce 优化，再进下一阶段）

## Round 33：allreduce 同步优化（6 sync → 2 sync/层）
- **动机**：host-staging allreduce 每层 6 次 `cudaStreamSynchronize`（run_layers_tp2 的 2 次 compute-stream 预同步 + allreduce 内 D2H 后 2 次 + H2D 后 2 次）。WSL2 下每次 sync 经 GPU 虚拟化往返，是 13.9 ms/token 屏障的主体。
- **改动**（3 文件）：
  - `src/core/tp/device_pair.h`：`allreduce` 签名加 `cudaStream_t stream_a, stream_b`（compute stream）
  - `src/core/tp/device_pair.cu`：host-staging 路径把 D2H/H2D 拷贝发到 **compute stream**（不再用 DevicePair 独立 stream）。D2H 天然排在 FFN kernel 之后（省 2 次预同步），H2D 天然排在 residual_add 之前（省 2 次 H2D 同步）→ 每层只剩 2 次 D2H 同步。P2P 路径（WSL2 不用）保留预同步 + 独立 stream
  - `src/models/qwen3_5/execution/text.h`：调用点传 `ctx_.stream, peer.ctx_.stream`，删掉 2 次预同步
  - `tests/test_tp_device_pair.cpp`：传 per-device non-blocking stream，回读前补 `cudaStreamSynchronize`（镜像生产路径里 residual_add 同 stream 的隐式排序）
- **正确性论证**：compute stream 是 `cudaStreamNonBlocking`（非阻塞），与默认流无隐式排序。生产路径里 H2D 与后续 `residual_add` 在同一条 compute stream 上，天然有序，无需 H2D 同步。测试回读走默认流（`DeviceBuffer::copy_to_host` 用同步 `cudaMemcpy`），故测试需显式同步 compute stream
- [x] 构建 GREEN（job pwsh-13，28/28，仅预存 narrowing 警告）；注意 WSL 侧 /home/zhuojun/ninfer 是独立副本，改动文件需先 cp 同步（tools/tp_bootstrap/sync_changed.sh）
- [x] ninfer_tp_device_pair_test **PASS**（p2p_available=0，host-staging 路径，4 组尺寸含 1MiB）
- [x] 用户接管测试：llama.cpp 占着 GPU（nvidia-smi 显示 5060 Ti 各 15GB，probe_mem 证实部分为记账残留），nsys 复测放弃，由用户手动启停 serve 实测
- [ ] 用户功能回归：真实请求无 bad_alloc / 输出正确 / 实测 tok/s（预期 ~22，从 ~20 提升；日志发回核对）
- [ ] 进入下一阶段：MTP/dflash2 组件切分、多请求并发冒烟、perplexity 健全性

## Round 34：in-kernel allreduce + mixer 头切分（追平/超越 llama.cpp 31.65 tok/s）
- **llama.cpp 机制（已读源码 C:/llama.cpp ar3-opt 分支 allreduce.cu）**：小 allreduce 走"分块 kernel 路径"——单 kernel 在两 GPU 同时跑，`cudaHostAllocMapped`+`cudaHostGetDevicePointer` 让设备直接读写 pinned host 内存，**跨 GPU 同步在 kernel 内部 busy-wait 一个 host arrival 标志**（volatile store + `__threadfence_system()`），完全绕开 host 端 `cudaStreamSynchronize`。这就是它 ~30µs/AR 而我们要 2 次 host 同步（~217µs）的原因
- **WSL2 关键前提已实测验证（tools/tp_bootstrap/mapped_probe.cu）**：`cudaHostAllocMapped`+`cudaHostGetDevicePointer` 可用；device-b 能读到 device-a 写入的同一块 mapped host 内存；**in-kernel allreduce 10KB = 0.038 ms/次（38µs），64 层 = 2.44 ms/token**，对比 host-staging 13.9 ms/token
- **重算预期**：现状 50.5ms/token(21tok/s) → 仅 in-kernel AR 39.0ms(~25.6) → +头切分 32.1ms(~31) → +CUDA graph 29.3ms(~34，超 llama.cpp)
- **in-kernel AR 是头切分前置**：把单次 AR 217µs→38µs，头切分让 AR 次数翻倍(64→128)才不亏(4.9ms < 省 9.4ms 计算)

### 步骤 1：in-kernel allreduce（先做先测，21→~25.6 tok/s）
- 改 `src/core/tp/device_pair.{h,cu}`：移植 llama.cpp chunked kernel 路径。`cudaHostAllocMapped` 分配 per-device staging + arrival ring（64B 间隔防 false sharing）；kernel 三阶段：① local→host_mine ② thread0 写 arrival token、自旋等 peer（`__nanosleep(100)`）③ 读 host_other、`__hadd` 写回 delta（in-place）。两设备 compute stream 各启一次 kernel，**无 host 同步**
- 保留 host-staging 作为 fallback（mapped 不可用时）
- [x] 实现 in-kernel AR kernel + DevicePair 集成（`ar_inplace_bf16` kernel：Phase1 local→mapped host、Phase2 thread0 写 arrival token+自旋等 peer `__nanosleep(100)`、Phase3 读 peer+`__hadd` in-place；mapped pinned 1MiB staging + 64B arrival ring；构造函数分配、析构释放、move 转移；mapped 不可用时 fallback host-staging）
- [x] 编译 GREEN（142/142）+ `ninfer_tp_device_pair_test` PASS（20480/16/1MiB 尺寸 + 移动语义，in-kernel 路径数值正确）
- [ ] 用户端到端测速（停 llama.cpp → serve → 预期 ~25.6 tok/s；确认 64 次/forward 连续 in-kernel AR 不死锁、logits 正确）

### 步骤 2：mixer 头切分（~31 tok/s）
- **Qwen3.8-27B 几何（GGUF 元数据 tools/tp_bootstrap/gguf_meta.py 已解析）**：attention 24q/4kv/head_dim256/rotary_dim64；GDN 16k头/48v头/head_dim128/inner6144/conv4；64 层=16 full-attn+48 GDN
- **切分后每卡**：attention 12q/2kv（qkv 14336→7168 行、o_proj K 6144→3072）；GDN 8k/24v（wqkv+gate 16384→8192 行、ssm_out K 6144→3072、beta/alpha 48→24）；KV cache 每卡 2 kv 头；GDN state 每卡 24 v 头
- **llama.cpp 切分参考（qwen35.cpp）**：attention Q(query+gate 融合)按 n_head 头切、K/V 按 n_head_kv 头切、wo row-parallel+AR；GDN wqkv 按头块切、wqkv_gate 按 v 头切、ssm_conv1d 按通道切、ssm_beta/alpha/dt/a 按 v 头切、ssm_out row-parallel+AR、recurrent state 按 v 头切
- **改动范围**：tp_split_spec.cpp（mixer 投影改 GatherRows/RowParallel）+ attn/gdn op 几何参数化（当前硬编码全模型几何）+ KV cache 每卡 2 kv 头 + GDN state 每卡 24 v 头 + o_proj/GDN-out allreduce
- [x] 实现头切分（权重 + attention + GDN + KV cache + allreduce）
  - **加载层**（tp_split_spec.cpp / tp_shard_views.cpp / load/text.cpp）：attention 融合权重 GatherRows {q,k,gate,v}、GDN 融合 GatherRows {q,k,v,z}、conv GatherCols {q,k,v}、o_proj/ssm_out RowParallel；GDN gating（a/b + a_log/dt_bias）保持复制（全 48 头）
  - **执行层**（execution/text.{h,cpp}）：TextContext 加 per-shard TextConfig（head 数减半）+ set_shard_config/shard_config()；attn_mix/gdn_mix 用 per-shard config 分配 workspace + tensor view；**融合投影拆解为通用 ops::linear**（per-shard 融合权重 [q|k|gate|v] / [q|k|v|z]，GEMM 后 slice+view，镜像 ffn_delta，绕开硬编码全模型行数的融合 GEMM kernel）；gdn_mix 里 g/beta 按全 48 头分配（gating 复制）再 slice 到本卡 24 头喂 delta net；mixer 输出投影写 delta（不直接加 residual）
  - **驱动层**（tp2_generation_core.{h,cpp}）：build_shard 构造 per-shard TextConfig，KV cache（2 kv 头）+ GDN state（24 v 头）按 per-shard 几何分配，context->set_shard_config；run_layers_tp2 每层 mixer 后加一次 delta allreduce（64 mixer + 64 FFN = 128 AR/token，均 ≤10KB 走 in-kernel 路径）
- [x] 编译 GREEN（BUILD_EXIT=0；ninfer-serve 已重建，含 bind_gdn 修复）
  - 加载层补漏①：src/ops/weight_input.cpp 的 input_projection dense 几何校验原为精确值（attn q=={6144,5120}/k=={1024,5120}、GDN q=={2048,5120}/v=={6144,5120}），头切分后 per-shard 形状（attn q=3072/k=512、GDN q=1024/v=3072）会触发 require 抛错，改为结构性校验（attn q=gate、k=v、q=6k；GDN q=k、v=z、v=3q；四块共享 hidden 维），全模型与 per-shard 形状均通过。per-shard 融合对象行连续（[q|k|gate|v]/[q|k|v|z] 顺序），故 p.projection 为单一 LinearParameters（非 PairedProjectionWeights），执行层 std::get<LinearParameters> 正确
  - 加载层补漏②（bind_gdn 头数 bug）：bind_gdn 原把 a/b + a_log/dt_bias 绑到 2*heads（误以为 loader 传 per-shard config），但 loader 传**全模型** config（48 v 头）→ 绑成 96 头，与物理 a/b 对象 (48,5120) 不符，触发「logical shape differs from Binding coverage」。改为绑 heads（=48，全模型值），a/b 复制全 48 头正确。TP-2 load 测试的 shape 校验已通过此修复（不再报 binding 错）
  - **TP-2 单测（test_tp2_load / test_tp2_forward）已更新为头切分断言**：load 测试改查 per-shard mixer 形状（attn q [3072,5120]、GDN q [1024,5120]）；forward 测试按 per-shard config（head 减半）分配 KV/GDN state + 调 set_shard_config。两测试需**停掉 llama.cpp** 后跑（2×5060 Ti 各需 ~14GB 权重，llama.cpp 占着时 cudaMalloc OOM）——由用户停 llama.cpp 后跑，或用户直接 serve 实测
  - 执行层补漏③（gdn_mix 三分支重构）：TP-2 decode 走 Phase::Verify（batch=1,width=1），原代码进 batched snapshot 分支（gdn_projection_snapshot，硬编码全模型 16384 行）→ per-shard 权重触发「unsupported single-parent profile」。重构为三分支：(a) batched_verify（Verify 且 batch>1 或 width>1，投机解码用）保留 gdn_projection_snapshot/record；(b) T>1（单卡 prefill）保留 gdn_projection 融合 GEMM；(c) T=1（单 token decode，含 TP-2）走通用 ops::linear 对 per-shard 融合权重（8192 行）做 GEMM，输出 [q|k|v|z] 写入 fused 缓冲，qkv=前 6144 行切片作 conv 输入，z=后 2048 行切片作 output gate，qc/kc/vc 用 workspace 独立缓冲作 conv 输出（避免 conv 输入与输出重叠被 require_split_nonoverlap 拒绝）。attn_mix 无重叠问题（fused 切片只被下游读，attention 写独立 a/KV 缓冲）。state 访问一致（conv_slot(gidx,0)==layer_view(gidx).conv，slot_count=1）
  - 执行层补漏④（conv split row profile 注册表）：causal_conv1d_silu_split 的 resolve_split_geometry 只注册了两个全模型 row profile（8192/2048/2048/4096 与 10240/2048/2048/6144），per-shard GDN profile（x=5120, out0=1024, out1=1024, out2=3072）未注册 → 触发「split received an unregistered row profile」。新增 CausalConvSplitGeometry::Rows1024x1024x3072 枚举值 + wrapper 注册 + smallt/prefill 两个 launcher 的 switch case（模板参数 <1024,1024,3072>）。kernel 本身是 shape-generic（CausalConvSplitOutput3 模板），只需注册 profile 即可。文件：src/ops/launcher/causal_conv1d.{h,cu}、src/ops/wrapper/causal_conv1d_silu.cpp
  - **切分层补漏⑤（names_by_object 用 bool 当对象索引——本轮关键 bug）**：tp_split_spec.cpp 构造「物理对象索引 -> 参数名」反查表时写成 `names_by_object[binding.whole_object].insert(name)`，但 `Binding::whole_object` 是 **bool**（src/artifact/schema.h:61），被隐式转成 size_t 后所有整对象绑定都落到 key=1。于是 `names_by_object.find(idx)` 找不到名字，`has_name("attention/output")` / `has_name("gdn/output")` 全部返回 false，两个 mixer 输出投影（本应 RowParallel，K 维切半）掉进最终 else 的 **Replicated**，per-shard 仍持全宽权重（n=5120, k=6144）。运行期 `ops::linear(x=[3072,1], w=[5120,6144], out=[5120,1])` 因 x.ne[0]!=w.k 抛「linear: expected [K,T] x [N,K] -> [N,T]」（诊断打印证实：x.ne=[3072,1] w.n=5120 w.k=6144 out.ne=[5120,1]）。修复：绑定一定带 parts（整对象绑定也存一个覆盖全对象的 part），故一律按 `p.object.index` 建表，删掉 whole_object 分支。同时把 untied output_head 显式置 Replicated（TP-2 forward 契约：每 shard 独立算全词表 logits 且两者逐位一致，head 不做 all-reduce；本模型 head 与 token_embedding 权重共享，先被 token_embedding 分支截获）。修复后实际行为变化仅 attention/output 与 gdn/output → RowParallel。诊断手段：临时在 ops::linear 的 validate_linear_semantics 失败分支加 fprintf 打印 x/w/out 形状（已定位，待移除）
  - **算子支持补漏⑥（attention per-shard head geometry 12/2）**：TP-2 头切分后单卡 attention 为 12 q 头 / 2 kv 头，但 causal_softmax_attention 的 require_causal_geometry 只接受 (24,4) 与 (16,2)，且 kernel 按 Geometry 模板实例化 → layer 3（首个 full-attn 层）报「causal_softmax_attention: unsupported head geometry」。修复：(a) geometry.cuh 新增 CausalD256H12Kv2 = CausalAttentionGeometry<12, 2, 2>（GroupSize=12/2=6，与 24/4 相同，故 q->kv 连续映射不变；SmallTSplitScale 取 2，与既有窄几何 16/2 一致，且 static_assert 85*scale<=256 满足）；(b) require_causal_geometry 接受 (12,2)；(c) causal_attention_chunk_tokens 把「q_heads==16 → 6」改为「q_heads!=24 → 6」，与 causal_attention_split_capacity 的 tokens 上限校验 (q_heads==24?8:6) 保持一致；(d) split capacity 增加 12 分支；(e) 8 个 launcher 文件（small_t / small_t_fp8 / small_t_k8v4 / small_t_nvfp4 各 2 处、prompt / prompt_fp8 / prompt_k8v4 / prompt_nvfp4_non_rdc）在 H24 分支后、H16 兜底前插入 12 头分支，共 13 处。kernel 自身对 QHeads 是泛型的（QHeads 只作 grid 维与 QHeads==24 的 constexpr 特化判据，12 走非 24 路径），故属机械实例化
- [x] **TP-2 两个测试均通过**（2026-09-18）：ninfer_qwen3_5_tp2_load_test → 「TP-2 dual-shard load passed devices=0,1 layers=64 shard_gate=[8704,5120] shard_down=[5120,8704] head=[248320,5120] replicated」；ninfer_qwen3_5_tp2_forward_test → 「all 11 probes shard-consistent; 32-step autoregressive decode shard-consistent」。ninfer-serve 已重建（21:57）。加载层/执行层/算子层全部对齐 per-shard 几何。附带：load 测试原断言 output_head=[124160,5120]（旧 ColumnParallel 设想）已改为 [248320,5120] replicated，与 forward 的全词表 logits 契约一致
- [x] **端到端实测通过（2026-09-18，llama.cpp 已停、GPU 空闲，agent 自测）**：
  - 两个 TP-2 测试仍 PASS（load / forward 11 probes + 32 步自回归）
  - serve（`--devices 0,1`，端口 8088）engine ready 35.2s、warmup 3.1s，每卡 11.5GB，无 OOM
  - **decode 29.09–29.11 tok/s（34.37 ms/token）**：短请求 32 tok、中请求 128 tok、512 tok 长生成三次一致（29.09/29.10/29.09）→ llama.cpp 基线 31.65 的 **92%**
  - 长 prompt（2063 tok）decode **31.53 tok/s**（31.7 ms/token），但 prefill 逐 token **69.6s**（TTFT ~70s，llama.cpp 403 tok/s 差 13.6×）→ Round 35 的现实动因
  - 生成质量：`1+1=2` reasoning 正确、`Paris` 命中 stop token（finish_reason=stop）、haiku/总结连贯；两卡 logits 一致 + 输出语义正确
  - **补漏⑦（timings 全 0）**：TP2GenerationCore 从不填 `timings.prompt_wall_seconds/generation_wall_seconds/prefill_seconds/decode_seconds/first_token_seconds`，且 `computed_prefill_tokens_` 只在 prefill 最后一个 token 自增（每请求记 1）。已按单卡 engine_core.h:826-876 的语义补齐（first_token 边界 = prefill 循环结束），并把 prefill token 计数改为 `+= prompt_tokens`。修复后 serve 响应 timings 可用、日志 prefill/decode 速率行恢复
  - **补漏⑧（设备索引提示误导）**：`--devices` 取的是 **CUDA 运行时索引**（默认 CUDA_DEVICE_ORDER=fastest-first），与 `nvidia-smi -L` 的 PCI 顺序**不同**——本机 CUDA 0,1 是两张 5060 Ti，而 nvidia-smi 的 1 是 T10（在 2）。守卫报错原提示"run nvidia-smi -L"，实测把 agent 引到 `--devices 0,2`（=T10）上。现改为直接列出 CUDA 可见设备（index=name(sm_xx)），不再引用 nvidia-smi
- [x] **nsys kernel 级分解（22:10 profile，llama.cpp 已停）**：墙钟 33.90 ms/token；dev0 GPU busy 30.99 + idle 2.91；每 forward 1009 个 kernel、1011 个间隙（均值 2.9 µs）
  - **计算 25.6 ms**：每卡权重流量 ≈10.15 GB/token（gate_up nvfp4 17408×5120×56、down 5120×8704×56、fp8 各 8 层、attn 7168×5120×16 + 5120×3072×16、GDN 8192×5120×48 + 5120×3072×48、head 248320×5120）→ 22.7 ms@448GB/s 理论，实测 89% 峰值，已近带宽下限
  - **AR 5.38 ms（128 次 × 42 µs = 15.9%）**：与 bench 的 38 µs 一致；偏大主因是两卡均速差 + 每层全屏障
  - **lm_head 2.94 ms/次（8.5%）**：fp8 248320×5120 = 1.27 GB，每卡各算全词表一遍（replicated 契约），是最大的单项可优化点
  - **launch 间隙 2.91 ms（8.6%）**：1009 个 kernel/forward；CUDA graph 是唯一有效手段
- **下一步（本阶段收尾 → 追平/超越 31.65）**：① lm_head 按词表切分（每卡 124160 行 → head 2.94→1.47 ms，再经 pinned 缓冲交换对半 logits 给 shard A 采样，交换 ~0.06 ms）≈ −1.4 ms；② TP-2 decode 前向接 CUDA graph（需先把 AR 的 host 端 token 计数改成 kernel 自增到达标志，否则 graph 重放会等同一个 token 值而挂死）≈ −2.5 ms。合计 ≈30.0 ms → ~33 tok/s，超基线 4%

### 步骤 3（后续）：CUDA graph（~34 tok/s，超 llama.cpp）
- TP2GenerationCore 当前 bypass CUDA graph（2.8ms/token 启动开销）；接入 graph 后 → ~34 tok/s

## Round 36：MTP（决定性超越）
- 模型带 nextn 头（gguf 里 blk.64.nextn.*），llama.cpp 未用
- 一次 forward 出 2-3 token → 60+ tok/s

## Round 34（旧版，已被上方取代）：mixer 头切分（追平 llama.cpp 31.65 tok/s）
- **背景（用户提供的 llama.cpp 数据）**：llama.cpp 跑 **Q6_K（~22 GB，比 NVFP4 19.1 GB 大 15%）**，`-sm tensor -ts 1,1` 头切分 + CUDA graph（`graphs reused=2726`）+ 未开 MTP，稳态 **31.65 tok/s**；prefill **403 tok/s（2.48 ms/token）**。我们 decode 21 tok/s、prefill ~47 ms/token（逐 token 循环，差 19×）
- **差距定位（nsys 对账，全部吻合）**：
  - llama.cpp 每卡读 ~11 GB/token → 计算 ~25 ms + allreduce 64×0.05≈3 ms + graph 0 = 31.6 ms ✓
  - 我们每卡读 13.5 GB/token（mixer 7.45 GB 两卡各读一遍）→ 计算 33.8 ms + allreduce 13.9 ms + 启动 2.8 ms = 50.5 ms ✓
  - 差距 100% 结构性：① mixer 复制（+9.5 ms）② allreduce 同步贵（+11 ms，Round 33 已改一半）③ 无 CUDA graph（+2.8 ms）。kernel 效率我们更高（396 GB/s=峰值 88%）
- **模型几何（nsys kernel 形状反推，Qwen3.8-27B）**：64 层 = 16 full-attention + 48 GDN。attention：q 24 头、k/v 16 头、head_dim 256（qkv 合并 14336×5120）；GDN：in_proj 16384×5120（q 24×256 + k/v 16×256 + gate 2048）。FFN 8704 中间维（55 层 nvfp4 + 8 层 fp8）。lm_head 248320×5120 复制
- **改动范围**：
  1. 权重加载：q/k/v/gate 投影按头切（column shard），o_proj/GDN out 按行切（row shard）；KV cache 每卡 8 头
  2. attention kernel：每卡 12 q 头 × 8 kv 头（GQA 1.5:1）
  3. GDN kernel：每卡 12 q 头 × 8 k/v 头，recurrent state 按头切
  4. allreduce：o_proj 输出 + GDN out 输出（每层 1 次，共 64 次/token，与 FFN delta 合并或独立）
- **预期**：每卡计算 33.8 → ~24.7 ms（13.5→9.8 GB），allreduce 13.9 → ~6 ms（Round 33 后），wall ≈ 24.7+6+2.8 ≈ 33.5 ms → **~30 tok/s**；若 allreduce 压到 llama.cpp 水平（0.05 ms）→ ~32 tok/s
- [ ] 读 attention/GDN/权重加载/KV cache 代码，设计头切分
- [ ] 实现头切分（权重加载 + attention + GDN + KV cache + allreduce）
- [ ] 编译 + 单测 + 用户功能回归
- [ ] nsys 复测 decode（目标 ~30-32 tok/s）

## Round 34 补丁：并发崩溃/挂死（用户真实流量暴露）

- **现象（用户 64k 实测日志）**：req#2/#3 同时到达 → 6 ms 内双双 HTTP 500 internal error；之后 req#4/#5 的
  `cudaMemsetAsync(shard.state_backing)` 报 `cudaErrorIllegalAddress`（context 已被污染）。
  agent 复现：**3 个并发请求全部挂死 >10 分钟**（三个 "req started" 同一毫秒，之后无任何进展）
- **根因（TP-2 core 无串行化）**：`Engine::submit` 对 TP-2 走 `core->submit()`（只建 Submission），
  `GenerationHandle::wait` → `TP2GenerationCore::Submission::wait` → `execute()` **直接跑在调用方（HTTP）线程**。
  单卡路径靠 EngineCore 的 worker 线程 + pending 队列串行，TP-2 没有等价物，而 serve 自身的
  `request_capacity` 用的是 **serve 选项**（≠ 引擎 normalize 后的 max_concurrency=1/pending=1），故并发请求直接进
  `execute()`。后果：两个线程共用同一 workspace **arena**（非线程安全）→ bump pointer 互相踩 → 通配指针/非法访问；
  且 `DevicePair::allreduce` 的 host 端 token 递增被打乱 → 两卡到达标志错配 → in-kernel AR **无限自旋**（这就是挂死）
- **修复**：`TP2GenerationCore` 加 `std::mutex execution_mutex_`，`Submission::wait` 先取锁再 execute（arrival order），
  与单卡 EngineCore 的单 worker 契约一致；排队期间被取消的请求仍会进 execute，其首个取消检查很便宜
- **验证**：3 并发请求 2.9s / 5.6s / 8.3s 串行完成，全部 HTTP 200，无挂死、无非法访问

## Round 34 补丁：prompt 前缀复用（长上下文多轮的唯一可行手段）

- **动机**：TP-2 无前缀缓存，每轮把**整段历史**重算（34 ms/token）。64k 下 10k 历史 = 每轮 5.7 分钟，实际不可用
- **关键事实**：KV 页启动时一次性 materialize、原地复用、跨请求不清空 → position < 复用边界的 K/V 天然有效，
  只需 GDN/conv 状态在那个边界的快照
- **模板实测（`[reuse-diag]` 临时打印 lcp/cached/new，已移除）**：
  - 相同 message 列表重复请求：lcp == 全长（渲染是确定的，无 per-request 变量）
  - 连续两轮对话：lcp 落在上一轮 prompt **末尾前 2–27 个 token**（上一轮末尾是生成用的
    `<|im_start|>assistant\n<think>\n`，而下一轮把该 assistant 轮按历史渲染 → 尾部几 token 不一致）
  - 结论：**稳定可复用的边界很靠近上一轮 prompt 末尾**，不需要按消息结构猜测
- **设计**：每卡 `kReuseSnapshotCount = 3` 份状态快照（live + slot0=prefill 末尾 + slot1/2=递归回退深度），
  `DeviceArena((1+3)*state_bytes)` ≈ 308 MiB/卡（state ≈77 MiB）。回退深度**自适应**：
  用上一对 prompt 的 gap 预测 `rewind_near = gap+2`、`rewind_far = 4*near+8`（clamp 4..4096）。
  下次请求取「≤ LCP 且 < 新 prompt 长度」的最深边界；快照在 prefill 循环内按 stream 顺序取（天然精确到边界）
- **验证（--greedy，保证逐 token 可比）**：req#2 cache 48/78（61.5%）、req#3 cache 74/94（78.7%）TTFT 683 ms；
  失效后重建的全量 prefill 与复用运行 **输出完全一致（MATCH）**；无关请求 cache 0
- **遗留**：gap 的解释（为何 reasoning 回显时 gap 达 18–27）未查清；当前靠自适应深度兜住

## 运维教训（务必遵守）

- **启动 ninfer-serve 必须先杀旧实例并确认其退出**：每个 serve 进程会在 host 端 staging 22.6 GB 工件，
  两个实例同时存在 → 宿主内存爆掉（本次导致用户机器几乎卡死），且两卡显存被重复占满
- `tools/tp_bootstrap/serve_run.sh` 已加固：pkill -9 → 轮询等待 30 s → 仍在则**拒绝启动**并打印进程表 →
  打印 `free -h` → exec。启动后必须用 `pgrep -a -f 'build/apps/ninfer'` + `nvidia-smi` + health 三重确认只有一个实例
- WSL `/tmp` 会被清空（nsys report 丢过一次）→ 剖析产物放 `/home/zhuojun/prof/`
- **WSL 发行版会在最后一个会话结束时自行 poweroff**：`setsid nohup` 起的后台 serve 也保不住，
  因为 WSL 判定「最后一个 client 断连」后会 `systemctl poweroff`（journal 里可见
  `Operation canceled @p9io.cpp:258 (AcceptAsync)` → `The system will power off now!` →
  `WSL ... systemctl poweroff did not terminate the instance in 10000 ms, calling reboot(RB_POWER_OFF)`）。
  2026-09-18 23:53 的 serve 就是这样凭空消失的（不是崩溃、不是 OOM，dmesg 无 OOM 记录）
  → **必须让一个 wsl 会话持续存在**：用后台 job 前台跑 `serve_run.sh`（job 不结束，会话不断），
  不要在别人测试期间跑会 `pkill ninfer-serve` 的脚本
- **2026-09-19 01:03 二次事故（同一根因，已加固）**：`run_r35_tests.sh` 里自带 `pkill -9 -f ninfer-serve`，
  而当时 8088 正由 supervisor 托管 → supervisor 立刻重启引擎（`starting engine`），
  同时测试进程正在加载第二份模型 → 用户看到「两个服务」、显存/内存双份占用。
  加固：**测试脚本一律不自杀服务**，改为检测到 `ninfer-serve` 就拒绝运行并提示；
  停止服务只有唯一显式入口 `tools/tp_bootstrap/serve_stop.sh`（**先杀 supervisor 再杀引擎**，否则必被重启）。
  正确顺序：`serve_stop.sh` → 独占测试 → `serve_supervise.sh <ctx> <port>`

## Round 35：TP-2 批量 prefill（TTFT 19× 提升）

- **现状**：TP-2 prefill 逐 token 循环（2063 token = 69.6 s，34 ms/token），llama.cpp 批量 403 tok/s。
  这是当前最大的可用性瓶颈（首轮长 prompt 无前缀可复用）
- **为什么此前做不了**（Round 34 的三处结构性阻塞）：
  1. `forward_tp2` 单 token 写死（[N,1] 张量、envelope {pos+1,pos+1}、position 标量）
  2. `gdn_mix` 的 T>1 分支用全模型行数的融合 `gdn_input_proj`（16384 行），per-shard 权重 8192 行不支持
  3. `run_layers_tp2` 的 mixer/FFN delta 分配在**层循环之外**的作用域，T>1 时每层累积 2×hidden×T×2 字节
     （T=1024、64 层 = 1.34 GB，远超 128 MiB workspace）
- **已具备条件**（此前补漏的成果，本轮直接复用）：per-shard NVFP4 形状全部注册且 T-generic
  （n7168k5120 attn 融合、n8192k5120 GDN 融合、n17408k5120 gate+up、n5120k3072 o_proj、n5120k8704 down；
  A16 分块 ≤32 token、A4 MMA ≥8 token 到任意长度）；attention 12/2 几何已进 prompt/small_t/fp8/k8v4/nvfp4 全部
  launcher；conv split `Rows1024x1024x3072` 已注册 smallt+prefill 两条路
- **改动**：
  1. `TextContext::forward_tp2_prefill(peer, pair, ids, first_position, logits, logits_peer)`：
     T token 的 embedding → lockstep 层循环（`Phase::Prefill`）→ final norm → 只对**最后一列**做 lm_head。
     绑定与单卡 `prefill_impl` 一致：positions=[T] 绝对位置、envelope={first+T, first+T}、kv_table_rows=[1]=0、
     `active_sequence_batch_=0`（单序列 prefill 路径）、state slots=[1]=0
  2. `gdn_mix`：融合 `gdn_input_proj` 分支条件收紧为 `T > 1 && shard_config_ == nullptr`；
     TP-2 shard 的 T>1 走与 T=1 相同的分解式路径（通用 `ops::linear` + `causal_conv1d_silu_split` + `gated_delta_net`）
  3. `run_layers_tp2`：层循环体内加 per-layer workspace scope
  4. `TP2GenerationCore::execute`：prefill 改为按 `kTP2PrefillChunk = 256` 分块；
     复用快照只能在块边界取，故按目标深度（`prompt_tokens - rewind_near/far`）**截断块长**，使目标正好落在块边界
  5. `DevicePair`：in-kernel allreduce 的 mapped pinned staging 从 1 MiB 提到 8 MiB（T=256 的 delta = 2.6 MB
     超过旧上限会掉到 host-staging 全同步路径），并向量化（uint4 + 4 路展开读取）+ 线程数 256→1024
- **实现中暴露的三类阻塞（本轮全部修掉）**：
  1. 融合投影的 dim-0 切窗不连续：`Tensor.ne[0]` 是**最内层连续轴**，T>1 时 `fused.slice(0,off,n).view(...)`
     直接抛 `view requires a contiguous tensor`；同理 GDN 的 `g[i*24:(i+1)*24]`/`beta` 切片进不了
     `gated_delta_net`、`fused` 的 [q|k|v] 切片进不了 split conv（两者都要求连续）。
     新增 `copy_row_block()`（`cudaMemcpy2DAsync`，dpitch=行块宽、spitch=父行宽）把块搬进已分配的连续
     workspace 缓冲；T=1 仍走零拷贝别名路径
  2. FP8 A8 / NVFP4 W4A4 激活量化器只按 K∈{5120,6144,17408} 实例化，per-shard 的 K=3072（o_proj/GDN out）与
     K=8704（FFN down）抛 `unsupported K`（T=1 走 gemv 不触发 A8/A4，故此前从未暴露）；补齐
     `Fp8Activation{3072,8704}` 与 `Nvfp4Activation{3072,8704}`
  3. `const auto projection = ...` 使 `projection.query` 等成为 const 视图，不能作为 memcpy 目标
- **验证**：`test_tp2_forward` 的 oracle 扩为三组（同一权重、同一 KV 行、同一 300 token 序列）：
  - **路由等价**（T=7，A16，低于 A4 的 8-token 阈值）：分块 `Phase::Prefill` vs 逐 token `Phase::Verify`，
    argmax 相同、top5 差 0.56 —— bf16 残差流在 64 层里重排归约的固有漂移（0.4%×√64）
  - **分块等价**（T=300，A4）：`300→128+172` 与 `300→64+236` 两种切分与「一次 300」**逐位相同**（max diff 0），
    这是 KV 页与 GDN conv/recurrent 状态跨块续接的最强证据；`300→256+44` 因 44 列余块换用更窄的 A4 MMA
    schedule（且其前 3 列走 conv 的 state 分支而非输入 tap）而 top5 差 0.63、argmax 不变
  - **信息项**：A4 批量 vs A16 逐 token，max logit 差 1.99、argmax 相同；两 shard 的 logits 逐位相同
- **实测结果（64k serve，`--greedy` 与普通采样各测一轮）**：
  - 新鲜 2063 token prompt：**TTFT 1.5 s = 1.38k tok/s**（改造前 69.6 s / ~30 tok/s，**~46×**），
    已超 llama.cpp 的 403 tok/s（3.4×）
  - 同一 prompt 再问：cache 2051/2063（99.4%）、TTFT **42.7 ms**
  - decode 稳定 **32.1–32.2 tok/s**（此前 29.1）—— AR staging 8 MiB + uint4 向量化在 T=1 也生效，
    现已**超过** llama.cpp 基线 31.65 tok/s

## Round 35b：prefill 深度剖析（块宽可调 + allreduce 真因定位）

- **改动（全部保留）**：
  1. TP-2 prefill 块宽从硬编码 256 改为**读取引擎选项** --prefill-chunk（normalize_engine_options 对 TP-2
     分支不再强制覆盖，改为钳制到 [128,1024] 并对齐 128）；上限 kPrefillChunkMaximum = 1024。
     serve 默认 1024，可运行时扫参，无需重编
  2. workspace 128 -> **384 MiB**（实测 1024 宽块的 arena 峰值 = **122 MiB**，原 128 MiB 会 std::bad_alloc）
  3. DevicePair staging 8 -> **24 MiB**，并改为**按调用奇偶双缓冲**：peer 在读取前就发布 arrival，
     原单缓冲存在"本卡为第 k 次调用覆写、对端还在读第 k-1 次"的**真实跨调用竞态**（此前只是时序上没触发）；
     双缓冲后第 k 次写入的缓冲，对端在第 k-2 次之后已读完，竞态彻底消除
  4. ar_inplace_bf16 支持**多 block 分片**（每 block 一个 arrival slot，块内连续切片，块间无需协调），
     block 数按载荷取 1..16（<=64 KiB 的 decode 载荷仍走 1 block 以保延迟）
- **正确性（test_tp2_forward，全部通过）**：KV 容量提到 2048 token；新增 **T=1024 一次成块 vs 4x256 分块**
  用例，与既有 128/64 分块一样**逐位相同**（max logit diff 0，top5 gap 0）；两 shard logits 逐位一致；
  32 步 decode 序列不变
- **实测（64k serve，--no-prefix-reuse，1257 token prompt，chunk 扫描）**：

  | chunk | TTFT | prefill |
  |---|---|---|
  | 128 | 1146 ms | 1.15k tok/s |
  | 256 | 1032 ms | 1.33k tok/s |
  | 512 | 990 ms | 1.36k tok/s |
  | 1024 | 976 ms | 1.38k tok/s |
  | 1024（多 block AR 后） | 954 ms | **1.42k tok/s** |

  即：块宽从 128 提到 1024 只换来 **+20~27%**，预期的权重流摊薄没有兑现 -> 瓶颈不在权重流。
- **根因（nsys + 逐步排除）**：ar_inplace_bf16 占**内核总时间 48%**（1024 宽块内占该块约 60%），
  每次调用约 3.2 ms 且两卡完全对称。逐项排除后定位到**硬件链路不对称**：

      索引 0  RTX 5060 Ti  pci 01:00.0  Gen5 x8  (~20 GB/s 实测)
      索引 2  RTX 5060 Ti  pci 08:00.0  Gen4 x4  (~7 GB/s 实测)

  **CUDA device 1（第二张 5060 Ti）插在芯片组给的 Gen4 x4 槽上**。TP-2 每层两次 allreduce，
  载荷 [hidden=5120, T] bf16，写+读往返 = 4xhiddenxT 字节/层/卡；折算到每 token 是
  **2x64x5120x2Bx2 = 2.6 MB/token**，在 7 GB/s 上 = **约 375 us/token 约 2.67k tok/s 上限**。
  实测边际斜率 **0.368 ms/token = 2.7k tok/s** —— **正好贴着这个硬件上限**，且与块宽无关。
- **被否掉的方案（都有实测依据，已回退）**：
  - AR 多 block（1/2/4/8/16）：TTFT 859/877/879/879/830/861 ms，**无差异** -> 不是单 SM 发不出请求
  - copy engine 搬数据 + 极小 signal/wait 内核做会合：**更慢**（951 ms vs 857 ms）且两 shard logits 不一致
    （会合/双缓冲语义不成立），已整体回退，只保留内核路径 + 双缓冲 + 多 block 结构
- **结论**：prefill 的 allreduce **已经贴住硬件地板**，继续在软件层面榨 prefill 的剩余收益上限约 1.4x
  （把 AR 之外的 270 ms/块压掉），但**最大单项杠杆是硬件**：把第二张卡从 Gen4 x4 槽移到 CPU 直连的
  Gen5/x8 槽，AR 上限从 2.67k 提到约 7.6k tok/s。
- **下一步**：按用户既定优先级，prefill 到此为止，转 **Round 36（MTP）** 攻 decode（31 -> 60+ tok/s）；
  decode 的 AR 载荷只有 10 KB，不受 Gen4 x4 影响（实测 AR 中位 9-18 us，已从 42 us 改善）

## Round 35c：AR 有序切片流水（再拿 15%）

- **动机**：1024 宽块归因（dev0，655 ms 窗口，98% busy）：**AR 413 ms (63.1%)**、fp8_mma 104.7 ms (16.0%)、
  nvfp4_w4a4_mma 83.1 ms (12.7%)、其余（silu/state_passing/prepare_wy_wu/residual/attention/conv）约 35 ms。
  AR 是唯一大项，且是纯 PCIe 时间。假设是「PCIe 全双工 -> 写读重叠能再省一半」。
- **独立微基准（两张 5060 Ti，复刻引擎协议）**：tools/tp_bootstrap/bench_ar_protocol.cu

  | payload | 串行（改动前协议） | 4 切片有序 | 8 切片有序 | 16/32 切片 |
  |---|---|---|---|---|
  | T=1024 10.5 MB | **3.180 ms**（与引擎实测完全一致） | 2.724 | **2.704** | 3.84 / 6.24 |
  | T=189 1.94 MB | 0.596 | **0.519** | 0.561 | 0.85 / 1.55 |
  | T=1 10 KB | **0.023** | 0.042 | 0.109 | 0.31 / 0.77 |

- **关键结论（推翻「全双工翻倍」假设）**：这条 **Gen4 x4（走芯片组）链路双向合计只有约 7.9 GB/s**，
  不是每个方向 7.9 GB/s。21 MB 往返的物理下限 = 2.66 ms，**8 切片有序流水已达 2.704 ms = 98.5%**，
  即 AR 已无剩余空间可挖（除非减少字节数，那要改并行布局或降精度，均不可取）。
  切片数必须自适应：>16 切片被每片握手延迟吃掉（16 片 3.84 ms、32 片 6.24 ms），decode 必须保持 1 片。
- **实现**（src/core/tp/device_pair.cu，无模型层改动）：
  1. ar_inplace_bf16 增加**写序链**：block b 先自旋等 order[b-1] == token 再写自己的切片，
     使各切片按序落地、对端可提前读前段；arrival 槽位协议不变
  2. ar_slices() 策略：载荷 <=256 KiB 用 1 片（保 decode 延迟），否则 512 KiB/片、上限 8 片
  3. 槽位数组扩为 2 个 cache line（arrival 数组 + order 链），共用同一次 cudaHostAlloc
- **验证与结果**（64k 服务等价配置，--no-prefix-reuse，1257 token 新鲜 prompt，chunk 1024）：
  - 正确性：test_tp2_forward 全绿，**T=1024 一次成块 vs 4x256 仍逐位相同**（max logit diff 0，top5 gap 0）
  - **TTFT 954 -> 808 ms（-15.3%）**，prefill **1.42k -> 1.50k tok/s**，decode 31.2-31.5 tok/s 不变
- **prefill 剩余空间评估（本轮结束时的诚实账）**：1024 宽块约 580 ms = AR 约 350 ms (60%) + MMA 188 ms (32%)
  + 其它约 35 ms。AR 已贴死链路，**唯一剩下的大杠杆是「AR 与 MMA 重叠」**（AR 只用 PCIe、GEMM 只用 HBM/张量核，
  资源不冲突），需要：AR 移出计算流 + 子块流水（子块间按层传递 GDN/KV 状态）+ 事件同步 + 环形 staging，
  上限约 1.5x（prefill -> 约 2.1k tok/s），但有跨卡 rendezvous 死锁风险，属大重构
- **更便宜的等价杠杆（硬件）**：把第二张卡从 Gen4 x4 槽移到 CPU 直连的 Gen5/x8 槽，AR 上限从 2.67k 提到约 7.6k tok/s
  —— 收益大于上面的大重构，且零代码风险

## Round 36：MTP（决定性超越）
- 模型带 nextn 头（gguf 里 blk.64.nextn.*），llama.cpp 未用
- 一次 forward 出 2-3 token → 60+ tok/s
**侦察结论（关键）：MTP 在本仓库已完整实现，只是 TP-2 路径把它关掉了。**
- 单卡路径：CLI --spec mtp --draft-tokens K（K=1..5）；config.mtp 由 options.speculative==Mtp 打开，
  loading::bind_mtp 绑定 mtp/*（input_projection [h,2h]、embedding_norm、hidden_norm、final_norm、
  mtp/layers/0/ 的 FullAttention block、token_embedding/output_head 与目标共享）
- 算子层齐备：speculative_prepare_verify_inputs/ids、speculative_accept_greedy_drafts（greedy + 采样两条路，
  工作区容量函数也在）、mtp_prepare_next_round（轮次簿记）、mtp_pack_fc_input / mtp_split_attn_in（MTP 输入打包）、
  kSamplePurposeSpeculativeAccept/Correction/Bonus
- 执行层齐备：TextContext 已带 MTP 支持，**构造时传入 MTP KV 视图 + MTP cache 即启用**；
  mtp_forward_batch（桥接首轮提议）+ mtp_forward_ar_step（后续 K-1 轮自回归提议）+ proposal_argmax
- TP-2 现状：model_instance.cpp:88 在 TP-2 分支 **options.speculative = {} 静默丢弃**；
  tp2_generation_core.cpp:118 load.speculative = None；build_shard 里 enable_mtp=false、
  mtp_physical_page_groups=0，且 TextContext 两个 MTP 参数都传 empty_mtp
- **移植设计（关键决策）**：**MTP 提议完全跑在 shard A 上、MTP 权重整份复制到两卡**。
  理由：MTP 只有 1 层（权重约 0.2 GiB/卡），提议 1 个 draft token 的成本是 1/64 权重流（约 0.4 ms）+ 头（约 0.3 ms），
  分片只能省零点几毫秒，却要引入 peer lockstep + 2 次 allreduce + shard 视图改造。
  而 MTP 输入 [enorm(embed(t)) | hnorm(hidden)] 在两张卡上**逐位相同**（embedding 复制、hidden 经 allreduce 同步，
  测试已验证两 shard logits 逐位一致），所以 shard A 单卡即可算出正确提议，**零 allreduce、零新 forward 代码**。
  验证阶段才用两卡（主模型批量 forward）。
- **分阶段实施**：
  - **36.1 加载与状态（已完成并验证）**：normalize_engine_options 不再丢弃 TP-2 的 --spec mtp（dflash/dflash2 明确报错）；
    ctor 把 backend/proposal_head 传给 plan_load；tp_split_spec 把 mtp/* 强制 Replicated；
    shard 0 打开 enable_mtp、规划并物化 MTP KV（独立页表 + 执行行）；
    **验证**：test_tp2_load 新增断言（input_projection [5120,10240]、attention query [6144,5120] **两卡都整份**）
    在 --spec mtp 下通过、无 spec 路径无回归；真实 serve --spec mtp --draft-tokens 3 起来后
    prompt 55 / decode 31.5 tok/s（解码回路尚未接 MTP，符合预期）。
    附带修复：build_r35.sh 改为目录级 rsync（逐文件清单漏同步头文件已踩坑两次）；
    serve_supervise.sh 支持第三个参数透传额外 serve 选项
  - **36.2 提议（已完成并验证）**：单卡侧三处接线 + 两个"约定陷阱"：
    1. **MTP 输入是 final norm 之后的 hidden**（等于 lm_head 输入），不是 final norm 前的 residual。
       证据：单卡 prefill_chunk 把 `xf`（= rmsnorm(x, final_norm)）传给 mtp_prefill_chunk；
       ordinary_decode_batch 的 hidden 输出同样是 norm 后的。→ forward_tp2/forward_tp2_prefill 新增
       `mtp_input_hidden` 出参（可选），返回 norm 后 hidden。
    2. **MTP 的 embedding 右移一位**：`plan_mtp_alignment_window` 给出 `shifted_embedding_begin =
       chunk_begin + 1`，即第 i 列的 embedding 是 token[i+1]；最后一列用刚采出的 token。
       → priming 用 shifted ids，final chunk 的最后一列在设备上覆盖为 sampled token
       （所以 final chunk 的 priming 必须移到采样之后）。
    3. **第 q 列的输出预测的是 t_{q+2}**（不是 t_{q+1}）：full-prefix bridge 用
       position = prompt_tokens-1 + next_token = io.token，而 speculative_prepare_verify_inputs 把
       draft0 验在 frontier+1。→ **验收口径必须是"滞后一步"**：第 p 步的 draft1 对比第 p+1 步的
       target argmax。我最初的"同位置 argmax"口径是错的（0/2 是口径错，不是接线错）。
    - 实现：`mtp_propose_window()`（bridge: mtp_forward_batch + K-1 次 mtp_forward_ar_step，逐字对齐
      `mtp_bridge_and_propose`）、RoundState（begin/complete_round_state_layout + RoundStateSpec
      {backend=Mtp, draft_window=K, batch_capacity=1}，memset 0 → rope_delta=0，
      backend_kv_table_row=0）、TextContext 传入 MTP 执行视图 + batch cache +
      set_mtp_proposal_extent(K)；`DecoderStateSpec.mtp_kv_heads`（MTP 层复制 → 用**全量** kv heads，
      这是新增字段，因为 text KV 仍按 shard 切）。
    - **实测**：64-token 贪心生成，draft1 接受率 **52/67 = 77.6%**（正常 MTP 水平）→ 权重/KV/位置/移位
      全部正确。代价：提议在关键路径上且尚未接受，解码 31.6 → 22.4 tok/s（下一步接受后才回本）。
  - **36.3 验证/提交机制（已勘明，待实现）** —— 单卡的设计非常干净，TP-2 可以照搬：
    1. 验证 forward 用 `GdnStateAction::RecordForReplay` + `GdnReplayRecords` 存储
       （`target_verification.cpp:14`）：GDN 层把每列的 (q,k,v,{g,beta}) 写入 record 平面，**不动 live state**
       （`ops::gated_delta_net_replay_record`，text.cpp:1175）。注意 `UpdateInPlace` 且 width != 1 会直接抛错
       （text.cpp:1031）→ 多列验证必须走 RecordForReplay。
    2. `ops::speculative_accept_greedy_drafts(target_tokens, target_logits, drafts, current_extents,
       frontiers, anchors, licensed_tokens, licensed_counts, accepted_drafts, public_token_count,
       sampling, work, stream)` 决定接受前缀（贪心/采样两条路）。
    3. `ops::gdn_replay_fold`（include/ninfer/ops/gdn_replay.h，GdnReplayFoldRow{source_state_slot,
       destination_state_slot, commit_columns}）把前 commit_columns 条 record 折回到状态槽 →
       **commit_columns = accepted + 1**，无需快照/回滚，一次 forward 完成一轮。
    4. **关键简化**：`forward_tp2_prefill(T=K+1, first_position=窗口起点)` 本身就是验证 forward ——
       envelope/因果掩码/位置绑定完全一致（窗口内所有列都有效，无需 valid_columns 过滤），
       只差两点：(a) 需要**每列 logits** [V,T]（现在只算最后一列），(b) 调用前后设置 RecordForReplay。
       窗口 = [刚采出的 t_{p+1}, d0, d1, ..., d_{K-1}]，位置 = [p+1 .. p+K]；
       target 每列 argmax 与 d_j 比对 → 接受前缀；bonus = 第 j 列的 argmax。
    5. 待读（下一步）：`GdnReplayRecords`/`GdnReplayFoldPlan` 的规划与 workspace 分配位置
       （program_impl.h:596 replay_records；planning/startup.cpp），以及 accept 的 sampling/
       public_token_count 实参取法。
    6. **本阶段的验收口径**：批量验证算出的接受数必须与 36.2 的逐步口径一致（≈77.6%）；
       最终以 36.5 的"greedy MTP == 纯 greedy 逐位相同"为准。
  - **36.3/36.4 轮次语义（已推导完毕，实现清单见下）** —— 最关键的是**状态纪律**：
    1. 窗口 = [anchor, d0, ..., d_{K-1}]，位置 = [frontier .. frontier+K]，宽度 T = K+1；
       anchor = 刚采出的 token，**它的 GDN 状态转移尚未提交**（基准状态 = 它之前的那个 token 之后）。
    2. 验证 forward 用 RecordForReplay 只写 record、不动状态；每列 target argmax 与 d_j 比对。
    3. `speculative_accept_greedy_drafts`：接受 A 个 draft，licensed = [d0..d_{A-1}, bonus]，
       bonus = 第 A 列的 argmax（即位置 frontier+A+1，**它是下一轮的 anchor 且其转移待提交**）。
    4. `gdn_replay_fold` 的 `commit_columns = A+1`（= licensed 数 = 窗口列 0..A，即 anchor + 被接受的 draft），
       与单卡一致（prefill.cpp:782/792：`committed = accepted_tokens`）。
       → 状态槽必须 ping-pong：TP-2 现在 **`.slot_count = 1`**（tp2_generation_core.cpp:212），
       需改为 2（state_bytes 会自动翻倍，arena 公式 `(1+kReuseSnapshotCount)*state_bytes` 不变），
       source/destination 交替。
    5. 下一轮 MTP bridge 需要**接受边界处的 hidden**（位置 frontier+A 的 final-norm hidden）：
       单卡用 `speculative_select_accepted_hidden(target_hidden, accepted, selected_hidden)` +
       `scatter(selected_hidden, destination_slot, continuation_hidden_store)`（target_verification.cpp:41-44）。
       TP-2 版：验证 forward 顺带输出每列 final-norm hidden [hidden,T]，选第 A 列，存到按 destination slot 索引的
       continuation store（或直接存进一个 [hidden, max_slots] buffer）。
    6. 子部件清单（下一步实现）：
       (a) `forward_tp2_prefill` 增加可选出参：每列 logits [V,T] + 每列 hidden [hidden,T]（现有只投影最后一列）；
       (b) TP-2 core 分配 `GdnReplayRecords`（`plan_gdn_replay_records`，spec: layers=GDN 层数(48)、
           record_capacity=B(1)、width=T、conv_channels/qk_heads/value_heads/key_dim/value_dim 取 **shard** 配置）
           与 `ops::GdnReplayFoldPlan`（program_impl.cpp:156 的构造范式）；
       (c) 轮次循环替换现有 in-place 解码循环（关键：每个 token 的状态转移只能发生一次）；
       (d) accept 的实参：`token_domain = public_token_count`、`configs = sampling_a`（设备端 SamplingConfig）、
           workspace 用 `speculative_accept_greedy_drafts_workspace_capacity_bytes()` 预留；
       (e) 必须用 `RecordForReplay`（`UpdateInPlace` 且 width != 1 直接抛错，text.cpp:1031）。
    7. 本轮验收：批量验证的接受数 == 36.2 逐步口径（≈77.6%）；36.5 用 greedy MTP == 纯 greedy 逐位相同收口。
  - **36.3 实现状态（代码已写完但被一个算子白名单阻塞）**：
    - 已实现并编译通过：验证 forward 的每列 logits/hidden（forward_tp2_prefill 新出参）、
      GdnReplayRecords + ops::GdnReplayFoldPlan（**两卡各一份**——验证是两卡都跑，所以 record/fold 不能只在 shard 0）、
      mtp_anchor_hidden、priming 简化为"只补 MTP K/V + 抓最后一列 hidden"、
      轮次循环（提议 → RecordForReplay 验证 → accept → 输出策略 → fold → 下一轮 anchor/hidden）。
    - **修正**：B=1 时 fold 用 **in-place**（source=destination=slot 0）是算子明确允许的
      （gdn_replay.h: "A row may be in-place"），所以**不需要** slot_count=2（原计划第 3 条作废）。
    - **阻塞**：启动时 FATAL "gdn_replay_fold: unsupported record geometry"。
      根因：fold 只注册了两套**全量**几何 —— src/ops/linear_attention/gated_delta_net/replay.cpp:154-160
      只放行 (48, qk16, val48, conv10240) 与 (30, qk16, val32, conv8192)；
      record 算子自身的校验 replay.cpp:107（qk_heads==16 && value_heads in {48,32}）同样会拦；
      fold kernel 派发 recurrent.cu:190-208 只实例化 FoldGeometry48x48 / FoldGeometry30x32。
    - **TP-2 分片几何**：layers 48、qk_heads 8、value_heads 24、conv_channels 5120
      （全量 16/48/10240 的一半；模板要求 kValueHeads % kQkHeads == 0 → 24/8 = 3，分组比与原几何一致，
      且 kernel 对 head 数是泛型的：recurrent.cuh:447/477/504）。
    - **修复方案**：注册第三个几何 (48, qk8, val24, conv5120) —— geometry 结构体 + launch_replay_fold 派发分支 +
      两处白名单（replay.cpp:107、:154）；按算子契约，新实例化需要**独立数值 oracle**，
      端到端再用 36.5 的 greedy 逐位一致收口。
    - 几何注册之前 **MTP serve 无法启动**（fold plan 构造即抛），服务须以不带 --spec mtp 运行。
  - **36.1 原始设计要点**：normalize_engine_options 不再丢弃 TP-2 的 --spec mtp；ctor 把 backend 传给 plan_load；
    build_shard 对 shard A 打开 enable_mtp、规划 MTP KV（mtp_physical_page_groups、页分配 + 表行发布，与 text KV 同构）、
    把 MTP KV 视图/cache 传进 TextContext；tp_split_spec 把 mtp/* 对象强制 Replicated（否则按形状会被 GatherRows 切掉）；
    以 test_tp2_load --spec mtp 验证两卡加载成功
  - **36.2 提议**：shard A 上调 mtp_forward_batch + K-1 次 mtp_forward_ar_step；
    oracle：同一前缀下 TP-2 的 draft token 必须与单卡引擎一致
  - **36.3 验证**：主模型批量 forward T=K+1 并输出**每列 logits**（现有 forward_tp2_prefill 只算最后一列，需扩展），
    再走 speculative_accept_greedy_drafts（greedy）/采样接受路径
  - **36.4 回滚与轮次**：验证前快照 GDN 状态池（KV 按位置寻址，部分接受时被后续轮覆盖，无需回滚），拒绝则恢复；
    mtp_prepare_next_round 推进 anchor/frontier/budget；多 token 提交走 preview_model 已有的前缀接受
  - **36.5 正确性与性能**：**greedy MTP 的 token 流必须与纯 greedy 逐位相同**（最强 oracle）；
    采样路径按 spec 概率接受做数值 oracle；然后实测 tok/s（K=1/2/3，目标 60+）
- [ ] 遗留（非阻塞）：capacity 日志行 TP-2 下显示 pages 0/0（memory_summary 未接 shard 池）；
  timings 字段 prompt_ms/predicted_ms 全 0（TP-2 路径未填）
- 后续：MTP 落地 → 多请求并发冒烟（当前 max_concurrency=1）→ perplexity 健全性



## Round 36b：验证窗口的免费优化、折叠几何 oracle、"贪心逐位一致"口径作废

### 1. 免费优化：验证窗口不再重复投影最后一列（已完成并验证）
- forward_tp2_prefill 的 `logits`/`logits_peer` 改为**可选指针**：普通 prefill 传 [V,1]（只要最后一列），
  验证窗口传 `logits_columns` [V,T]（要全部列）并给 `nullptr, nullptr`。原来两者同时算：
  `logits_columns` 的最后一列与 [V,1] 完全同一份数据，却**再读一遍 lm_head**（每轮每卡一次权重流）。
- MTP 分支里两个从未被读取的 [V,1] buffer 一并删除；`forward_tp2_prefill` 增加
  "二者只能给一个"的显式校验。
- 实测（同一 96-token 贪心请求、前缀已缓存）：K=3 long 47.99 tok/s；改动前同一口径 46.7 tok/s
  ——收益与预估的 1.4 ms/轮同量级，其余被 allreduce 掩盖。

### 2. `gdn_replay_fold` 新几何的独立数值 oracle（已完成，通过）
- `tests/ops/test_gdn_replay_fold.cpp`：`FoldProfile` 增加 `qk_heads` 字段（原来是文件级常量 16），
  新增 7 个 {48, 8, 24, 5120}（TP-2 分片几何）用例，并修正 `kProfile` 与 `Hq=` 后缀。
- `build/tests/ninfer_gdn_replay_fold_test` → `OK gdn_replay_fold`（exit 0）。该用例自带主机侧
  独立 oracle：折叠后的 recurrent 槽、conv 历史、guard 字节逐字节比对。
- 结论：**折叠（状态提交）机制本身正确**，新几何实例化没有索引/数值问题。

### 3. 性能实测（temperature 0，同一 prompt，前缀命中缓存）
| 路线 | 96-token 请求 | long 请求（predicted_n == completion，自洽） |
|---|---|---|
| 纯 decode（无 --spec） | 31.49 tok/s | — |
| MTP K=1 | 41.99 | 42.43 |
| MTP K=2 | **50.47** | **54.03** |
| MTP K=3 | 47.92 | 47.99 |

→ **`--draft-tokens 2` 是当前最优点（+71% vs 纯 decode）**；K=3 的第 3 个 draft 边际接受率低，
每轮却多付一次 bridge+AR 与更宽的验证。

### 4. "贪心 MTP == 纯贪心逐位相同"不成立（已定性，原 36.5 口径作废）
- 现象：同一 prompt、temperature 0，四条路线**各自确定**（同路线重复请求逐位相同），但两两不同：
  按生成顺序（reasoning+content）plain vs k1 第 9 字符、plain vs k2 第 79、plain vs k3 第 72、
  k2 vs k3 第 72 就分叉。
- 推理：贪心验收算子（`speculative_accept_greedy_drafts`，kernel 注释明确 temperature<=0 且无 penalty
  时退化为"最长匹配 draft 前缀 + 分歧列 argmax"）保证**提交的 token 一定是验证窗口自己的 argmax**。
  因此只要各路线 target logits 相同，提交流必然相同；四条路线互不相同 ⇒ **验证窗口的 target logits
  依赖窗口宽度 T**。
- 随 K 变化的唯一输入就是窗口宽度（窗口内容 = 同一锚点 + 同一批 draft 前缀），所以这与 draft 质量、
  MTP KV、折叠状态都无关（第 2 节 oracle + 各自确定性排除了状态类缺陷）。
- 根因层：TP-2 的"验证"直接复用 `forward_tp2_prefill`（Phase::Prefill 的 chunk 核，T=K+1 一次算完），
  普通解码走 `forward_tp2`（T=1）；单卡引擎有一条**专门的 Phase::Verify 路径**
  （`target_verify_batch`，text.cpp:790 `run_layers(x, Phase::Verify, tap)`）。TP-2 没有 Verify 相位，
  把"验证"和"prefill"合成一个入口，代价就是验证列的 argmax 与 T=1 解码不逐位一致。
- 定性：输出仍是**验证路径自身 numerics 下的合法贪心续写**；每 ~15–20 个 token 出现一次近似平局的
  argmax 翻转，不属于状态/记账缺陷。但**不满足"跨路线逐位相同"**。
- 若要恢复逐位一致（下一步，未做）：给 TP-2 增加 Verify 相位（causal 相位与 valid_columns 语义对齐
  decode 核），或让验证的**列 0**走与 T=1 相同的投影/注意力核。先做最小定位实验：
  验证窗口 T=2 的列 0 logits vs T=K+1 的列 0 logits（同状态、同 token；RecordForReplay 宽度 2 合法），
  量化 max|Δ| 与逐列 argmax 一致率，判断是投影 tile 差异还是注意力/GDN 核差异。

### Round 36b 运维 note
- 切换 K 必须重启 serve（`--draft-tokens` 是启动参数）：`serve_stop.sh` → `: > serve_supervised.log`
  → 后台 job 跑 `serve_supervise.sh 65536 8088 '--spec mtp --draft-tokens N'` → `r36b_wait.sh` 等
  "listening on" 再发请求（不能用旧日志里的 listening 行）。
- 测量脚本：`r36b_fold_test.sh`、`r36b_wait.sh`、`r36b_req.sh <K>`、`r36b_cmp.sh`、`r36b_timings.sh`。

### Round 36c：采样参数下的服务（用户要求 --temp 0.7 --top-k 20 --top-p 0.80）
- 正确的 CLI 是 `--temperature`（没有 `--temp` 别名），top-k 上限 20：
  `serve_supervise.sh 65536 8088 '--spec mtp --draft-tokens 2 --temperature 0.7 --top-k 20 --top-p 0.80'`
- 实测（请求体不带 temperature/top-k/top-p，用服务端默认）：128-token 请求 50.03 tok/s，
  39-token（stop token 提前结束）48.26 tok/s；比贪心 K=2 的 54.03 略低
  （采样验收分支要按列建截断分布 + 随机接受，且 draft 是 one-hot 贪心提议，接受率下降）。
- 功能可用：http=200、输出连贯；走 `speculative_accept_greedy_drafts` 的 temperature>0 采样分支
  （kernel 注释：按截断目标分布做分布正确的拒绝采样）。
- 遗留：
  1. `[mtp] round ... rate=hit/checked` 的语义目前是"被接受 draft 数 / 轮数"，可以 >100%，误导；
     应改成"被接受 draft 数 / 提议 draft 数"（够 `mtp_draft_checked_ += mtp_drafts_` 一行）。
  2. TP-2 采样路径尚无分布正确性的统计 oracle（单卡侧有对应测试），当前只验证了功能可用 + 输出连贯。

## Round 36d — 用户报告的"输出死循环"与 MTP 崩溃

### 结论 1：死循环与 MTP 无关（已定性，可用）
- 用户测试提示词："创建一个HTML，内容是SVG绘制一个鹈鹕骑自行车的2D动画"。
- MTP 路由（K=2 + temp0.7/top-k20/top-p0.80，服务端默认 thinking xhigh）：700 token 全落在 reasoning，
  content=0，finish=length，reason=2491 字符，模型一直停在"构思/复述任务"。
- 同一请求走**非 MTP** 路由：同样 700 token 全部落在 reasoning，content=0（reason=2549）→ 与 MTP 无关。
- 请求体加 `"reasoning_effort": "none"`（非 MTP，max_tokens 1200）：reasoning=0、content=2893 字符
  （完整中文说明 + 完整单文件 HTML），32.3 tok/s。→ 用户所见"死循环"是**思考阶段失控**（默认 thinking xhigh
  吃掉全部输出预算），与采样参数无关（用户也观察到加参数前就存在）。
- 可用手段：请求体 `reasoning_effort: none|minimal|low`；服务端 `--default-thinking-budget N`；加大 max_tokens；
  必要时 presence/frequency penalty。

### 结论 2：MTP 路由存在真实崩溃（阻塞 MTP 稳定，待修）
- 现场：`tp2_generation_core.cpp:443` 的 D2H `cudaMemcpyAsync` 报 `cudaErrorIllegalAddress`。
  这是 prefill 之后第一个 stream 同步点，真正出错的 kernel 很可能在 prefill/priming 阶段而非 proposal 本身。
- 复现：MTP K=2 服务启动后第一个用户请求即崩（pelican + reasoning_effort=none, max 1200）；
  water-cycle + max 900 + thinking xhigh 亦崩两次（约 9s）。warmup 自身（pos=55/56）正常。
- compute-sanitizer(memcheck) 抓不到：workspace 是整块 384 MiB cudaMalloc，越界落在 arena 内部时不可见，
  只有越出 arena 才成 illegal address。
- 下一步：用 `CUDA_LAUNCH_BLOCKING=1`（tools/tp_bootstrap/r36d_launchblock.sh）让错误在该 launch 的
  CUDA_CHECK 处直接报出，定位到具体 op。
### Round 36e — MTP 崩溃根因与修复（已修，A/B 验证通过）
- 定位过程：在 TP-2 代码里临时插入 cudaStreamSynchronize 分阶段检查（prefill forward / priming /
  proposal bridge / ar step / verify / accept / fold），再在 mtp_prefill_chunk 内部细分；
  第一次收敛：prefill forward 干净、final priming 报错；第二次收敛：priming 内部 mtp_forward_stem。
- 根因：mtp_prefill_priming 把提示词最后一个 token 写进 MTP 输入 ids 时用了
  ids_t.data + (length - 1)；Tensor::data 是 void*，该表达式是**字节偏移**（GNU 扩展），
  于是采样 token 的 4 字节跨写到相邻两个 int32，产生 >= 2^24 的非法 token id，
  ops::embedding 按该 id 取行远离 embedding 表 -> cudaErrorIllegalAddress。
  短 prompt（warmup）时被破坏的 int32 恰好未超出合法 id 范围，所以只在真实请求上崩。
- 修复：改成 static_cast<std::int32_t*>(ids_t.data) + (length - 1)（元素偏移）。
- 附带修复 1（潜在越界）：MTP KV 物理页数原来与文本 KV 相同。参照单卡规划
  （planning/startup.cpp 的 mtp_extra_pages）：投机窗口会在 commit frontier 之后多写
  mtp_drafts_ 个位置，故 MTP pool 需 + ceil(mtp_drafts_/kPagedKVPageSize) 页，
  且 execution row 只映射逻辑容量（避免 publish 越界写表）。
- 附带修复 2：mtp round 的 rate 统计由「接受 draft / 轮数」改为「接受 draft / 提议 draft」
  （mtp_draft_checked_ += mtp_drafts_），不再出现 >100%。
- 验证（MTP K=2 + temp0.7/top-k20/top-p0.80，同一批先前必崩请求）：
  pelican+reasoning none x2 -> http=200，65/62 token，finish=stop，47-48 tok/s；
  water-cycle max900 -> http=200，70 token，52.4 tok/s；服务进程不再退出。
  接受率样本 rate=105/184（57%），与修复前按轮统计 ~55% 一致，说明 ids 破坏对 draft 影响有限。
- 排查工具（保留）：tools/tp_bootstrap/r36d_dbg.sh（自包含起服+连发请求+状态检查）、
  r36d_verify.sh、r36d_show.sh、r36d_analyze.sh。调试用的阶段同步已从源码移除。### Round 48 — 第 4 项端到端证明：118,869-token 提示成功
- 推荐配置（plain + fp8 KV + `--max-context 131072`）实测：**prompt_tokens=118,869**，http=200，96.8 s；
  另一例 prompt_tokens=19,869，http=200，19.5 s。=> 131,072 容量真实可用（bf16 KV 下上限仅 65,536）。
- 脚本 `tools/tp_bootstrap/r48_longctx.sh`；证据写入 `docs/tp2-dual-5060ti.md`。
- 注：两次 `content_chars=0` 属已知的思考预算行为（max_tokens=220 被 reasoning 占满），非错误。### Round 47 — W1 稳定阻塞点：`rope.cu:189`（两次独立假设下复现）
- 新增运行时开关 `NINFER_TP2_VERIFY_PHASE`（默认 Prefill，验证完已移除）用于低成本迭代；
  又补了「Verify 相位使用 [width, batch] 形状 positions」的绑定（两卡各绑）。
- 结果：仍稳定崩在 `src/ops/launcher/rope.cu:189 cudaErrorIllegalAddress`，说明 RoPE 在分片 width>1 下的
  张量布局/工作区假设未被满足，需在 RoPE op 与其调用处（rope 位置/形状契约）做更深的改造。
- 处置：停止继续试探（上下文预算耗尽风险高），核心调用恢复为 `Phase::Prefill` 并重新编译；
  `text.cpp` 中的 Verify 接线（sequence/valid-columns/backend-rows/positions 绑定 + 两处分片门控）保留、默认不启用。
- **W1 结论固定**：MTP 与 plain 的逐 token 一致需要完成分片 width>1 Verify 路径（当前阻塞 RoPE），
  本会话无法完成；MTP 以实验特性交付，推荐默认 plain + fp8 KV + 131k 配置（已验证）。### Round 46 — 交付文档数字口径更正
- 发现并更正 `docs/tp2-dual-5060ti.md` 中的 prefill 口径错误：早前 r39 的「长提示」填充脚本未生效，
  实测 prefill 请求只有 **279 token**，故 0.453 s 对应 **~615 tok/s**（非 ~3.3k tok/s）；已在表中标注 token 数。
- 另记录：新起服务后的**第一个请求**含 CUDA graph 捕获（fp8/131k 首请求 14.4 s），文档注明以稳态数字为准。
- 推荐配置（plain + fp8 + 131072）稳态：decode 256 token 8.10–8.14 s（与 bf16 持平）。### Round 45 — W1 收敛结论：TP-2 分片 batched verify 是引擎级缺口
- 本轮已尝试的接线（均已实现并可编译，默认关闭）：`forward_tp2_prefill(..., Phase)`；Verify 相位的
  sequence 绑定（batch/width/state-slot/valid-columns/backend-kv-rows，两卡各绑）；`batched_verify` 与 GDN
  `ph == Verify` 分支对分片做门控（走 decomposed 递推，两个相位数值一致）。
- 依次暴露的故障点：`tensor dimensions must be positive` → `gdn_input_proj_conv_record workspace:
  unsupported single-parent profile` → `residual_add` 非法访存 → `cudaMemcpy2DAsync` 非法访存 → **`rope.cu:189`**
  非法访存（`CUDA_LAUNCH_BLOCKING=1` 下 warmup 挂死）。=> 分片上的 width>1 Verify 路径缺多处形状/工作区接线。
- 结论：W1（MTP token-for-token parity）在本会话预算内无法完成；MTP 按实验特性交付（吞吐 +25% 端到端 /
  +50% 纯 decode），推荐默认使用 plain 路线。### Round 44 — 临时探针清理完成
- 脚本 `tools/tp_bootstrap/strip_tp2_probes.py` + `strip_tp2_probe_comments.py` 删除 4 处 env 探针块
  （ROUND_DETAIL/PARITY/TOKEN_TRACE/SKIP_FOLD）与 7 行残留注释；`grep NINFER_TP2_` 命中 0；编译 BUILD_EXIT=0。
- 保留的有意行为：verify 仍走 Phase::Prefill（附 TODO 说明 Phase::Verify 未完成）；MTP 为实验特性。### Round 43 — fp8 KV 质量证据（W3 完成度提高）
- 推荐配置（plain + fp8 KV + 131072 上下文）4 次散文采样：content 1365/859/1382/1602（中位 ~1374），
  最长 0 段全 0；对照 bf16 plain 基线 1233/763/1469/1456/1430/1621（中位 1443，0 段全 0）=> **fp8 KV 不损质量**。
  已写入 `docs/tp2-dual-5060ti.md` 的 KV 小节。
- 仍待办：(1) 清理 `tp2_generation_core.cpp` 的 4 处临时探针（830/851/922/953 行，需精确文本）；
  (2) W1 的 TP-2 batched verify（未完成，goal 不标 complete）；(3) 最终交付汇报。### Round 42 — W5 交付文档落地
- 新增 `docs/tp2-dual-5060ti.md`：推荐配置（`--kv-dtype fp8 --max-context 131072 --kv-capacity auto`）、
  benchmark 表（prefill/decode、plain vs MTP K=3、显存、KV dtype 对照、131k/262k 结果）、KV 量化说明、
  MTP 实验特性与偏差根因、本分支修复清单。
- 8088 当前运行推荐配置（fp8 + 131072 + auto，temp 0.7/top-k 20/top-p 0.80）。
- 剩余：清理 `tp2_generation_core.cpp` 里的临时探针（NINFER_TP2_PARITY/ROUND_DETAIL/TOKEN_TRACE/SKIP_FOLD），
  然后做最终 goal 评估（W1 未达成 => 不得标记 complete）。### Round 41 — W3/W4 解决：per-device smem opt-in（fp8/nvfp4/int8 KV 全部可用，131k 上下文）
- 根因（高置信，已修复并验证）：`prompt_fp8.cu:19-22` 等量化 prompt 注意力 launch 用**函数内 `static const`**
  调 `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 92,416 B)`。该属性是**每 device** 的函数属性，TP-2 两个分片
  各自 launch，第二个 device 从未 opt-in => 请求 >48 KiB 动态 smem 的 launch 报 `cudaErrorInvalidValue`。
  bf16 路径 smem <48 KiB 不需要 opt-in，因此只有量化变体失败。
- 修复：改为**按 device 缓存**的 opt-in（`cudaGetDevice` + `static bool attr_done[64]`），改动文件：
  `prompt_fp8.cu`、`prompt_nvfp4_non_rdc.cu`、`prompt_k8v4.cu`（脚本 `tools/tp_bootstrap/fix_per_device_attr.py`）。
- 验证（`r42_fp8check.sh`，`--devices 0,1 --max-context 65536 --kv-capacity auto`，256-token 请求）：
  bf16 8.153 s / fp8 8.091 s / nvfp4 8.093 s / int8 8.140 s，全部 http=200，容量行分别显示 KV 65,536 tokens <dtype>。
- 更大上下文（`r43_bigctx.sh`）：**fp8 + `--max-context 131072` OK**（KV 131,072 tokens，13,904 MiB/卡，请求 8.102 s）；
  **int8 + 131072 OK**（13,952 MiB，8.088 s）；`fp8 + 262144` OOM 失败（预期，KV 需 ~2× 显存）。
- 结论：**2 倍上下文（65k→131k）在 fp8/int8 KV 下显存与吞吐基本不变** => 推荐配置改为
  `--kv-dtype fp8 --max-context 131072 --kv-capacity auto`。### Round 40 — W3 范围定位：所有量化 prompt 注意力变体同点失败
- 实测：TP-2 + `--kv-dtype nvfp4` 失败位点 `prompt_nvfp4_non_rdc.cu:33 CUDA_CHECK(cudaGetLastError()) failed:
  cudaErrorInvalidValue`，与 fp8 的 `prompt_fp8.cu:36` **同类同点** => 不是单个 dtype 的问题，而是量化 prompt
  注意力路径共同缺失的前置条件（最可能：KV cache 的 scale 平面/view 未按该路径接线；bf16 无 scale 故正常）。
- 单卡对照实验**无效**：`--devices 0` + `--kv-capacity 32768` 被 CLI 拒绝（打印 help 退出），需先用 `--help` 确认
  单卡设备参数与 kv-capacity 约束后重做；本轮已把 `--help` 过滤结果记录在会话中。
- 下轮 W3 步骤：(1) 用 `--help` 修正单卡参数并重跑 r40 对照；(2) 若确认 TP-2 专属，则在量化 KV cache 创建处
  （scale 平面分配/绑定）与 `prompt_*_fp8/nvfp4.cu` 的 launch 参数上加最小诊断打印，定位缺失项。### Round 39 — W2/W4 实测数据（已入库）
- 环境：2×(5060 Ti 16GB) TP2，Qwen3.8-27B NVFP4，贪心，`--kv-capacity auto`，`tools/tp_bootstrap/r39_bench.sh`。
- prefill（~1500 token 提示，输出 8）：plain 0.453 s（≈3.3k tok/s），MTP K=3 0.526 s。
- decode（256 token，端到端含 prefill）：plain 8.153 s = **31.4 tok/s**；MTP K=3 6.532 s = **39.2 tok/s**（+25%）。
  注：更早的纯 decode 计数（token trace）为 plain 31.5 / MTP K=1 42.0 / K=2 50.5–54.0 / K=3 47.9–48.0 tok/s，
  与端到端口径不同，报告时需分别标注口径。
- 显存：plain 13,888 MiB/卡；MTP 14,582 / 14,322 MiB（权重 11.1 GiB + KV 65,536 token bf16）。
- **`--max-context 131072` 在 bf16 KV 下启动失败**：`cudaMalloc failed: cudaErrorMemoryAllocation` =>
  更大上下文必须启用 KV 量化（fp8/int8）或降低 kv-capacity；这是 W3 的直接动机。
- W3 fp8 失败位点细化：`src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cu:27-36` 的 kernel launch
  报 `cudaErrorInvalidValue`；同函数 19-22 行的 `cudaFuncSetAttribute`（`kCausalPromptFp8SmemBytes`）已成功、
  22 行未抛 => 不是 smem 超限。优先排查：(a) TP-2 跨卡/跨 context 传入了无效 `cudaStream_t`；
  (b) fp8 cache 的 `k_scale_pages`/`v_scale_pages` 未按分片分配；(c) `q.ne[2]`（tokens）在分片下异常。### Round 38c — W1 处置决定（证据充分）与后续路线
- 新证据：`CUDA_LAUNCH_BLOCKING=1` 下 TP-2 verify 版本 warmup **挂死**（非阻塞时崩溃）=> batched-verify 在
  head-split 分片路上存在底层缺陷（读写非法/未映射内存的典型双态），修复深度不可控。
- 机制结论（已由 37g/37h 定量支持）：chunk verify（Prefill 核）与 decode（Verify 核）的求和顺序/舍入不同，
  逐列 10–20% argmax 翻转，且被接受的 token 的 KV 行由 chunk 核写入 => 漂移永久累积 => MTP 轨迹持续偏离
  plain => 长生成退化（0 复读）。**结论：MTP 的 token-for-token parity 需要完整实现 TP-2 batched verify，
  本轮无法完成**；MTP 保持 opt-in 实验特性并在 docs 明确限制。
- 交付口径（按目标 1–5 重排优先级）：
  1. MTP：保持可用但标注 experimental；报告中给出实测数字（K=1/2/3 吞吐、质量对比、已知偏差与原因）。
  2. W2/W4：用 plain 路线完成 prefill/decode 基准矩阵（context × prefill-chunk × MTP on/off）与显存分解，
     给出更大上下文的最优参数（prompt cache / max-context / kv-capacity / prefill-chunk）。
  3. W3：修 fp8/int8 KV（`prompt_fp8.cu:36` invalid value、int8 行拷贝延迟报错）；修好后重跑 r38_kvsweep。
  4. W5：benchmark 文档 + 推荐 serve 配置 + README/docs 更新，清理临时脚本与 PLAN.md。### Round 38b — Verify 路径仍故障：需 CUDA_LAUNCH_BLOCKING 定位真实核
- 补齐 `active_valid_columns_`（I32[1]=tokens）与 `active_backend_kv_table_rows_`（I32[1]=0）绑定后重跑
  `r37_colparity.sh`：MTP 服务仍崩，报错点漂到 `text.cpp:91 cudaMemcpy2DAsync` `cudaErrorIllegalAddress`。
- 纠错：`copy_row_block`（text.cpp:76-98）**不是 bug** —— 该文件 all activation tensors 的 dim 0 是最内层连续轴，
  故 dpitch/width = rows*element、spitch = parent_rows*element、height = columns 是正确的；错误编辑未生效，未改动该函数。
- 结论：`cudaMemcpy2DAsync` 为异步调用，其报错可能是**更早的核故障延迟暴露**（先 residual_add，后此点）=>
  下一轮第一步：以 `CUDA_LAUNCH_BLOCKING=1` 启动 Verify 版本，拿到**真正的出错核与位置**，再针对性修形状/绑定。
- 启用 Verify 只需把 `tp2_generation_core.cpp` 的 verify 调用加回 `qwen::TextPhase::Verify`（当前为 Prefill + TODO）。
- W3 现状（本轮测得）：bf16 4 样本 content 1874/1797/0/0（仍有 188/206 字符 0 段）；fp8、int8 启动即失败，
  错误分别落在 `prompt_fp8.cu:36`（cudaErrorInvalidValue）与 `text.cpp:91`（cudaErrorIllegalAddress，疑似同源延迟报错）。### Round 38a — W3 现状：量化 KV 路径不可用（需修）
- `--kv-dtype fp8` 启动失败：`src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cu:36`
  `CUDA_CHECK(cudaGetLastError()) failed: cudaErrorInvalidValue`（prompt/prefill 段 fp8 因果注意力核启动参数非法）。
- `--kv-dtype int8` 启动失败：`text.cpp:91 cudaMemcpy2DAsync` `cudaErrorIllegalAddress`（KV 行拷贝的源/目标步长不匹配）。
- bf16（默认）4/4 请求 http=200，单请求 34.7/37.8/36.4/36.6 s（散文 400+ token，MTP K=3）。
- 结论：W3 不是「配置即可」，而是「两条量化 KV 路径在 TP-2/head-split 几何下有 bug」，需分别定位与修复；
  修复后重跑 `r38_kvsweep.sh` 取质量/显存/速度对比。### Round 37h — 根因模型收敛：bf16 KV 的核间累积漂移 + TP-2 缺 batched-verify 绑定
- 关键解释（与全部观测一致）：KV cache 为 bf16。chunk verify（Phase::Prefill）为「被接受的 token」写入的 KV 行
  由 chunk 核算出，而 plain 路由同一 token 的 KV 行由 decode 核算出 => 两者 bf16 舍入不同 => 逐 token 累积漂移
  => 约 54–80 个位置后 argmax 翻转（逐列 parity 首差 pos=80；早前单列探针 pos=55）。
  这解释了 37g 的「每一列 10–20% 不一致」「授权映射正确却仍偏离」「首分叉出现在 index 3–25」。
  推论：**只要 verify 不是 decode 等价核，两条路由的 KV/状态就无法逐位对齐**，W1 的 token-for-token 判据必须靠 Verify 相位。
- 单卡 MTP 路径已具备 width>1 的 batched verify（可复用为参考实现）：
  `text.cpp:396-418` MTP attention 的 batched 分支（用 `active_sequence_batch_`/`active_sequence_width_`、
  `active_valid_columns_`、`active_backend_kv_table_rows_`、`batch_mtp_kv_->batch_layer_view(0)`），
  `text.cpp:854-855` 是 batched verify 的绑定样板：`active_backend_kv_table_rows_` 与 `active_valid_columns_`。
- TP-2 在 Verify 相位缺的正是这套绑定：我只绑了 batch/width/source-slot，于是 MTP attention 走进 batched 分支后
  解引用了未绑定的 `active_valid_columns_`/`active_backend_kv_table_rows_` => 崩溃点为 `residual_add`（形状/指针不一致）。
- 下一轮最小步（按序）：(1) 在 `forward_tp2_prefill` 的 Verify 分支补齐 `active_valid_columns_`（I32[width*batch]，
  窗口列全 1）与 `active_backend_kv_table_rows_`（窗口列的后端 KV 行，可从 prefill 已有的行表取）；
  (2) 用 `r37_colparity.sh` 判定（期望 col* diff 大幅下降/为 0）；(3) 若 GDN 的 width>1 记录路径仍报
  `unsupported single-parent profile`，给分片加回退（复用 decomposed 记录路径）。
- 兜底交付口径（若 (A) 最终不可行）：MTP 作为 opt-in 实验特性，默认发布 plain 路线的最优配置，并在 docs
  中明确说明核间 bf16 漂移导致的 MTP 数值下限；但先继续 (A)。### Round 37g — 逐列 parity 定量：所有列 10–20% 不一致 => 必须走 Verify 相位（无便宜捷径）
- 探针升级为「逐列 T=1 重放」：探针前后各恢复一次 scratch 状态（保证只比较核差异），用 `forward_tp2`（=plain
  路由同一函数，Phase::Verify）按 `mtp_position + j` 顺序重放窗口每列，与 chunk verify 的 `target_tokens[j]` 比较。
- 结果（K=2，width=3，贪心，单请求 400 token）：col0 same=228/diff=24（9.5%，首差 pos=80）、
  col1 same=200/diff=52（21%）、col2 same=214/diff=38（15%）。
- 结论：**chunk verify（Phase::Prefill）与 decode（Phase::Verify）逐列数值不一致**，因此 draft 接受与 bonus 都可能
  与 plain 不同 => MTP 轨迹偏离 => 思考失控/0 复读。授权映射本身正确（37d 已证）。
- 唯一正确修法：(A) 让窗口校验走带 width>1 的 Verify 相位。当前阻塞：启用后 warmup 崩在
  `src/ops/launcher/residual_add.cu:27 cudaErrorIllegalAddress`（形状/工作区未按 width 推导）。
  下一步：定位 `text/layers/... verify columns=3` 诊断上下文与 mixer/mlp 在 Verify 相位下的形状推导
  （buffer/view 是否按 1 列而循环按 3 列），补齐后重跑 `r37_colparity.sh`（期望 col* diff=0）再跑 `r36h_trace.sh`。
- 探针保留：NINFER_TP2_PARITY（逐列 parity）、ROUND_DETAIL、TOKEN_TRACE、SKIP_FOLD。### Round 37f — Phase::Verify 接线的第二层缺口（形状/工作区）
- 修复尝试：`batched_verify` 增加 `shard_config_ == nullptr` 门控（head-split 分片没有全量行数的 fused record
  workspace，必须走分片 decomposed 路线；该门控保留）。
- 结果：warmup 越过 input projection，但在 **`src/ops/launcher/residual_add.cu:27`** 崩在
  `cudaErrorIllegalAddress`（http=000，无 FATAL 行）=> Verify 相位在 width>1 下需要 TP-2 的**张量形状/工作区**
  接线（residual/窗口 buffer 的列数与 width 绑定不一致）。
- 当前状态：verify 调用**退回 Phase::Prefill**（附 TODO），MTP 服务可用；`phase` 参数、sequence 绑定、
  shard 门控三处改动保留备用。
- 两条继续路线（下一轮择一，建议先 (B) 拿可验证收益）：
  (A) 完成 Verify 相位接线：查 `residual_add` 的 shape/workspace 契约与 `active_sequence_width_` 在 Verify 分支
      下的 buffer 尺寸（text.cpp 的 run_layers/attn_mix/mlp 侧），补齐 TP-2 的形状推导与 workspace 计划。
  (B) 混合数值修正（小步、可立刻判定）：chunk verify 只用于判定 draft 接受，**最后一个 licensed token（bonus）**
      改用 T=1 的 `forward_tp2`（天然 Phase::Verify）从同状态重算后再提交；观测到的错误正是 bonus 与 decode 不一致。
      判据：`r36h_trace.sh` 首分叉应显著后移（期望仅在 draft 误接受处偶尔分叉）。### Round 37e — Phase::Verify 接线尝试：两步阻塞点已精确定位
- 已实现（保留，default 关）：`forward_tp2_prefill(..., Phase phase = Phase::Prefill)`，phase 透传给
  `run_layers_tp2`；phase == Verify 时按单卡 decode 的做法绑定 `active_sequence_batch_ = 1`、
  `active_sequence_width_ = tokens`、`active_linear_state_source_slots_`（I32[1] = 0，两卡各绑）。
- 阻塞点 1（已解决）：不绑定 sequence 时 `text/layers/0 verify columns=3: tensor dimensions must be positive`。
- 阻塞点 2（未解决）：绑定后 `text/layers/0 verify columns=3: gdn_input_proj_conv_record workspace:
  unsupported single-parent profile` => **TP-2 的 workspace 计划没有为 width>1 的记录型 GDN op 注册 profile**。
  需要查 `src/models/qwen3_5/program/planning/startup.cpp`（已有 3 处 Verify 引用）与 workspace 规划：
  要么给 TP-2 路由此 phase/width 注册对应 profile，要么让 verify 的记录路径复用单卡已注册的 profile。
- 当前代码状态：`tp2_generation_core.cpp` 的 verify 调用**暂时退回 Prefill**（附 TODO），保证 MTP 服务能启动；
  phase 参数与绑定代码保留待用。
- 备选方案（若 profile 接线成本过高）：混合数值修正 —— 用 chunk verify 只做「draft 是否被接受」的判定，
  而**最后一个 licensed token（bonus）改用 T=1 的 `forward_tp2`（Phase::Verify）从同状态重算**并提交；
  观测到的错误正是「chunk 的 bonus argmax 与 decode 不一致」（pos=79: target[0]=13 vs decode 15）。
  代价：每轮多一次 T=1 前向（约 +30–50% 目标算力），但可立刻得到与 plain 一致的 bonus。### Round 37d — W1 根因分离完成：状态正确，缺陷是 verify 相位数值不一致
- round 细节探针（NINFER_TP2_ROUND_DETAIL=1，逐轮打印 drafts 与 verify 逐列 argmax）：
  `[round] pos=70 drafts=[97913,97237] target=[97913,97237,99986]`、`pos=73 drafts=[101729,4960] target=[101729,4960,130621]`、
  `pos=76 drafts=[16,15] target=[16,15,15]` 均正确（target[j]==drafts[j] 时授权、bonus=target[A]）；
  但 `pos=79 drafts=[15,96356] target=[13,15,98003]`：**target[0]=13 而 decode/plain 在该位置给 15** —— MTP 提交了错误 token
  （draft 本来是对的）=> 授权映射正确，错的是 verify 的逐列 argmax 本身。
- parity 探针（NINFER_TP2_PARITY=1，同状态下 verify 列 0 vs T=1 decode）：K=1 208 轮 190 一致、K=2 179 轮 127 一致，
  **首次不一致都在绝对位置 55**，之后约 10–20% 轮次因近平分翻转 => 与 plain 首分叉（index 3）指向同一位置。
- 结论：**状态/fold/conv/授权全部正确**；剩余质量缺陷与「0 复读/思考失控」源于 verify 用 Phase::Prefill（T=K+1 chunk kernel）
  与 decode 的 **Phase::Verify**（T=1）数值不同。
- 修复方向（已定位代码）：`src/models/qwen3_5/execution/text.cpp:1394` 显示 TP-2 的 T=1 decode 用 `Phase::Verify`；
  `text.cpp:1021` 的约束 `(ph == Phase::Verify) && (active_sequence_batch_ > 1 || active_sequence_width_ > 1)`
  说明 Verify 相位本支持 width>1（投机校验）。下一步：把 MTP 窗口校验从 `forward_tp2_prefill`(Phase::Prefill)
  切到 width>1 的 Verify 相位入口（查 text.h 的 forward_tp2_* 声明确认是否有 phase 参数或专门的 batch/verify 入口），
  然后用 `r36h_trace.sh` 判定（期望首分叉大幅后移/消失），再跑 6 次采样统计（期望 content 与 plain 同量级、无 0 复读）。
- 临时探针（修好后统一删除）：NINFER_TP2_TOKEN_TRACE、NINFER_TP2_PARITY、NINFER_TP2_ROUND_DETAIL、NINFER_TP2_SKIP_FOLD。### Round 37c — K 扫描（trace 首分叉）与质量扫描结果
- trace（贪心，与 plain 逐 token 比）：K=1 首分叉 3（content 707）、K=2 首分叉 10（content 0）、K=3 首分叉 7（content 401）。
  => 分叉很早且各 K 都发生；K=1/3 在该次运行里仍能产出正文。
- 质量扫描（temp0.7/top-k20/top-p0.80，essay 提示词，各 4 次，max 1400）：
  K=1 content = 0,0,2332,0（退化 3/4，longest0 = 0,0,2,0）
  K=2 content = 939,0,1850,0（退化 2/4，longest0 = 173,0,111,0）
  K=3 content = 434,1951,1632,1963（**退化 0/4**，中位 1791；longest0 = 93,158,511,221）
  plain 对照（6 次）：content 1233/763/1469/1456/1430/1621，中位 1443，longest0 全 0。
- 结论：K=3 质量已与 plain 同量级（中位 1791 vs 1443），但 **MTP 各 K 都出现 0 连续段（plain 从不出现）**
  => MTP 路由仍有质量缺陷，K=3 只是最轻。修复前建议 MTP 用 K=3。
- fold 契约核对（`include/ninfer/ops/gdn_replay.h:19-47`）：commit_columns 为 [0,T] 的 record 前缀；
  conv 历史 = tail_3(old_history || conv_record[0:commit_columns])；source_state_slot = 产生该行 record 时所用 slot。
  按此契约我当前的 restore(slot0 <- scratch) + fold(source=dest=0, commit=committed) 在纸面上是正确的
  => 剩余偏差更可能是「logits 数值/近平分」而非状态错位。
- 下一步（决定性 oracle）：在 trace 里同时打印两路由每个位置的**该 token 的 logit 值**（把 window_logits/采样前的
  logits 对应元素 D2H 回来），比较同位置同一 token 的 logit 差：若差极小而是 argmax 近平分 => 数值问题（需专用
  Verify 相位对齐 decode 数值）；若差异大 => 仍是状态/列语义错误。### Round 37b — 修正 37a：record plane **确实包含 conv**（新假设：record 列与窗口列的对齐/差一）
- `src/ops/linear_attention/gated_delta_net/recurrent.cu:113-117` 引用 `records.conv.data` 与 `states.conv_layer0.data`
  （带 `conv_layer_stride_bytes`），说明 record plane 里有 conv 记录、fold 也会重放 conv；我在 37a 的
  「record 不含 conv」判断是只看了 `validate_replay_record` 前 10 行（key/value/gate）导致的误判，已作废。
- 因此 restore+fold 的 conv 处理**应该**是对的，首分叉 4 -> 10 仍不一致的原因需重查。新假设（按优先级）：
  (h1) **窗口列与 record 列的对齐差一**：我的 verify 窗口是 [anchor, d0..d_{K-1}]（宽度 K+1），而 record 的
       `width`/列语义可能是「只含新 token」的 K 列；若如此，`commit_columns = committed` 会少推进 anchor 自身
       一列（状态滞后一列）=> 第 2 轮起出错，与首分叉 10 相符。需读 `GdnReplayRecords` 的 spec/width、
       `validate_replay_record` 全文、以及 record 写入时窗口列 -> record 列的映射，再决定是
       `commit_columns = committed ± 1` 还是把 verify 窗口改成不含 anchor 的 K 列。
  (h2) record 的 `rows`/batch 维度与 TP-2 单行用法不匹配（rows=1 时应为 1，需核对）。
  (h3) fold 必须在 verify 的 stream 之后同步（两卡 stream 顺序/依赖）：目前 fold 与 verify 同 stream 应已有序，
       但 restore 的 memcpy 与 fold 的 kernel 之间需确认无跨流依赖。
- 判定工具不变：`r36h_trace.sh` 看首分叉 index（当前 10；旧 4；skip-fold 25），改一处跑一次即可二分。### Round 37a — W1 关键发现（record plane 不含 conv 状态）
- `src/ops/linear_attention/gated_delta_net/replay.cpp:129-133`：record plane 只有 `key_record`
  [kStateDim, qk_heads, width, rows]、`value_record` [kStateDim, value_heads, width, rows]、
  `gate_record` [2, value_heads, width, rows] —— **没有 conv/短卷积的记录**。
- 而 `src/models/qwen3_5/state/state_image.h:34` 的 `LayoutRegion linear_conv` 表明 conv 状态是状态镜像的一部分，
  verify 的 `RecordForReplay` 会把它按整窗（K+1 列）推进。
- 结论：`restore + fold` 只复原/重放了线性注意力区域，conv 区域被 restore 退回却无人推进 => conv 滞后，
  这正好解释「restore+fold 首分叉 10 反而差于不 restore 的 skip-fold 25」；skip-fold 保持不变时线性+conv 都按
  K+1 走，全接受轮恰好正确，直到第一次部分接受（≈token 25）才分叉。
- 正确修法（二选一）：
  (i) 扩展录制：让 RecordForReplay 也记录 conv 转移、replay_fold 也重放 conv 区域（改动 ops + records 结构，通用但工作量大）；
  (ii) **推荐**：restore+fold 之后，用最后 `conv_width` 个**已提交** token 重建 conv 状态（等价于 prefill/`state_passing` 的 conv 预备逻辑），
       代价只有几个 token，且不依赖录制机制。
- 判定：改完后跑 `r36h_trace.sh`，要求首个分叉 index 大幅后移（理想为 400 token 内无分叉），再做 6 次采样统计。
## Round 37 — 长任务目标（用户授权自主推进，完成前不汇报）
目标：修好 MTP 一致性 -> 基准测试 -> prefill/decode 优化 -> 显存优化 + KV fp8/q8_0 量化 -> 放开不影响质量的
参数以支持更大上下文 -> 交付“双卡 5060Ti TP2 极致优化”的可用配置与文档。
### W1 MTP 一致性（判据：MTP 与 plain 贪心流逐 token 一致 + 6 次采样 content 与 plain 同量级、无 0 复读）
- 已确认根因：RecordForReplay 会按窗口（K+1 列）推进活 GDN 状态，旧代码又 fold 一次 => 双倍推进（首分叉 4）。
- 已实现：kRoundScratchSlot 快照 + fold 前 restore（首分叉 4 -> 10，仍未一致，MTP content 仍 0）。
- 待查：(c1) conv/短卷积状态是否在 record plane 内（若不在 => 状态滞后，与 10 差于 skip-fold 的 25 一致）；
  只读排查 decoder_state.cpp 的线性注意力状态/记录面布局与 records 的层/列覆盖。
  (c2) commit_columns 列起点差一。(c3) 兜底：活状态写 scratch 不动，或 UpdateInPlace 逐列（正确但慢）。
- 之后删插桩 NINFER_TP2_TOKEN_TRACE / NINFER_TP2_SKIP_FOLD，重跑 6 次统计 A/B。
### W2 MTP 性能/负载均衡
- 现状 GPU0 100%/GPU1 ~72%（MTP 层 + argmax/accept/fold/采样/策略只在 shard A）。
- ①两卡 lockstep 冗余跑 MTP 层（均衡，墙钟不变）②按 vocab 切 MTP 提案 lm_head + 窗口 argmax（真正缩短
  关键路径）③重扫 draft-tokens K=1/2/3。
### W3 KV cache 量化（fp8/q8_0）
- 先查现有能力（--help / docs / plan_cache 的 dtype 路径）；支持则测质量+显存+吞吐，不支持则评实现成本。
### W4 显存与上下文
- 量化每卡显存构成（权重/KV/workspace/state arena((2+kReuseSnapshotCount) 份)/prompt cache slot/页大小），
  逐项调参并记录对 max-context 的影响，给出推荐 --max-context 与并发。
### W5 基准与交付
- 基准矩阵（prefill/decode tok/s × 上下文长度 × MTP on/off × KV 量化 on/off，记录命令与硬件）。
- 交付：更新 README/docs（TP-2 + MTP + 量化 + 推荐参数）、清理临时脚本与插桩、收尾移除 PLAN.md。
### Round 36j — 快照+fold 修复：部分改善（首分叉 4 -> 10），尚未一致
- 实现：header 新增 `kRoundScratchSlot`（快照数组 `kReuseSnapshotCount + 1`、Arena `(2 + kReuseSnapshotCount)`）；
  每轮 verify 前 `snapshot_state(a/b, kRoundScratchSlot)`；fold 前把 scratch 拷回活状态再 fold
  （`source_state_slot = destination_state_slot = 0, commit_columns = committed`）。
- 结果（同一贪心 token trace）：首个分叉 index 4（旧：fold 叠在已被 verify 推进的状态上）**-> 10**；
  旧诊断「跳过 fold」是 25。=> 方向正确但未复原真实状态：fold 重放与真实逐列推进仍有差异。
  候选原因：①conv/短卷积状态是否在 record plane 内；②`commit_columns` 列起点偏移 1；
  ③record 列与窗口 token 的对齐。MTP content 仍为 0（reason 425/493）=> 未完成。
- 下一步（同一 harness 二分）：(a) restore + skip fold（验证 restore 本身不吃掉状态）；
  (b) restore + fold with `commit_columns = committed - 1`；(c) 把 conv 状态纳入快照/重放后重测。
- 判据不变：MTP 与 plain 贪心流逐 token 一致，且 6 次采样 content 与 plain 同量级、无 0 复读。- 补充（已核实）：`src/models/qwen3_5/execution/text.h:32-35` 的 `GdnStateAction` 只有 `UpdateInPlace` 与
  `RecordForReplay` 两个值，即 `RecordForReplay` 的本意就是「录制转移 + 照常推进活状态」，
  所以正确做法必须是从 **verify 前的快照**重放，而不是在被推进过的状态上再 fold：
  fold 行应写 `source_state_slot = <scratch>, destination_state_slot = 0, commit_columns = committed`。
- scratch slot 必须是 **新的** slot（不能复用 `kReuseSnapshotCount` 那几个：它们保存 prefill 边界状态供跨请求
  复用，decode 轮次写它们会破坏缓存），因此需要把每卡 GDN 状态 slot 数 +1（plan_cache / 状态分配处）。
- 实施步骤：(1) 状态 slot +1；(2) 每轮 verify 前把 slot0 拷进 scratch；(3) fold 用 source=scratch、dest=0；
  (4) 用 `r36h_trace.sh` 验证 MTP 与 plain 贪心流**逐 token 一致**（判据）；(5) 删除 NINFER_TP2_SKIP_FOLD 与
  token trace 临时插桩；(6) 再重跑 6 次统计 A/B（判据：content 中位数与 plain 同量级、无 0 复读）。
- GPU 负载观察（用户）：MTP 开启时 GPU0 100%、GPU1 约 72%。原因：`mtp_propose_window` 只跑在 shard A
  （每轮 K+1 次 MTP 层前向）+ 全部 argmax/accept/fold/采样/策略串行工作都在 A，B 在 text mixer lockstep 后
  空等。结构上属正常；优化方向：①两卡 lockstep 冗余跑同一份 MTP 层（利用率好看但不变快）；
  ②把 MTP 提案的 lm_head 与窗口 argmax 按 vocab 切开（每卡一半 + 少量交换）= 真正缩短关键路径。
  建议先修正确性再谈负载切分（当前 MTP 输出错误，性能数字无意义）。
### Round 36i — 根因确认：verify 推进活 GDN 状态 + fold 二次推进（双倍前进）
- 差分实验：`NINFER_TP2_SKIP_FOLD=1`（跳过 `replay_fold->execute`，其余不变）重跑同一贪心 trace：
  首个分叉 index 4 -> **25**（一致前缀延长 6 倍）。
- 机制：`tp2_generation_core.cpp:812-820` 用 `set_gdn_state_action(RecordForReplay, &records)` 跑 verify 的
  `forward_tp2_prefill`；该路径除录制转移外**也把窗口（K+1 列）的状态写进了活 slot 0**，随后 875-883 又对
  slot 0 做 `commit_columns = committed` 的 fold => 状态双倍前进（第 1 轮全接受时 6 列而非 3 列）。
  第 1 轮的 token 在状态被污染前算出，所以前 3 个 token 正确、第 2 轮起全错 —— 与 index=4 首分叉完全吻合。
  跳过 fold 时状态只前进窗口的 K+1 列，全接受轮恰好正确，直到第一次部分接受（committed<K+1）才错 => index 25。
- 影响面：状态错位后模型思考失控/复读 `0`/不产出正文；这是 MTP 路由 6/6 退化的直接原因（非采样参数、非 token 域）。
- 正确修法（下一步实现）：每轮 verify 前把活状态快照进一个 scratch slot，接受后以
  `source_state_slot = scratch, destination_state_slot = 0, commit_columns = committed` 执行 fold，
  使状态只由“已提交列”重建，不依赖 verify 是否顺手写了活状态（两卡各一份 slot，需相应扩 slot 容量）。
- 临时插桩（定案后删除）：NINFER_TP2_TOKEN_TRACE（提交点 token 跟踪）、NINFER_TP2_SKIP_FOLD（跳过 fold）。
- 服务状态：8088 = plain（MTP 在修好前不可用）。
### Round 36h — 贪心 token 流逐 token 对比：定位首个分叉
- 方法：临时在共享提交点加环境变量门控的 token 跟踪（NINFER_TP2_TOKEN_TRACE=1，打印 plain/mtp 每轮提交的
  绝对序列位置与 token id），同一提示词、temperature 0，两路由各跑一次，逐位置比较。
- 结果（index=生成的绝对位置）：
  plain: 97913 97237 99986 101729 4960 130621 16 ...
  mtp  : 97913 97237 99986 1710   97995 4960  130621 ...
  前 3 个 token 完全一致（第 1 个 MTP 轮次正确），**首个分叉 index=4（第 2 个 MTP 轮次）**；
  MTP 在第 4/5 位提交了两个目标贪心不会选的 token，之后流与 plain 相差 2 个位置。
- 结论：不是数值蝴蝶效应（1710 与 101729 无关联），而是第 2 轮起目标 logits/状态已错；第 1 轮用的是
  prefill 状态且正确 => 嫌疑集中在**第 1 轮 fold 之后的状态推进**（record plane 列起点 / commit_columns
  多列提交语义 / 录制列与窗口列的对应）。
- 下一步（决定性差分）：在 MTP 轮次中强制每轮只提交 1 个 token（只 fold 1 列、commit=1）再跑同一 trace：
  若逐 token 等于 plain => 单列路径正确，缺陷在多列 fold；若仍分叉 => verify 列 0/状态读取本身错。
- 临时插桩：tp2_generation_core.cpp 提交点处 NINFER_TP2_TOKEN_TRACE 门控打印（定案后删除）。
- 服务状态：8088 = MTP K=2（temp0.7/top-k20/top-p0.80）。
### Round 36g — MTP 路由输出退化：定位与域实验
- 统计 A/B（同一提示词、temp0.7/top-k20/top-p0.80，各 6 次）：MTP content 中位数 0（6/6 退化，reason 108–2611）；
  plain 中位数 1443（1233/763/1469/1456/1430/1621，0/6 退化）。贪心（temperature 0）同样复现：
  MTP 900 token 全思考 content=0，plain reasoning 177 + content 1524。=> 不是采样方差、不是提示词、不是采样参数。
- token 域实验：MTP 校验的 argmax/accept 原用 public_token_count(248077)，与 TP-2 plain 路由的 vocab(248320) 不一致。
  改成 vocab 后：dom 3 次中 2 次正常（content 639/415），dom2 4 次中 1 次正常（458）=> 3/7 vs 0/6，弱正向但不充分。
  决定：保留 vocab（TP-2 内部各路由必须同分布）；把「single-device 用 public_token_count、TP-2 plain 用 vocab」
  的域不一致记为待办：需统一，改 plain 路由必须单独验证。
- 精化假说：plain 与 MTP 的贪心流在约 79 字符（≈30 token）后才分叉 => 不是 GDN 状态错位（否则 1–2 token 内即分叉），
  而是 verify 走 Phase::Prefill 的 T=K+1 chunk kernel、与 decode 的 T=1 kernel 数值不一致，蝴蝶效应放大为思考失控。
- 下一步：给 TP-2 实现数值等价于 decode 的 Verify 相位（对应 36c 遗留），或逐列 decode 复算做 oracle，定位首个分叉列。
- 服务状态：8088 = MTP K=2（temp0.7/top-k20/top-p0.80）。
### Round 36f — MTP 路由输出偏离目标模型（新发现，未修）
- 用户实测（DSH GUI -> 8088 MTP K=2）：同一提示词 3 小时前（无 MTP）是完整 1000 字文章，
  现在出现「人工智能的000」「精准医疗的000」以及数百字符的 0 连续段（token 15 = "0" 的复读），
  17s/969 token 后被用户停止。服务端日志显示该请求末尾 `[mtp] round ... anchors=15 accepted=3` 反复出现。
- 受控 A/B（同一提示词、同一 max_tokens、同一采样参数，各 2 次）：
  MTP K=2 -> 278/140 token，content 360/0，finish=stop；plain -> 705/1387 token，content 1241/1081（完整文章）。
- **决定性的贪心 A/B（temperature=0，排除采样方差）**：
  MTP K=2 -> 900 token 全为 reasoning、content=0（思考失控）；plain -> 900 token，reasoning 177、
  content 1524（完整文章）。=> MTP 路由提交的 token 流与目标模型不一致，是真实缺陷（非采样参数问题）。
- 假说（待验证，优先级从高到低）：
  1. verify 走 Phase::Prefill chunk kernel + RecordForReplay，而 decode/plain 走 T=1 的 forward_tp2；
     窗口各列（尤其 col>=1 的 bonus 列）的目标 logits 可能与真实因果推进不一致（状态滚动/相位语义）。
  2. accept op 的 current_extents / round_lengths（=mtp_position）/ round_anchors 语义偏差，
     导致 accepted prefix 选择错误。
  3. fold 的 commit_columns 与 record plane 的列起点（anchor 列是否计入）不匹配，导致 GDN 状态错位。
- 下一步（决定性的判定实验）：对同一个窗口，逐列比较 verify 的 argmax（window_logits）与 plain 路径
  按 token 逐个 decode 的 argmax；首个不一致的列即定位点。
- 当前服务状态：8088 已切回 plain（无 MTP），MTP 在输出质量修复前不建议用于测试。

### Round 49 — W1 结案：根因是分片未写 replay 记录（已修复并验证）
- 方法：对照 `C:\llama.cpp` 的 Qwen3.5 MTP（`src/models/qwen35.cpp` 的 `graph_mtp`、`common/speculative.cpp:1324` 的
  `common_speculative_impl_draft_mtp`、`src/llama-memory-recurrent.cpp:193-203` 的快照回卷）来定判据与结构。
- 门控重估（推翻 36c/36g 的相位假说）：`Phase` 在 head-split 分片上**没有语义**——`attn_mix` 的 phase 形参未被使用
  （`text.cpp:871-989`），`gdn_mix` 的两处 `ph == Phase::Verify` 都被 `shard_config_ == nullptr` 排除
  （全文件 `shard_config_` 只有 1024/1069/1159 三处命中），`mixer_layer`/`mlp_layer`/`tp_mlp_delta` 的 `prefill`
  只用于 NVTX 名与异常文本 ⇒ 改相位不改变任何数值，36c 的「decode 等价 Verify 相位」计划作废。
- 判据重估：llama.cpp 的 verify 也不是 decode 等价核（`n_seq_tokens > 1` 走 chunked `GDN_CH`，`==1` 走 `GDN_AR`，
  `delta-net-base.cpp:433-446`）；被接受 token 的 KV 由该窗口写入后不回改（`server-context.cpp:3992-3998` 只截尾）；
  bonus token 取自同一次 K+1 verify（`sampling.cpp:697-702`）；`docs/speculative.md:213` 明确「需要精确一致时用 greedy」。
  ⇒ 「与 plain 逐 token 一致」为不可达判据，替换为「无退化 + 质量同档 + 有加速」。
- **根因**：TP-2 的 head-split 分片不产生 replay 记录。`gdn_mix` 的两条记录分支都以 `shard_config_ == nullptr`
  为条件（融合 width>1 记录算子只注册全量 16384 行 fused parent，`gdn_input_proj.cpp:860-864` 的 workspace 查询
  正是 36c 报出的 `unsupported single-parent profile`），而 `tp2_generation_core.cpp:872-886` 每轮仍然执行
  「恢复轮前快照 + `gdn_replay_fold(commit_columns=committed)`」，fold 消费的是**从未写过的记录平面** ⇒
  48 个线性注意力层每轮被回放成空日志（recurrent 不前进、conv 历史被同一份空日志重建），16 个全注意力层正常
  ⇒ 前几 token 连贯、随后 "0" 复读。与 36f/36g 现象吻合。
- **修复 1**（`text.cpp` 的 `gdn_mix`）：分片在自身 decomposed 路线上记录——conv 记录 = 该路线已物化的 `[q|k|v]`
  原始卷积输入（直接拷贝；通道序与布局同记录平面）；recurrent 改用 `ops::gated_delta_net_replay_record`
  （`replay.cpp:107-110` 显式注册分片几何 `qk_heads=8, value_heads=24`；契约规定其输出与归一化 `gated_delta_net`
  逐位相同）⇒ 窗口 logits/接受判定不变，只有状态转移变得可回放。另补 `causal_conv1d_silu.h` 的 row profile 文档
  （`(1024,1024,3072)/C=5120` 实现早已支持）。
- 副产物（避免重走弯路）：`src/ops/launcher/rope.cu:189` 是 `CUDA_CHECK(cudaGetLastError())` 检查点而非出错核，
  多设备交错下报错点不必等于故障核 ⇒ 36c 的「rope 维度契约 → 引擎级 rank-3→rank-4 改造」前提不成立。
- 验证 1：端到端 A/B（`r36f_stats.sh`，同提示词、temp0.7/top-k20/top-p0.80、各 6 次）MTP content
  1519/1318/1294/1195/1683/0（中位 1306、6/6 `longest0=0`）vs plain 1165/1452/1546/672/224/1237（中位 1201）；
  吞吐 MTP K=2 稳态 44.3–49.2 vs plain 31.6–31.8 tok/s（无回归）。唯一那个 0 是 reasoning 预算 artefact，plain 同样出现。
- **修复 2**（新发现的同族缺陷）：`small_t_fp8.cu`/`small_t_k8v4.cu`/`small_t.cu` 的小 T 核同样需要 >48 KiB
  动态 smem opt-in，但用的是**进程级 `static`**。单 token decode 用 32 KiB tile（低于 48 KiB 默认值）所以缺陷潜伏；
  MTP 窗口 T=K+1 切到 64 KiB tile 才暴露 ⇒ MTP + 量化 KV 的每个首请求都 `small_t_fp8.cu:56 cudaErrorInvalidValue`
  （kCausalHeadDim=256：4*64*256=64 KiB）。四路统一到 `src/ops/common/cuda_smem.h` 的按 (kernel, device) opt-in。
- 验证 2：fp8 KV @131072 + MTP K=2 三次采样 content 1530/1481/1708、`longest0` 全 0、finish=stop（`r49_fp8mtp_smoke.sh`）。
- 回归测试全 PASS：`gated_delta_net_replay_record`、`gdn_replay_fold`（含 48×8×24×5120 分片几何）、
  `gdn_input_proj_conv_record`、`gdn_input_proj_conv_snapshot`、`gdn_input_proj`、`tp_device_pair`、
  `softmax_attention`、`kv_cache_append`、`context_kv_materialize`、`sliding_window_attention`、
  `qwen3_5_tp2_load/forward --artifact`；`BUILD_EXIT=0`。
- 服务状态：8088 = 推荐配置（fp8 KV @131072 + MTP K=2 + temp0.7/top-k20/top-p0.80）。

## Round 50 — 提议头 A/B：`--lm-head-draft`

- 起因：`--lm-head-draft` 把 draft 侧输出头从与主头 weight-tie 的全量头（248320 行 Q8_G32_FP16）换成
  artifact 里的 indexed proposal head（131072 行频率短表、Q4_G64_FP16，附 `proposal/token_ids` 行→真实 id 映射；
  加载时经 `load.cpp:83-84` 直接替换 `mtp/output_head` / `draft/output_head`）。verify 仍用全量头，
  所以它只改「提议什么」，不改最终输出的 token。
- 脚本：`r50_proposal_head_ab.sh`（中文长文）、`r50b_code_ab.sh`（Python 代码）。两路配置一致：
  `--devices 0,1 --kv-dtype fp8 --max-context 131072 --kv-capacity auto --temperature 0.7 --top-k 20 --top-p 0.80
  --spec mtp --draft-tokens 2`，各 6 次采样、串行单请求，仅 opt 路加 `--lm-head-draft`。
- 中文长文：full 45.18 tok/s（43.20–52.25）、接受 2780/5118=54.3%、content 中位 1294；opt **50.10**（47.53–55.35）、
  2749/5136=53.5%、中位 1336；两路 `longest0=0`、无退化 ⇒ 吞吐 +10.9%、接受 −0.8pp。
- Python 代码：full 48.59（45.62–51.07）、接受 4532/7730=58.6%；opt **49.58**（48.85–54.87）、4326/8142=53.1%；
  两路都 finish=length（1400 token 预算全耗在 reasoning），内容不可比 ⇒ 吞吐 +2.0%、接受 −5.5pp。
- 显存：opt 每卡 +340 MiB（14600→14940、14338→14678），与 131072×5120 Q4_G64_FP16 ≈ 0.33 GiB 吻合 ——
  证实 `proposal/head` / `proposal/token_ids` 不以 `text/`、`mtp/` 开头，`tp_split_spec.cpp:51` 的名字反查表不收录它们，
  于是落到默认分支 `Replicated`：两张卡各存一份。
- 结论：短表遗漏 + indexed 头的 Q4 量化都会把它的 argmax 推离全量头（散文 −0.8pp、代码 −5.5pp），接受率随域下降；
  但 head 读取变便宜仍盖过损失（散文 +10.9%、代码 +2.0%）⇒ 是吞吐权衡而非白拿。8088 采用 `--lm-head-draft`
  （主用途是散文/对话），回退只需去掉一个 flag。
- 口径坑：TP-2 路径响应里的 `timings.draft_n` 恒为 0，接受率只能用 serve 日志的累计计数器
  `[mtp] round pos=… accepted=… rate=accepted/drafted`（drafted 恰为轮数×K）。

## Round 51 — 14.9 GiB/卡的显存构成拆解

- 方法：受控配置扫描（每次启动后只读 `nvidia-smi memory.used`，不发请求），配置间作差归因；再对 23.7 GB
  artifact 用 `tools/artifact/reader` + `tp_split_spec` 的切分规则算出每卡权重字节。全部对得上（误差 ≤2 MiB）。
- 实测（MiB，shard0/shard1）：2048 无投机 11872/11872；2048+MTP K2 12314/12306；131072 无投机 13904/13904；
  131072+MTP K2 14600/14338；131072+MTP K2+`--lm-head-draft` 14940/14678。
- 配置事实（来自 artifact config）：hidden 5120、vocab 248320、64 层（16 full + 48 linear）、attn 24 heads /
  **4 KV heads / head_dim 256**、GDN key 16×128、value 48×128、conv kernel 4。
- 关键推导：文本 KV = 16 层 × 每卡 2 个 KV head × 256 × K+V × 1 B = **16 KiB/token/卡** ⇒ 131072 恰好 2 GiB/卡
  （bf16 要 4 GiB，这就是它只能到 65536 的原因）；MTP 层 KV = 4 heads（未做 head-split）× 256 × 2 × 1 B
  = 2 KiB/token ⇒ 256 MiB，且只在 shard 0 ⇒ 这是 shard0 比 shard1 大的唯一主因（实测差 262 MiB）。
  GDN state = 48 × (128×128×24×4 + 5120×3×2) = 73.4 MiB，arena ×5 = 367 MiB。
- 每卡账本：分片文本权重 8480 + 复制文本权重 2472（embedding 1213 + lm_head 1213 + norms 46）+ MTP 权重 430
  + 提议头 341 + 文本 KV 2064 + GDN state 367 + workspace 384 + CUDA 上下文/对齐 137 + MTP KV 262（仅 shard0）
  + replay 记录 ~2 ⇒ 14939/14677，与实测 14940/14678 吻合。
- 可优化点（按收益排序）：① embedding 与 output head 各 1213 MiB 在两张卡上重复（合计 4.8 GiB），
  做词表维切分 + 一次 gather 可省 ~1.2 GiB/卡；② state arena 5 份中有 3 份是前缀复用快照、1 份是轮次 scratch，
  只降快照数可省 ~220 MiB/卡；③ workspace 固定 384 MiB/卡，而单层 prefill 峰值约 140 MiB。
- 顺带确认：本机 nvidia-smi 有 3 张卡，index 1 是 Tesla T10（0 MiB，未使用），两张 5060 Ti 是 index 0 和 2。

## Round 52 — 262,144 上下文：词表/隐层并行切分 + 显存腾挪

- 目标：artifact 的 `max_position_embeddings` = 262,144 token。Round 51 的账本给出三个方向（词表并行、
  快照数、workspace），本轮全部落地并逐条验证。
- **权重切分**（`tp_split_spec` / `tp_shard_views` / `weight_splitter` / `execution/text.cpp`）：
  - `text/output_head`（248320×5120 FP8 RowScale）与 `proposal/head`（131072×5120 Q4_G64_FP16）按**词表行**切；
    `text/token_embedding`（同为 FP8 RowScale）按**隐层列**切（不按词表：ids 保持全域，杜绝越界行）。
  - 两卡各算各的半块，再用 `write_row_block` + 一次 in-place allreduce 合并回 `[V,T]` 才采样/argmax；
    合并 = 两个不相交行块 + 另一半清零求和 ⇒ 按构造逐位精确，采样与 `speculative_accept_greedy_drafts` 不变。
  - 踩过的坑：`tp_shard_views` 的 ColumnParallel 必须用分片本地偏移（`begin=0, end=(n/2)*k`）；FP8 切片的
    `scale_ne[0]`、`scale_nb[1..3]`、`group`/`group_size` 要跟着新行数走（否则 `invalid FP8 weight`）；
    合并时 peer 半块要写到 `peer.shard_index_*`（写进自己那份会让 argmax 整体偏移 V/2=124160）；
    ids 广播要按 16 字节补齐（allreduce 要求 %16==0），并用 padded 缓冲的前缀 view 保持 `ne[0]==ids.ne[0]`。
  - Q4 是**枚举 (n,k) 表**（`q4_dispatch.cpp`），没有 65536×5120 这一档 ⇒ 新增 `select_q4_n65536_k5120`
    （与 131072 档同构：`gemv_r4_w1_direct` / `ksplit<65536,5120,4|8>` / `mma_r64_c128`）。
  - FP8 embedding gather 原先只认 hidden=5120 ⇒ kernel 按 D 模板化 + `embed_gather_fp8_supports_width` 查询，
    新增 `[248320,2560]` 域。
  - `proposal/head`、`proposal/token_ids` 不以 `text/`、`mtp/` 开头，正是 Round 50 它落到 `Replicated` 的原因；
    本轮把 `proposal/` 纳入名字表，并加 `TpSplitOptions::split_proposal_head`。
- **显存腾挪**：前缀复用快照 3→2（state arena 367→294 MiB：聊天模板的轮次间隙只需要 near rewind，
  实测 16k/65k 相同 prompt 重入 0.30/0.33 s）；workspace 384→192 MiB（单层 prefill 峰值约 140 MiB，
  超出是报告的 arena overflow 而非越界）。
- **结果**：262144 正常起服，`nvidia-smi` 每卡 **15614 / 15094 MiB**（余 697 / 1217）；台账 shard0
  `weights+ctx 11494.6 | kv 4644.2 | state 293.6 | record 2.6 | workspace 192.0`。预填 16057 token / 10.11 s
  = **1588 tok/s**；解码 512 token / 8.68 s = **59.0 tok/s**（223 轮、2.3 token/轮、每 draft 接受 64.6%），
  比 131072 时代的 46–49 tok/s 高约 20%（拖步切到两卡并行）；相同长 prompt 重入 0.30 s。
- **逐位一致的证明**：5 条 greedy 提示在 131072（优化前 `ab-control`）与 262144（`ab-final`）上 hash 全同：
  `c8540bca7a3c20de` / `589e247eccd554b7` / `b83495f177889c2a` / `7199385e4eb468b0` / `cc73c73125c6f621`。
- **有记录价值的死路**：把 MTP 层 KV 降到 nvfp4 可省 228 MiB、接受率几乎不变（392/510 vs 391/510），
  但 5 条 greedy 提示有 3 条输出不同 —— 不是精度泄漏进输出，而是 verify+fold 与单 token 解码是不同执行形状，
  **draft 模式一变、近似并列的取舍轨迹就变**。最终保留 fp8 MTP KV，买下「逐位一致」这条性质。
- 测试全绿：`tp2_load`（plain / `--spec mtp` / `--spec mtp --lm-head-draft`）、`tp2_forward`、`embedding`、
  `linear_tp2_split_fp8_head`、新增 `linear_tp2_split_grouped_head`（**真实** [131072,5120]、T=1/2 与全量 Op 逐位比对）、
  `linear_tp2_split_nvfp4`、`tp_device_pair`。

## Round 11 — ① TP-2 verify 上 exact-batch CUDA Graph（并修掉轮末跨卡竞态）

- 目标：PLAN §12 Round 11 的 ①（exact-batch CUDA graph）+ ③（AR 与 compute 重叠）+ ④（MTP draft 链下沉），
  不做 ②（权重 NVFP4 化，属部署期变体）。本轮完成 **①**。
- **基线测量**（新增 env-gated 相位计时 `NINFER_TP2_TIMING=1`，7 个 CUDA event + host 时钟）：
  `decode rounds=52 committed=127 avg_round=38.90ms mtp=3.70 verify=34.23 accept=0.08 copy=0.09 sync_wait=0.01 fold=0.00`。
  即 verify 占 88%，MTP 链 3.70 ms 次之；发射间隙实测 4.4 ms，比 PLAN:642 的 2.91 ms 估计更大。
- **实现**（`src/core/decode_graph.{h,cpp}`、`src/core/arena.{h,cu}`、`src/models/qwen3_5/execution/text.{h,cpp}`、
  `src/runtime/engine/tp2_generation_core.{h,cpp}`）：
  - `DecodeGraphDefinition::capture_group(defs, streams, body)`：**多流同时 capture**，每卡一张图；跨卡通信仍然
    只有 in-kernel AR 自旋，host 不参与，因此不必改 AR 协议（只把 epoch token 从内核参数改成设备端计数器）。
  - `TextContext::forward_tp2_window(...)`：capture 安全的窗口前向。每轮 ids/positions 走**可移植 pinned** 缓冲
    （capture 成 memcpy node，每次 replay 重读），attention envelope 固化为捕获期常量。
  - 桶取 `mtp_graph_profiles(max_context, mtp_drafts_)`：K=2、131072 下 8 个桶，每轮按可见上界选桶。
  - **arena 地址可复现**是上图的硬前提：新增 `DeviceArena::rewind(watermark)` + `position_arena(arena, floor, target)`，
    每轮丢弃 proposal 链的占用回到固定 watermark；已捕获桶的 watermark 低于当前请求时视为未捕获并重捕
    （启动 warm-up 请求的 watermark 更小，所以首个真实请求会重捕一次）。
  - `NINFER_TP2_VERIFY_GRAPH=0` 关闭、默认开启；eager 路径与捕获路径共用**同一个** `forward_tp2_window` 和
    **同一个**桶 envelope，所以 A/B 只差发射方式（这也是「逐位一致」的可信度来源）。
- **结果**（同一 bench 配方，graph 关 → 开）：decode 短提示 62.8 → **71.5 tok/s**（+13.9%），2048 上下文
  64.1 → **72.4 tok/s**（+13.0%），轮时间 38.90 → **34.47 ms**，其中 verify 34.23 → 29.80 ms（差额全在发射/调度，
  两路径的 kernel 参数与输入完全相同）；prefill 512 持平（1312 → 1334 tok/s）。
- **正确性**：5 个 prompt × 最多 256 token 的贪婪 A/B，eager 与 graph **逐字节一致**（含 reasoning_content）；
  graph 路径对是否插桩不敏感、跨会话可复现。
- **顺带修掉一处真实竞态（改动前就存在）**：一开始 eager 与 graph 在 5 个 prompt 里有 3 个不一致，而**任何**
  额外同步/回读（哪怕加在轮末）都会让 eager 变成与 graph 一致 ⇒ eager 路径对时序敏感。根因：解码轮结束时
  只 `cudaStreamSynchronize(shard_a_.device.stream)`，随后 host 立刻做 fold / state restore / 下一轮 window 与
  arena 复用，而 shard B 的 verify 尾部可能还在跑 —— 两卡之间**唯一**的顺序保证是 AR 自旋，所以对端尾部与
  host 的下一轮之间没有任何顺序。eager 路径靠「每轮上千次发射让 host 始终落后 GPU」长期掩盖了它，换成 graph
  replay 后 host 反超 GPU，才稳定地暴露出来。修法：轮末同时同步 A、B（`sync_wait` 实测仍为 0.01 ms）。
  修后 eager 与 graph 在全部 5 个 prompt 上逐字节一致。
- **MTP 输出质量**（非本轮引入）：与 `-Plain` 单 token 路径贪婪对照，5 个 prompt 只有 1 个逐字节一致，其余在
  近似并列处翻转 —— 与 Round 52 记录的性质一致（verify 窗口与单 token 解码是不同执行形状），不是 ① 的回归。
- 未做：④（draft 链去 host 化）、③（prefill AR∥MMA 子块流水）。
- **附带修掉的第二个缺陷（真实崩溃，非 ① 引入）**：全量测试里 `ninfer_qwen3_5_tp2_forward_test` 在第一层内
  报 `cudaErrorIllegalAddress`，而 HEAD 同一二进制通过。二分：强制 allreduce 走 host-staging 即通过 ⇒ 是本轮
  「设备端 token」的 in-kernel AR 引入。根因：token 计数器分配在各自设备的 device 堆上，但「哪个 shard 驱动
  pair」由调用方决定（该测试会让两个 shard 各驱动一次，于是 `stream_a` 属于 device 1），内核在 device 1 的
  context 里解引用了 device 0 的裸指针。修法：token 改放 **mapped pinned host memory**（两设备都可见），kernel
  侧每 block 读一次 + shared memory 广播 —— 若每线程直读系统内存，单次 allreduce +4 us、整轮 +0.5 ms。
  修后 12 个相关测试全绿（含 tp2_load / tp2_forward），eager/graph 仍逐字节一致。
- **最终实测（修完两处缺陷后的同一二进制，graph 关/开）**：avg_round 39.6 → **35.3 ms**，其中 verify 35.0 → **30.6 ms**；
  等输出的 2048 上下文解码 65.6 → **76.5 tok/s**（+16.6%）；prefill_512 两侧都是 1419 tok/s。
- **④ 复核结论（本轮结论，勿重做）**：PLAN 对 ④ 的描述（「每 draft 一次 `cudaStreamSynchronize`」，旧版
  `tp2_generation_core.cpp:543`）在当前代码里已经不成立：`mtp_propose_window` 整个窗口只有 **1 次 D2H + 1 次 sync**，
  每 draft 只剩一次 D2D 拷贝 + 一次 increment 内核；`mtp_forward_batch` / `mtp_forward_ar_step` / `mtp_forward_core`
  内部无 host 同步、无 D2H（grep 实证）。实测 `mtp=3.70~3.75 ms` 是真实 GPU 工作（两次 T=1 的 MTP 层前向，权重流受限），
  且与 verify 串行（同一对流）。剩余机会只有「fold 与 MTP 链重叠」≤0.8 ms（≈2%；fold 走 `replay.cpp`，不含 allreduce，
  理论上可另开流），但要改 AR token 协议与流序，收益/风险比不划算 ⇒ 判定 ④ 已完成，不再改。下一项做 ③。

---

## Round 12 — ③ prefill AR∥MMA 子块流水：实现、实测、否决

③ 是 PLAN §12 表格里 ①③④ 的最后一项（prefill 的 allreduce 与 compute 重叠）。本轮把它**完整实现并实测**，
结论是**在 TP-2 上无收益、慢约 5%，已整体回退**，只留下本记录。

### 实现（完整可用，未保留）

- `DeviceContext` 加第二条流 `collective_stream`（与 `stream`/`transfer_stream` 同生命周期、同
  move/dtor 语义）。
- `forward_tp2_prefill` 在 `NINFER_TP2_PREFILL_OVERLAP` 打开、phase=Prefill、`pair.in_kernel_allreduce()`、
  驱动卡 == `pair.a()`、tokens ≥ 256 时把 chunk 切成 A=[0,ta)、B=[ta,T) 两块，逐层交错推进：
  A.mixer → arm(0) → B.mixer → arm(1) → AR(A.mixer)/AR(B.mixer)（collective 流）→ A.res+A.mlp → B.res+B.mlp
  → A.res2 → B.res2。每个 AR 一对 event：`produced` 记在 compute 流、collective 流 wait，
  AR 后 `done` 记在 collective 流、compute 流 wait —— 依赖链 L,A → L,B → L+1 保持，
  GDN conv/递推状态与 KV 可见性由 compute 流自身的顺序保证。
  子块绑定用嵌套 `ScopedPositions`/`ScopedEnvelope`：positions/rope 是外层 tensor 的 slice，
  envelope 分别是 {first+ta,first+ta} 与 {visible_end,visible_end}。
- **关键约束（踩坑，值得记住）**：切分点必须落在**激活调度块边界**上。初版 ta=tokens/2（300→150+150）
  让 `ninfer_qwen3_5_tp2_forward_test` 的 chunk-split 不变量失败（`300 -> 128+172` 的
  max_logit_diff=1.14~1.60，因为参照的「单块 300」现在内部走 150+150，落到了不同的 MMA/tile 调度类）；
  改成 64 对齐（`ta = max(64, (tokens/2) & ~63)`，300→128+172）后**逐位一致**：
  `300 -> 128+172: max_logit_diff=0`、`300 -> 64+236: 0`、`T=1024 one chunk vs 4x256: 0`，
  且 300 与 1024 的 top5 与**未切分基线**逐个数值相同（`348=22.875 621=14.25 …`）。
  即：**层间 A/B 交错 + 双流 AR 在数值上与单块顺序执行完全等价**，这是 ③ 唯一确定的技术收益。

### 实测（2072 token 单请求 TTFT，同一二进制，前缀缓存不命中）

| 配置 | run1 | run2 | run3 |
|---|---|---|---|
| `NINFER_TP2_PREFILL_OVERLAP=0` | 1249 ms | 1189 ms | — |
| `NINFER_TP2_PREFILL_OVERLAP=1` | 1258 ms | 1314 ms | 1513 ms（冷启） |

稳定态 ≈1219 ms vs ≈1286 ms ⇒ 子块流水**慢约 5%**（方向在两次独立测量里一致）。

### 为什么模型错了

1. PLAN:651 的成本模型把 AR 当**串行**开销（"AR 占 prefill 48%，重叠后 ≈2.6k tok/s"）。实测不成立：
   in-kernel AR 的两个内核是对称的，两卡各自 drifted 推进时，本卡自旋等对端的时间本来就被**对端仍在跑的
   compute 掩盖**，可重叠余量远小于模型估计。
2. 子块切分的代价是真实的：T 减半后 GEMM tile 变窄，且**权重每层被流读两次**（prefill 是权重流敏感形状），
   实测这两项加起来盖过 AR 收益。
3. 因此 ③ 的最终形态就是**不带子块流水的单块 prefill**；`collective_stream` 与整套流水代码已回退，
   工作树干净回到 ① 的提交 `8f6a812e`。若将来出现真正 compute-bound 的 prefill（更宽 chunk、更快权重路径），
   可复用本节的 64 对齐约束与 event 拓扑重新评估。

---

### Round 13 — 删除 decode 循环里的 [mtp] 刷屏打印

用户报告：推理时 cmd 窗口被 `[mtp] round pos=… anchors=… accepted=… rate=…/…` 刷屏，`--log-level error` 也关不掉。

- **根因**：那行是 `tp2_generation_core.cpp` 里的裸 `std::fprintf(stderr, …)`，**直接写 stderr、不经 logger**，
  也没有 env 门控（同文件的 `[tp2-time]` 由 `NINFER_TP2_TIMING=1` 门控）。每 decode 轮一行，
  出厂配置 59.0 tok/s ÷ 2.3 token/轮 ≈ **26 行/秒**。它是历史 `NINFER_TP2_*` 探针清理的漏网之鱼，
  留下只因 `r53_decode_analysis.sh` 还在解析它。
- **实测写入成本**（同一格式行、无缓冲 `os.write(2, …)`、20000 行）：`2>NUL` 1.8 µs/行、
  `2>文件` 1.5 µs/行、**真实控制台 20–25 µs/行**（Windows 控制台写入同步、要过 conhost 一跳）。
  ⇒ 26 行/秒 × 20 µs ≈ 0.5 ms/秒 = **0.05%**；单轮 20 µs / 35.3 ms = **0.057%**。吞吐影响可忽略。
- **真正的风险不是慢而是卡**：cmd 默认开 QuickEdit，在窗口里点击/选中会让控制台停止消费输出、
  `fprintf` 阻塞在生成线程上，整个推理停顿到松手为止；stderr 接慢管道（`2>&1 | tee`）同样会背压。
- **改动**：删除该 `fprintf` 与只服务于它的两个计数器 `mtp_draft_checked_`/`mtp_draft_hit_`
  （`tp2_generation_core.{h,cpp}`，cpp −7 行 / h −7 +4 行）。接受率数据仍可得：
  `NINFER_TP2_TIMING=1` 的 `[tp2-time] decode rounds=R committed=C` ⇒ 接受 draft 数 = C−R、
  接受率 = (C−R)/(R·K)，K = `--draft-tokens`。
- `docs/tp2-dual-5060ti.md` 中"接受率取自 `[mtp] round` 计数器"的说法已同步改为上式。
  `tools/tp_bootstrap/r53_decode_analysis.sh` 是 Linux 时代的归档脚本（硬编码 `/home/zhuojun/prof/…` 日志路径），
  **未改动**：它的 acceptance 列现在会退化成 `no mtp`，需要时应改从 `[tp2-time]` 取。

---

### Round 14 — “续写边界”（方案 2）：实现、真实流量证伪、撤回

用户在 agent 会话里暴露的问题：稳态 `cache` 命中 99% 但 TTFT 仍不低，且周期性出现 `cache 0`。
先按方案 2 实现“把复用边界推到上一轮生成序列末尾”：新增状态槽持有 decode 结束时的 GDN 状态，
`cached_tokens_` 记 prompt + `generated[0..G-2]`、boundary = `prompt + G - 1`（最后那个生成 token 是下一轮
anchor，transition 故意未折叠），每卡 +73 MiB（`state` 293.6 → 367.0 MiB），并加 `NINFER_TP2_REUSE_TRACE=1` 诊断。

**实测（DSH agent 真实流量，36 条请求）**：`slot=0`（上一条 prompt 末尾）命中 35 次、`slot=1`（rewind）1 次、
**续写槽 0 次**；`continuation > shared` 36/36 成立，其中 33/36 条 `shared == prefill_end` **精确相等** ——
客户端下一条 prompt 与上一条 prompt 逐 token 相同直到末尾，然后在**第一个生成 token 处**分叉。
旁证：req#1 生成 499 token，而 req#2 的 prompt 只比 req#1 长 203 token（渲染出的 assistant 回合明显短于生成流）。
即 DSH 不回传模型生成的 token 流，而是重新渲染 assistant 回合，“续上上一轮输出”在协议层不成立。

**决定：撤回**（`tp2_generation_core.{h,cpp}` 回到 `1cdfab97`），保留 `NINFER_TP2_REUSE_TRACE` ——
它是判断“客户端是否原样回传历史”的唯一手段（每请求打印 `prompt/cached/shared/prefill_end/rewind -> reuse/slot`），
`state` 台账回到 293.6 MiB/卡。

**同轮定位到、但尚未做的三件事**：

1. 新会话/上下文压缩后的第一轮：req#16 prompt 63,182 只与缓存共享 14,768（= system prompt + 工具定义），
   而缓存的两个边界都在 106k 附近 ⇒ `reuse=0`、TTFT **46.9 s** 全量重算；约 11 s 是白丢的
   （在前缀末尾之后再加一个 system+tools 结束位置的锚点即可）。
2. 工具返回的大段新内容：req#4/5/6 的 suffix 8,966 / 14,634 / 8,895 token，前推只有 0.95–1.13k tok/s ⇒
   TTFT 8.0 / 13.7 / 8.9 s，且缓存已命中 82–90%，剩下的是**必须新算**的 ⇒ 这是 prefill 吞吐问题，不是缓存问题
   （冷启也只有 1.35–1.38k tok/s）。
3. 每请求 ~0.28–0.35 s 固定开销：18 token 的 suffix → TTFT 272 ms，519 token 的 suffix → 657 ms（边际只需 0.38 s），
   这是 TTFT 的地板。

**日志口径**：`| prefill X tok/s` 是 `(prompt − cache)/prefill_seconds`，在近乎全命中的请求上分子只有十几 token、
分母被固定开销主导，用户据此把 `prefill 70.6 tok/s` 误读成“从头 prefill”。`operational_log.cpp` 现在同时打印分子
（`prefill 70.6 tok/s (18 tok)`）。

## Round 15 —— 工具调用被当成文本：A/B 判定与量化无关

**现象**：agent 会话里模型偶尔把工具调用当正文吐出（`<tool_call> <function=todo_write> …`），前端提示
“工具调用方式不对”；有时被纠正后恢复。

**定位**：ninfer 的 `WARN req#N tool markup returned as text | <reason>` 给出四类原因
（`invalid tool name` / `undeclared tool` / `malformed structure` / `trailing content`）。逐字复述探针
（把截图里的标记原样喂回模型，换行版与空格版各一次）都被正确解析成 `tool_calls`；namespace 分组往返也正确
（声明 `namespace=dsh` 的 `todo_write` → 模型答出模板渲染的 `dsh__todo_write`，响应还原为
`name=todo_write, namespace=dsh`）⇒ 解析器与 chat_template 均无问题。

**根因（模型侧）**：请求只声明 **1 个**可直接调用的函数（日志 `tools 1`；口径已验证：1 个 namespace 包 2 个
函数时日志写 `tools 2`），而系统提示以 `tools.read` / `tools.pwsh` / `tools.todo_write` … 列出整套 SDK 工具。
模型有时直接对这些“能读到、不能直接调”的名字发结构化调用，或照抄成带点的 `tools.read`，被
`enforce_declared_names` 拒收后原样返回文本。

**A/B（判断 HF → ninfer 转换是否造成退化）**：3 个诱导 prompt（todo / read / pwsh）+ 1 个对照，
`temperature 0.7`，每臂两轮共 75 条工具请求：

| 模型 | 工具请求 ok | 标记类失败 | 撞输出上限 |
| --- | --- | --- | --- |
| 官方 `qwen3_8_27b_nvfp4.ninfer` | 38/75 | 27/75（21 个 `tools.*` + 6 malformed） | 10 |
| 转换 `qwen3_8_27b_w4a4_w8a8.ninfer` | 50/75 | 19/75（17 个 `tools.*`/`read` + 1 malformed + 1 undeclared） | 7 |

标记类失败率 36% vs 25%，差异不显著（两比例 z≈1.4，p≈0.16），**失败形态完全相同**（都是照抄 `tools.read`
这类带点名字）；对照 prompt 两臂都 10/10 ⇒ 转换没有造成工具调用能力退化。另一条轴是“思考过长撞输出上限”，
探针只有 300 token 上限放大了它，真实会话 32k 不会触发。

**结论**：不是 ninfer bug、不是 chat_template、不是量化；是“唯一可调用工具是 run_code、提示里却列出整套 SDK
工具名”这一提示结构下的模型侧照抄行为。ninfer 严格拒绝未声明调用是正确行为——放行等于让模型绕过 run_code
直接触发 pwsh/write。

**可复用探针**：`%TEMP%\ab_probe.ps1` + `%TEMP%\probe_t0..t3.json`，
`pwsh -File %TEMP%\ab_probe.ps1 -Tag <name> -Repeats 15`；脚本自取 `/v1/models` 的 id，
按 ok / bad-name / malformed / empty 分类并导出 `%TEMP%\ab_<tag>.csv`。

**顺带查明**：ninfer-serve 在 PATH 缺少 ffmpeg/libcurl 的 `bin` 目录时以 `0xC0000135`（DLL not found）
静默退出、零输出；从工具环境启动必须先把这两个目录加进 PATH（见 `tools/win_port/serve.ps1:69`）。

---

## Round 16 — TP-2 host 侧状态检查点（已实现，待端到端验证）

**问题**（Round 15 之后两次实测）：客户端在 turn 边界重渲染历史，新 prompt 比上一条**短**，分叉点落在历史中段。
TP-2 核心只有 2 个**显存**边界（上一条 prompt 末尾 + 一个 rewind 点），判据 `boundary <= shared_prefix` 都不满足
⇒ `reuse=0` ⇒ 整条 prompt 全量重算：

| 请求 | prompt | cached | shared | reuse | TTFT |
| --- | --- | --- | --- | --- | --- |
| req#24 | 12,143 | 12,362 | 11,872 | 0 | 7.8 s |
| req#43 | 57,378 | 58,611 | 43,239 | 0 | 42.0 s |
| req#51 | 74,536 | 90,424 | 57,376 | 0 | 57.6 s |

req#51 那段用 250 ms NVML 采样确认：两卡全程 98–100%，**不存在「只有一块卡在跑」，也没有真空期**
（采样里唯一一直 0% 的是 `gpu1` = Tesla T10，NInfer 不用它）；`done` 行的 `prefill 1.30k tok/s (74,536 tok)`
与 TTFT 57.6 s 一致。注意 `throughput` 窗口行的 `prefill 14.9k tok/s (74,536 tok)` 是另一套口径
（窗口内结算的 token ÷ 窗口长度 5.0 s，`src/serve/operational_log.cpp:340`），不是真实 prefill 吞吐；
真实值在 `done` 行（`operational_log.cpp:274-279`，除以 `prefill_seconds`）。

**根因**：不是 KV。walk 从边界开始、边界之前的 KV 页本来就不重算（`tp2_generation_core.cpp` 的注释）。
缺的是 **GDN（线性注意力）状态**——循环累积量，既不能从 KV 反推也不能平移；全注意力层可以按 LCP 随便截断
（llama.cpp 的 `n_past = slot.prompt.tokens.get_common_prefix(input_tokens)`，`tools/server/server-context.cpp:3103`，
只解决这一半），hybrid 模型必须靠上下文检查点兜底：llama.cpp `--ctx-checkpoints` 默认 32、
`--checkpoint-min-step` 默认 8192（`common/common.h:611-615`），创建在 `llama_decode()` 之前
（`server-context.cpp:3508-3518`），恢复时从新到旧找 `<= LCP` 的检查点，找不到才 `do_reset` 全量重算（`:3236-3249`）。

**实现**（显存 0 增量，检查点全部放 pinned host 内存）：

- 沿用引擎既有的 host state-image 预算 `--host-state-slots`（`ContextCacheOptions::host_state_slots`，
  `include/ninfer/types.h:136`）：TP-2 分支不动这个预算，其它 disabled-cache 路由照旧清零
  （`src/runtime/engine/model_instance.cpp:100`、`:135-142`）。
- `tp2_generation_core.h/.cpp`：`Shard::HostCheckpoint` 环（`PinnedHostBuffer`，一份 = `state_backing.bytes` ≈ 73.4 MiB/卡）；
  步长 `kReuseCheckpointStride = 8192` 起步，按 `ceil(max_context / 槽数)` 放大到 128 的倍数，保证
  `槽数 × 步长 ≥ max_context`（默认 8 槽 / 131072 → 16384；脚本里配 16 槽 → 8192）。
- prefill 每跨过一个步长，在同一个 shard stream 上 D2H 一份状态，记下 frontier 与 prefill id；
  复用扫描在原有 2 个显存边界之外，再取「**上一次 prefill** 的、位置最深且 ≤ `shared_prefix`」的主机检查点；
  `begin_gdn_state` 相应走 H2D 分支；KV 路径完全不动（前缀页本来就保留，suffix 照旧重算）。
- 正确性规则：只用**上一次完成的 prefill** 产生的检查点（prefill id 标记）——它的 `[0, shared)` 与本条逐 token
  相同，和显存边界是同一套论证；更老的检查点不保证同一位置上是同一批 token。被取消的 prefill
  （`cancellation.requested()` 提前 `return`）不发布 id，它写进环里的检查点自然被排除。
- 启动账本新增一行：`[mem] host-checkpoints shard N slots S x 73.4 MiB | stride X tok | pinned Y MiB`；
  `NINFER_TP2_REUSE_TRACE=1` 的 `[tp2-reuse]` 行多了 `host=<槽数> stride=<步长> ... src=host|device`。

**验证状态**：`ninfer_engine` 目标编译通过（`cmake --build build-win --target ninfer_engine`，BUILD_EXIT=0），
`git diff --check` 干净。**未做端到端验证**：需要重新链接 `ninfer-serve.exe` 并重启服务，而当前服务在跑、exe 被占用。

**待做的验证**：重启后起新一轮会话，让第二轮 prompt 比第一轮短且分叉在中段，确认 `[tp2-reuse]` 出现
`reuse>0` 且 `src=host`，`done` 行的 `cache` 从 0% 变成 `shared` 量级，TTFT 从 ~57.6 s 降到
`(prompt − checkpoint) / 1.3k + 固定开销`（16 槽 / 步长 8192 下 req#51 预计 ~13-15 s）。

### Round 16.1 — 第一次重启没生效 + 尾部检查点窗口

**用户把 `--max-context` 改成 204800 并重启后仍很慢。日志检查结论：跑的是旧二进制。**

| 证据 | 值 |
| --- | --- |
| `build-win/apps/ninfer-serve.exe` 时间戳 | `01:39:37` |
| `tp2_generation_core.cpp` 修改时间 | `02:58:20` |
| 日志里 `host-checkpoints` 行数 | 0 |
| 日志里 `src=host` 行数 | 0 |
| `[tp2-reuse]` 格式 | 旧的（没有 `host=/stride=/src=`） |

即：改了上下文、重启了服务，但没有重新链接 exe，所以 Round 16 的修复完全没跑起来。
只有**最终链接**会被运行中的 exe 挡住（Windows 锁定正在运行的映像，`link.exe` 报 LNK1104）；
编译与 `ninfer_engine.lib` 一直正常，所以库里是新代码、exe 还是旧的。停掉服务后完整构建一次通过。

**日志里的铁证**（旧二进制，`NINFER_TP2_REUSE_TRACE=1`）：

| 请求 | prompt | cached | shared | reuse | TTFT | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| req#9 | 53,167 | 56,462 | 11,872 | 0 | 39.0 s | 只共享 22% |
| req#12 | 53,763 | 53,750 | **53,592** | **0** | **39.0 s** | **只差 171 token 也全量重算** |

req#12 的两个显存边界是 `prefill_end=53,750`、`rewind=53,746`，都**在分叉点 53,592 之后**，
`boundary <= shared_prefix` 不满足 ⇒ `reuse=0`。rewind 槽停在末尾−4 的原因：`rewind_near_` 由**上一对**
prompt 的 gap 预测，而 req#11 那对的 `gap == 0`（完全命中，`if (gap != 0)` 不更新），更早的 req#10 那对 gap=2
被 `kReuseRewindMinimum=4` 夹到 4。**当前 gap 只有算完 LCP 才知道，结构上无法预测** ⇒ 检查点位置必须与预测无关。

**新增：尾部检查点窗口**（`kReuseTailWindow = 8192`）。除步长网格外，walk 的最后 8192 token 内**每个 chunk 末尾**
都存一份 host 检查点（对齐 llama.cpp 的 `near_prompt_end` 例外，它同样不受 `checkpoint-min-step` 限制）。
效果（`--prefill-chunk 1024`、32 槽 / 步长 8192）：req#12 的上一条 prompt 是 53,750，chunk 末尾为 1024·k 与 53,750；
尾部窗口给出 46,080 … **53,248** 这 8 份 ⇒ 下一条 `shared=53,592` 命中 53,248，只需重算
53,763 − 53,248 = **515 token ≈ 0.4 s**（原 39.0 s）。

**诚实的边界**：这只对「共享前缀长、只有尾巴被重渲染」有效。req#12 型（99.7% 共享）→ 0.4 s；
req#51 型（77% 共享，上一条 90,424）→ 命中 57,344，重算 17,192 ≈ 13 s（原 57.6 s）；
req#43 型（75% 共享）→ 命中 40,960，重算 16,418 ≈ 12 s（原 42 s）；
而 req#9 型（只共享 22%，客户端删掉 44k 历史）命中 8,192，仍要重算 44,975 ≈ 35 s ——
那 41k 后缀是真正的新内容，**物理上必须算**。

**200k 的显存余量提醒**（本次重启的 `[mem]`）：

```
[mem] shard 0 capacity 204800 | ... kv 3628.3 | ... free 176.0 of 16310.6 MiB
[mem] shard 1 capacity 204800 | ... kv 3225.0 | vision 826.5 | free 0.0 of 16310.6 MiB
```

B 卡余量 0.0 MiB（vision 塔也在 B 卡），任何额外显存分配（大图、图捕获）都可能 OOM；
host 检查点只占主机内存，不加重这一点。131072 时 B 卡还有 ~1 GB 余量。

**构建结果**：停掉 ninfer-serve 后 `cmake --build build-win -j 12` 一次通过，
`ninfer-serve.exe` / `ninfer.exe` / `ninfer-perplexity.exe` 全部重新链接（03:13），修复已进二进制。

### Round 16.2 — 有效性规则改正（谱系剪枝）+ 取消即失效

**发现**：Round 16 原来的「只认**上一次** prefill 产生的检查点」规则会砸掉最常见的那种场景。看 req#11 →
req#12：req#11 的 prompt 是 53,750、`reuse=53,594`，**它自己只走了 156 个 token** ⇒ 那一次 prefill 只会写出
一个 chunk 末尾（还是被跳过的 prompt 末尾）⇒ **它根本不会留下任何检查点**；而 req#10 那次留下了 46,080…53,248
这些尾部检查点，却被「只认上一次」的 id 规则整体排掉 ⇒ req#12 依旧 `reuse=0`。也就是说旧规则下这个修复在最关键的
场景里是空转的。

**改法：谱系有效性（lineage validity）**。每个检查点带 `valid`，规则两条：

1. **选择前剪枝**：`position > shared_prefix` 的检查点一律置 `valid = false`。论证：检查点的状态是「写它的那次
   prefill 在前 p 个 token 上的状态」，只有当整条谱系（上一次 prefill → 当前 prompt）对 p 之前的 token 一致时它
   才是当前 prompt 的状态；而 `shared_prefix = lcp(cached_prompt_tokens_, prompt)` 正是这个界限，越界者**永不可能**
   再被用上（后续 prompt 必须匹配一个当前 prompt 已经分叉的 token）。剪枝后取最深的幸存者。
2. **发布时生效**：prefill 成功走完后，把 `prefill_id == 本次 live id` 的槽标成 `valid`（id 精确圈定本次写的槽，
   环形回绕也不会认错）。没走完就取消的 prefill 永远不会执行这一步 ⇒ 它写的槽保持不可用。

这样一来，req#11（只走 156 token，不产生检查点）之后，req#12 仍然能用 req#10 留下的**尾部检查点 53,248**：
`shared = 53,592` 覆盖它 ⇒ 只需重算 53,763 − 53,248 = **515 token ≈ 0.4 s**。

**顺带修掉一个潜伏隐患**：prefill 中途被取消时原代码直接 `return`，`cached_state_valid_` 仍为 `true`，
于是「上一次 prefill 的边界 + 快照」被当成有效——可取消的那次已经改写了 KV，也可能覆盖过 rewind 槽的快照 ⇒
下一次请求可能拿到不一致的状态/KV。现在取消分支会 `cached_state_valid_ = false` 并清空整环，下一个 prefill 重新发布。

**验证**：`cmake --build build-win -j 12` 全量通过（`ninfer-serve.exe` 03:15:15），`ninfer_engine_options_test` 通过，
`git diff --check` 干净。trace 现在还会打印有效检查点数：`[tp2-reuse] … host=<valid>/<slots> stride=… -> reuse=… src=host`。
### Round 16.3 — 检查点环拆成 grid + tail 双子环（含实测与验证）

**动机来自实测**：用户会话的 trace 在 77k 上下文就到 `host=29/32`，req#18 直接 `host=32/32` 全满。原因是每个
prefill 都会贡献最多 8 个尾部锚点，而环按位置顺序覆盖 ⇒ **位置网格（深分叉唯一的覆盖手段）被逐步挤掉**，
200k 上下文下必然更糟。

**改法**：

- `kReuseTailCheckpointCount = 8`；环分两段：`[0, grid_slots)` 放位置网格（永不被尾部锚点覆盖），
  `[grid_slots, size)` 放尾部锚点（每次 prefill 刷新）。Shard 新增 `host_checkpoint_grid_slots` /
  `host_checkpoint_tail_next`，两个游标各自在自己的子环里循环。
- stride 按 **grid 槽数** 算：`max(8192, ceil(max_context / (slots - tail_slots)))` ⇒ 32 槽 / 200k 时
  `ceil(204800/24) = 8534 -> 8576 tok`（覆盖 24 x 8576 = 205,824 ≥ 204,800 ✓）。
- 尾部窗口自适应：`tail_span = min(8192, prefill_chunk x tail_slots)`，窄 chunk 不会灌爆尾部子环。
- 有效性规则不变（发布时按 prefill id 标 valid、选择前按 `shared_prefix` 剪枝、取消即失效）。

**验证**（停服务 -> 构建 -> 起服务，全部由我执行）：

- 停服务后 `cmake --build build-win -j 12` 全量成功，`BUILD_EXIT=0`，`ninfer-serve.exe` 03:37:40。
- 新账本：`[mem] host-checkpoints shard 0 slots 32 (grid 24 + tail 8) x 73.4 MiB | stride 8576 tok | pinned 2349.0 MiB` ✓
  （stride 8576 本身就是拆环生效的证据；旧版是 8192）。
- probe：`openai-chat` 58 token -> 203 ms，`cache 44 (75.9%)`，decode 76 tok/s ✓。
- `--preserve-thinking` 生效：`req#1 started | ... | preserve thinking` ✓（服务器默认写进有效语义）。

**本会话（用户会话）实测收益**（对照理论最小值 = prompt - shared）：`src=host` 命中 req#5/#11/#12/#19；
每条请求的实际重算量与理论最小值差额 <= 1,022 token（一个 chunk）；对照旧二进制：req#5 10.2 s -> 1.3 s、
req#11 37.6 s -> 28.6 s。剩下的耗时都是「客户端重渲染/工具结果带入的真新 token」，按 1.0-1.5k tok/s 必算。

**`--preserve-thinking` 的语义**（A/B 用）：`chat_template.cpp:611` 决定已结束轮次的思考是否保留；
Responses 路由的取值规则见 `openai_responses_state.cpp:159-165`：请求显式给了就用请求的（并标记语义变化），
没给且**有父记录**就继承父值，没给且**无父记录**就用服务器默认（`translate.cpp:118`）。用户客户端每轮发全量历史
（无 `previous_response_id`），所以重启后即为服务器默认 true。观察点：`shared` 是否还会中途塌陷。

**端口**：启动脚本里 `$Port` 曾被改成 3456，而客户端 `.dsh/settings.yaml:18` 指向 `8099`，已按 8099 恢复；
以后改端口两边必须同步，否则客户端连不上。

## Round 15b — 工具调用约束解码（路线 A）：实现、验证与两处既有缺陷（已修）

承接 Round 15 的诊断（不是量化、不是 chat_template、不是解析器、不是客户端配置；llama.cpp 做 lazy GBNF 约束解码，
NInfer 没有约束层）。本轮把「路线 A：文本化轻量版」落地，并在真实流量里连带修掉两处**既有** TP-2 缺陷。

### 1. 路线选择（A/B）

「工具名跨多 token 时掩码算错」这类正确性风险在 llama.cpp 上同样存在，但被其实现方式压得很低：
`llama_grammar_apply_impl`（`src/llama-grammar.cpp:1354`）把每个候选 token 用 `token_to_piece` 解码成文本、用
`llama_partial_utf8` 处理跨 token 的半个 UTF-8，再拿「已累积文本 + 片段」去 GBNF parse 栈校验——**按文本约束、对分词
不敏感**，且 parser 长期生产验证。若在 token 序列上建前缀树则分词敏感、易掩错。故分两条路线：

- **路线 A（本轮落地）**：状态机累积「已生成文本」（同 llama.cpp 的 `trigger_buffer`），每步把候选 token 用 NInfer
  tokenizer 解码成文本片段（处理跨 token 半个 UTF-8），校验「累积文本 + 片段」是否为已声明工具名/参数名的合法前缀
  （用声明名的 trie 加速），否则置 `-INF`。对分词不敏感，正确性风险对齐 llama.cpp，代码量远小于完整 GBNF。
  代价：约束激活的每步有 O(vocab) 的 CPU 掩码计算（~数 ms），但只作用于工具调用区（几个 token），摊销可忽略。
- **路线 B（Phase 2 选项，未做）**：直接搬 `llama-grammar.cpp` 的栈式 parser（~1500 行）+ 为 Qwen 生成 grammar。
  正确性风险最低（生产验证），天然支持完整 schema 校验。若 A 在真实流量仍偶发边界问题、或要做完整参数 schema
  校验，升级到 B。

不做「未声明也当 tool_call 返回」：等于放行模型绕过 `run_code` 直接触发 `pwsh`/`write`，破坏 PTC 契约。

### 2. 实现

- **新 op `apply_token_mask`**：`include/ninfer/ops/token_mask.h` + `src/ops/kernel/token_mask.cuh` +
  `src/ops/launcher/token_mask.{h,cu}` + `src/ops/wrapper/token_mask.cpp`（注册进 `src/ops/basic_sources.cmake`）。
  逻辑形状：BF16 `logits[rows, columns]`（dim0 连续）与同形状 U8 掩码，元素 `(v,c)` 在 `v + c*rows`；掩码为 0 处写成
  `-inf`（`__float2bfloat16(-CUDART_INF_F)`）。契约：rank-2、dtype/形状匹配、`ne[2]/ne[3]==1`、连续、非空、掩码只读、
  两侧不得别名。**在 CUDA Graph 之外调用**（graph 只捕获 verify forward，`window_logits` 从 graph 出来后再掩码）。
- **新 frontend 模块 `tool_call_constraint.{h,cpp}`**：
  - `ToolCallMaskTable`：每个 vocab id 一份解码字节 + 特殊 token 位；由 `Frontend` 用 `shared_ptr<const Tokenizer>`
    持有并**只建一次**（`frontend.cpp` 构造时 `build_tool_call_mask_table(tokenizer)`）；表行数 = tokenizer 公开词表，
    无效行给空片段 + special=1。
  - `ToolCallNameTrie`（256 叉）、`ToolCallGrammar`、`ToolCallGrammarState`（字节级状态机，模式
    `Free / FunctionLiteral / FunctionName / FunctionClose / ParameterName / ParameterValue / ToolClose`）、
    `ToolCallConstraint`（共享表 + 不可变语法 + 每请求可变状态，故按请求构造的是**非 const** 对象）。
  - 语法细节：触发词 `<tool_call>` 用**滚动匹配**（不是贪心整段匹配）；`FunctionClose` 用 `alive_` 位掩码同时跟踪
    `</function>` 与 `<parameter=`；`</parameter>` 用 KMP；任何错配即 `dead_`（本位置不再约束、不再有合法前缀）；
    空白只在 `FunctionLiteral / FunctionClose / ToolClose` 的 `progress_ == 0` 处跳过（`FunctionName` **不跳过**）。
  - **安全阀**：某位置若除纯空白外没有任何候选能推进语法，则报「不受约束」而不是发全 0 掩码——空白被跳过、不推进
    语法，掩到它会让模型一直吐空格直到 context 用尽；全 0 掩码则会卡死请求。`build_mask` 因此返回 `advances`
    （只有存活片段含非空白字节时才为真）。
  - **`dead_` 只在探测副本上读取**，历史不会污染候选判定。
- **`OutputSession`**：保留 Content 通道的原始字节流（`raw_content_text()`）并暴露 `in_reasoning()`。掩码与工具调用
  parser 吃**同一条**字节流 ⇒ 跨 token 的工具名、半个 UTF-8、跨越结构字面量的 token 都不会被误判。
- **`tool_call_parser.h`**：把 Qwen framing 常量（`<tool_call>` / `</tool_call>` / `<function=` / `</function>` /
  `<parameter=` / `</parameter>`）收敛为 `inline constexpr std::string_view`，删掉 `.cpp` 里的本地重复。
- **`frontend`**：`make_tool_call_constraint` 返回非 const `shared_ptr<ToolCallConstraint>`；契约为空 / 未
  `enforce_declared_names` / `tools` 为空时返回空。
- **`tp2_generation_core`**：plain decode 在 `ops::sample` 之前、MTP 在每个 verify 列 `ops::argmax` 之前应用掩码；
  每列掩码 = 「committed 文本 + 该列之前的 drafts」对应的语法位置（`build_mask_after`），draft 本身不掩码（非法
  draft 与掩码后的 argmax 不一致，因而被 `speculative_accept_greedy_drafts` 拒收）；约束只在 `!in_reasoning()` 时激活。
  掩码只在 `FunctionLiteral / FunctionName / FunctionClose / ParameterName / ToolClose` 生效，`ParameterValue` 与
  自由文本不掩码。
- **测试**：`tests/ops/test_token_mask.cpp`（GPU；逐位精确比对、掩码只读、全掩、形状/rank/dtype/别名校验；注册进
  `tests/ops/tests.cmake`）与 `tests/test_tool_call_constraint.cpp`（纯 CPU；惰性触发、只许已声明名、多 token 名字、
  完整名后必须 `>`、参数名、自由值、回到 Free、跨字面量 token、不可拼写名的回退、纯空白死路、二次调用、
  packed-domain 尾部；注册进 `tests/models/qwen3_5/tests.cmake`）。

### 3. 验证

- `ninfer_token_mask_test` / `ninfer_tool_call_constraint_test` / `ninfer_tool_call_parser_test` 全部通过；同一批
  `ninfer_sampling_test` / `ninfer_argmax_test` / `ninfer_engine_options_test` 也通过（6/6）。
- `ninfer_qwen3_5_frontend_test` 仍是 PLAN.md 703–707 记录的**既有**失败（`unsupported frontend/chat_template.jinja`，
  与本改动无关；曾用「临时关掉掩码表构建」bisect 证实 HEAD 同样失败）。
- **真实服务端到端**：用户 11:46 重启后 req#51（`openai-responses` 流式、`tools 1`、prompt 111,124 / output 733、
  `cache 110,158 (99.1%)`、TTFT 1.2 s）**正常结束、无任何报错** —— 域修复生效（修复前该请求必 500）。
- **未做**：对照 Round 15 的 75 条探针复测（需按同一探针脚本重跑）。

### 4. 既有缺陷 1：TP-2 的采样域是打包行数（已修）

约束表覆盖不到 logits 域时暴露：TP-2 核心有 4 处把**物理行数** `vocab` 当作有效域传出——`ops::sample`（prefill 首
token、plain decode）、`ops::argmax`（MTP target）、`speculative_accept_greedy_drafts`（接受核）。契约本来就把两者
分开（`sample` 校验 `token_domain ∈ [1, physical_rows]`，`argmax` 参数名即 `valid_rows`，accept 文档写 `token_domain`），
单卡路径（`text.cpp:2279/2283`、`decode.cpp:258`、`draft.cpp` 的 selector 域）都传 `public_token_count`；那枚**算了却
从未使用**的 `public_tokens` 正是这个意图的残留。

两个不同的词表概念：`resources.public_token_count` 是 tokenizer 公开词表（合法 id 域），`config.text.vocab_size`
（本机 248320）是**打包后**的 embedding/logits 行数；`frontend/resources.cpp:12` 只要求 `count <= vocab_size`。
修复：声明上提到 `vocab` 旁，4 处全部改传 `public_tokens`；`sampling_workspace_capacity_bytes(vocab, 1, 1)` 保留
（容量上界）。约束掩码同样按 **logits 域**生成（`build_mask(domain, mask)` / `build_mask_after(prefix, domain, mask)`），
tokenizer 未定义的行一律置 0（排除）。

**影响面**：越界 id 不会越界访存（embedding/lm_head 都是 248320 行，仅读到填充行），但会进 `OutputSession` 的
`Tokenizer::decoded_token`（`output_session.cpp:458/563`，无守卫）抛 `out_of_range` ⇒ 单个请求 500。TP-2 路径缺的
正是单卡 `program/decode.cpp:256 validate_licensed_tokens` 那层守卫。触发条件是「打包行胜过所有真实候选」（贪心要求
它是全域最大；采样要求它进 top-20 且活过 top_p/min_p），正常分布下几乎不会发生。

**风险论证（为何近乎无操作）**：`token_domain` 在核里只作遍历上界（`sampling.cuh:33/41/127`），两域的 `cap` 都是
`min(20, domain)`，RNG 抽取发生在截断后的候选集上 ⇒ 修复只在本该触发的那些步改变结果，正常步逐位不变。

### 5. 既有缺陷 2：penalty 的计数数组恒为空（已修）

`make_sampling_config` 把 `token_counts` 置空，而惩罚项只从该数组取 `c_v`（`sampling_device.cuh:252`：
`cnt = c.token_counts ? c.token_counts[v] : 0`，再加轮内 overlay），故 TP-2 路径的 presence/frequency penalty 对
**跨轮**重复完全无效——只剩 MTP verify 的轮内 overlay（同一窗口内的 draft）还起作用。

修法按单卡 `install_sampling`（`decode.cpp:145-149`）：仅当 penalty 非零时在请求 workspace 建 `I32[public_tokens]`
计数数组、`cudaMemsetAsync` 清零（arena 会交回上一请求的字节）后挂到 `sampling_config`；`ops::sample`（四个
finalize kernel 都 `atomicAdd`）与 `speculative_accept_greedy_drafts` 自行累加产出的 token。未配 penalty 时不建
数组、`c_v` 恒为 0，与单卡路径一致。该路由没有 forced-token 路径，故不需要 `increment_token_counts`。

**A/B 实测**（同 seed 12345、同 prompt、同参数；临时 revert `64db25f1` 编出修复前二进制，两份二进制各独占起一次
服务）：

| 运行 | 修复前 | 修复后 |
|---|---|---|
| penalty = 0 | 532 chars, `apple x30`, 最长同词连续 15 | **与修复前逐字节相同** |
| presence = 2.0 | **与 penalty=0 逐字节相同** | 不同（580 chars, `apple x15`, 去重词 69） |
| frequency = 2.0 | **与 penalty=0 逐字节相同** | 不同（717 chars, `apple x9`, 最长连续 2, 去重词 99） |

要点：修复前三种配置**逐字节相同**（penalty 完全无效）；修复后无 penalty 路径与修复前**逐字节相同**（无回归）；
修复后 penalty 生效且 frequency=2.0 把重复彻底压掉。统计含模型在 reasoning 里复述 prompt 的那些 `apple`（约 16 个），
所以 `apple x30` 不是纯生成量，但三次之间的相对变化不受影响。同 seed 重复两次逐字节相同 ⇒ 差异可归因于 penalty 而
非随机性。

### 6. 未做的观察（留档）

- `tp2_generation_core.cpp` 的 `sampling_b` 是死代码（只有 `(void)sampling_b;`），`sampling_buf_b` 也只为它而建。
- TP-2 路径建议补一个 `validate_licensed_tokens` 同款守卫（见 §4 影响面）。
- 服务 `/v1/models` 发布的 id 取决于启动参数/工件（实测同一工件：默认启动发布 `qwen3.8-27b`，另一次发布
  `qwen3.8-27b-w4a4-w8a8`，而 stderr 的 `engine ready` 行始终打印后者）。**请求里的 model 必须从 `/v1/models` 取**，
  不要从日志或硬编码推。

### 7. 运维注记（Windows 原生 / DSH 代理沙箱）

- DSH 文件策略处于 `workspace-write` 时 **ninja 无法执行任何子进程**：`cmake --build` 静默挂住，连一个平凡单边工程
  的 `build.ninja` 也一样（`ninja -t` / `ninja -n` 正常）。构建须在 `danger-full-access` 下进行。
- `pwsh` 后台作业里用 `Tee-Object`（或任何写进该作业 stdout 的管道）会在管道写满后中途卡死；改
  `cmd /c "... > log 2>&1"` 再轮询文件。
- `ninfer-serve` 必须由用户在自己的终端启动：`Start-Process` 起的服务是「调用方 shell 的 job object 子进程」，
  agent 工具调用结束即被回收（实测：单次工具调用内起的服务能顺利跑完并被我杀掉，跨调用即消失）。
  停服务 + 重链 exe 由 agent 执行没有问题（停掉的旧进程不占 exe，重链后由用户重启）。
- 编译前必须 `. ./tools/win_port/vcvars.ps1`；跑测试需要把 FFmpeg / curl 的 DLL 目录加进 `PATH`，否则加载器报
  `0xC0000135`。
- 端口：历史问题见 Round 16.3 末尾（脚本 `$Port` 与客户端 `.dsh/settings.yaml` 必须同步）；本轮实测服务端为 3456。

## Round 16b — 新会话分叉锚点：把分叉位置存进 host ring（Round 14 ① 遗留项已修）

Round 14 ① 记的那笔浪费在 Round 16 落地 host ring 之后**并没有消失**。Round 16 的 ring 是两个子环：位置网格
（`grid_slots` 个槽，步长 `max(8192, ceil(max_context/grid_slots))`，保证覆盖整个上下文）和尾部锚点
（`kReuseTailWindow` = 8192 内的 chunk 前沿）。而**实际启动没有传 `--host-state-slots`**（`tools/win_port/serve.ps1`
不含该选项），故取默认 8 ⇒ `tail=4, grid=4` ⇒ `stride = max(8192, ceil(131072/4)) = 32768`。新会话只共享
system+tools（~14.8k），**落在第一个网格点之前**；被继承 prompt 的尾部锚点全在它的末尾（~106k），按 pruning 规则
（`position > shared_prefix` 即失效）全部作废 ⇒ `reuse=0`，全量重算。

### 改动

新增第三个子环 `divergence`（`kReuseDivergenceCheckpointCount = 1`），**加在 `--host-state-slots` 预算之外**
（每 shard +73.4 MiB pinned host；若从 grid 里抠槽，小 ring 会把 stride 从 32768 推粗到 43776，反而伤到"深处分叉"
那个场景）。prefill 走查时若 `shared_prefix` 严格落在本次走查区间内，就把 chunk 截断到该位置并 D2H 冻结一份状态；
下次渲染同一稳定块的请求即从该处重启。命中判据复用现有扫描，未新增路径。

同一次改动还修了 `snapshot_host_checkpoint` 的一个既有缺陷：原实现只覆写 `position`/`prefill_id` 就发起 memcpy，
而末尾的 publish 扫描只在走查正常结束时才跑；若 prefill 中途抛异常，该槽会保留**上一次 prefill 的 `valid=true`**
配上新 `position` 和可能撕裂的 buffer，后续请求会据此恢复一份错误状态。现在写入前先 `valid = false`。

### 关键修正：锚点不能落在观测到的分叉点上

第一版直接实测失败：

```
B: prompt=35361 cached=35463 shared=17457 host=1/9 -> reuse=47  slot=8 src=host
C: prompt=17472 cached=35361 shared=17456 host=0/9 -> reuse=0   slot=0 src=device
```

B 把锚点写在 17457，而 C 与 B 的 shared prefix 是 17456 —— **差一个 token**。"首个不同 token 的下标"取决于稳定块
末尾那个跨界 token 吸收了多少后续字符，而该 pair 的后续文本不同。锚点落在分叉点上时，下一次 pruning 会把它直接
杀掉（`host=0/9`）。改为锚定在分叉点前 `kReuseDivergenceMargin = 8` 个 token：代价是 8 个 token 的重算（~5 ms），
换来整段前缀不丢。

### 实测（原生 Windows，`--max-context 131072`，默认 `--host-state-slots`）

场景几何刻意贴近真实部署：A = nonce + HEAD(~17.5k) + tailA(~17.5k)，B = 同 nonce + HEAD + tailB，C = 同 nonce +
HEAD + 短尾。HEAD 末尾距 B 的末尾 ~17.5k token，**远超尾部窗口（4096）**，所以尾部锚点够不着。

| 请求 | prompt_n | cache_n | prompt_ms |
|---|---|---|---|
| A_long | 35421 | 44 | 23774 |
| B_deep | 35363 | 0 | 23401 |
| C_short | 23 | **17451** | **71.8** |

服务端追踪（`NINFER_TP2_REUSE_TRACE=1`）：

```
B: prompt=35363 cached=35465 shared=17459 ... host=0/9 stride=32768 -> reuse=0     slot=0 src=device
C: prompt=17474 cached=35363 shared=17458 ... host=1/9 stride=32768 -> reuse=17451 slot=8 src=host
```

C 命中的是 `slot=8`，即 **divergence 子环**（grid 0-3、tail 4-7、divergence 8），不是尾部分窗；且 C 的 shared
(17458) 比 B 的 (17459) 少一个 token，那次漂移正被 8 token 余量吸收。B 付全量 35,363 token / 23.4 s（即改动前每个
新会话的代价），C 只走 23 token / 71.8 ms。启动账本：

```
[mem] host-checkpoints shard 0 slots 9 (grid 4 + tail 4 + divergence 1) x 73.4 MiB | stride 32768 tok | pinned 660.7 MiB
```

**收益**：同一稳定块的第 2 个及以后的新会话各省 ~10.8 s。第一个分叉请求无法受益——它的分叉点在走查前不可知，
锚点是它自己写下的。

### 验证与遗留

- 改动的 TU 单独编译通过（`ninja -C build-win src/runtime/CMakeFiles/ninfer_engine.dir/engine/tp2_generation_core.cpp.obj`），
  全量 `ninja ninfer-serve` 链接通过。
- 端到端如上表；服务端追踪 + 启动账本双重佐证，无需对比旧二进制。
- TP-2 的 ring **仍无自动化测试覆盖**：`test_tp2_forward` 只测 `forward_tp2`，不构造 `TP2GenerationCore`。
  要覆盖它需要真工件 + 双卡，故本轮仍以端到端实测为准。

## Round 17 — TTFT 地板拆解：0.28-0.35 s 不是"固定开销"，一半随 prompt 长度增长

Round 16b 之后剩下的疑问是"每请求 0.28-0.35 s 的固定开销在哪"。结论：它不是一个常数。在 100k context 下它的构成是
~32 ms 的 Engine 常量 + ~81 ms 随 context 线性增长的前向 + ~10 ms 的 host 状态恢复（仅新会话）+ ~70 ms 的服务端
渲染/分词 + ~100 ms 的 HTTP/JSON/序列化/客户端。

### 定位手段：`NINFER_TP2_TIMING` 的 prefill 分段计时

TP-2 路由上 `engine_timing` 完全未填充（`host_exposed_seconds`、`units` 全 0），`prefill_seconds` 又是
`execute()` 顶部（1090 行）到首 token 的整段宿主墙钟，无法归因。因此在 `NINFER_TP2_TIMING=1` 下加了
scan / state / walk / post 四段（段间插设备同步，仅该环境变量开启时生效）：`scan` 覆盖复用扫描与 pruning，
`state` 覆盖 GDN 状态恢复，`walk` 覆盖走查与首 token 采样，`post` 覆盖快照与 publish。

12-token suffix、默认 `--host-state-slots`、同源对比（复用来源由 `NINFER_TP2_REUSE_TRACE` 标注）：

| context | src | scan | state | walk | post |
|---|---|---|---|---|---|
| 15.3k | device | 0.02 | 0.55 | 53.83 | 0.03 |
| 15.3k | host | 0.02 | **10.95** | 53.24 | 0.03 |
| 30.5k | device | 0.03 | 0.55 | 65.43 | 0.05 |
| 30.5k | host | 0.02 | **10.94** | 65.36 | 0.05 |
| 45.3k | device | 0.04 | 0.54 | 77.51 | 0.05 |
| 45.3k | host | 0.03 | **11.06** | 77.49 | 0.05 |

- **scan 只有 0.01-0.04 ms**：那个 O(context) 的宿主复用扫描（52k 次整数比较）不是问题；先前按截距外推怀疑它是错的。
- **state 的 host 路径固定贵 10.4 ms**（0.55 → 10.95），与 context 无关。147 MiB H2D 被 GPU1 的 Gen4 x4
  （约 7 GB/s）卡住，是链路地板。
- **walk = 31.9 ms + 0.81 µs/token × context + 0.79 ms/token × suffix**。三个 context 点线性，且与复用来源无关。
- post 0.03-0.05 ms，可忽略。

### 0.81 µs/token 落在 walk 里，是窄 query 块对长 KV 的遍历

走查只处理 12 个 token，前向却随 context 线性增长。多出 30k context 时每卡多读约 557 MB KV，耗时 23.7 ms，即
**有效带宽约 23 GB/s**（该卡 448 GB/s 峰值的 5%）。只有 12 个 query，并行度不足以掩盖 KV 读延迟。这是 kernel
效率问题，不是宿主开销——item 2 的目标。

### host 恢复的 10.4 ms 不值得换 device 平面

divergence 锚点迁到 device 快照只需 **+73.4 MiB/卡**（一个状态平面；`state_arena` 由 4 平面变 5，
293.6 → 367.0 MiB/卡；紧的那张卡剩余 1580 MiB 的 4.6%），同时可省掉 host ring 的第 9 槽（−73.4 MiB pinned/卡）。
但收益只有 **9.9 ms × 每会话一次**：host 恢复只在复用边界既非 prefill end、也非自适应 rewind 点时发生。Round 17
扫描里 3 次稳态探测有 2 次 `src=host` 是探测形状的产物——三轮 prompt 等长，pair 间 gap 恒为 12，预测的下一共享
前缀 15304-12-2 比实际共享前缀 15292 低 2 个 token，被可达性守卫 `prompt_tokens - depth >= reuse` 判为"本轮没
走到"而丢弃（trace 中 p2 留下 `rewind=0`，p3/p4 落到 host 槽 6）。真实聊天每轮 prompt 在增长、共享前缀还伸进
助手回复，守卫通常成立，走 device。**结论：放弃该杠杆。**

### 服务端渲染/分词：98-99.7% 是分词，渲染只有 0.1-0.3 ms

`preparation_seconds` 写在 **`request_start`** 事件里（在准备完成之后写出），不是缺失也不是全 0——按
`request_done` 找会误判。7 个样本的 `total` 与 `tokenize` 之差只有 0.04-0.34 ms：

| prompt_tokens | prepare total | tokenize | 其余 |
|---|---|---|---|
| 15299 | 5.922 | 5.806 | 0.116 |
| 45299 | 29.884 | 29.545 | 0.338 |

即 `chat_template.render()` 不是成本，**每个请求重新分词整段历史**才是。`--request-log-jsonl` 的
`timings_seconds.prepare` 与 `preparation_seconds.total` 在 chat 路径上一致（差 0.02 ms），所以主路径只有一次
分词——`Frontend::count_tokens` 只服务 Anthropic `/v1/messages/count_tokens`。

### 修复：BPE 每个词两次堆分配

`append_normalized_bpe_ids`（`tokenizer.cpp`）对每个 pre-tokenizer 词都新建一个 `std::vector<BpeNode>`
与一个 `std::priority_queue`。长 prompt 约 3-4 万词，即 10 万次以上堆分配。改为把两个缓冲提到词循环之外复用
（`nodes.resize` / `heap.clear()`），并把 `std::priority_queue` 换成同一个 vector 上的 `std::push_heap` /
`std::pop_heap` 加 `LaterBpeCandidate`。

这**不改变语义**：`std::priority_queue::push` 的规定实现就是 `c.push_back(v); std::push_heap(...)`，
`pop` 就是 `std::pop_heap(...); c.pop_back()`，`top()` 就是 `c.front()`——插入序列与比较器完全相同，
因此弹出顺序逐位一致。顺带淘汰了 `#include <queue>`。

固定内容探针、同一真实工件、两次服务进程（改动前/后）：

| cache_n + prompt_n | tokenize 前 (ms) | tokenize 后 (ms) | 加速 |
|---|---|---|---|
| 1211 | 0.330 | 0.226 | 1.46x |
| 5851 | 1.178 | 0.766 | 1.54x |
| 1812 | 0.856 | 0.439 | 1.95x |
| 7037 | 2.715 | 1.613 | 1.68x |
| 27913 | 13.623 | 6.178 | 2.20x |
| 39213 | 14.568 | 8.329 | 1.75x |
| 67 | 0.064 | 0.056 | 1.13x |

合计 33.3 → 17.6 ms（**1.89x**）。100k context 下这一项从约 38 ms 降到约 22 ms。

### 验证

- **逐位等价**：同一探针改动前后，7 个请求的 `cache_n + prompt_n` 全部相同（含 CJK、组合字符
  e+U+0301 与 a+U+0300、emoji 的 mixed 用例），`prompt_ms` 不受影响。
- `ctest -R qwen3_5_frontend` **通过**（含 `test_bpe_merge_order`、`test_boundary_aware_tokenization`）。
- 顺带修掉一个既有环境问题：`tests/fixtures/frontend/` 下的 *.jinja 在本 checkout 里是 CRLF，而 `.gitattributes`
  声明 `*.jinja text eol=lf`，于是模板 sha256 不在白名单里，前端测试以 `0xC0000409`（未捕获异常 fastfail）
  退出且**无任何输出**——`main` 不接异常，失败点不可见。把两个 fixture 归一化为 LF（对 git 不可见，`git status`
  仍干净）后整栈通过。此崩溃在 stash 掉分词改动后同样复现，与本轮改动无关。

### 遗留

- 分词剩余约 0.22 µs/token（100k 约 22 ms）。彻底消除需要按 added-token 段缓存 `BoundaryEncodedText` 前缀
  （`encode_with_boundaries` 已按特殊 token 分段、段间归一化本就独立，因此按段切分可证等价），代价是 Frontend 里
  引入带并发保护的缓存与一段前缀拼接——相对约 22 ms 收益风险偏高，本轮未做。
- 地板大头仍是 walk 的 0.81 µs/token（100k 约 81 ms），即 item 2。

## Round 18 — TP-2 窄 chunk 的注意力路由遗漏：walk 的 0.81 µs/token 来自"用 prompt 内核跑十几行 query"（已修）

Round 17 把 walk 的随 context 项（0.81 µs/token，100k 约 81 ms）记为 item 2。本轮定位到根因：**不是访存带宽不够，而是路由把 TP-2 分片的窄 chunk 交给了 prompt 内核**，后者的 KV 方向并行度不足以喂满显存。

### 根因

`ops::causal_softmax_attention`（`src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`）按 `q_heads`/`width`/`max_visible_keys` 在三条私有路线里选一条：`SmallT`、`ChunkedSmallT`（都是 flash-decoding 式 split-KV，先出 partial_acc/m/l 再合并）、`Prompt`（大 T 内核，沿 query 行并行，KV 方向基本串行）。

- 单卡（24 q 头）在 `width <= 16` 时由第一分支接管：fp8 存储下 `width >= 9` 的 prompt 上限只有 320 keys，超过就落到 `ChunkedSmallT`。
- `q_heads == 16`（另一款分片）在 `max_visible_keys > prompt_visible_keys` 时同样落到 `ChunkedSmallT`。
- **`q_heads == 12`（Qwen3.8-27B 的 TP-2 分片：24q/4kv/head_dim256 的一半）没有任何分支**，`width <= 6` 之外全部落到末尾的 `return Prompt`。

三条路线的实现早已存在且对 12 头实例化完毕：`causal_attention_chunk_tokens` 对非 24 头返回 6、`causal_attention_split_capacity` 有 `CausalD256H12Kv2` 分支（`small_t.cu:249`）、bf16/i8/fp8/nvfp4/k8v4 五个 small-T 启动器都带 12 头分派。12 头几何由 `548a426a feat(tp2)` 引入，而路由条件来自更早的 `a7818988`——**没同步扩展，是遗漏**，不是设计取舍。12 行 query 沿 KV 串行走 45k keys，实测约 20 GB/s，即 0.81 µs/token。

### 改动

- `causal_softmax_attention.cpp`：把长 context 的 split-KV 条件从 `q_heads == 16` 放宽到 `(q_heads == 12 || q_heads == 16)`。`width <= kMaximumVerifyTokens(16)` 的守卫保留，因此大 chunk 仍走 prompt 内核。工作区容量函数本就按宽度逐个重跑 `causal_attention_resolve_route`，规划侧无需改动。
- `tests/ops/softmax_attention/causal_cache.cpp`：把 `d256-h12-kv2` 注册进 `kGeometries`（此前只有 24/4 与 16/2），长包络用例的 `q_heads == 16` 门限放宽到 `!= 24`，并新增 `verify_route_selection()` 直接断言路线选择（12 头长 context → ChunkedSmallT、12 头短 context → Prompt、12 头 6 token → SmallT、24 头 → ChunkedSmallT）。

### 实测

同一探针（12-token suffix，`NINFER_TP2_TIMING=1` 的 `[tp2-prefill]` 分段），单位 ms：

| context | walk 前 | walk 后 | prompt_ms 前 | prompt_ms 后 |
|---|---|---|---|---|
| 15.3k | 53.83 | 44.12 | 64.3 | 55.2 |
| 30.5k | 65.43 | 45.71 | 76.4 | 56.7 |
| 45.3k | 77.51 | 47.46 | 88.7 | 59.0 |

- 随 context 项 **0.81 → 0.111 µs/token（7.3x）**：每 30k context 从 +23.7 ms 降到 +3.3 ms。
- 常量项没有变差：`chunks=1`，权重仍是每个 prefill chunk 流一次，split-KV 只在注意力算子内部按 6 token 再分两段。
- 外推 100k：walk 从约 113 ms 降到约 44 ms，**约 −69 ms**。
- 1028-token suffix 同步受益（35k：891.7→851.7；45.3k：985.1→926.1），但该段主要是每 token 权重流。

### 验证

- `ninfer_softmax_attention_test` **通过**：12/2 几何现已在全部存储（BF16/INT8/FP8/NVFP4/K8V4）、三种 block-table 映射、A1/A3（有/无 mask）用例下对**独立 FP64 oracle** 校验，新增的长包络用例正是新路线。该 Op 的公开契约明确要求"每条 cache 路线直接对独立数学 oracle 校验，路线间一致性只算旁证"，把分片几何注册进测试矩阵是这一契约的一部分。
- 端到端生成冒烟：`17*23` 得 391、Peru 首都 Lima，正常。
- 全量 `ctest -j 2`：**126/127 通过**，4 个缺真实产物跳过，唯一失败是既有的 `ninfer_request_log_test`（`bfec5c03` 改了 pretty 行但未同步 `tests/test_request_log.cpp:487`，与本轮无关）；该过期期望已顺手补齐 `(300 tok)`，补后通过。

### 剩余

- walk 的常量项（约 32.9 ms）与 suffix 项（0.79 ms/token）未动；100k 下 walk 约 44 ms 已是地板的小头。
- 分片几何的 `prompt_visible_keys`（512/1024 keys）沿用 16 头分片的调参，本轮只测了 ≥15.3k context，更短 context 的交叉点未测（收益量级也小）。
