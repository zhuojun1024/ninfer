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
the repository root. Three PLAN.md snapshots are archived verbatim below under their own
`PLAN.md archive` headers: the 2026-09-21 upstream cherry-pick re-plan, the 2026-09-22
delivery-cleanup rewrite, and the 2026-09-24 post-protocol-refactor state.

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
  （后续修复：TP-2 现在填充 `SpeculativeStats`，响应直接带 `draft_n`/`draft_n_accepted`。）

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

---

# PLAN.md archive (archived 2026-09-21, Rounds 48-55 + worklog Rounds 11-19)

Archived verbatim from the working PLAN.md at the point of the upstream cherry-pick re-plan. The completed sections are summarized in the new PLAN.md; this is the full record.

# NInfer TP-2 计划（2× RTX 5060 Ti · Qwen3.8-27B NVFP4）

> **唯一的活动计划**，也是跨上下文压缩的持久记忆。整理时间：Round 48 收尾。
> - 完整历史记录（原 PLAN.md 全文，Round 1–48 逐轮证据与失败尝试）：`docs/tp2-dual-5060ti-worklog.md`
> - 交付说明、推荐配置与实测数据：`docs/tp2-dual-5060ti.md`
> - 状态：**交付范围 ①–⑤ 全部完成。**① 的判据已从「与 plain 逐 token 一致」（不可达，见 §5.5）改为
>   「无退化 + 质量同档 + 有加速」，并已达成（§5）。
> - **Round 52 收尾（已完成）**：词表/隐层并行切分（§10 的 Step 1+2+3）与显存腾挪把上下文顶到
>   artifact 上限 **262,144 token**（15,614 / 15,094 MiB/卡），greedy 输出与优化前逐字节一致。
>   详细数据见 `docs/tp2-dual-5060ti.md` 与本文件 §10。
> - **Round 53（进行中）**：TP-2 视觉识别 —— 「静态分片分工」（MTP 只 shard 0 / vision 只 shard 1），
>   见 §11。用户已批准；目标是显存足够时不引入单图 token 上限配置。
> - **§13（已完成并实测通过）**：TP-2 多会话 KV 池 —— host KV 换入换出，
>   显存始终只有一份激活 KV，内存最多存 5 份非激活会话（LRU 淘汰最旧），会话回来时整体召回、
>   只 prefill 新后缀。见 §13 进度小节。

---

## 1. 目标（用户原始要求）

1. 修好 MTP 一致性问题；
2. 做基准测试；
3. 查找 prefill/decode 优化空间；
4. 优化显存占用，开启 KV cache fp8/q8_0 量化，调整对性能/质量影响不大但占用资源的参数，以开启更大上下文；
5. 交付一个针对本机双卡 5060Ti TP-2 极致优化的推理框架。

原始计划的补充完成条件（记录在归档 §0）：TP=1 路径零回归、切分 Op 有独立 oracle、端到端质量确认、
性能下限（MTP0 ≥ 28 tok/s、MTP3 ≥ 70 tok/s）、C=1/2/4 稳定性。

---

## 2. 完成状态

| 目标 | 状态 | 证据 |
|---|---|---|
| ① MTP 可用（无退化 / 质量同档 / 有加速） | **完成**（§5） | 6 次采样 content 1519/1318/1294/1195/1683/0（中位 1306、`longest0` 全 0）vs plain 1165/1452/1546/672/224/1237（中位 1201）；稳态 decode 46–49 vs 31.6 tok/s |
| ② 基准测试 | 完成 | prefill 279 token / 0.453 s（615 tok/s）；decode 端到端（含 prefill）256 token：plain 8.153 s（31.4 tok/s）、MTP K=3 6.532 s（39.2 tok/s）；纯 decode：plain 31.5、K=1 42.0、K=2 50.5–54.0、K=3 47.9–48.0 tok/s |
| ③ 显存 / KV 量化 | 完成 | 五种 dtype（bf16/int8/fp8/nvfp4/k8v4）在 TP-2 全部可用；fp8 质量与 bf16 同档（1365/859/1382/1602 vs 1233/763/1469/1456/1430/1621，均无 0 复读）；显存 plain 13,888 MiB/卡 |
| ④ 更大上下文 | 完成 | 65,536 → 131,072 → **262,144 token**（fp8 KV，15,614 / 15,094 MiB/卡）；端到端 118,869-token 提示 HTTP 200 / 96.8 s，65k-token 提示 48.1 s |
| ⑤ 文档与推荐配置 | 完成 | `docs/tp2-dual-5060ti.md`（推荐配置、基准表、KV dtype 表、MTP 限制与根因、修复清单） |

**未达成的原有门槛**（诚实记录）：MTP3 ≥ 70 tok/s 未达到（实测纯 decode 上限 K=2 50.5–54.0）；
TP-2 路径未跑 perplexity 评测（质量证据改用同提示词多采样 A/B）；per-shard arena 的
`memory_summary()` 仍报 `pages 0/0`、`runtime 0 B`（仅显示口径问题，未影响功能）。

---

## 3. 推荐运行配置（8088 当前按此运行）

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh`（Round 53c 新增）是**进程级默认思考强度**：不带该 flag 时
沿用模板自带的默认（本 artifact 为 `xhigh`），请求体的 `reasoning_effort` 优先。实测同一问题
`xhigh` 思考 172 tokens、`medium` 46 tokens。
**`--chat-template` 不存在且本架构不支持**：模板是 artifact 资源
（`resource/text/chat_template.jinja`），必须与 `tokenizer_config.json.chat_template` 逐字节相同，
再按 sha256 白名单编译——只有 `e84f32a2…`（thinking-toggle）与 `c3cf9e34…`（reasoning-effort，
本 artifact 用的就是它）两种语义。详见 §11.8。

262,144 是 artifact 的 `max_position_embeddings`：预填 1588 tok/s、稳态 decode 59.0 tok/s。
MTP（`--spec mtp --draft-tokens 2 --lm-head-draft`）已在 bf16 与 fp8 KV 两条路线验证无退化、
质量同档；去掉这三个 flag 即回到 plain。

`--vision`（Round 53 新增可用）：静态分片分工把 MTP 放 shard 0、Vision 塔放 shard 1，
启动台账多出 `vision 413.3` 一行（shard 1），两卡余量 699 / 951 MiB。多模态请求与纯文本请求
走同一套机制（MTP + 前缀复用，见 §11.6），冷启动 greedy 逐字节相同。

---

## 4. 已完成的关键工作

- **prefill/decode 优化（目标 3）**：批量 prefill（TTFT 19×）、块宽扫描与 allreduce 真因定位（Round 35b）、
  AR 有序切片流水（+15%，Round 35c）、allreduce 6 sync → 2 sync 与 in-kernel allreduce + mixer 头切分
  （Round 33/34，追平 llama.cpp）。细节见归档。
- **KV 量化修复（目标 4 的关键前提）**：量化 KV 原本在 TP-2 上完全无法启动。根因是
  `prompt_fp8.cu` / `prompt_nvfp4_non_rdc.cu` / `prompt_k8v4.cu` 用**函数内 `static`** 做
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`（92,416 B > 48 KiB 默认值）的 opt-in；该属性是
  **每 device** 的函数属性，第二个分片所在 device 从未 opt-in → 每次 launch 报 `cudaErrorInvalidValue`。
  改为按 device 缓存后五种 dtype 全部可用（`tools/tp_bootstrap/fix_per_device_attr.py`）。
- **更大上下文（目标 4）**：fp8/int8 KV 使 131,072 token 在 13.9 GB/卡内可用，并以 118,869-token 提示端到端证明。
- **MTP 状态机修复**：每轮 verify 前快照 GDN 状态到新 scratch slot（`kRoundScratchSlot`），fold 前先 restore，
  消除「verify 推进活状态 + fold 二次推进」的双倍前进（首分叉 4 → 10）。
- **临时插桩清理**：`NINFER_TP2_ROUND_DETAIL/PARITY/TOKEN_TRACE/SKIP_FOLD/VERIFY_PHASE` 全部删除
  （`grep NINFER_TP2_ ` 为 0 命中）。

---

## 5. 已完成：① MTP 输出退化（根因：分片未写 replay 记录）

### 5.1 现象（修复前）
MTP 可用且更快（端到端 +25%、纯 decode +50%），但输出退化：temp0.7/top-k20/top-p0.80 下 6 次采样
content 中位数 0（6/6 出现 "0" 复读，`longest0` 达数百），plain 中位数 1443、0/6 退化。贪心同样复现。

### 5.2 根因（已确认）
TP-2 的 head-split 分片**根本不产生 replay 记录**：`gdn_mix` 里两条记录分支都以 `shard_config_ == nullptr`
为条件（融合的 width>1 记录算子只注册了全量 16384 行 fused parent，分片拿不到）。而每轮 MTP 仍然执行
「恢复轮前快照 → 用 `commit_columns` 跑 `gdn_replay_fold`」，fold 消费的是**从未写过的记录平面**：
48 个线性注意力层的 recurrent 状态每轮都被回放成空日志（等于不前进），conv 历史也被同一份空日志重建；
16 个全注意力层照常工作 ⇒ 前几个 token 局部连贯、随后塌缩成 "0" 复读。与现象完全吻合。

### 5.3 修复（`src/models/qwen3_5/execution/text.cpp` · `gdn_mix`）
- 分片在自身的 decomposed 路线上记录：conv 记录就是该路线已物化的 `[q|k|v]` 原始卷积输入，直接拷贝到
  记录平面（通道序与布局一致，fold 需要的就是 `conv_record[0:committed]`）；
- recurrent 改用 `ops::gated_delta_net_replay_record`：它显式注册了分片几何（qk 8 heads / value 24 heads），
  且契约规定其输出与归一化 `gated_delta_net` **逐位相同** ⇒ 窗口逐列 logits 与接受判定不变，只有状态转移
  变成可回放（正是 fold 需要的）；
- `causal_conv1d_silu.h` 的 row profile 文档补上分片 profile `(1024,1024,3072) / C=5120`（实现早已支持）。

### 5.4 证据
- 端到端 A/B（`r36f_stats.sh`，同一提示词、temp0.7/top-k20/top-p0.80、各 6 次）：MTP content
  1519/1318/1294/1195/1683/0（中位 1306），**6/6 `longest0 = 0`（零复读）**；plain 1165/1452/1546/672/224/1237
  （中位 1201）。唯一那个 0 是 reasoning 预算 artefact，plain 同样出现（两例把预算全花在 reasoning）。
- 吞吐无回归：MTP K=2 稳态 44.3/46.2/45.5/46.1/49.2 tok/s，plain 31.6–31.8 tok/s。
- 第二个缺陷（同属"per-device 属性用进程级 static"）：`small_t_fp8.cu`/`small_t_k8v4.cu`/`small_t.cu` 的小 T
  核也需要 >48 KiB 动态 smem 的 opt-in，但单 token decode 用 32 KiB tile 恰好低于 48 KiB 默认值，缺陷一直潜伏；
  MTP 窗口（T=K+1）切到 64 KiB tile 才暴露 ⇒ `small_t_fp8.cu:56 cudaErrorInvalidValue`。四路已统一到
  `src/ops/common/cuda_smem.h` 的按 (kernel, device) opt-in。验证：fp8 KV @131072 + MTP K=2 三次采样
  content 1530/1481/1708、`longest0` 全 0（`r49_fp8mtp_smoke.sh`）。
- 回归测试全 PASS：`gated_delta_net_replay_record`、`gdn_replay_fold`（含 48×8×24×5120 分片几何）、
  `gdn_input_proj_conv_record`、`gdn_input_proj_conv_snapshot`、`gdn_input_proj`、`tp_device_pair`、
  `qwen3_5_tp2_load --artifact`、`qwen3_5_tp2_forward --artifact`。

### 5.5 判据修正（重要）
「与 plain 逐 token 一致」是**不可达判据**，已废弃：verify 窗口与单 token decode 本来就是不同执行形状
（本引擎走 chunked attention/conv，llama.cpp 走 chunked GDN 核），且被接受 token 的 KV 由该窗口写入后不回改。
llama.cpp 的 Qwen3.5 MTP 结构相同，其文档也只要求「需要精确一致时用 greedy」。W1 判据因此改为
**无退化 + 质量同档 + 有加速**，并已达成。

### 5.6 被否定的两条诊断（留档，避免重走）
- **相位说**：曾判定「verify 走 `Phase::Prefill`、decode 走 `Phase::Verify`，两核舍入不同」。错误：`Phase`
  在 head-split 分片上没有语义——`attn_mix` 完全不用它的 phase 形参，`gdn_mix` 的两处 `ph == Phase::Verify`
  都被 `shard_config_ == nullptr` 排除——所以改相位不会改变任何数值。原 §5.5 计划的「rank-3 → rank-4
  引擎级布局改造」随之作废。
- **`rope.cu:189` 说**：那是 `CUDA_CHECK(cudaGetLastError())` 检查点而非出错核；多设备交错下异步非法地址
  会在**第一个**检查点浮现，报错点不必等于故障核。

---

## 6. 可选后续（非阻塞）

- **W2 负载均衡**（**已完成**）：MTP 开启时 GPU0 100% / GPU1 ~72% 属结构性（MTP 层 + 全部 argmax/accept/fold/采样/策略只在
  shard A；B 在 text mixer lockstep 后空等）。方向：①两卡 lockstep 冗余跑同一份 MTP 层（墙钟不变）；
  ②按 vocab 切 MTP 提案的 lm_head + 窗口 argmax（真正缩短关键路径）；③重扫 K=1/2/3。
- **KV dtype 扫描补全**：fp8/int8 已有质量与显存数据，nvfp4/k8v4 只验证了可启动与吞吐。
- **工具调用约束解码**（**已完成**，路线 A；诊断见 Round 15，实现与验证见 worklog Round 15b）：对齐 llama.cpp 的 lazy
  tool-call grammar，把 `<function=` 后的工具名与参数名掩码限制在本次请求声明的工具上（文本化轻量版；完整 GBNF 的
  路线 B 见 Round 15）。掩码在 CUDA Graph 之外应用，MTP 按 verify 列取各自语法位置、draft 不掩码，思考阶段不约束。
- **TP-2 host 侧状态检查点 + 短 prompt 按 LCP 截断复用**（**已实现并完成端到端验证**，诊断见 Round 16，实测见 Round 16.3）：把 GDN 状态快照放进 pinned host
  环形池（0 显存增量，`--host-state-slots` 语义），步长 8192，复用扫描取最深的 `<= shared_prefix` 检查点。
  修掉「客户端在 turn 边界重渲染出更短 prompt ⇒ reuse=0 ⇒ 整条全量重算（实测 42.0 / 57.6 s）」。实测 `src=host` 命中：req#5 10.2 s → 1.3 s、
  req#11 37.6 s → 28.6 s，重算量与理论最小值差额 ≤ 1 个 chunk。

---

## 7. 交付前收尾清单

- [x] 删除运行时插桩（`NINFER_TP2_*`）
- [x] 完整记录归档：`docs/tp2-dual-5060ti-worklog.md`
- [x] 删除只向 PLAN.md 追加历史文本的 `tools/tp_bootstrap/r36b_plan_append.sh`、`r36c_plan_append.sh`
- [x] W1（①：MTP 退化已修复、判据已修正为「无退化 + 质量同档 + 有加速」，见 §5）
- [ ] 用推荐配置复测并发 C=1/2/4 与流式/stop 冒烟（Round 31/34 的结论基于 bf16 配置）
- [ ] 交付时按 AGENTS.md 移除本文件（PLAN.md）；`tools/tp_bootstrap/` 下有约 300 个一次性诊断脚本，
      需决定保留/清理范围（其中 `build_r35.sh`、`serve_*`、`r37+r38+r39+r4x` 系列与 `docs/tp2-dual-5060ti.md`
      引用为可复现流程）

---

## 8. 本机环境与运维要点

| 项 | 值 |
|---|---|
| 编译/运行 | WSL2；Windows 工作树 `D:\Documents\workbench\ninfer`，构建树 `/home/zhuojun/ninfer` |
| 构建 | `bash tools/tp_bootstrap/build_r35.sh`（rsync `src tests apps include bench tools` 后 `cmake --build build -j 8`；成功标记 `BUILD_EXIT=0`，约 2–3 分钟） |
| 服务 | 8088；`serve_stop.sh` → 后台 `wsl -e bash .../serve_supervise.sh <ctx> 8088 '<extra args>'` → 轮询 "listening on http://127.0.0.1:8088"（加载 30–37 s） |
| 首请求 | 付 CUDA graph 捕获（实测 14.4 s），稳态数字才有意义 |
| 约束 | 本会话不终止/重启/新启动任何 llama.cpp 进程；只管理 ninfer 进程 |
| 工具链注意 | 嵌套 `pwsh` → `wsl -e bash -lc "..."` 会吞 `$var` 与重定向 ⇒ 把命令写成脚本文件放 `tools/tp_bootstrap/` 再执行；单次阻塞调用上限 600 s ⇒ 长任务用后台作业 + ≤420 s 轮询 |

---

## 9. 证据索引

- 脚本：`build_r35.sh`（构建）、`serve_supervise.sh`/`serve_stop.sh`（服务）、`r37_colparity.sh`（逐列 parity）、
  `r36h_trace.sh`（逐 token 分叉）、`r38_kvsweep.sh`/`r42_fp8check.sh`/`r44_fp8quality.sh`（KV 量化）、
  `r39_bench.sh`/`r43_bigctx.sh`/`r48_longctx.sh`（基准/长上下文）、`r45_verify.sh`/`r46_recconfig.sh`/`r47_verify.sh`（验收）、
  `fix_per_device_attr.py`/`strip_tp2_probes.py`/`strip_tp2_probe_comments.py`（源码批改）、
  `verify_gdn_replay_fix.sh`/`verify_tp2_artifact.sh`（① 修复的回归）、`r36f_stats.sh`（质量 A/B）、
  `r49_fp8mtp_smoke.sh`（MTP + fp8 KV 冒烟）
- 日志：`/home/zhuojun/prof/`（`serve_supervised.log`、`kvsweep-out.log`、`r39/bench-out.log`、`r42_fp8check.log`、
  `r43_bigctx.log`、`r44_fp8quality.log`、`r45_verify.log`、`r48_longctx.log`、`kvdiag-out.log`、`ab_stats.log`、`r49_smoke.log`）
- 源码改动清单：`docs/tp2-dual-5060ti.md` 的 "Fixes applied on this branch" + 本文件 §4/§5.3；逐轮文件引用见归档

> 说明：① 的修复改动为 `src/models/qwen3_5/execution/text.cpp`（`gdn_mix`）与 `include/ninfer/ops/causal_conv1d_silu.h`（文档）；其余历史改动清单见 `docs/tp2-dual-5060ti.md` 的 "Fixes applied on this branch"。

---

## 10. 已完成：词表/隐层并行切分（Step 1 + 2 + 3）与 262,144 上下文

动机（Round 51 显存账本）：每卡有一份完整的 `text/token_embedding`（1,213 MiB）与 `text/output_head`
（1,213 MiB），两张卡各一份 ⇒ 合计 4.8 GiB。词表维切分后每卡只留一半 ⇒ 每卡 −606 MiB/表，
两张表都做完 −1.2 GiB/卡；换算每省 1 GiB/卡 ≈ +65,536 token 上下文，262,144 还需要卡 0 再腾 ~930 MiB。

**Step 1（本次）**：`text/output_head` → `WeightSplitKind::ColumnParallel`（每卡行区间
`[shard*V/2, (shard+1)*V/2)`）。

- 载入：`tp_split_spec` 新增 `TpSplitOptions::split_output_head`；`materialize_model_tp2` 按
  `purpose == Generation && (!speculative_enabled() || proposal_enabled())` 打开。理由：选 Full 提议头时
  `text/output_head` 与 `mtp/output_head` 是同一个 artifact 对象（weight-tied），而 MTP 层只在 shard 0 上跑，
  该路线的提案 argmax 需要全量头 ⇒ 保持 replicated（行为与今天完全一致），留到 Step 3 用 peer 参与解决。
- 执行：新增 `TextContext::project_head_tp2(peer, pair, hidden, hidden_peer, logits, logits_peer)`：
  - 未切分：两次全量 `project`（现状）。
  - 已切分：每卡先 `project` 出 compact `[V/2, T]`，把 `[V,T]` 缓冲清零后用 `cudaMemcpy2DAsync` 写进自己的半区，
    再 `pair.allreduce` 求和（对侧半区为 0）⇒ 两卡都拿到完整 `[V,T]` logits，采样/接受判定逻辑不变。
  - 三个调用点：`forward_tp2`（decode T=1）、`forward_tp2_prefill` 的 `[V,1]` 分支、verify 的
    `logits_columns`（`[V,width]`，原本只有 shard A 有缓冲 ⇒ 函数内为 shard B 新开一个）。
- 验收（全部通过）：`ninfer_qwen3_5_tp2_load_test` 三条路线（无 spec / `--spec mtp` /
  `--spec mtp --lm-head-draft`）、`forward_test`（11 探针 + 32 步 AR + 批量 prefill）、
  `ninfer_tp_device_pair_test`；新 op oracle `ninfer_linear_tp2_split_fp8_head_test`（真实头形状
  [248320,5120] 的 T=1/3/5、[16384,5120] 的 T=1/4，全部 "exact"）；8088 端到端 greedy 对照
  （`top_k=1` 固定）5 个提示词逐字节一致（`ab-control.jsonl` vs `ab-split.jsonl`，hash 全等）。
- 显存：14,940/14,678 → **14,334/14,072 MiB/卡**（−606 MiB/卡）。
- 顺带修复 `weight_splitter` 的 FP8 分片描述符：`slice_fp8_rows` 补 `scale_ne[0]`/`scale_nb[1..3]`，
  `slice_fp8_cols` 补 `group_size`/`group`（此前分片 Weight 与 payload 不自洽，只有经 geometry 重建
  描述符的生产路径掩盖了该问题）。

**Step 2（已完成）**：`text/token_embedding` → `WeightSplitKind::RowParallel`（按 hidden 列切分，每卡
`[V, D/2]`）。选列切分而非词表行切分：ids 始终是全域（不存在越界行），且合并路径与输出头完全同构。

- 执行：`TextContext::embedding_tp2(peer, pair, ids, ids_peer, x, x_peer)`：两卡各查自己那半列到 compact
  `[D/2, T]`，再用新抽出的 `merge_local_row_blocks`（与头共用）清零+写半区+allreduce；文本路径
  （`forward_tp2`/`forward_tp2_prefill`）直接传 peer 的 ids。MTP stem 只在 shard 0 跑 ⇒ 新增
  `TextContext::set_tp_peer`（TP-2 core 在两卡建好后注册），把 ids 广播到对侧（16 字节对齐填充 + allreduce；
  形状用前缀视图保持 `T` 与输出一致）再合并；未注册 peer 且表已切分时硬报错。
- 阻塞与解决：FP8 embedding 的注册域原本硬编码 `[248320,5120]`。把 FP8 gather 内核按隐藏宽度模板化
  （`embed_gather.cuh`）、launcher 按宽度分派（5120/2560 各自的 block 划分）、wrapper 用
  `embed_gather_fp8_supports_width` 校验（词表不再限制，因为内核与行无关），并更新 op 文档与
  `ninfer_embedding_test`（`Fp8Table` 参数化宽度，2560 走完整 T 扫描 + CUDA Graph + 对齐用例）。
- 验收（全部通过）：`ninfer_embedding_test`（5120+2560 全量）、`tp2_forward_test`、两条 load test；
  8088 端到端 greedy 对照 5 个提示词与 Step 1 构建逐字节一致（`ab-embed.jsonl` vs `ab-split.jsonl`）。
  期间发现并修复两个只有真实服务才能暴露的问题：`allreduce` 要求字节数是 16 的倍数；广播缓冲形状必须
  等于 ids 个数（否则 gather 的 T 校验失败）。
- 显存：14,334/14,072 → **13,728/13,466 MiB/卡**（再 −606 MiB/卡；两步合计 −1,212 MiB/卡）。

**Step 3（已完成）**：`proposal/head`（131072×5120 Q4_G64_FP16，341 MiB）按**词表行**切分，每卡 65,536 行
（171 MiB）。`proposal_argmax` 走 `hidden` 广播 → 两卡各算自己那半 → `merge_local_row_blocks` 合成
`[131072,T]` → shard 0 argmax + token-id 重映射。阻塞点：Q4 A16 是**枚举 (n,k) 表**，没有 65536×5120 这档，
新增 `select_q4_n65536_k5120`（`q4_shapes.h`/`q4_dispatch.cpp`/`sources.cmake` + 新 `.cu`）。oracle：
`ninfer_linear_tp2_split_grouped_head_test` 在真实 [131072,5120]、T=1/2 上与全量 Op 逐位比对全 "exact"。

**Round 52 结果（262,144 落地）**：

- 上下文：131,072 → **262,144 token**（artifact 上限）。`nvidia-smi` 15,614 / 15,094 MiB/卡（余 697 / 1,217）。
- 显存腾挪：state arena 367→294 MiB（前缀复用快照 3→2；`rewind_depths` 只剩 `{0, rewind_near_}`，
  实测 16k/65k 相同 prompt 重入 0.30/0.33 s ⇒ 聊天轮次模式够用）；workspace 384→192 MiB。
- 性能（t=0，usage 计数）：预填 16,057 token / 10.11 s = **1588 tok/s**；解码 512 token / 8.68 s = **59.0 tok/s**
  （223 轮、2.3 token/轮、每 draft 接受 64.6%），比 131072 时代 46–49 tok/s 高约 20%。
- 逐位一致：5 条 greedy 提示在 131072（优化前）与 262,144 上 hash 全同（`r52_ab.sh final` + `r52_cmp3.sh`）。
- 死路记录：MTP 层 KV 降 nvfp4 省 228 MiB、接受率不变（392/510 vs 391/510），但 3/5 提示输出变化 ——
  verify+fold 与单 token 解码执行形状不同，draft 模式一变近似并列的取舍就变；因此保留 fp8 MTP KV。
- 回归全绿：3 条 load route、`tp2_forward`、`embedding`、`linear_tp2_split_fp8_head`、
  `linear_tp2_split_grouped_head`（新）、`linear_tp2_split_nvfp4`、`tp_device_pair`。

---

## 11. 进行中：TP-2 视觉识别（静态分片分工）

**目标**：在双卡 TP-2 路由上支持 `--vision`。采用**静态分工**而不是动态换入换出：
`mtp/*` 只放 shard 0（shard 1 今天那份 430.4 MiB 是纯浪费 —— MTP 本来就只跑 shard 0），
`vision/*` 与视觉编码 workspace 只放 shard 1，shard 0 完全不动。用户决定：**若显存足够，就不要引入
「单图 token 上限」配置项**。

### 11.1 为什么是这个方案（Round 53 论证，已完成）

现状（三个独立障碍，按严重度）：
1. `normalize_engine_options` 在 `device_b >= 0` 分支硬置 `options.enable_vision = false`
   （`src/runtime/engine/model_instance.cpp:96`）；serve 层还有第二道闸
   （`src/serve/generation_service.cpp:313,368` 的 `vision_disabled`）。
2. `TP2GenerationCore` 全文没有视觉执行路径（只有 `load.vision` 与 `frontend.vision_enabled` 两处透传）；
   编码器会话 `VisionPrefillSession` 只存在于单卡 Program 路径。
3. 显存。

显存账（本机实测）：
- 分量字节（`tools/artifact/reader.py` 直接解析目录）：text 19431.7 / dflash2 2123.6 / mtp 430.4 /
  proposal 340.5 / **vision 282.0** MiB（333 张量、27 层、hidden 1152）。
- vision 在 TP 切分表里无规则 ⇒ `tp_split_spec.cpp:157-159` 的 `else` ⇒ `Replicated`（两卡各一份）。
- 编码 workspace 峰值（`vision.cpp:88-195` 布局 + `startup.cpp:720-725` 的
  `merged = min(context, kMaximumVisionItemTokens=16384)` ⇒ 65,536 patches）：x 144 + patch_bf16 192 +
  qkv 432 + attention_norm 144 + mlp_up 538 + mlp_norm 144 MiB，作用域复用后 **约 1.0–1.2 GiB**（估算，
  ±20%），另有 160 MiB output handoff。
- 现余量 697 / 1217 MiB（nvidia-smi，262,144 配置）。分工后：
  - shard 1：1217 + 430(MTP 移走) − 282(vision) = **1365 MiB** → 装 ~1.0–1.2 GiB workspace，余 165–365 MiB；
  - shard 0：**保持 697 MiB 不动**。
- 与动态换出对比：峰值完全相同，但静态分工零切换、零 pinned host 副本、无请求期分配、
  失败发生在启动期而非请求期，因此不采用动态换出。

### 11.2 Step A —— 按分片放置（可独立验证，先做）

- `tp::TPObjectSplit` 增 `shards` 掩码（默认 `0b11`）：`tp_materialize.h`。
- `object_shard_bytes`/`shard_capacity_bytes` 只统计会收到该对象的卡（`tp_materialize.cpp:77-115`）。
- `materialize_tp2` 对掩码外的卡不分配、不拷贝、`storage.device` 留空（`tp_materialize.cpp:173-261`）。
- `build_tp_split_spec`：`mtp/*` → `shards = 0b01`；`vision/*` → `shards = 0b10`
  （名字前缀已在同一循环内可判，不需要新 option）。
- `shard_views` 跳过本卡没有的对象（`tp_shard_views.cpp:38-52` 的 `device_parent` 会失败）。
- `Parameters`/`TextContext` 容忍缺失：shard 1 不得解引用 MTP 权重，shard 0 不得构造 VisionContext。
- **验收（2026-09-19 完成，全绿）**：
  - 9 项回归全通过（3 条 load route、`tp2_forward` 11 探针 + 32 步 AR + 批量 prefill、`embedding`、
    `linear_tp2_split_fp8_head`、`linear_tp2_split_grouped_head`、`linear_tp2_split_nvfp4`、`tp_device_pair`）。
  - 台账：shard 1 `weights+ctx` **11494.6 → 11064.6 MiB（−430.0）**；shard 0 不变。
    nvidia-smi 15,614 / **14,664** MiB/卡 ⇒ shard 1 余量 **1,217 → 1,647 MiB**。
  - greedy 逐字节：`ab-final.jsonl` 与参考 `ab-embed` **IDENTICAL**（5/5 hash 全同）⇒ 分片放置数值中性。
  - 实现：`TPObjectSplit::shards` 掩码 + 按卡容量/上传 + `shard_views` 保序留空 +
    `Model::has_weight` + `Parameters` 按存在性构造 mtp/vision 块。
  - shard 1 的 1,647 MiB 中，扣除 vision 权重 282 MiB 后余 **1,365 MiB** 给编码 workspace（预计需 1.0–1.2 GiB）。

### 11.3 Step B —— TP-2 视觉执行路径（已实现，待实测）

**执行层（`execution/text.h|.cpp`）**

- 新增 `Tp2VisionChunk`：`control`（本 chunk 覆盖的已编码 item，可为空）、`embeddings`
  （`[hidden, merged]`，属 Vision 卡）、`positions`（prompt 全量 `[3, T]` MRoPE 表）、`prompt_tokens`。
- `forward_tp2_prefill(..., Phase phase = Phase::Prefill, const Tp2VisionChunk* vision = nullptr)`：
  - `make_bind` 在有 `vision` 时为每个 chunk 建 **`[tokens, 3]`** 的 rope 张量（token 维连续，
    与单卡 `[tokens, axes]` 绑定一致；`ne[1]==3` 才是 RoPE op 读 MRoPE 表的分支），
    否则沿用原 `[tokens]`（cache position 与 rope position 仍重合）。
  - `embedding_tp2` 之后，若本 chunk 与某 item 相交：取该 item `scatter_indices` 在
    `[first_position, first_position+tokens)` 内的**连续**列段（`[visual_begin, +count)`），
    shard 1 从 handoff 张量 D2D 拷到本 chunk workspace 的 `source`，shard 0 同样大小的
    `staging` 清零，一次 `pair.allreduce(staging, source)`（与 0 相加 ⇒ 精确拷贝）后两卡各自
    `ops::scatter(staging/source, local_indices, x/x_peer)`。
    代价只与 chunk 相关（≤1024 token ⇒ ≤10 MiB/次），不随图片大小一次性占用。
- `PreparedPromptAccess::mutable_view`（`frontend`）：session 需要可变 prompt 以释放已编码的
  host patch payload。

**核心层（`runtime/engine/tp2_generation_core.*`）**

- 启动：shard 1 + `enable_vision` 时按单卡同一 planner 建 plan
  （`plan_workspace(config, params, min(capacity, 16384), kWorkspaceBytes)`）并独占一个 arena，
  台账新增 `vision %.1f` 列。
- 每请求：`plan_vision_control` → 组装 `VisionPrefillPlan`（`uses` 全量、`control_index==prepared_item_index`、
  `max_merged_count`）→ `build_vision_control(data, plan, 0)` → `VisionPrefillSession`（shard 1）。
- prefill 循环：`vision_session->prepare_chunk(t0, length)` 先按 item 边界截断，再套用 snapshot 边界
  （rewind 边界优先，下一 chunk 重新进入同一 item）；把 `Tp2VisionChunk` 传给 `forward_tp2_prefill`。
- **多模态请求与纯文本请求走同一套机制**（Round 53b 修正，见 §11.6）：MTP 照常开启、
  前缀照常复用；视觉 item 只在"落在复用前缀之外"时才进 session。
- `vision_seconds` 单独上报，`prefill_seconds = 总时长 - vision_seconds`（与单卡语义一致）。
- `model_instance.cpp` 去掉 TP-2 分支的 `enable_vision = false`（DFlash 限制保留）；
  serve 侧 `--vision` → `EngineOptions.enable_vision` → `LoadOptions.vision` 链路本就存在。

### 11.4 Step C —— 端到端与显存裁决（已完成，通过）

- **判据通过**：`load_00/03/07/11.png` 各问「图里的数字是多少」→ 回复 `00 / 03 / 07 / 11`
  全部正确（`tools/tp_bootstrap/r53_numbers.py`）；另 3 张图的自由描述与图内容一致
  （同心圆环、彩色竖条纹、左上角白色标签框）。prompt 1,092 tok，TTFT ~870 ms，
  prefill 1.57k tok/s。
- **显存裁决**：16,384 的完整 envelope **装不下**（首个 `cudaMalloc` 被拒；本机 WSL2 上
  `cudaMemGetInfo` 又不可信，所以判据只能是「分配是否被拒」）。自适应循环落到
  **item ceiling 8,192 vision tokens**、arena **413.3 MiB**（`encode peak 413.3` / `handoff 80.0`）。
  8,192 token ≈ 2,900×2,900 px，已覆盖实际输入。**没有新增任何用户可见配置**；
  超出 ceiling 的单图在请求期被明确拒绝，错误信息指向 `[mem] vision` 台账行。
- 运行期峰值与纯文本完全相同：shard 0 15,614 MiB（余 699）、shard 1 15,360 MiB（余 951）——
  arena 常驻，图片请求不再做大块分配（设计意图）。
- 为什么不是 16,384：`test_vision_workspace.cpp` 自身断言最大 item 的 plan < 827 MiB
  （≈ 8,192 的 413 MiB 线性外推），所以更像**碎片**导致 ~800 MiB 连续块拿不到，而非总量不足。
  把 arena 挪到 KV 之前分配也许能拿到 16,384，但会让 KV/state（4.1 GiB + 293 MiB）先垫底失败，
  风险大于收益，故保持「KV 先、vision 自适应」。
- 图片/带图会话的吞吐见 §11.6（MTP 开启后 68–71.5 tok/s，优于纯文本实测的 59）。
- `ninfer_qwen3_5_vision_workspace_test`（需要 `NINFER_TEST_ARTIFACT`）在本机**不适用**：
  它走单卡 Program，单张 16 GiB 卡装不下这个 27B 模型，实跑 OOM。它断言的最大 item plan
  <827 MiB 只作为量级参考引用，不作为本机证据。

### 11.5 Round 53b —— 修掉「发过图片后整个会话掉到 33 tok/s」（已完成）

**现象与根因**：多轮会话里客户端每一轮都会把图片重新放进 history，所以**后续每一轮请求都仍然带
media**。首版为缩小风险设的两条范围收敛（`use_mtp = mtp_enabled_ && !media`、图片请求 `reuse = 0`）
于是让整个会话都失去 MTP 与前缀复用：decode 33 tok/s、TTFT 每轮 ~0.8 s。

**修法：多模态请求与纯文本请求走同一套机制**

- 取消 `!media`，MTP 照常（回到 `mtp_enabled_`），解码循环/位置推进/解码阶段都用同一个标志。
- 前缀复用恢复：视觉 item 只在 `token_end > reuse`（落在复用前缀之外）时才进 `VisionPrefillPlan`
  与 session，`control_index = prepared_item_index - first_item`（与单卡 `request_plan.cpp:702-724`
  + `materialization.cpp:295-312` 一致）；落在复用前缀内的 item 直接释放 host patch payload。
- **复用前缀覆盖全部 item 时不建 session**（`vision_plan.uses` 为空）——这正是"带图会话的第 2 轮起"
  的常态：不再重编码、不再 scatter，但**只要 prompt 带 media，每个 chunk 仍绑定 3 轴 RoPE 表**
  （那是 prompt 的属性，与是否编码无关）。
- `Tp2VisionChunk` 的传参条件从「有 session」改成「prompt 带 media」，control/embeddings 由 session 填。

**实测（shipped 配置 262,144 + `--vision`，`tools/tp_bootstrap/r53_chat.py`）**

| | prompt | cached | TTFT | decode | 回答 |
|---|---|---|---|---|---|
| 第 1 轮（图片，全新 prefill） | 1,092 | 44 (4%) | 807 ms | **68.0 tok/s** | "07" ✓ |
| 第 2 轮（同会话追加文本问题） | 1,288 | **1,082 (84%)** | **169 ms** | **71.5 tok/s** | "21" ✓ |

- MTP 接受率 `rate=170/216 = 78.7%`，高于纯文本 Round 52 实测的 64.6%。
  ⇒ MTP priming 对图片列仍用占位 token 的 embedding（视觉 embedding 只进文本 KV）这一点
  **没有可测损失**，因此不再增加"把 shifted 视觉 embedding 接进 MTP priming"的复杂度。
- 冷启动 greedy A/B 与 golden `embed` **逐字节相同**（`v53f == embed == v53b`）。
- 顺带查清一个**既有**特性（非本次引入）：结果对"上一轮请求留下的 rewind 深度"敏感——warm 状态下
  A/B 的 req0 会变（`v53c=f4f52ac3`；同一 history 完全可复现 `v53d==v53e=5b4a9783`；冷启动回到
  golden）。触发它的是一轮**长纯文本**请求也会复现同样偏移（`v53g==v53c`），说明与 Vision/MTP 无关：
  `rewind_near_` 决定 prefill chunk 切分点 → 不同 FP 归约次序在近似平局处翻转贪心 token。
  shipped 配置是 `temperature 0.7/top-k 20`，实际使用不可见。
- 纯文本仍逐字节不变：`ab-v53b.jsonl` 与参考 `ab-embed` **IDENTICAL**（5/5）。

### 11.6 已知未决点

- 图片请求的 MTP 已开启（见 §11.5），接受率 78.7%：MTP priming 对图片列用的是占位 token 的
  embedding，实测无损失；若以后要把它做精确（shifted 视觉 embedding 接进 `mtp_prefill_priming`），
  那是约 100 行的工作量，目前没有收益证据。
- 单图 token 上限被硬编码在自适应搜索里（8,192）；若以后要把 shard 1 的 KV 或 state 让出一点，
  可以重新拿到更大的 envelope，无需改接口。
- 视觉编码跑在 shard 1 时 shard 0 在编码期空闲（编码约百毫秒量级，可接受）。

### 11.7 decode 瓶颈测量（Round 53d，仅测量未改动）

同一文本提示（prompt 75 token，output 256，greedy 请求）在 262,144 配置下扫 draft 窗口：

| 配置 | decode | 说明 |
|---|---|---|
| plain（无 MTP） | **32.8 tok/s** | 30.5 ms/次前向（1 token/次） |
| `--spec mtp --draft-tokens 2`（shipped） | **56.8 tok/s** | ~39–41 ms/轮、~2.0–2.3 token/轮 |
| `--draft-tokens 3` | 58.5 tok/s | 与 d2 在噪声内，略优 |
| `--draft-tokens 4` | 55.5 tok/s | 窗口变长的边际收益已为负 |

decode 期间实测（`nvidia-smi -lms 300`）：gpu0 **sm 2820 MHz / mem 13801 MHz / 92.8 W / util 100%**，
gpu2 2805 MHz / 13801 MHz / 84.7 W / util 99% ⇒ **满频、未撞功耗墙（180 W）、无热降频**，
所以不是时钟/功耗问题。`nvidia-smi topo -m` 显示两卡之间是 **SYS**（无 NVLink/P2P，peer 流量过主机桥）。

**结论（瓶颈拆分）**：一轮 ≈ 39–41 ms，其中
- **~30.5 ms（约 74%）是目标模型的一次 verify 前向**：每卡每次读 ~9.7 GB 权重，折合 ~318 GB/s，
  只有 448 GB/s 峰值的 ~71%；差距主要嫌疑人＝每层 row-parallel 的跨卡 allreduce（走 SYS 主机桥）
  + 无 CUDA Graph 的 ~150 次 kernel launch + NVFP4 GEMV 在 5060 Ti 上的实际效率。
- **~9–10 ms（约 24%）是 MTP 起草链 + 窗口组装 + argmax/接受 + state 恢复/折叠 + 每轮 host sync**
  （`mtp_propose_window` 是 host 串行 AR，每个 draft 一次同步；d4 变慢即边际 draft ~3 ms）。

**可压榨清单（按性价比排序）**
1. `Phase::Verify`（decode 等价的 batched GDN/attention）——目前 verify 走近似 `Phase::Prefill`，
   接受率是 token/轮的直接乘数；这正是代码里标注的 "known-incomplete TP-2 path"。
2. 跨卡集合通信：减少每层 allreduce 次数（把 attention-out 与 MLP-down 两次合成一次）、
   或让它与下一层权重流重叠；P2P 在消费级驱动上不可用，只能减少次数/字节。
3. 每轮 9–10 ms 固定开销：draft 链改成 device 侧 AR（去掉每 draft 的 host sync）；
   TP-2 core **完全没有 CUDA Graph**（`cudaGraph` grep = 0），单卡路线是有的。
4. NVFP4 GEMV 按 5060 Ti（448 GB/s、SM 更少）重新调参：用 ncu 看 top kernel 的 DRAM 达成率。
5. 长上下文：262k 时 fp8 KV 读取约占流量 30%，`--kv-dtype k8v4|nvfp4` 可再砍（质量取舍）。
6. 不可行的：超频/功耗墙（已满频、93/85 W）、P2P（驱动屏蔽）。

### 11.8 进程级思考强度（Round 53c）+ 为什么没有 `--chat-template`

- **新增** `ninfer-serve --reasoning-effort low|medium|xhigh`（进程默认，请求体优先）：
  `serve_options.{h,cpp}`（解析 + usage）、`translate.cpp`（`effective_reasoning_effort` 回退链：
  请求字段 → 进程默认 → 模板默认）、`generation_service.cpp`（启动即校验模板是否支持该档）、
  `request_log.cpp`（`server_start` 记录 `default_reasoning_effort`）、`docs/serving.md`（flag 表 +
  说明）、`tests/test_serve_options.cpp`（解析 / usage / 解析语义三处断言，`ninfer_serve_options_test` 通过）。
  实测：`--reasoning-effort high` 被拒（提示 `must be low, medium, or xhigh`）；8088 启动 11 s 健康；
  请求行显示 `thinking medium`；17×23 回答 391、思考 46 tokens（同题 `xhigh` 为 172）。
- **`--chat-template` 不可用**，三条独立证据：
  1. 产品里没有这个 flag（`ninfer-serve: unknown argument: --chat-template`；CLI `ninfer` 也没有）。
  2. 模板是 **artifact 资源**：`frontend.cpp:194-202` 强制
     `tokenizer_config.json.chat_template` 与 `resource/text/chat_template.jinja` 逐字节相同，随后
     `chat_template.cpp:414-424` 按 sha256 只接受两种语义——`e84f32a2…`（thinking-toggle）与
     `c3cf9e34…`（reasoning-effort，本 artifact 内嵌的就是它，所以 `reasoning_effort` 本来就能用）。
  3. 放在模型目录的 `chat_template.jinja` 是**第三个**模板：28,234 B、sha256 `e57684ba…`、
     `template_version = "qwen3.8-froggeric-v22.5"`，带 XML/JSON 工具调用格式、
     `auto_disable_thinking_with_tools`、`max_tool_arg_chars`、`preserve_reasoning` 别名等。即使加了
     flag 也会被白名单拒绝，因为它的渲染语义在本引擎里没有实现（前端不是通用 Jinja 解释器，而是每种
     语义一套手写 C++ 渲染器）。要用它 ＝ 新增一种前端语义（含工具调用/多模态/effort 别名），属独立功能，
     需另行确认范围。

---

## 12. 进行中：Windows 原生移植（分支 `feat/windows-native-port`）

**目标**：同一 artifact、同一套产品代码，在原生 Windows 上跑通——先单卡 CLI/serve，再做 TP-2 spike。

**平台面普查（Round 54，已确认）**
- 很薄：`mmap`/`madvise`/`epoll`/`pthread`/`NUMA`/`setrlimit`/线程亲和 **全无**；线程与同步用 `std::thread`；
  HTTP 服务端是 cpp-httplib，`src/serve/http_transport.cpp` 已有 `#if defined(__linux__)` 结构可直接补分支。
- 待改代码：
  - `src/artifact/file_io.cpp`：`::open`/`::pread`/`O_CLOEXEC` → `CreateFileW` + `ReadFile` + `OVERLAPPED`
    定位读（注意 >4 GiB 偏移、`FILE_FLAG_SEQUENTIAL_SCAN`），约 50 行。
  - `src/product/logging/logging.cpp`、`src/serve/request_log.cpp`：`localtime_r`/`getpid`/`isatty`/`strerror_r`。
  - `src/product/media_acquire/acquire.cpp`：POSIX socket 出站抓图 → 建议改用仓库已有的 cpp-httplib 客户端。
- 工具链（已核实存在）：VS 2022 Community `D:\Program Files\Microsoft Visual Studio\2022\Community`
  （`vcvars64.bat` ✓）、Windows CUDA **v13.3**（`sm_120a` 支持 ✓；WSL 侧是 13.1，对比时要记住这个差异）。
- **最大未知工作量**：`.cu`/`.cpp` 里是否存在 GCC 扩展（`__attribute__`/`__builtin_*`）——需一次 MSVC+nvcc
  编译扫描才能定价。
- **最高风险（决定 TP-2 版 Windows 是否成立）**：`cudaHostAllocMapped` 的双卡 in-kernel allreduce
  （`src/core/tp/device_pair.cu:195`）在 Windows/WDDM 下的可用性与性能。
- 可参考：三卡分支 `C:\llama.cpp` @ `ar3-opt`，HEAD 为
  `92758c647 ggml-cuda: add 3-way AllReduce kernel (on by default) + timing instrumentation`——
  与我们的跨卡集合通信瓶颈（见 §11.7）直接相关。

**步骤**
1. 单卡 `ninfer` CLI：文件 IO + 日志杂项 + MSVC/CMake 构建；跑 5 个 greedy 提示词与 Linux 结果对照。
2. 单卡 `ninfer-serve`：HTTP、流式、媒体输入行为对照。
3. **TP-2 spike**：mapped pinned 双卡是否成立，量 decode 每轮毫秒并与 Linux（~39 ms/轮）对照。
4. 收尾：Windows 构建/运行写进 `docs/`，脚本 `.ps1` 化，回归清单。

**预期**：机械移植 3–5 天；性能上 WDDM 很可能略逊于 Linux（我们已在 448 GB/s 峰值上只拿到 71%，
见 §11.7），所以移植的收益主要是**部署便利**，不是速度。

**进度（Round 54 第 1 轮，已实测）**
- CMake 配置在原生 Windows 上走到了很深：MSVC 19.44.35228 + nvcc 13.3.73（host=MSVC）+ CUDAToolkit 13.3 全部识别成功，
  `sm_120a` 被接受，Threads 找到。CMake 用的是 pip 装的 **4.4.2**（仓库要求 ≥3.28；注意 CMake 4 可能对
  `third_party` 里 `cmake_minimum_required(<3.5)` 报策略错误，必要时加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`）。
- `cmake/NinferTargets.cmake` 里**没有任何 GCC 专用 flag**（只有 CUDA 的 `-lineinfo`），原先担心的
  "GCC 扩展"风险从构建系统这侧看比预想小得多。
- **唯一 configure 阻塞点**：`cmake/Dependencies.cmake:3` 的 `find_package(PkgConfig REQUIRED)`。
  它被用来找两样东西：
  - `FFMPEG`（libavformat/libavcodec/libavutil/libswscale，**无条件 REQUIRED**）：`src/media/decode/decode.cpp`
    的 60 处调用就是媒体的实际解码器 ⇒ 图片/视频解码依赖它，**不是可选项**。
  - `libcurl>=7.85`（仅 `NINFER_BUILD_PRODUCT_SUPPORT`）：只被 `src/product/media_acquire/acquire.cpp`
    用来抓远程 http/https 媒体（26 处）。
- 机器上既没有 pkg-config/pkgconf，也没有 vcpkg/MSYS2/conda；`C:\WINDOWS\system32\curl.exe` 只是运行时。
  `C:\ffmpeg-8.1.2-full_build` 只有 `bin`，**没有 `include`/`lib`（dev 文件缺失）**，所以不能直接拿来链接。

**下一步决策（第 2 轮）**
1. 取 MSVC 可用的 ffmpeg dev 文件：优先 gyan.dev 的 `ffmpeg-release-full-shared`（带 `include/`+`lib/*.lib`）；
   直连不通时按项目规则走本地代理 `127.0.0.1:7897`；备选是 vcpkg 源码构建（耗时更长）。
2. 把 `cmake/Dependencies.cmake` 改成平台感知：Unix 保留 pkg-config，Windows 用
   `NINFER_FFMPEG_ROOT` + `find_path`/`find_library` 定位同一组库；libcurl 同样处理，或把
   `media_acquire` 的抓取换成 WinHTTP 以彻底去掉该依赖（该文件本身就是 POSIX socket 待改点）。
3. 重新 configure → 首次编译扫描，定价 `.cu/.cpp` 的 MSVC 兼容性。
- 已取到 MSVC 可链接的 FFmpeg dev（BtbN win64-gpl-shared）：
  `D:\ffmpeg-dev\expanded\ffmpeg-master-latest-win64-gpl-shared`，`include/libavcodec|libswscale` 与
  `lib/{avcodec,avformat,avutil,swscale}.lib` 全部就位。下一步把它接进 `cmake/Dependencies.cmake` 的
  Windows 分支（`NINFER_FFMPEG_ROOT` + `find_path`/`find_library`，Unix 侧继续走 pkg-config）。

**进程占用纪律（用户要求）**
- Windows 侧一旦跑起 `ninfer`/`ninfer-serve`，同一个 23.7 GB artifact 会被再装一遍（显存 + host KV/state），
  与 WSL 侧 8088 服务抢显存和内存。**移植期间 WSL 的 8088 服务保持停止**（Round 54 已停：三卡
  `memory.used` 均为 0 MiB、health 无响应）；需要跨平台对照测量时再临时启动，测完立即停。
**进度（Round 54 第 1 轮）：核心库在 Windows 上编译通过**

工具链 VS2022 19.44 + CUDA 13.3 nvcc + Ninja，`-DCMAKE_CUDA_ARCHITECTURES=120a`；
`tools/win_port/configure.bat` + `build.bat`（`NINFER_JOBS`，默认 12）。当前配置 apps 关
（`-DNINFER_BUILD_APPS=OFF`），`cmake --build build-win` → `BUILD_EXIT=0`，10 个静态库
（artifact/core/engine/media_decode/model_loading/model_runtime/nvfp4_non_rdc/ops/runtime_support/text）
+ 394 个目标文件。

移植改动（已落地）：

| 位置 | 问题 | 处理 |
|---|---|---|
| `cmake/Dependencies.cmake`、`src/{media,product}/CMakeLists.txt` | 本机无 pkg-config | 平台分支：Windows 走 `NINFER_FFMPEG_ROOT`/`NINFER_LIBCURL_ROOT` + `find_path`/`find_library`；导入目标改名 `ninfer::ffmpeg`/`ninfer::curl` |
| `CMakeLists.txt` | CUDA 13 的 CCCL 要求 `/Zc:preprocessor`；`windows.h` 的 min/max 宏 | MSVC 下加 `/Zc:preprocessor`（CUDA 经 `-Xcompiler`）+ `NOMINMAX WIN32_LEAN_AND_MEAN` |
| `src/artifact/file_io.{h,cpp}` | `open/fstat/pread/close`、`unistd.h`、`off_t/ssize_t` | `CreateFileW`（顺序扫描 / `FILE_FLAG_NO_BUFFERING`）+ `ReadFile` 带 `OVERLAPPED` 偏移做定位读；句柄类型 `NativeFileHandle`（Windows 为 `void*`） |
| `src/ops/linear/nvfp4/nvfp4_w4a4_tma.{cuh,cu}`、`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.{cuh,cu}` | C2719：128 字节对齐的 TMA 描述符不能按值传参 | 新增 `Nvfp4TmaDescriptorStaging`（`cudaMallocAsync` + `cudaMemcpyAsync` 到设备内存，传指针，仅 `_WIN32`）；内核体内用引用，保证两条路径共享同一份代码 |
| `src/core/uint128.h`（新） | MSVC 无 `__int128`/`__uint128_t` | 新增 `Uint128`（`_umul128` 或原生 128 位 + 饱和加法/右移/比较）；改 `runtime/contract/resources.h`、`runtime/engine/context_cache/context_cost.cpp`、`context_cache/materialization_planner.h` |
| `src/core/host_process.h`（新） | `getpid`/`isatty`/`localtime_r` | `host_process_id()`/`host_stderr_is_interactive()`/`host_localtime()`；`context_cost.cpp` 已改用 |
| `src/text/CMakeLists.txt` | utf8proc C2491（静态库里定义 dllimport） | `target_compile_definitions(ninfer_text PRIVATE UTF8PROC_STATIC)` |
| `src/models/qwen3_5/execution/text.cpp` | C2397 非恒定窄化（GCC 只警告）；`void*` 算术是 GCC 扩展 | 显式 `static_cast`；先 `static_cast<const std::byte*>` 再做字节偏移 |

经验：GCC 把「非恒定窄化」当警告，MSVC 直接报 C2397，所以 Linux 能编的代码在 Windows 上会逐个暴露，需逐个加显式转换。

Windows 侧资源与路径：

- FFmpeg dev：`D:\ffmpeg-dev\expanded\ffmpeg-master-latest-win64-gpl-shared`（BtbN shared，含 `include/`+`lib/`）。
- libcurl：`D:\curl-dev\expanded\curl-8.22.0_1-win64-mingw`（**只有 `lib/libcurl.dll.a`，mingw 版**；MSVC 能否直接链接待验证，否则换 vcpkg 或改 WinHTTP）。
- 模型 artifact：`D:\LLM\qwen3_8_27b_nvfp4.ninfer`（23,719,715,076 B，与 WSL 侧同一个 artifact，可直接做跨平台对照）。
- 自定义模板：`D:\LLM\chat_template.jinja`（同 WSL 侧那份 28,234 B）。

下一步（apps 关→开）：`src/product/logging/{logging,startup_log}.cpp`（`unistd.h`/`localtime_r`/`isatty`/`sys/ioctl.h` TIOCGWINSZ →
`host_process.h` + `GetConsoleScreenBufferInfo`）、`src/serve/request_log.cpp`（`getpid`）、
`src/product/media_acquire/acquire.cpp`（POSIX socket + libcurl）、`src/serve/http_transport.cpp`（`#if defined(__linux__)` 分支）；
之后单卡 `ninfer` CLI 与 Linux 输出对齐（贪婪基准哈希见 §11），再进 TP-2 mapped-pinned spike。
**进度（Round 54 第 2 轮）：CLI / serve 全部编译链接通过，并能在 Windows 启动**

配置 `-DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_FFMPEG_ROOT=… -DNINFER_LIBCURL_ROOT=…` →
`BUILD_EXIT=0`，产物 `build-win/apps/{ninfer,ninfer-serve,ninfer-perplexity}.exe`（各约 188 MB）；
`ninfer.exe --help` 与 `ninfer-serve.exe --help` 均正常输出、退出码 0（说明 DLL/CRT/链接已就位）。

本轮新增修复：

| 位置 | 问题 | 处理 |
|---|---|---|
| `CMakeLists.txt` | spdlog 内置 fmt 要求 `/utf-8` | MSVC 下加 `/utf-8`（CUDA 走 `-Xcompiler`） |
| `src/core/host_process.{h,cpp}` | `ioctl(TIOCGWINSZ)`、`gmtime_r` | 新增 `host_terminal_columns()`（`GetConsoleScreenBufferInfo`）与 `host_gmtime()`；`src/core/CMakeLists.txt` 注册 `host_process.cpp` |
| `src/product/logging/{logging,startup_log}.cpp`、`src/serve/request_log.cpp` | `localtime_r`/`isatty`/`getpid`/`unistd.h` | 改用 `core/host_process.h` 的平台函数 |
| `src/product/media_acquire/acquire.cpp` | `<arpa/inet.h>` 等 POSIX 头；`relative.native().starts_with("..")`（Windows 是宽字符） | 平台分支 `<winsock2.h>/<ws2tcpip.h>` + 一次性 `WSAStartup`；改判首个路径分量是否为 `path("..")` |
| `src/product/CMakeLists.txt` | WinSock 符号未解析 | 链 `ws2_32` |
| `cmake/Dependencies.cmake` | `find_library` 选中**静态** `libcurl.a`（mingw 静态档，MSVC 无法链接） | 优先 `find_file(... libcurl.dll.a)` 取导入库，找不到再回退 `find_library` |
| `apps/perplexity/main.cpp` | `gmtime_r` | 改 `ninfer::host_gmtime`（该文件处于全局匿名命名空间，必须写限定名） |

已验证：MSVC 能链接并**运行** mingw 版 `libcurl.dll.a` + `libcurl-x64.dll`（探针 `tools/win_port/probe_curl_link.bat`，
打印 `curl libcurl/8.22.0 LibreSSL/…`，退出码 0），因此 `media_acquire` 在 Windows 上继续用 libcurl，暂不需要 WinHTTP 重写。

运行环境（重要）：两个 exe 运行时需要 `PATH` 含

    D:\ffmpeg-dev\expanded\ffmpeg-master-latest-win64-gpl-shared\bin;D:\curl-dev\expanded\curl-8.22.0_1-win64-mingw\bin

（`libcurl-x64.dll` 必须可见）。启动前确认没有第二个模型进程：`nvidia-smi memory.used` 三卡均为 0 MiB。

下一步：单卡 CLI 真实生成（`D:\LLM\qwen3_8_27b_nvfp4.ninfer`）并与 Linux 贪婪基准哈希对齐 → serve 端到端 →
TP-2 mapped-pinned spike（重点验证 `src/core/tp/device_pair.cu` 的 `cudaHostAllocMapped` 在 WDDM 下是否可用）。
**进度（Round 54 第 3 轮）：TP-2 在 Windows 上跑通，性能与 Linux 持平；mapped-pinned spike 通过**

端到端实测（`tools/win_port/serve.ps1`，两卡 5060 Ti，artifact `D:\LLM\qwen3_8_27b_nvfp4.ninfer`）：

| 项目 | Linux (WSL2) | Windows |
|---|---|---|
| TP-2 纯 decode | 32.8 tok/s | 32.9 tok/s |
| TP-2 MTP `--draft-tokens 2` | 56.8 tok/s | 56.9 tok/s |
| 图片轮 decode | 53b 修复后正常 | 60.4 tok/s，MTP 持续（rate=106/174） |
| 引擎加载（23.7 GB artifact） | — | 35–42 s |
| 真实 agent 流量（流式 / tools / 图片） | — | 已服务，41–73 tok/s |

KV 容量与显存余量（实测，带 `--vision`）：capacity 8192 → kv 145 MiB、free 4180/3930 MiB；
capacity 131072 → kv 2322 MiB、free 2002/1580 MiB。**Windows 推荐 131072**（262144 是 WSL 配方，
WDDM 下余量过小）。

**进度（Round 10）：与 llama.cpp `ar3-opt` 的同机对照 —— decode 打平的原因不在内核**

新增/修改脚本：`tools/win_port/bench_serve.ps1`（同一套工作负载压任意 OpenAI 兼容服务，数字全取服务端自报的 `timings`）、
`tools/win_port/serve_llama.ps1`（把本机调好的 llama.cpp 配方固化，含 `-Stop`/`-Status`；与 `serve.ps1` 互斥，同一时刻只驻留一个引擎）、
`serve.ps1` 增加 `-DraftTokens`。协议：temperature 0、256 输出、medium 思考、两边都开 MTP、同题面、一次一个引擎。

| 工作负载 | ninfer | llama.cpp | 比值 |
|---|---|---|---|
| prefill 1.9k | 1,401 tok/s | 764 tok/s | 1.83x |
| prefill 7.2k | 1,667 tok/s | 1,040 tok/s | 1.60x |
| prefill 28.5k | 1,552 tok/s | 1,016 tok/s | 1.53x |
| decode 256 贪婪 token（3 次） | 56.7 / 56.7 / 56.7 tok/s | 51.7 / 52.6 / 52.7 tok/s | 1.08x |

同一 decode 题面的单轮预算：ninfer K=2 38.2 ms/轮、2.18 committed、58.8% 单 draft 接受率；ninfer K=3 40.7 ms、2.27、42.4%；
llama.cpp `n_max=3` 49.4 ms、2.60、53.8%（llama.cpp 每 256 个 token 复用 95–96 次 CUDA graph）。

结论（完整论证见 `docs/tp2-dual-5060ti.md` 的 Cross-engine comparison 一节）：

1. **batch=1 decode 是权重流不是算力题**：`tok/s = 每轮 committed / 轮时长`，轮时长下限 = 每卡权重字节 / 448 GB/s。
   ninfer 每卡 10.15 GB/forward，llama.cpp 8.56 GB（用 `llama-gguf` 实测其 GGUF 15.94 GiB 的类型分布：NVFP4 MLP 10.38 GiB、
   q5_K 注意力/GDN 3.09 GiB、q6_K 词表 1.55 GiB、q8_0 输出头 1.29 GiB）。两者分别落在各自下限的 59% / 39%。
   两张 5060 Ti 合计 896 GB/s = 一张 5090（1,792 GB/s）的一半，这正是同一 artifact 在 5090 上 71.2 tok/s、在这里 ~57 tok/s 的原因：
   **TP-2 是容量决策（27B 装不进单张 16 GiB），不是提速决策**。
2. ninfer 的定向优化确实生效在"算得动"的地方：每轮比 llama.cpp 快 23%，prefill 快 1.5–1.8 倍。
3. 吃掉这点优势的四项：**draft 深度**（2.18 vs 2.60；本 artifact 上 K=3 反而掉到 55.7）、**FP8 权重多读约 26% 字节**、
   **TP-2 关掉了 CUDA graph**（`model_instance.cpp:99` `use_cuda_graph=false`，~1,009 kernel/forward、~2.9 ms 启动间隔）
   且**头切分把 allreduce 翻倍到 128 次/token**、融合投影内核在分片行上被绕过（`text.cpp:1104-1113`）、
   **MTP 链每轮 7–10 ms 串行在 host**（`mtp_propose_window` 每个 draft 一次 `cudaStreamSynchronize`）。
4. 后续优化顺序（按证据强度）：(a) 给 TP-2 的 verify forward 上 exact-batch CUDA graph（~7.6%/轮）；
   (b) 把 attention/GDN/embedding/head 与末 8 层 FFN 从 FP8 换 NVFP4（每卡每 token −2.6 GB，约 +18%，有精度代价）；
   (c) 减少 collective 次数（不切头或合并同层两次 AR）；(d) 让 draft 链与下一轮 verify 重叠或下沉到设备端。

**Round 10 追加：优化 ①③④ 的理论收益核算（模型推算，非实测）**

decode 单轮 38.2 ms / 2.166 committed（=56.7 tok/s）拆账：权重流 25.6 ms（448 GB/s 理论 22.7 ms，效率 89%）
+ allreduce 1.7 ms（128 次 × 9–18 µs，已从 42 µs 优化过）+ 发射间隙 2.91 ms（1009 kernel/forward，nsys 实测）
+ MTP 链 / accept-fold / host 同步 7.7 ms（= 轮时长 − verify 30.5 ms）。
prefill 每 1024-token 块 660 ms 拆账（按 worklog Round 35c 的 580 ms = AR 350 + MMA 188 + 其它 35 换算到 Windows）：
AR ≈350 ms（53%）、MMA+其它 ≈310 ms；AR 已贴 Gen4 x4 链路地板（2.6 MB/token ÷ 7 GB/s → 2.67k tok/s）。

| 项 | decode 理论收益 | prefill 理论收益 |
|---|---|---|
| ① exact-batch CUDA graph | 发射间隙 2.91 → 0：38.2→35.3 ms，61.4 tok/s（+8.3%） | 块内间隙摊薄，≈ −0.4% |
| ③ collective 重叠（**不是删掉**，TP 必需） | −1.7 ms（与 draft 链重叠）：→33.6 ms，64.5 tok/s | AR 与 MMA 重叠：660→≈390 ms，**≈2.6k tok/s**（受链路 2.67k 封顶） |
| ④ draft 链下沉/重叠（保留 ~3 ms 设备端） | −4.7 ms：→28.9 ms，**74.9 tok/s（+32%）** | 不适用（prefill 不走 MTP） |

合计理论值：decode **≈75 tok/s**（绝对地板 22.7+3.0=25.7 ms → 84 tok/s）；prefill **≈2.6k tok/s**（worklog 自己的保守估计 1.5x → ≈2.1k）。
③ 的边界：worklog 已算过头切分的账——AR 次数 64→128 多花 4.9 ms，换回 9.4 ms 计算，净赚；
所以"减少 AR 次数"必须放弃权重切分，在 decode 上一定亏，唯一正确方向是"重叠"。

**①③④ 的显存账**（每卡；Windows 131072 实测 free 2002/1580 MiB，262144 配方 free 0/196 MiB）：

| 项 | 增量/卡 | 机制 |
|---|---|---|
| ① CUDA graph | ≈ 0–几 MiB（若用 graph 私有 pool 最坏 ≈ 192 MiB） | 所有 arena 启动即常驻（workspace 192.0、state 293.6、KV、权重），graph 只新增 exec 与节点簿记；硬前提是把 AR arrival token 从 mapped host 移到设备端（KiB 级）。风险点本仓库出现过：capture 内 `cudaMallocAsync` 会变成 graph memory node（TMA 描述符那次），故 capture 期的一切分配必须先预分配 |
| ③ AR∥MMA 子块流水 | **+190–580 MiB** | 2–4 个 in-flight 子块各自需要激活区（workspace 192 MiB × 流水级数）与 GDN 状态快照（state 293.6 由 4–5 份组成，≈59–73 MiB/份）。AR staging 本身是 `cudaHostAllocMapped` 的 host pinned（2×24 MiB × 2 缓冲 = 96 MiB/卡），扩容只吃主机内存 |
| ④ draft 链下沉/重叠 | 只去 host 同步 ≈ 0–10s MiB；若让 verify 与 draft **并发**则需第二份 MTP KV：**+258 MiB**（仅 shard 0，131072） | MTP 权重 430 MiB 已在两卡复制、MTP KV 已存在（262144 时 516.2 MiB 仅在 shard 0） |

合计 **+0.2–0.85 GiB/卡**：131072 能放下（余量降到 shard 0 ≈ 1.4–1.8 GiB、shard 1 ≈ 1.0–1.4 GiB），
262144 放不下（free 0/196）。反过来 ② 权重 NVFP4 化会**省** ≈2.15 GiB/卡——② 与 ③ 一起做等于既提速又把 262144 装回来。

**mapped-pinned spike（go/no-go 已通过）**：`tools/win_port/spike.ps1`（源 `tp_mapped_spike.cu`）实测
peer access 0→1 / 1→0 均为 0（SYS 拓扑，无 P2P）；`cudaHostAllocPortable|cudaHostAllocMapped` +
`cudaHostGetDevicePointer` 两卡均成功且 peer 看到**同一 UVA 地址**；跨卡读 mapped host 内存校验
`sum=match`，带宽约 1.4–1.9 GiB/s（本卡写 5.8 GiB/s）。结论：WDDM 下 in-kernel mapped-pinned allreduce
可用、与 WSL2 同档，这正是端到端 decode 与 Linux 持平的原因。

**重要环境事实**：`C:\Users\zhuojun\.dsh\settings.yaml` 已配置 `ninfer-win` provider
（`baseURL: http://127.0.0.1:8099/v1`）且 `agent-default-model` 指向它。也就是说：Windows 侧
ninfer-serve 跑在 8099 就是本机 DSH 的默认模型服务；测试期间它已服务过真实流式/tools/图片请求
（`req#4 media 1, prepared 7.91 ms`）。WSL 侧 8088 已停，全机只有一个模型进程。

**脚本教训**：`Start-Process` 起的服务是「调用方 shell 的 job object 子进程」，agent 工具调用结束时会被
连带杀掉（现象：进程在、GPU 0%、模型从未加载）。因此 `serve.ps1` 默认**前台**运行（终端用户拥有进程，
Ctrl+C 停止），只有 `-Background` 才用 `Start-Process`；自测服务时必须用 harness 的后台 job（跨调用存活）。

工具与文档：`tools/win_port/` 现全部为 PowerShell（`vcvars/build/serve/spike/fetch_ffmpeg/fetch_curl`，
旧 `.bat` 已删）；新增 `docs/windows.md` 并在 `docs/README.md` 登记；`docs/windows.md` 含构建、运行、
平台差异、实测数据与限制（单卡 CLI 需 ≥20.2 GiB 显存，本机 16 GiB 卡只能走 TP-2）。
**进度（Round 54 第 4 轮）：提交与回归验证**

- 提交：`453aaa6a feat: build and serve natively on Windows with VS2022 and CUDA 13.3`（分支
  `feat/windows-native-port`）。
- Linux 回归：`build_r35.sh` → `BUILD_EXIT=0`（全量重建 + 测试目标）。修复了一处**测试基建问题**：
  `build_r35.sh` 的 rsync 列表缺 `cmake` 与顶层 `CMakeLists.txt`，导致 WSL 树仍用旧的
  `cmake/Dependencies.cmake`（`PkgConfig::FFMPEG` vs 新的 `ninfer::ffmpeg`）而配置失败；现已补上
  （注意：该脚本会同步自身，改动要**跑第二次**才生效）。
- 受影响的 Linux 测试：`context_cost`、`resource_manager`、`pretty_logging`、`context_cost_measure`
  全部通过；`artifact|nvfp4|qwen3_5` 共 26 项 25 通过。
- **唯一失败为既有问题（与本分支无关）**：`ninfer_qwen3_5_frontend_test` 抛
  `unsupported frontend/chat_template.jinja (sha256 821b1c036748885c26b552d2a174878fe06199c390ccf612a6169326338cbaa5)`
  —— 即 53c 引入的模板摘要白名单拒绝了测试夹具；`src/models/qwen3_5/chat_template.cpp` 与 `tests/`
  在本分支均未改动，故 HEAD 上同样失败。
- 教训：Windows 上重新链接 exe 前必须先停服务（运行中的 exe 会锁住文件 → `LNK1104`）。

**剩余缺口（用户决定不做）**：目标里的「跑通单卡 ninfer CLI」在本机无法字面完成——CLI 仅支持单卡，而该
27B NVFP4 artifact 需要约 20.2 GiB 常驻权重（10.1 GiB/shard × 2），16 GiB 卡装不下（已写入
`docs/windows.md` 限制节）。要真正跑到端到端，需要另造一个小的 `.ninfer` 测试 artifact（转换工具链 +
合成小 checkpoint），属额外工作；当前 CLI 二进制本身已验证可运行（`--help`），且与 serve 共用同一套
引擎/加载器（已端到端跑通）。
**进度（Round 54 第 5 轮）：Windows 测试套件跑通，并抓到一处真实移植缺陷**

- 新增 `tools/win_port/test.ps1`：先把 FFmpeg/curl 的 `bin` 放进 PATH 再跑 ctest。**必须如此**：测试 exe 运行时
  加载 avcodec/swscale/libcurl，PATH 缺失时 Windows loader 会对每个 exe 弹「找不到 DLL」对话框，而不是报测试失败。
- Windows 全量测试目标编译链接通过（253 个目标，`BUILD_EXIT=0`）。为跑通测试补的平台分支：
  `test_pretty_logging`（`pipe/dup/dup2/read` → `_pipe/_dup/_dup2/_read` + `_setmode(_O_BINARY)`）、
  `test_context_cost`/`test_request_log`（`getpid` → `ninfer::host_process_id()`）、`tests/artifact/fixture.h`
  （`mkdtemp` → 唯一目录名循环）、`test_gdn_replay_records`（`std::aligned_alloc` → `_aligned_malloc`）、
  `test_host_timing`（`#undef near`：winnt.h 的遗留空宏）、3 处 `constexpr ... std::sqrt`（MSVC 不折叠）、
  1 处 `std::array` CTAD 显式化、4 个文件补 `<array>`。
- 唯一平台排除：`ninfer_artifact_materialization_test` 的 CUDA 故障注入依赖 GNU ld `--wrap`，MSVC 链接器无
  等价物；`tests/artifact/tests.cmake` 用 `if(NOT MSVC)` 排除该源文件与 `--wrap` 选项，其余检查照常。
- Python 工具平台分支：`tools/artifact/file_io.py`（`os.sysconf`/`posix_fadvise`/`fdatasync` 有则用、无则
  降级：丢弃页缓存变 no-op、`fsync` 兜底；新增 `read_at`/`write_at` 封装 `os.pread`/`os.pwrite`），
  `reader.py`/`writer.py`/`convert/sources/safetensors.py` 改用它。
- **真实产品缺陷（由测试套件抓到）**：Windows `open_handle` 只用 `FILE_SHARE_READ`，导致**读取 artifact 期间
  无法重写/替换/删除该文件**（POSIX 读语义允许），表现为 `ninfer_artifact_reader_test` 抛
  `ios_base::failbit`。已改为 `FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE`，与 POSIX 读语义对齐。
- 回归：Windows `artifact` 3/3 通过；Linux 23/23 通过（`artifact|context_cost|pretty_logging|request_log|
  host_timing|resource_manager|kv_cache|gdn|attn_input_proj|gated_delta`），Linux 全量构建 `BUILD_EXIT=0`。
**进度（Round 54 第 6 轮）：修掉 Windows A4 TMA 的真实缺陷（CUDA Graph 捕获）**

- 症状：`ninfer_linear_nvfp4_a4_test` 在 Windows 上抛 `cudaErrorIllegalInstruction`；`CUDA_LAUNCH_BLOCKING=1`
  把它定位到 `src/core/decode_graph.cpp:144` 的 `cudaGraphLaunch` —— 即失败发生在**捕获后的 CUDA Graph** 中。
- 根因：第 4 轮为绕开 MSVC C2719 把 tensor map 改成「`cudaMallocAsync` 设备暂存 + 从**栈上** host 对象
  `cudaMemcpyAsync`」。这条路在 graph 捕获下不成立：流序分配会变成 graph 的 memory node（地址到实例化时才
  确定），而 pageable 源指针在 replay 时早已失效 → TMA 读到垃圾描述符 → illegal instruction。
  Linux 不受影响（那边是 `__grid_constant__` 按值传参）。
- 修复：Windows 改用 **mapped pinned host arena** 存描述符（`cudaHostAllocMapped|Portable` +
  `cudaHostGetDevicePointer`，32 槽轮转；host 直接写、TMA 经同一 UVA 地址读），启动路径**不再发出任何 CUDA
  调用**，因此捕获安全；内核侧用 `fence.proxy.tensormap::generic.release/acquire.gpu` 把描述符发布给
  tensor-map proxy（PTX ISA 8.3，SM_90+）。linear 与 linear_swiglu 两个 A4 内核同步修改。
- 结果：`ninfer_linear_nvfp4_a4_test` Windows 通过；Windows 线性/TMA 面 30/30、Linux 同面 29/29。
- 全量 Windows 套件（124 项）修复前 121 通过/3 失败，现 a4 已修；`ninfer_context_kv_materialize_test` 在整轮
  串行运行时撞 300 s 超时、单独运行通过（判为干扰，非缺陷）；`ninfer_qwen3_5_frontend_test` 与 Linux 同样
  存在的既有模板白名单问题。
**进度（Round 54 第 7 轮）：单卡 CLI 条款的取证与判定**

- 事实：Windows 侧 `ninfer.exe` 可运行、参数处理正确（`ninfer_cli_options_test` 在该平台 **exit 0**），CLI 之下
  的加载/引擎/KV/graph/采样全部是已被 TP-2 服务端到端验证过的同一套引擎（Windows 与 Linux 性能持平）。
- 判据：本机唯一 artifact（27B NVFP4）单卡需 ~20.2 GiB 常驻权重 vs 16.3 GiB 显存，且**没有预分配容量检查**
  （grep 未发现 "does not fit" 类前置校验），强行单卡运行会被 WDDM 把缺的 ~4 GiB 换页到主存而不是干净报错——
  正是用户要求避免的风险，故**不执行**。
- 结论：该条款属硬件约束（不是移植缺口）。若要字面满足，需要另造小 artifact：Windows 侧无 torch（miniconda
  3.13 未装），WSL 侧有项目 py311 环境可转换，但还需写一个 tiny 模型的 recipe —— 属独立子任务，等用户决定
  是否投入。
**进度（Round 54 第 8 轮）：单卡 CLI 小 artifact 子任务计划（跨压缩记忆）**

目标：在本机产出一个小到能装进单张 16 GiB 卡的 `.ninfer`，然后用 Windows `ninfer.exe` 单卡跑通
「加载 → 生成 → 采样」，为「跑通单卡 CLI」条款提供实测证据。

已确认的门槛与事实（勿重复调研）：
- 本机唯一 artifact：`D:\LLM\qwen3_8_27b_nvfp4.ninfer`（27B NVFP4，单卡需 ~20.2 GiB > 16.3 GiB）；引擎
  `src/runtime` 内**没有**预分配容量前置检查，强行单卡会被 WDDM 换页 → 禁止尝试。
- Windows python 为 miniconda 3.13，**无 torch**；WSL 侧无 miniconda/py311（AGENTS.md 里那是上游作者机器）。
- 转换器入口 `python -m tools.convert`（`tools/convert/__main__.py`），官方 recipe 只有 27B 级
  （`qwen3_8_27b_nvfp4` 等），tiny 模型需自写 recipe（`--recipe my_recipe.py[:configure]` 或 `--override`）。
- 文档：`docs/weight-conversion.md`（recipe/source/override 语义、safetensors 源）。

拟定步骤（每步可独立验证，失败即可回退）：
1. 装**CPU 版** torch（避免 2.5 GB CUDA 轮子）：优先 `pip install torch --index-url
   https://download.pytorch.org/whl/cpu`；直连失败再走本机代理 `http://127.0.0.1:7897`。若不可行，改走
   「纯 numpy 的合成 `LogicalSource`」，绕开 torch。
2. 读 `tools/artifact/*.py` 的 `LogicalSource` 接口 + `tools/convert/sources/safetensors.py` 的用法，确定合成源
   需要提供什么（张量名/形状/字节）。
3. 按 `src/models/qwen3_5/config.cpp` 的校验与 HF 命名写一个 tiny config（例如 2 层、hidden 1024、
   head_dim 128、q/kv heads 满足整除约束），生成随机权重（numpy/torch，fp32/bf16 即可）。
4. `tools/convert` 转换出小 artifact（先试 `--override` 只改 source/输出路径，必要时写 tiny recipe）。
5. Windows 单卡运行：`ninfer.exe <tiny.ninfer> --prompt "..." --max-new 8`，记录加载时间、是否生成、退出码；
   若引擎对极小 shape 有硬约束（静态断言/blocksize），按报错调大 config 的最小可行值。
6. 成功则写入 `docs/windows.md`（单卡 CLI 实测数据）并提交；长期不收敛则把该条款记为硬件+工具链限制。
**进度（Round 54 第 9 轮）：子任务前置条件就绪**

- CPU 版 torch 已装在 `build-win\torch-venv`（gitignored，不动 base 环境）：**torch 2.14.0+cpu / numpy 2.5.3** ✓
  （`build-win\torch-venv\Scripts\python.exe`）。
- 转换器接口已确认：`python -m tools.convert --model <checkpoint目录> --recipe <官方名|file[:func]> --out X.ninfer
  [--components text] [--override file.py] [--source NAME=PATH] [--name NAME] [--device DEVICE]`；
  `--model` 即「主 checkpoint/config + 默认资源」。
- 下一步（按代价从低到高）：① 先直接用官方 recipe 跑 tiny checkpoint（`--recipe qwen3_8_27b_nvfp4
  --components text`），若它按 config 驱动逐层映射即可零改动成功；② 不成功再写 tiny recipe/override；
  ③ 生成脚本放 `tools/win_port/tiny_model.py`（合成 tiny config + 随机 bf16 权重，HF 命名）。

---

## Round 11: ①③④ 的实现（TP-2 双卡优化）

目标（用户明确要求）：在 `feat/windows-native-port` 上实现 PLAN §12 表格里的 ①③④ 三项，**不做 ②**（权重 NVFP4 化，
它要求新 artifact + 逐形状 Op 准入，已判定为部署期变体而不是运行时开关）。

三项定义与理论值（PLAN:640-667 的核算，基线 decode 38.2 ms/轮、2.166 committed、56.7 tok/s；prefill 1552 tok/s@28.5k）：

| 项 | 内容 | decode 理论 | prefill 理论 |
|---|---|---|---|
| ① | TP-2 verify forward 上 exact-batch CUDA graph（现被 `model_instance.cpp:99` 关掉） | 发射间隙 2.91→0 ms：35.3 ms / 61.4 tok/s | 块内间隙摊薄 ≈ −0.4% |
| ③ | collective 与 compute **重叠**（不是删 AR；删 AR 会亏，见 PLAN:655） | −1.7 ms（AR 与 draft 链重叠） | AR∥MMA：660→≈390 ms / ≈2.6k tok/s |
| ④ | MTP draft 链下沉/重叠（每 draft 一次 cudaStreamSynchronize，见 tp2_generation_core.cpp:543） | −4.7 ms → 28.9 ms / **74.9 tok/s** | 不适用 |

单轮预算拆账（PLAN:642-644）：权重流 25.6 ms（448 GB/s 地板 22.7）+ AR 1.7 ms（128 次）+ 发射间隙 2.91 ms
（1009 kernel/forward）+ **MTP 链/accept/fold/host 同步 ≈ 8.0 ms**。prefill 每 1024-token 块 660 ms = AR ≈350 + MMA/其它 ≈310。

显存账：合计 **+0.2–0.85 GiB/卡**；131072 配方（free 2002/1580 MiB）能放下，262144（free 0/196）放不下。

### 已确认的接口事实（勿重复调研）

- `forward_tp2_prefill`（src/models/qwen3_5/execution/text.cpp:1766）是 TP-2 唯一的分块前向入口，解码轮的 verify
  也走它（Phase::Prefill，见 tp2_generation_core.cpp:999-1005）。它的 host 依赖只有三处：
  `copy_i32(ids.data(), b.ids, stream)`（:1826）、`ops::fill_i32_positions(b.positions, first_position, stream)`（:1827）、
  `b.envelope = {visible_end, visible_end}`（:1823）。device-driven/graph 化必须把前两者改成设备张量、
  把 envelope 变成按桶捕获的常量。
- 单卡路径的 graph 模式可直接照搬：src/models/qwen3_5/program/graph_execution.h（capture_graph/run_prepared）、
  program/decode.cpp:24-63（captured body 首尾各一次 **pinned** host↔device memcpy，program_impl.cpp:62 分配
  PinnedHostBuffer）、program/planning/graph_profiles.cpp:62-89 mtp_graph_profiles（capacity 131072、K=2 时
  ends = 124/508/2044/4092/8194/16386/32764 → **8 个桶**，envelope 按桶最大值捕获，逐轮真实位置由设备张量携带）。
- `DecodeGraphDefinition::capture` / `DecodeGraphExecutable::instantiate|launch` 在 src/core/decode_graph.cpp:59-146，
  平台无关，TP-2 可直接用（每卡一份 definition/executable）。
- AR 是 cudaHostAllocMapped|Portable 的 host pinned + 内核自旋（src/core/tp/device_pair.cu:90-143），
  两卡各自 capture 后交替 launch 即可，不必改 AR 协议。
- Windows 无 nsys/ncu（C:\Program Files\NVIDIA Corporation 下只有 App/FrameView），性能归因**只能用引擎内 CUDA event**。

### 执行顺序（每步独立可测、可回退）

1. **测量**：TP-2 解码轮加 env-gated 相位计时（host 时钟 + CUDA event），确认 8.0 ms 的 MTP 链/accept/fold 归属，
   并复现基线 56.7 tok/s。
2. **④**：draft 链去 host 化——drafts 留设备、window token 直接给设备版前向、去掉冗余 D2H/H2D 与逐 draft 同步。
3. **①**：把整轮 capture 成两张图（每卡一张），pinned ingress/egress，按 mtp_graph_profiles 桶捕获；每轮只填 ingress + launch。
4. **③**：prefill 的 AR 与 MMA 重叠（子块流水；需两套激活区与两份 GDN 状态快照，见 PLAN:663 的显存账）。
5. 验证与文档：贪婪 A/B（与改造前逐 token 一致）+ bench_serve.ps1 实测 + 更新 docs/tp2-dual-5060ti.md / worklog。

### 进度

- [x] 计划落盘（本节）。
- [ ] 步骤 1 测量。

#### Round 11 侦察结论（子代理只读调研，勿重复）

- **层循环**：\`run_layers_tp2\` 是 text.h:411-458 的模板（不在 text.cpp）。每层顺序固定：
  \`tp_mixer_layer\`（双卡，行并行输出投影写 partial delta）→ **AR#1**（text.h:436）→ \`residual_add\`×2 →
  \`tp_mlp_delta\` → **AR#2**（text.h:449）→ \`residual_add\`×2。层内**无任何 host 同步/回读**（\`src/ops\` 全目录
  grep \`cudaStreamSynchronize|DeviceToHost\` 零匹配）。
- **allreduce**（src/core/tp/device_pair.cu:289-395）：签名 \`allreduce(void* a, void* b, bytes, stream_a, stream_b)\`；
  in-kernel 路径 host 侧只连续发两个 kernel、无同步；epoch \`token = ++ar_call_\`（单调不减，device_pair.h:64），
  按 \`token&1\` 双缓冲（24 MiB/卡），block 间有 \`order_mine[b-1]==token\` 写序链，对端自旋 \`arrival_other[b]!=token\`。
  **graph 危害**：token 是**内核参数**，capture 会把它固化，replay 复用旧 token 会让对端自旋提前通过（读到上一轮的旧数据）
  → 上 graph 前必须把 token 改成**设备端计数器**（\`increment_i32_scalar\` 或内核内自增）。
- **positions**：\`fill_i32_positions\` 只接受 host base（src/ops/kernel/position.cuh:7）；\`offset_i32_positions\`
  支持 **[1] 设备 delta 广播到 [T]**（kernel: \`dst[i]=src[i]+delta[0]\`）→ 设备化路径 = 静态 arange + 设备 base。
- **fold**：\`GdnReplayFoldPlan::execute(std::span<const GdnReplayFoldRow>, stream)\` 在 **host 校验并打包**，
  \`commit_columns\` 进 \`__grid_constant__\` 内核参数（replay.cpp:266-325、recurrent.cu:104-128），
  头文件自述「改变 host row 描述符必须重新 capture」→ **fold 不能带变异 commit 数进 graph**，本轮保持 eager。
- **state 账**：state pool \`slot_count=1\`，live backing = **73.4 MiB**；PLAN 里 293.6 MiB = live + 3 份快照
  （2 份 prefix-reuse + 1 份 round scratch）。每轮 snapshot+restore = 2×73.4 MiB D2D/卡（不是 293.6×2）。
- **accept**：\`speculative_accept_greedy_drafts\` 已输出设备端 \`accepted\`，但**无人读取**；提交数始终来自 host 输出策略。
- **forward_tp2_prefill 的 host 依赖**（text.cpp:1766-2021）：\`copy_i32(ids)\` H2D（:1826）、
  \`fill_i32_positions(b.positions, first_position)\`（:1827）、\`envelope{visible_end,visible_end}\`（:1798/:1823，
  host 端决定 attention 路由/chunk/split）；\`Phase::Verify\` 分支（:1931-1965）MTP 轮**不走**（走 Prefill）。
- \`set_i32_scalar\` 只支持 host 值（scalar.cu:9）；\`assign_i32_scalar\` 是 D2D memcpy；\`increment_i32_scalar\` 设备端。
- **verify 窗口的显存**：K=2 时 T=3，激活峰值只有几 MB（最大项 \`logits_columns\` [248320,3] BF16 = 1.5 MB），
  所以 graph 私有 workspace **不需要** 192 MiB（PLAN:662 的最坏估计过于保守）。
- **③ 的子块流水前提**：所有 op 形状泛化（T 任意），但 (a) token 范围由 forward_tp2_prefill 每 chunk 一次性绑定
  （positions/envelope/kv_table_rows），(b) GDN conv/递推状态按 source/destination slot 顺序推进，
  (c) full-attention 子块 2 的查询要能看到子块 1 已 append 的 K/V ⇒ 每层依赖链 L,A→L,B→L+1 必须保持，
  能做的是**按 slot 的 A→B 流水**（AR 走链路时另一子块走 SM）。
- **踩坑（已修）**：给解码轮加 CUDA event 计时时，事件在**创建时的当前设备**上绑定；若与
  \`shard_a_.device.stream\` 不是同一设备，\`cudaEventRecord\` 会返回 \`cudaErrorInvalidResourceHandle\`，
  而该错误会被**锁存**到下一次不相关的 \`cudaGetLastError()\`（表现为 \`scalar.cu:10\` 的 set_i32_scalar 崩溃）。
  修法：在 shard A 设备绑定后再创建事件，且计时调用失败即丢弃错误并自我禁用。





## Round 11 进度：① 已完成（已实测），并修掉一处真实跨卡竞态

### 步骤 1 测量（已完成）

在 tp2_generation_core.cpp 加 env-gated 相位计时（NINFER_TP2_TIMING=1，7 个 CUDA event + host 时钟）。
基线（graph 关闭 = eager 路径，bench_serve.ps1 同一配方）：

    decode rounds=52 committed=127 avg_round=38.90ms mtp=3.70 verify=34.23 accept=0.08 copy=0.09 sync_wait=0.01 fold=0.00

即 38.9 ms/轮 = MTP 链 3.70 + verify 34.23 + accept 0.08 + copy 0.09 + host 策略/收尾（差额约 0.8 ms）。
verify 占 88%，其发射/调度开销实测比 PLAN:642 的 2.91 ms 更大；MTP 链 3.70 ms 是第二大项。

### 步骤 3 ① exact-batch CUDA Graph（已完成）

实现位置：src/core/decode_graph.{h,cpp}、src/core/arena.{h,cu}、src/models/qwen3_5/execution/text.{h,cpp}、
src/runtime/engine/tp2_generation_core.{h,cpp}。

- DecodeGraphDefinition::capture_group(defs, streams, body)：多流同时 capture，每张图只记录本流上的 kernel；
  跨卡仍只靠 in-kernel AR 自旋，host 不参与。
- TextContext::forward_tp2_window(...)：prefill 前向的 capture 安全变体。每轮输入（ids/positions）走
  可移植 pinned host 缓冲（capture 成 memcpy node，每次 replay 重读），attention envelope 变成捕获期常量。
- 桶来自 mtp_graph_profiles(max_context, mtp_drafts_)（K=2、131072 时 8 个桶，按可见上界分派）。
- arena 布局必须逐轮可复现：新增 DeviceArena::rewind(watermark) 与 position_arena(arena, floor, target)。
  每轮先丢弃 proposal 链的 arena 占用、回到固定 watermark，图里固化的地址才成立；replay 时按捕获 watermark
  对齐（更低 rewind，更高 alloc_bytes，低于本请求 watermark 直接报错）。已捕获桶的 watermark 若低于当前请求
  则视为未捕获并重捕（warm-up 的 watermark 更小，因此首个真实请求会重捕一次）。
- NINFER_TP2_VERIFY_GRAPH=0 关闭（A/B 用），默认开启；eager 参考路径与 capture 路径共用同一个 window 函数
  与同一个桶 envelope，所以 A/B 只比较发射方式。

实测（同一 bench 配方，graph 开/关）：

| 指标 | ① 关（eager） | ① 开（graph） | 差 |
|---|---|---|---|
| avg_round | 39.6 ms | 35.3 ms | **−4.3 ms** |
| verify | 35.0 ms | 30.6 ms | **−4.4 ms** |
| decode_at_2048（两侧都是 64 token） | 65.6 tok/s | 76.5 tok/s | +16.6% |
| prefill_512 | 1419 tok/s | 1419 tok/s | 持平 |
| sync_wait | 0.01 ms | 0.01 ms | — |

（tok/s 还受本轮文本决定的 draft 接受率影响，所以以轮时间与等输出的 2048 对照为准。）
verify 少的 4.4 ms 全是发射/调度开销（两路径的 kernel 参数与输入完全相同）。
数值正确性：5 个 prompt × 最多 256 token 的贪婪 A/B，eager 与 graph 逐字节一致（含 reasoning_content）；
graph 路径对是否插桩不敏感、可复现。

### 顺带修掉一处真实跨卡竞态（改动前就存在，非 ① 引入）

排查中发现：未插桩的 eager 路径在 5 个 prompt 里有 3 个与 graph 不一致，而任何额外同步/回读
（即使放在轮末）都会让它变成与 graph 一致 —— 即 eager 路径对时序敏感。根因是解码轮结束时只同步了 shard A
（cudaStreamSynchronize(shard_a_.device.stream)），随后 host 立刻做 fold / state restore / 下一轮 window 与
arena 复用，而 shard B 的 verify 尾部可能仍在跑；两卡之间唯一的顺序保证是 AR 自旋。
修法：轮末同时同步 A、B（tp2_generation_core.cpp）；实测 sync_wait 仍为 0.01 ms（B 只落后极少）。
修后 eager 与 graph 在全部 5 个 prompt 上逐字节一致。

### 顺带修掉的第二处缺陷：in-kernel allreduce 的 token 归属（真实崩溃）

带上 ① 的全量测试里 ninfer_qwen3_5_tp2_forward_test 报 cudaErrorIllegalAddress（第一层内），而 HEAD 的同一
二进制通过。二分定位：把 allreduce 强制走 host-staging 路径即通过 ⇒ 本轮改成「设备端 token」的 in-kernel AR 引入。
根因：token 计数器原先分配在各自设备的 device 堆上，但「哪个 shard 驱动 pair」由调用方决定（该测试会让两个 shard
各驱动一次，于是 stream_a 属于 device 1），内核在 device 1 的 context 里解引用了 device 0 的裸指针。
修法：token 改为 mapped pinned host memory（两个设备都可见），kernel 侧每 block 读一次并用 shared memory 广播
（若每个线程都直接读系统内存，单次 allreduce 多 ~4 us、整轮多 ~0.5 ms）。修后 12 个相关测试全绿
（含 tp2_load / tp2_forward），eager 与 graph 仍逐字节一致。

### 未决（非本次范围）

MTP 输出质量：与 plain decode（-Plain，单 token TP-2 路径）贪婪对照，5 个 prompt 中只有 1 个逐字节一致，
其余在近似并列处翻转。这是 docs/tp2-dual-5060ti.md §MTP 已记录的**接受行为**：verify 窗口与单 token 解码是
不同的执行形状（分块注意力/卷积路由），逐列 logits 必然有差异，llama.cpp 的 Qwen3.5 MTP 同样如此。
**注意**：不要再用「Phase::Prefill 导致近似」解释它 —— 同一文档已记录那是被否定的旧诊断（Phase 在
head-split 分片上无任何语义作用）。这不是 ① 的回归。

### 下一步

- [x] 步骤 1 测量。
- [x] 步骤 3 ①（exact-batch graph）+ 轮末双卡同步。
- [x] 步骤 2 ④ 复核：**该项按原描述已经不需要做**。PLAN 的 ④ 依据（"每 draft 一次 cudaStreamSynchronize"，
  指旧版 tp2_generation_core.cpp:543）在当前代码里已不成立：mtp_propose_window 整个窗口只有 **1 次 D2H + 1 次
  sync**，每 draft 只剩一次 D2D 拷贝 + 一次 increment 内核；mtp_forward_batch / mtp_forward_ar_step / mtp_forward_core
  内无任何 host 同步或 D2H（grep 实证）。实测 mtp=3.70~3.75 ms 是**真实 GPU 工作**（两次 T=1 的 MTP 层前向，
  权重流受限），且与 verify 串行。残余机会只有「fold 与 MTP 链重叠」≤0.8 ms（≈2%，fold 走 replay.cpp，不含
  allreduce，理论上可另开流），但要动 AR token 协议/流序，收益风险比不划算 ⇒ 判定 ④ 已完成，不再改。
- [x] 步骤 4 ③：prefill AR 与 MMA 重叠——**已实现、已实测、已否决**（见下）。
- [x] 步骤 5：复测与文档更新（③ 的结论写入本文件与 worklog）。

## Round 12：③ prefill AR∥MMA 子块流水（已实现，实测无收益，已回退）

### 实现（完整可用，最终未保留）

- `DeviceContext` 增加第二条流 `collective_stream`（与 `stream`/`transfer_stream` 同生命周期）。
- `forward_tp2_prefill` 在 `NINFER_TP2_PREFILL_OVERLAP` 打开且 phase=Prefill、`pair.in_kernel_allreduce()`、
  驱动卡 == `pair.a()`、tokens ≥ 256 时走两子块流水：A=[0,ta)、B=[ta,T)；每层顺序
  A.mixer → arm(0) → B.mixer → arm(1) → AR(A.mix)/AR(B.mix) → A.res+A.mlp → B.res+B.mlp → A.res2 → B.res2，
  4 个 AR 各配一对 event（`produced` 在 compute 流、`done` 在 collective 流），
  子块绑定靠嵌套 `ScopedPositions`/`ScopedEnvelope`（positions/rope 为外层 tensor 的 slice，
  envelope 分别为 {first+ta,first+ta} 与 {visible_end,visible_end}）。
- **关键约束（踩坑）**：切分点必须落在激活调度块边界上。初版 ta=tokens/2（300→150+150）会让
  `tp2_forward_test` 的 chunk-split 不变量失败（max_logit_diff=1.14/1.60）；改成 64 对齐
  （ta=max(64,(tokens/2)&~63)，300→128+172）后**逐位一致**：
  `chunking 300 -> 128+172: max_logit_diff=0`、`64+236: max_logit_diff=0`、
  `T=1024 one chunk vs 4x256: max_logit_diff=0`，且 300/1024 的 top5 与**未切分**的基线逐位相同 ⇒
  层间 A/B 交错与双流 AR 在数值上完全等价（这是 ③ 唯一确定的技术收益）。

### 实测（同一二进制、无缓存前缀、单请求 2072 token prompt，TTFT 端到端）

| 配置 | run1 | run2 | run3 |
|---|---|---|---|
| `NINFER_TP2_PREFILL_OVERLAP=0` | 1249 ms | 1189 ms | — |
| `NINFER_TP2_PREFILL_OVERLAP=1` | 1258 ms | 1314 ms | 1513 ms（冷启） |

稳定态 ≈ 1219 ms vs ≈ 1286 ms ⇒ 重叠**慢约 5%**。

### 结论（为什么不做）

1. 原成本模型（PLAN:651，AR 占 prefill ~48%）把 AR 当作**串行**开销，实测不成立：in-kernel AR 的两个
   内核是对称的，两卡各自 drifted 推进时，本卡自旋等对端的时间本来就被**对端仍在跑的 compute 掩盖**，
   可重叠的余量远小于模型估计。
2. 子块切分的代价是真实的：T 减半后 GEMM tile 变窄、**权重每层被流读两次**（prefill 是权重流敏感的形状），
   A/B 实测这两项加起来盖过了 AR 的收益。
3. 因此 ③ 在 TP-2 上不是有效优化：**保留最终方案为不带子块流水的单块 prefill**，`collective_stream`
   与整套流水代码已回退（工作树干净回到 8f6a812e）。
4. 若将来出现真正 compute-bound 的 prefill 形状（例如更宽 chunk、更快的权重路径），可重新评估；
   届时本节的 64 对齐约束与 event 拓扑可直接复用。

---

### Round 13 — 清理 decode 循环的 [mtp] 刷屏打印（已完成）

`tp2_generation_core.cpp` 的 decode 循环里有一条裸 `std::fprintf(stderr, "[mtp] round pos=…")`：
不经 logger（所以 `--log-level error` 关不掉）、无 env 门控，每轮一行（≈26 行/秒）。
实测写入成本 `2>NUL` 1.8 µs/行、文件 1.5 µs/行、**真实控制台 20–25 µs/行**
⇒ 单轮 20 µs / 35.3 ms = **0.057%**，吞吐影响可忽略；真正的风险是 cmd 的 QuickEdit 选中会让控制台
停止消费、阻塞生成线程（不是慢，是卡）。

已删除该打印与仅供其使用的 `mtp_draft_checked_`/`mtp_draft_hit_` 两个计数器；
接受率改由 `NINFER_TP2_TIMING=1` 的 `[tp2-time] decode rounds=R committed=C` 推导：
接受 draft 数 = C−R，接受率 = (C−R)/(R·K)。`docs/tp2-dual-5060ti.md` 已同步。

---

### Round 14 — 前缀复用“续写边界”：实测证伪并撤回（已完成）

实现了方案 2（续写边界：`cached_tokens_` 延伸到上一轮生成序列末尾 + 新状态槽），用真实 DSH 流量（36 请求）验证：
`slot=0` 35 次、`slot=1`(rewind) 1 次、**续写槽 0 次**；33/36 条 `shared == prefill_end` 精确相等 —— 客户端下一条 prompt
与上一条逐 token 相同到末尾，然后在第一个生成 token 处分叉（不回传生成流，而是重新渲染 assistant 回合）。
结论：该方案对本客户端零收益，**已撤回**（引擎两个文件回到 `1cdfab97`，`state` 回到 293.6 MiB/卡）；
保留 `NINFER_TP2_REUSE_TRACE=1`（每请求打印 `prompt/cached/shared/prefill_end/rewind -> reuse/slot`），
它是判断“客户端是否原样回传历史”的唯一手段。

**下一步候选（用户暂缓）**：① system+tools 前缀末尾加锚点，修新会话/压缩后第一轮的 46.9 s（可省约 11 s）；
② prefill 吞吐（suffix 0.95–1.13k tok/s、冷启 1.35–1.38k）；③ 每请求 ~0.3 s 固定开销。

---

### Round 15 — 工具调用的约束解码（路线 A 已落地）

**现象**：DSH 接 NInfer 时偶尔把工具调用当正文吐出（`<tool_call> <function=todo_write> …`），前端提示
“工具调用方式不对”；同一套 DSH 接 llama.cpp 从未出现。

**已定位（诊断细节见 worklog Round 15）**：不是量化（官方 `nvfp4` 与转换 `w4a4_w8a8` 各 75 条工具请求，标记类失败
27/75 vs 19/75，z≈1.4 不显著）、不是 chat_template、不是解析器、不是客户端配置。根因是**引擎策略差异**：llama.cpp 对
声明工具做 lazy GBNF 约束解码（`common/chat.cpp:1286` 为每个工具生成 `<function=NAME>` 规则，`server-common.cpp:1332`
注入生成参数），未声明的名字在采样层不可达；NInfer 只在解析层用 `enforce_declared_names` 拒收后原样返回文本。

**决策（路线 A/B）**：llama.cpp 的 grammar 按**文本**校验（`token_to_piece` + `llama_partial_utf8` + GBNF 栈），对
分词不敏感；在 token 序列上建前缀树则分词敏感、易掩错。故分两条路线：

- **路线 A（本轮落地）**：状态机累积已生成文本，每步用 tokenizer 把候选 token 解码成片段（处理跨 token 半个
  UTF-8），校验「累积文本 + 片段」是否为已声明工具名/参数名的合法前缀，否则置 `-INF`。对分词不敏感，正确性风险
  对齐 llama.cpp，代码量远小于完整 GBNF。
- **路线 B（Phase 2 选项，未做）**：移植 `llama-grammar.cpp` 的栈式 parser + 为 Qwen 生成 grammar，天然支持完整参数
  schema 校验。仅当 A 在真实流量仍偶发边界问题、或要做完整 schema 校验时升级。

**状态**：路线 A 已实现并通过单测。实现落在 `include/ninfer/ops/token_mask.h`（+ `src/ops/{kernel,launcher,wrapper}`）、
`src/models/qwen3_5/frontend/tool_call_constraint.{h,cpp}`、`frontend.{h,cpp}`、`output_session.{h,cpp}`、
`tool_call_parser.h`、`src/runtime/engine/tp2_generation_core.cpp`；`ninfer_token_mask_test` /
`ninfer_tool_call_constraint_test` / `ninfer_tool_call_parser_test` 通过，真实服务 `tools 1` 请求端到端正常。
要点：掩码只在工具调用结构区生效（`ParameterValue` 与自由文本不掩码）、只在非思考阶段激活、在 CUDA Graph **之外**
应用；MTP 按 verify 列取各自语法位置，draft 不掩码（非法 draft 与掩码后的 argmax 不一致而被拒）。
实现细节、边界处理与验证见 worklog Round 15b。

**不做**：「未声明也当 tool_call 返回」等于放行模型绕过 `run_code` 直接触发 `pwsh`/`write`，破坏 PTC 契约。

**同轮修掉的两处既有缺陷**（细节见 worklog Round 15b）：

- TP-2 把**物理行数**当有效域传给 `ops::sample` / `ops::argmax` / `speculative_accept_greedy_drafts`，可以采到
  tokenizer 未定义的打包行（248320 行 vs 公开词表）；现全部改传 `public_token_count`，正常步逐位不变。
- `make_sampling_config` 的 `token_counts` 恒为 null ⇒ presence/frequency penalty 对跨轮重复完全无效；现按请求建
  计数数组。A/B 实测（同 seed，修复前/后两份二进制各起一次服务）：修复前三种 penalty 配置输出**逐字节相同**，
  修复后 `frequency=2.0` 把重复从「最长连续 15」压到「2」，且无 penalty 路径逐字节不变。

**剩余（可选）**：

1. 真实模型端到端复测：对照 Round 15 的 75 条探针，确认未声明名字不再出现。
2. 轻量诊断：WARN 里打印模型实际写出的工具名，线上即可看出是 `tools.read` 还是 `todo_write`。
3. 路线 B（完整 GBNF / 参数 schema 校验）。
4. 可选加固：TP-2 补一个 `validate_licensed_tokens` 同款守卫。

---

### Round 16 — TP-2 host 侧状态检查点 + 短 prompt 按 LCP 截断复用

**状态**：已实现并完成端到端验证（`src/runtime/engine/tp2_generation_core.{h,cpp}`、`src/runtime/engine/model_instance.cpp`、
`tests/test_engine_options.cpp`）；`ninfer_engine` 编译通过、新增的选项契约单测通过；实测 `src=host` 命中（req#5 10.2 s → 1.3 s、req#11 37.6 s → 28.6 s，详见 worklog Round 16.3）。

**问题（Round 15 后续实测）**：客户端在 turn 边界重渲染历史，新 prompt 比上一条**短**（实测 −219 / −1,233 /
−15,888 token），分叉点落在历史中段。TP-2 核心只保留 2 个**显存**边界（上一条 prompt 末尾 + 一个 rewind 点），
判据要求 `boundary <= shared_prefix`，两个边界都落在分叉点之后 ⇒ `reuse=0` ⇒ **整条 prompt 全量重算**：
req#43 57,378 token / TTFT 42.0 s，req#51 74,536 token / TTFT 57.6 s（NVML 采样证实这 59 s 两卡都是 98–100%，
与 `done` 行的 `prefill 1.30k tok/s (74,536 tok)` 一致）。

**根因不是 KV**：walk 从边界开始，边界之前的 KV 页本来就不重算（`tp2_generation_core.cpp:1042-1045` 注释）。
缺的是 **GDN（线性注意力）状态**：它是循环累积量，既不能从 KV 反推、也不能平移；全注意力层的 K/V 可以随便
截断复用，线性层不行。llama.cpp 正是因此才额外做 context checkpoints（PR #15293）：它的
`n_past = slot.prompt.tokens.get_common_prefix(input_tokens)`（`tools/server/server-context.cpp:3103`）只解决
注意力层，hybrid 模型靠检查点兜底 —— `n_ctx_checkpoints = 32`、`checkpoint_min_step = 8192`（`common/common.h:611-615`），
创建在 `llama_decode()` 之前（`server-context.cpp:3508-3518`），恢复时从新到旧找 `<= LCP` 的检查点，找不到才
`do_reset` 全量重算（`:3236-3249`）。

**方案（显存 0 增量）**：检查点放 **pinned host 内存**，引擎已有这套语义：`--host-state-slots`（文档表里叫
「完整 Host StateImages」，`src/serve/serve_options.cpp:220-223`）、`HostStatePool` +
`StateImageStore(device_pool, host_pool, capacity)`（`program_impl.cpp:125-144`）、启动阶段 `HostStatePin`；
TP-2 核心本来就在用 `PinnedHostBuffer`（`tp2_generation_core.cpp:279`）。

1. 一份状态 = **73.4 MiB/卡**（`state_bytes`；台账 `state 293.6 = (2+2)×73.4`）。16 槽 = 2.3 GB host、
   32 槽 = 4.7 GB host、**显存 +0**；恢复一次 = 每卡 73.4 MiB H2D（pinned，≈5–20 ms），相对几十秒可忽略。
2. prefill 过程中按 `kReuseCheckpointStride = 8192`（对齐 llama.cpp 的 `checkpoint_min_step`）把 `state_backing`
   异步拷进环形 host 池，并记下 frontier 位置。
3. 复用扫描从「2 个显存边界」扩成「2 个显存边界 + host 检查点」，取最深的 `<= shared_prefix`
   （`tp2_generation_core.cpp:891-897` 就是扩展点）；`begin_gdn_state`（`:976-987`）增加 host→device 分支，
   KV 路径完全不动。
4. **正确性规则（谱系有效性）**：检查点带 `valid`；每次选择前把 `position > shared_prefix` 的置为失效（越界者
   永不可能再被用上），prefill 成功发布时用 prefill id 把本次写的槽标为有效。**不能**只认「上一次 prefill」——
   上一次若只走了很短的后缀（完全复用），它自己不留检查点，那样最常见的小尾巴分叉会依旧 `reuse=0`（Round 16.2）。
   另外 prefill 中途取消时清 `cached_state_valid_` 并清空整环（取消走的 KV 与 `cached_prompt_tokens_` 不再一致）。
5. 一个 131k 上下文按 8k 步长最多 16 个检查点，所以 16 槽即可覆盖满上下文（超出部分环形淘汰）；
   200k 建议 32 槽。
6. **尾部窗口**：最后 8192 token 内每个 chunk 末尾额外存一份（对齐 llama.cpp 的 `near_prompt_end` 例外），
   覆盖「只重渲染尾巴」这一最常见情形（实测 req#12：只差 171 token 却全量重算 39.0 s → 命中尾部检查点后 ~0.4 s）。

**验收**：同一会话连续两轮，第二轮 prompt 比第一轮短且分叉在中段 ⇒ `[tp2-reuse]` 报 `reuse > 0` 且来自 host
检查点，`cache` 命中从 0% 升到 `shared` 量级，TTFT 从 ~57 s 降到 `(prompt − checkpoint)/1.3k + 固定开销`；
回归：现有 TP-2 用例 + 新增「短 prompt 命中 host 检查点」场景；`git diff --check` 干净。

---

## 13. 已完成：TP-2 多会话 KV 池（host KV 换入换出）

**目标**：多个聊天会话（A/B/C…）交替使用同一个 TP-2 引擎。任意时刻显存里只有一份激活会话的 KV；
最多 5 份非激活会话的 KV + 状态保存在 pinned host 内存，会话回来时整体召回（H2D），只 prefill 新后缀；
超出 5 份时淘汰最旧（LRU）。

**背景与现状（证据）**

- TP-2 核心（`src/runtime/engine/tp2_generation_core.{h,cpp}`）绕过单卡路线的整套 context cache
  （Program/ResourceManager/checkpoint/资源事务）；`model_instance.cpp` 的 `normalize_engine_options`
  在 `device_b >= 0` 时把 `context_cache` 重置为 disabled（只保留 `host_state_slots`）⇒
  `--max-private-continuations` / `--host-kv-mib` 当前在 TP-2 路线**静默无效**（不报错）。
- TP-2 核心自维护的状态：单行 paged KV（`kv_table_rows = 1`，池按 max_context 建，**上一次 prefill 的
  KV 常驻显存**，新请求直接回收页）、2 个 GDN 状态快照（显存，prefill 末尾 + 1 个 rewind）、
  host 检查点环（Round 16：只存 GDN 状态、单谱系、步长 8192）。
- Round 16 的环解决「同一会话内客户端重渲染较短历史」（GDN 状态召回 + KV 从检查点位置 interval 重算）；
  本计划解决「**跨会话** KV 保留」（KV + 状态整体召回、零重算）。两者正交，环保留。
- 量化（Qwen3.8-27B：16 full-attention 层 × 4 KV heads × head_dim 256，另有 48 层 GDN）：
  文本 KV 16.125 KiB/token/卡（FP8，TP-2 显存台账）；GDN 状态 73.4 MiB/卡/会话；MTP KV 2 KiB/token（shard 0）。
  换入/换出耗时（PCIe 4.0，~25 GB/s 实测口径）：32K 会话 ≈ 40 ms；128K ≈ 160 ms；
  相对全量 prefill（200k ≈ 90–160 s）可忽略。

**关键设计决策**

1. **不做抢占（v1）**：换入换出只发生在**请求边界**——激活会话的响应已完整生成并流式发完，
   新会话请求到达时才换。A 的客户端零停顿；B 的 TTFT 只多一次换入。
   「B 打断 A 生成到一半」需要暂停 lockstep decode 循环、保存部分输出、之后恢复 A，列为 v2。
2. **显存零新增**：device KV 池与 GDN 状态池不变（仍只服务一个激活会话）。
3. **精确 token 匹配**：incoming prompt 分词后扫目录（≤6 条）取最长精确前缀匹配；
   渲染不一致 ⇒ 全量 prefill（优雅降级，与今天一致）。
4. **保留 Round 16 环**：它服务同一会话的 in-prefill 检查点；内存紧张时可把 32 槽调小
   （8 槽 ≈ 1.2 GiB，步长变粗、重算变多）。
5. **两阶段换入换出 + 校验**：D2H（双卡）→ 校验 → 释放显存页 → H2D（双卡）→ 校验 → 映射。
   任一阶段失败时系统处于一致状态（各会话在显存或内存之一），报错后下一请求可重试。
   参照单卡路线 ResourcePlan 的 commit/abort 语义，TP-2 版简化为两阶段 + 校验。

**工作项（按依赖顺序）**

| # | 工作项 | 位置 | 规模 |
|---|---|---|---|
| 1 | **Host KV arena**（每 shard 一份）：pinned host 存储，镜像 device KV 页几何（page-major `[X,P,H,N]`，P=64）；每会话一块 slab，按该会话实际 frontier 页数分配（不按 max_context 预分配满）。预算：5 会话 × 204800 token × 16.125 KiB/token/卡 ≈ 16 GiB/卡、共 32 GiB（即 `--host-kv-mib 32768`）；另 GDN 状态 5 × 73.4 MiB × 2 卡 ≈ 0.7 GiB | `tp2_generation_core.cpp` | ~1–1.5k LOC |
| 2 | **会话目录**：替换 `cached_prompt_tokens_`/`cached_boundaries_`/`cached_state_valid_` 单谱系三件套。≤6 条（1 激活 + 5 非激活），每条：token 历史（204800 × 4 B ≈ 0.8 MiB，可忽略）、frontier、驻留标志（device/host）、host KV slab、host GDN 状态、MTP KV（shard 0）、LRU 时钟 | 同上 | ~500 LOC |
| 3 | **会话匹配**：扩展现有 reuse scan（`tp2_generation_core.cpp:953` 起）——对单个 `cached_prompt_tokens_` 的线性 token 比较泛化为扫目录取最长精确前缀 | 同上 | ~200 LOC |
| 4 | **换出事务（D2H）**：请求边界触发（新会话准入时）。双卡 KV 页 D2H → 校验 → GDN 状态 D2H（复用现有 `PinnedHostBuffer` 机制）→ MTP KV D2H → 释放显存页 → 标记 host-resident。先拷后放，峰值显存不增加 | 同上 | ~500 LOC |
| 5 | **换入事务（H2D）**：反向。映射显存页 → KV H2D → GDN 状态 H2D → MTP KV H2D → 校验 → 标记 device-resident | 同上 | ~500 LOC |
| 6 | **LRU 淘汰**：目录满（5 非激活）且来新会话时，释放最旧会话的 host KV + 状态 + 历史 | 同上 | ~200 LOC |
| 7 | **prefill 集成**：换入后从会话 frontier 起只 prefill 新后缀。现有 prefill 已支持从 reuse 点开始（`reuse` token），把「reuse 点 = 会话 frontier + 已映射的恢复页」接上 | 同上 | ~300 LOC |
| 8 | **选项打通**：`model_instance.cpp` 把 `host_kv_capacity_bytes` / `max_private_continuations` 透传给 TP2GenerationCore（或加 TP-2 专用选项）；启动校验（arena ≥ 会话数 × 容量 × 页字节）+ `[mem]` 台账加一行 | `model_instance.cpp`、`serve_options` | ~300 LOC |
| 9 | **测试**：新增 `tests/models/qwen3_5/test_tp2_sessions.cpp`（参照 `test_tp2_forward.cpp` 的小模型 fixture）：A prefill→完成 → B prefill→完成 → **A 续轮**（断言命中召回、非全量 prefill；输出与从零 prefill 的 oracle **逐 bit 一致**——KV 字节相同则 attention 相同）→ C/D/E/F（断言 A 被 LRU 淘汰）→ A 再来（断言全量 prefill） | `tests/` | ~800–1k LOC |
| 10 | **文档**：`docs/tp2-dual-5060ti.md`（产品语义 + 推荐配置）+ 启动脚本注释 | `docs/` | — |

**验收标准**

- 功能：多会话交替场景下 `[tp2-session]` 诊断报命中/未命中/淘汰；命中时 TTFT ≈ 换入耗时 + 新后缀
  prefill（32K 会话 ≈ 40 ms + 后缀），而非全量 prefill。
- 数值：召回会话的输出与从零 prefill 的 oracle 逐 bit 一致（KV 字节相同 ⇒ attention 相同）。
- 回归：现有 TP-2 用例全过（`qwen3_5_tp2_forward --artifact` 等）；`git diff --check` 干净。
- 资源：显存零新增（nvidia-smi 台账不变）；pinned host ≈ 32 GiB（KV）+ 0.7 GiB（状态）+ 现有环
  4.7 GiB。**启动前确认物理内存 ≥ 48 GiB**（本机 WMI 读不出内存总量，用任务管理器核对）；
  不足则 `--host-kv-mib` 降档（204,800 token 口径下 16384 = 2 个满上下文会话）。

**风险与注意**

- 32 GiB pinned 内存：WDDM/Windows 的 `cudaHostAlloc` 有实际上限；现有环已 pin 4.7 GiB 且工作正常，
  32 GiB 需实测确认。
- `max_pending_requests = 1`（TP-2 强制）：同一时刻只有一个等待请求，聊天场景无影响。
- vision workspace 是 per-request 临时 arena，不属于会话状态，不参与换入换出。
- 会话 token 历史 ≤ 0.8 MiB/会话，可忽略。

**分期**

- **v1**：上表 1–10 全部（请求边界换入换出 + 5 会话 LRU + 精确匹配）。
- **v2（可选）**：生成中抢占（暂停/恢复 lockstep decode、部分输出续流）。

**工作量估计**：~4–5k LOC C++ + 测试，数周量级。物理传输可参照单卡路线
`src/models/qwen3_5/program/transactions/materialization.cpp` 的 D2H/H2D 事务，但 TP-2 核心是独立
代码路径，属移植而非启用。

**进度（Round 55，已完成并实测通过）**

已落地（工作项 1–10 的代码部分）：

- **主机 KV arena**：复用既有 `HostKVArena`（`src/core/host_kv_arena.h`）与
  `DeviceKVPagePool::copy_to_host/copy_from_host`，TP-2 不自己写 D2H 分页循环。每 shard 一个
  arena，容量 = `--host-kv-mib / 2`，**首次换出时才构造**（单会话负载永不 pin）。shard 0 的
  arena 同时注册 text 与 MTP 两套 geometry。会话 slab 按该会话 frontier 页数分配
  （`pages_for_tokens(frontier)`），不按 max_context 预分配。device KV 池保持启动时全量物化
  且页表不变，换入换出只覆盖数据，显存零新增。
- **会话目录**：`TP2GenerationCore::SessionEntry` = tokens（prompt + 已提交生成）、frontier、
  device/host 驻留标志、每 shard host KV slab、每 shard GDN 状态镜像、MTP slab、LRU 时钟。
- **匹配**：`session_recall` 在 execute 开头、prefix scan 之前运行。对每条目录项算精确 LCP；
  host 条目要求 `LCP >= frontier`（整体召回语义）；候选按 frontier 深浅比较，只换比**驻留谱系
  可复用深度**更深的。驻留深度由 live frontier / device snapshot / host 检查点三者求出，与 scan
  同源；深度为 0 时驻留会话先换出再新建（否则目录项会被新会话内容静默顶替）。
- **换出/换入事务**：`session_store_active` / `session_restore`。两个 shard 各自在本地 stream 上
  D2H/H2D KV + GDN 状态 + MTP KV，随后同步两卡才改驻留标志。换出失败（预算不足）时丢弃该条目
  并降级为全量 prefill；换入失败由 `execute` 的 catch 使驻留条目失效。
- **LRU**：目录容量 = `--max-private-continuations`（TP-2 默认 6，含驻留），满时淘汰最旧非驻留项。
- **prefill 集成**：scan 新增 **LiveState** 候选（reuse = 会话 frontier，state 不拷贝）；
  `ReuseSource{None,DeviceSnapshot,HostCheckpoint,LiveState}` 取代 `reuse_from_host_`；
  `cached_prompt_tokens_` 现在含已提交生成 token（decode 结束 publish），因此同会话续轮也能从
  frontier 复用。注意 frontier = prompt + generated − 1：最后采样的 token 尚未 forward。
- **正确性边界**：① 换入/新建会话时使 host 检查点环失效（环的状态属于被置换的谱系，位置可能
  仍落在召回历史内）；② 任何从 walk 抛出的异常由 `execute` 的 catch 使驻留条目失效并清
  `cached_state_valid_`；③ prefill 取消同样（保留其他 host 条目）。
- **选项打通**：`normalize_engine_options` 的 TP-2 分支保留 `host_kv_capacity_bytes` 与
  `max_private_continuations`，disabled-cache 校验对该路线跳过；`--host-kv-mib 0` 关闭会话保留。
  启动台账新增 `[mem] host sessions capacity ...`；`NINFER_TP2_SESSION_TRACE=1` 打印
  recall / evict 决策。`tools/win_port/serve.ps1` 暴露 `-HostKvMiB`（默认 32768）与
  `-PrivateContinuations`（默认 6）。
- **测试**：`tests/models/qwen3_5/test_tp2_sessions.cpp`（已注册，`SKIP_RETURN_CODE 77`）。
  需 `NINFER_TEST_ARTIFACT` + 两张同名 sm_120a 卡。场景：oracle（关闭保留）全量 prefill 对齐 →
  A/B/A 召回并断言 `reused_prompt_tokens == frontier` → 输出与 oracle 逐 token 一致 → 填满目录
  触发 LRU 淘汰 A → A 回来断言 `reused_prompt_tokens == 0` 且输出仍一致。设备扫描只在 sm_120a
  设备里挑同名对（本机 CUDA 还会枚举到 Tesla T10），foreign 设备不再让测试永久 skip。

**实测（Round 55）**

- 端到端测试 `ninfer_qwen3_5_tp2_sessions_test`：artifact `D:\LLM\qwen3_8_27b_w4a4_w8a8.ninfer`，
  两张 5060 Ti（CUDA 0/2），`max_context 2048`、`prefill_chunk 256`、greedy，跑两条路线：
  plain + bf16 KV，以及 MTP + fp8 KV（serve 推荐配置）。退出码 0，耗时 200 s，输出
  `TP-2 session retention (plain|mtp) passed: recall reused 71 prompt tokens bit-identically;
  LRU eviction forced a full prefill`。MTP 轮 shard 0 的 arena 打印 `2 layouts`（shard 1 为 `1`），
  证明 MTP 自己的 KV slab 随会话换入换出。`NINFER_TP2_SESSION_TRACE=1` 的决策链与目录状态自洽：
  `new session entries=0`（A）→ `new session entries=1`（B，A 换出）→
  `recall frontier=71 resident_depth=0 tokens=72 entries=2`（A 召回）→ `new session entries=2`（C）→
  `new session entries=3` + `evict frontier=55 tokens=56`（淘汰 B）→ `new session entries=3` +
  `evict frontier=95 tokens=96`（淘汰 A）→ A 回来走 `new session`（全量 prefill）。首次换出时才出现
  `host KV arena shard N: 64.0 MiB pinned, 1 layouts`，证实单会话负载不 pin。
- pinned 内存台账：`ninfer-serve --devices 0,1 --max-context 204800 --kv-dtype fp8 --host-kv-mib 32768
  --max-private-continuations 6` 实测启动
  `[mem] host sessions capacity 6 | host KV 16384.0 MiB/shard | retention enabled` +
  `[tp2-session] host budget 16384.0 MiB/shard holds 5 of 5 full-context sessions (3225.0 MiB each)`。
  204,800 × 16.125 KiB = 3225.0 MiB/shard，与文档公式逐位吻合；`--host-kv-mib 32768` + 默认目录 6
  （1 驻留 + 5 host）即推荐档。全程显存占用 0（arena 未 pin），停机后 GPU 归零。
- 数值一致性由测试的逐 token 对齐承担：召回 walk 与 oracle 全量 prefill 的 greedy 输出完全相同，
  说明 KV 字节与 GDN 状态镜像都被正确换入；LRU 淘汰后重新 prefill 也回到同一序列。

**已知边界**

- 预算口径：shard 0 的 arena 同时承载 text 与 MTP slab，对称切分下 MTP 会吃掉 shard 0 的一部分
  预算；容量不足时按 LRU 淘汰，必要时调大 `--host-kv-mib`。
- 客户端重渲染导致 `LCP < frontier` 的会话不召回（优雅降级为全量 prefill）；同一谱系内的部分
  复用仍由 Round 16 环 + device snapshot 承担。
- 运行模型测试需要 ffmpeg 与 libcurl 的 `bin` 在 PATH（`ninfer-serve` 同样），否则进程以
  `STATUS_DLL_NOT_FOUND`(0xC0000135) 退出；`tools/win_port/serve.ps1` 已代为设置。
- **既有 artifact 差异（非 §13 引入）**：`test_tp2_forward` 在 `qwen3_8_27b_w4a4_w8a8.ninfer` 上
  chunk-split 不变性不成立（300 -> 256+44，top5 gap 1.0625、max logit diff 11.53），在
  `qwen3_8_27b_nvfp4.ninfer`（文档记录的 artifact）上逐位一致（gap 0、diff 0）。该测试只用底层
  shard fixture、不构造 `ninfer::Engine`，与 §13 无关。`ninfer_qwen3_5_tp2_load_test` 在新
  artifact 上通过。
- **既有缺陷（非 §13 引入，未修）**：TP-2 路线 `bf16` KV + `--spec mtp` 在第一次 prefill 就崩
  （`prompt.cu:63` `cudaErrorInvalidValue`）。在 `--max-context` 2048 与 8192、且启动台账为
  `host sessions capacity 0`（会话保留关闭）时均复现，故与会话池改动无关；`fp8` + MTP 正常，
  测试的 MTP 轮因此走 fp8。排查入口是 bf16 prompt-attention 的 launch 参数（`prompt.cu`）。

### 13.1 会话恢复的后续加固（Round 18，已实现并实测）

用户要求：修掉会话恢复隐患 1/2/5，并排查「系统提示词缓存完全不命中」。

- **隐患 1（prefill 取消丢进度）**：取消时把已完成 chunk 的前缀发布进目录（`t0 > 0` 时
  `snapshot_state` + `cached_boundaries_` + `session_publish(tokens[0..t0), t0)`），重发同一 prompt
  从该前缀继续而不是从 0 重来。测试新增 `run_cancelled`：在第 2 个 chunk 前取消，重发断言
  `reused_prompt_tokens == 256` 且与 oracle 逐 token 一致。
- **隐患 2（prompt-end 状态镜像让回收失败）**：两张可选状态镜像改为 best-effort，分配失败只告警，
  会话仍以 frontier 镜像可召回（`host_prompt_end` 仅在两张镜像都拿到时置位）。
- **隐患 5**：删掉只写不读的 `host_valid`。
- **系统提示词缓存**（本次新增机制，三项）：
  1. 网格检查点对齐：prefill 的 chunk 末尾截断到 `next_host_checkpoint`，网格检查点正好落在 stride
     倍数上（此前落在 stride 之后的第一个 chunk 末尾，共享前缀短于该覆盖范围时没有可复用边界）。
  2. 退化召回防护：host 扫描要求 `reach < prompt_tokens`（必须还剩 token 可前向）；否则恢复整段
     历史后还要从 0 重新 prefill，并顺带清掉设备 lineage 与检查点环。
  3. 共享前缀状态镜像（`host_shared_state` / `host_shared_end`）：会话换出时，若新 prompt 正好在
     walk 起点处分叉（`reuse == shared_prefix`），把该边界状态冻进被换出会话自己的镜像；若分叉点
     落在 walk 内部，则把 divergence anchor 的状态拷进该镜像。之后任何与它共享同一稳定块的会话都
     能召回这条 entry（KV 用该 entry 自己的 slab，状态用该镜像）。`[tp2-session]` 新增
     `shared_end=` 与 `via=shared`。
- 实测（plain 路线，exit 0）：`entry 0 tokens=672 frontier=671 prompt_end=664 shared_end=512
  shared=512 reach=512 via=shared` → `recall frontier=512` → `reuse=512 src=live`；先有一个小型
  无关请求接管设备池，仍能召回。

**未决问题（重要：未修，勿当成已交付的正确性）**：新增「召回 walk 对比 from-scratch oracle」断言后
发现——`shared_c` 场景（第 3 个共享同一系统提示词的会话）的召回 walk 与 from-scratch prefill 在第
7/8 个 greedy token 分叉（变成重复 token）；而同一 boundary 走「设备 KV 仍在」的复用路径（`src=host`
的 `shared_b`）与 from-scratch oracle 逐 token 一致。本轮用设备侧回读把范围钉死：

- KV 往返**逐字节保真**：恢复后把设备页 D2H 回读与 slab 比较，`diff=0 first=<end>`（8 页 / 16 MiB）。
- **整个设备页内容（含 host 传输不覆盖的尾部）在 store 与 recall 两次取样完全一致**：
  `store entry=1 frontier=671 dev=56163be625b88263` == `restore boundary=512 dev=56163be625b88263`。
  即召回点的设备 KV 就是 `shared_b` 场景里被 from-scratch oracle 验证过的同一份内容。
- 状态镜像三处（capture / restore / ring slot）校验和相同；`begin_gdn_state` 各分支只碰 `state_backing`。
- 两个引擎对同一 prompt 的 from-scratch 输出逐 token 一致（`shared_a_first` vs oracle 的 `shared_a`），
  所以这不是两台引擎的可复现性差异。

**根因已定位（Round 19，措辞改判）**：用 `NINFER_TP2_DEBUG_LOGITS` 打印采样前的 top-1/top-2 后确认，
分叉**在 prefill 的最后一列就已经存在**（不是 decode 逐轮累积的）：

```
shared_c  oracle   reuse=0   src=0  prefill pos=639  top1=54 4.750000 top2=1646 4.562500 gap=0.187500
shared_c  召回     reuse=512 src=3  prefill pos=639  top1=54 4.406250 top2=220  4.406250 gap=0.000000
shared_b  oracle   reuse=0   src=0  prefill pos=639  top1=1703 6.468750 top2=4263 5.468750
shared_b  召回     reuse=512 src=2  prefill pos=639  top1=1703 6.218750 top2=4263 5.250000
shared_a  oracle   reuse=0   src=0  prefill pos=639  top1=197 4.781250 top2=220 4.593750
shared_a  本引擎  reuse=0   src=0  prefill pos=639  top1=197 4.593750 top2=220 4.562500
```

关键在最后两行：**两个 Engine 实例对同一个 prompt 各做一次 from-scratch prefill，logits 也不同**
（4.781250 vs 4.593750），而它们的 8 个 greedy token 仍然相同（测试一直在断言这个）。`shared_b` 走的
是**不恢复任何 KV** 的既有 ring 路径（`src=2`），同样偏了 ~0.25。所以：

- 不是 host slab 召回丢数据（上一轮已逐字节证明：KV 往返 `diff=0`、整页设备内容 store/recall 一致）；
- 不是共享前缀镜像的错；
- 而是 **prefill 的数值在不同 Engine 实例之间不可复现**，量级 ~0.2（对 logit≈5 是 4%，远大于归约顺序
  能产生的误差，怀疑 workspace/arena 历史导致的未初始化读取）。`shared_c` 只是恰好 oracle 的 gap 只有
  0.1875，第一个被这个偏移翻转的 token 出现在第 7 个。

**根因（Round 19 收尾，已定论）**：把最后一块 prefill chunk 的宽度一起打出来就清楚了 —— 同一
prompt、同一 `reuse=0` 的 from-scratch walk，logits 随**最后一块 chunk 的宽度**漂移：

```
#3  oracle    shared_a  reuse=0 chunk=550+90 snap1=550  top1=197 4.781250
#19 本引擎    shared_a  reuse=0 chunk=542+98 snap1=542  top1=197 4.593750
#30 本引擎重复 shared_a  reuse=0 chunk=582+58 snap1=582  top1=197 5.156250
#9  oracle    shared_c  reuse=0 chunk=590+50 snap1=590  top1=54  4.750000
#23 召回      shared_c  reuse=512 chunk=512+128 snap1=0 top1=54  4.406250
```

`snap1 = prompt_tokens - rewind_near_`，而 `rewind_near_` 是按引擎历史（上一个请求的分歧位置）
推出来的，**于是同一次 prompt 在每次 walk 里的 chunk 边界都不同**，prefill 的末列 logits 因此差
0.3~0.56（对 logit≈5 是 6~11%，远超浮点重排能产生的量级）。也就是说：

- 召回/host slab/共享前缀镜像**全部无罪**（KV 与 state 均已逐字节证明一致）；
- 分歧的真正来源是「**后缀 chunk 宽度随历史变化 ⇒ 同一 prompt 的前缀计算不可复现**」；
- 之前「两侧 chunk 相同」的判断是错的：`shared_c` 的 oracle 是 `590+50`，召回是 `512+128`。

**修复方向**：把 chunk 计划与 ring 的 rewind 目标解耦 —— chunk 宽度只由 `(reuse, prompt_tokens)`
决定（固定 `prefill_chunk`、`min(prefill_chunk, remaining)`），ring 检查点只落在这些固定边界上。
「把快照放在贴近 frontier 的位置」本来是省一次 rewind 的小优化，代价却是让数学结果依赖历史，
而 `kReuseTailCheckpointCount` 的密集尾部窗口已经把 rewind 成本限制在几十 token 内，所以这个优化
不值得保留。改完后同一 prompt 的 chunk 计划可复现，召回 walk 才能重新做逐 token 断言。

诊断代码：`tp2_debug_logits`（`NINFER_TP2_DEBUG_LOGITS`）与 `tp2_debug_hash_*` /
`NINFER_TP2_DEBUG_KV`，默认关闭，修复完成后删除。

---

## Round 17：plain decode 上图 + AR 按张量大小切传输策略（进行中）

**目标（用户明确要求）**：把下面两项列入计划并实现，每项做完做 A/B 并汇报。

| # | 项 | 理论收益 | 现状与依据 |
|---|---|---|---|
| ① | plain（`--spec` 缺省）单 token decode 轮上 exact-batch CUDA Graph | 去掉每轮 ~1000 个 kernel 的发射/调度开销。Round 11 在 MTP verify（同规模 kernel 数）上实测 `verify` 少 4.4 ms/轮 | plain 路由（`tp2_generation_core.cpp:1463-1494`）完全 eager；`forward_tp2` 用 `set_i32_scalar` 传 host token/position，**不可捕获**。MTP verify 已有图（Round 11 ①） |
| ② | `DevicePair::allreduce` 按 payload 大小切传输策略 | decode 侧每轮 128 次 AR、每次 9–18 µs（1.2–2.3 ms，占轮 4–7%）；prefill 侧单次 AR 已经贴链路地板 | 现状只有一条 in-kernel mapped-pinned 自旋路径（`device_pair.cu:338-444`），仅切片数按大小自适应（`ar_slices`） |

### 关键前置结论（勿重复调研）

1. **envelope 桶在数值上是精确的**。`small_t_fp8.cuh:147-179`：`window = last_pos + 1` 由**设备端 positions**算出；`active_split_count = min(default_splits(window), split_count)`，split 的 key 区间也全部由 `window` 推导；host 侧 `logical_capacity = envelope.max_visible_keys` 只用于越界保护（`last_pos >= logical_capacity ⇒ write_neutral`）。`causal_small_t_launch_capacity`（`small_t.cu:78-94`，配合 `small_t.cuh:81-94` 的 tier 末端 `{128,160,512,4096,5000,8198,16390}`）取 envelope 区间内 `default_splits` 的上确界，因此只要 host 的 `splits ≥ default_splits(window)`，桶 envelope 与精确 envelope **逐位一致** ⇒ 单卡 `ordinary_graph_profiles` 的桶做法可以照搬到 TP-2。
2. **TP-2 分片上 `Phase` 无数值作用**：`text.cpp:1233`、`:1399` 的 Verify 分支都被 `shard_config_ == nullptr` 挡住；`attn_mix`/`gdn_mix` 里 `ph` 只用于 nvtx 名字。
3. **但 `active_sequence_batch_` 有数值作用**：`text.cpp:1154` 起，`active_sequence_batch_ != 0` 才走 batched attention 路由。`forward_tp2`（plain decode）设 batch=1/width=1，而 `forward_tp2_window`（MTP verify 用）**没有**设 batch/width ⇒ **plain decode 上图不能直接复用 `forward_tp2_window`**，必须复刻 `forward_tp2` 的绑定。
4. `model_instance.cpp:99` 的 `use_cuda_graph=false` 只影响单卡 Program 路线，与 TP-2 无关。
5. `--spec` 缺省即 `SpeculativeBackend::None`（`speculative_options.h:11-16` 只解析显式传值）；`tools/win_port/serve.ps1 -Plain` 就是 plain 路线。

### ① 实现设计

- 新增 `TextContext::forward_tp2_decode_window(...)`（`models/qwen3_5/execution/text.{h,cpp}`）：与 `forward_tp2` 逐行同构，只有可捕获化改动 —— `ids` 与 `cache/rope positions` 从 pinned host 用 `copy_i32` 的 memcpy node 读入（不再是 `set_i32_scalar` 的 host 值），envelope 变成捕获参数；其余绑定（batch=1、width=1、`kv_table_rows=0`、state slots=0、`Phase::Verify`、终范数、`project_head_tp2`）完全照抄。
- `TP2GenerationCore`：`VerifyGraph` 改名 `WindowGraph` 并复用同一套 `round_base/arena_begin/arena_bytes` 簿记；新增 `decode_graphs_`（桶来自 `qwen::detail::ordinary_graph_profiles(max_context)`，`visible_begin = min+1`、`visible_end = min(max_context, max+1)`，与单卡 ordinary decode 图同源）与 `decode_window_host_`（pinned `[token, position]`）。
- 只有 `pair_.in_kernel_allreduce()` 时上图（P2P / host-staging 会同步 host，不可捕获）。
- 环境开关 `NINFER_TP2_DECODE_GRAPH`：未设/其它 = 捕获图；`0` = eager + **桶** envelope（纯发射机制 A/B）；`exact` = eager + 精确 envelope（= HEAD 行为，envelope A/B）。
- 采样留在图外（sampling 参数逐请求变化，`logical_pos_a` 与 tool mask 都是 host 驱动），图只覆盖 forward；logits 缓冲由调用方在图前按固定顺序分配，复刻 verify 图的位置约定。
- 验收：三个配置的贪婪逐 token 一致性 + `bench_serve.ps1` 实测 tok/s + `NINFER_TP2_TIMING=1` 的 `avg_round`。

### ② 实现设计

- 大小键控的传输策略（`NINFER_TP2_AR_STRATEGY`：`size` 默认按大小 / `kernel` 强制现路径并作为 A/B 参照）：
  - **小载荷（≤ 64 KiB，decode 的 10 KiB、verify 的 40 KiB 属此档）**：in-kernel 路径 + 把 `bump_ar_token` 融进 AR kernel（单 block 时线程 0 自增、`__threadfence_system()` 发布、经 shared memory 广播），省掉每次 AR 的一次 launch；staging 换成独立的 2×64 KiB mapped-pinned 区，parity slot 不再与 prefill 槽别名。线程数保持 `kArThreads`。
  - **大载荷（> 64 KiB，prefill 的 10 MiB 属此档）**：保持切片路径不动（实测已贴链路地板）。
- **未实现的 copy-engine 大载荷臂**（D2H → `cudaEventRecord` → 对端 `cudaStreamWaitEvent` → H2D 到 scratch → 设备端 add）：放弃理由有二 —— 一是其跨卡定序必须依赖跨设备 `cudaStreamWaitEvent`（`device_pair.cu:352-357` 的「copy engine 更慢」测得的是旧 host-staging 路径，不足以据此否决，但带宽口径下 copy engine 与 SM 写用的是同一根 PCIe）；二是本项的真实杠杆不在传输方式，而在**每轮 128 次调用的次数**，prefill 对照臂全程未动即证明了这一点。
- 数值契约不变：BF16 逐元素 `__hadd`，加法顺序不变（实测 7/7 逐位一致）。

### 进度

- [x] 计划落盘（本节）
- [x] ① 实现（forward_tp2_decode_window + DecodeStepMode + decode_graphs_）
- [x] ① 构建 + 数值比对 + A/B：decode +12.6% @1K / +12.1% @32K（31.11→35.03、29.79→33.39 tok/s，每轮省约 3.6 ms）；prefill 不变；graph vs 桶 7/7 逐位一致。详见 profiles/bench/tp2_decode_graph/report.md
- [x] ② 实现（payload ≤ 64 KiB 单 block + bump 融合进 AR kernel + 独享 2×64 KiB staging；> 64 KiB 走原切片路径；NINFER_TP2_AR_STRATEGY=size|kernel）
- [x] ② 构建 + 数值比对 + A/B：decode +0.63% @1K / +0.51% @32K（35.12→35.34、33.48→33.65 tok/s，avg_round 29.86→29.70 ms）；prefill 是天然对照，三臂均不变；token 7/7 逐位一致。线程数必须保持 kArThreads（256 线程版本是 −0.2%/−0.4% 回归）。详见 profiles/bench/tp2_decode_graph/report.md

---

# PLAN.md archive (archived 2026-09-22, upstream cherry-pick re-plan + DFlash2 TP-2 rounds)

Archived verbatim from the working PLAN.md when it was reduced to the current state, after the
upstream cherry-pick re-plan, the DFlash2 TP-2 rounds (B1-B7, verify CUDA Graph, MTP draft-chain
graph, declined-draft fix) and the delivery cleanup. Historical record only: the current state and
the remaining work live in `PLAN.md` at the repository root, product behavior and measurements in
[tp2-dual-5060ti.md](tp2-dual-5060ti.md).

# NInfer TP-2 计划（2× RTX 5060 Ti · Qwen3.8-27B NVFP4）

> **唯一的活动计划**，也是跨上下文压缩的持久记忆。整理：2026-09-21（上游 cherry-pick 重规划）。
> - 完整历史记录（Round 1–48 逐轮证据与失败尝试）：`docs/tp2-dual-5060ti-worklog.md`
> - 上一版 PLAN.md 全文（Round 48–55 + worklog Round 11–19，逐字归档）：同文件尾部
> - 交付说明、推荐配置与实测数据：`docs/tp2-dual-5060ti.md`；Windows 原生移植：`docs/windows.md`

---

## 1. 关键信息

- **分支**：`feat/windows-native-port`（Windows 原生 + WSL2 双树；与 origin/master 的 merge base `e360c4c0`，
  上游现领先 20 个提交，见 §4）。
- **硬件**：2× RTX 5060 Ti 16G（SYS 拓扑、无 P2P、448 GB/s/卡）。WSL2 构建树 `/home/zhuojun/ninfer`，
  Windows 工作树 `D:\Documents\workbench\ninfer`（构建树 `build-win`）。
- **主负载**：Qwen3.8-27B NVFP4，TP-2（权重按 shard 切半，本地自建半几何 shape）；MTP draft
  （K=2、`--lm-head-draft`）走全几何 shape（NVFP4 swiglu TMA / linear_add TMA / Q8）。
- **现状**：交付目标 ①–⑤ 全部完成；vision、Windows 移植、多会话 KV 池、decode graph + AR 策略均已落地实测。
- **关键数字**（262,144 配置，WSL）：prefill 1,588 tok/s；decode 59.0 tok/s（MTP K=2）；
  Windows 131,072 配置 decode 56.9 tok/s（与 Linux 持平）。

**推荐运行配置**（WSL 8088 / Windows 8099 同配方，Windows 侧 `--max-context 131072`）：

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh` 是进程级默认思考强度（请求体优先）；`--chat-template` 在本分支
**尚不存在**（模板按 sha256 白名单编译，只有 `e84f32a2…`/`c3cf9e34…` 两种语义）——§4 Tier 1 摘取后
新官方 artifact（模板 `a497db9e…`）与 `--chat-template` 才可用。

### 环境与运维要点

| 项 | 值 |
|---|---|
| 构建（WSL） | `bash tools/tp_bootstrap/build_r35.sh`（rsync + `cmake --build build -j 8`，成功标记 `BUILD_EXIT=0`，约 2–3 分钟；脚本会同步自身，改动要跑第二次才生效） |
| 构建（Windows） | `tools/win_port/configure.bat` + `build.bat`（VS2022 + CUDA 13.3，`-DCMAKE_CUDA_ARCHITECTURES=120a`） |
| 服务 | WSL 8088（`serve_supervise.sh`）；Windows 8099（`tools/win_port/serve.ps1`，默认前台；自测服务必须用 harness 后台 job，`Start-Process` 起的进程会随工具调用结束被杀） |
| 运行 PATH（Windows） | FFmpeg（`D:\ffmpeg-dev\…\bin`）与 libcurl（`D:\curl-dev\…\bin`）必须在 PATH，否则 `STATUS_DLL_NOT_FOUND` |
| artifact | `D:\LLM\qwen3_8_27b_nvfp4.ninfer`（23.7 GB，**旧** artifact，模板 `c3cf9e34…`）；新官方 artifact（模板 `a497db9e…`）需 §4 Tier 1 摘取后加载 |
| 进程纪律 | 全机同一时刻只有一个模型进程（WSL 8088 与 Windows 8099 互斥）；Windows 重新链接 exe 前先停服务（LNK1104） |
| 工具链 | 嵌套 `pwsh` → `wsl -e bash -lc` 吞 `$var` 与重定向 ⇒ 命令写成脚本文件放 `tools/tp_bootstrap/` 再执行；单次阻塞调用上限 600 s ⇒ 长任务用后台作业 |

---

## 2. 已完成工作摘要（细节见归档）

| 项 | 结论 |
|---|---|
| ① MTP 一致性 | 根因＝分片未写 replay 记录（fold 消费空日志）；已修复，判据修正为「无退化 + 质量同档 + 有加速」并达成（归档 §5） |
| ② 基准测试 | prefill 615→1,588 tok/s；decode plain 31.4 / MTP K=2 50.5–54.0 → 59.0 tok/s（归档 §2） |
| ③④ 显存 / KV / 上下文 | 五种 KV dtype 可用（修复 per-device smem opt-in 缺陷）；65,536 → 131,072 → **262,144**（词表/隐层并行切分，−1.2 GiB/卡）（归档 §10） |
| ⑤ 文档 | `docs/tp2-dual-5060ti.md` + `docs/windows.md` |
| Vision（Round 53） | 静态分片分工（MTP→shard 0、vision→shard 1）；多模态与纯文本同一机制（MTP + 前缀复用）；单图上限 8,192 tokens，无新增配置（归档 §11） |
| Windows 移植（Round 54） | VS2022 + CUDA 13.3 原生构建；CLI/serve/TP-2 全部跑通，TP-2 性能与 Linux 持平（decode 32.9/56.9 tok/s）；TMA 描述符 mapped-pinned 修复（C2719 + CUDA Graph 捕获两连）（归档 §12） |
| ①③④ 优化（Round 11–12） | ① exact-batch graph 落地（verify −4.4 ms/轮、decode +16.6%）；③ AR∥MMA 子块流水实现后实测慢 5%，**否决并回退**；④ 复核后原前提不成立，判定完成（归档 Round 11/12） |
| Round 13–16 | [mtp] 刷屏打印清理；前缀续写边界方案实测证伪并撤回；工具调用约束解码路线 A 落地（token 掩码、图外应用）；host 侧状态检查点（步长 8192，显存 0 增量）落地 |
| 多会话 KV 池（Round 55/18） | host KV 换入换出 + 5 会话 LRU + 共享前缀镜像；召回恢复的 KV 与 device 逐字节一致，结果与 from-scratch 在分块边界舍入范围内一致（精确并列仍可能翻转首个采样，见 §3.1）（归档 §13/13.1） |
| Round 17 | plain decode exact-batch graph（+12.6%）+ AR 按 payload 大小切传输策略（+0.63%）落地 |
| Round 19 | shared_c 召回分歧根因定论（chunk 宽度经 `rewind_near_` 依赖引擎历史）；修复已落地并验证（§3.1） |
| Round 20 | chunk 计划与 ring rewind 解耦（1a/1b/1c）：同一 prompt 不再依赖引擎历史，plain/mtp 逐位通过；边界敏感性定量（末块宽度不是变量、边界位置是、120 步 decode 零翻转）；host KV arena 改可分页 backing（`--host-kv-pinned` 才锁页，锁页被拒自动回退） |
| DFlash2（单卡基准；TP-2 未适配） | 已完整实现并在**单卡 5090** 路线合格（`docs/maintainer/dflash.md`、`docs/performance/*`、bench corpus）；TP-2 路线**有意拒绝** `--spec dflash2`（`model_instance.cpp` 归一化报错，worklog §36.1 与 `:2438` 明确保留该限制）；上双卡属「MTP/dflash2 组件切分」级工作，未排期，见 §3.6 |

**未达成的原有门槛（诚实记录）**：MTP3 ≥ 70 tok/s 未达到（纯 decode 上限 K=2 50.5–54.0）；
TP-2 路径未跑 perplexity 评测（质量证据用同提示词多采样 A/B）；per-shard arena 的
`memory_summary()` 仍报 `pages 0/0`（仅显示口径问题）。

---

## 3. 未完成事项

### 3.1 Round 19/20：chunk 计划与 ring rewind 目标解耦（已完成）

根因（已定论）：prefill 最后一块 chunk 的宽度由 `rewind_near_`（按上一请求的分歧位置推出）决定
⇒ 同一 prompt 每次 walk 的 chunk 边界都不同 ⇒ prefill 末列 logits 漂移 0.3–0.56（对 logit≈5 是
6–11%，远超浮点重排量级）。召回 / host slab / 共享前缀镜像全部无罪（KV 与 state 已逐字节证明一致）。

修复方向：chunk 宽度只由 `（reuse, prompt_tokens）` 决定（固定 `prefill_chunk`、
`min(prefill_chunk, remaining)`）；ring 检查点只落在这些固定边界上；放弃「把快照放在贴近 frontier
的位置」的小优化（`kReuseTailCheckpointCount` 的密集尾部窗口已把 rewind 成本限制在几十 token 内）。

**进展（Round 20，第一半已落地）**：`snapshot_at[1]` 改成「最后一块 chunk 的起点」
（`span = prompt_tokens - reuse`，`last = span % prefill_chunk`），chunk 计划不再读 `rewind_near_`。
连续两次运行实测：

- 所有 walk 的 prefill logits 逐位一致（`#3` 与 `#19` 同为 `chunk=512+128 top1=197 4.937500`，
  `#9` 为 `chunk=512+128 top1=54 4.406250`）⇒ **同一 prompt 的结果不再依赖引擎历史**；
- 修复前 oracle 的 `shared_c` 是 4.750000、召回是 4.406250，现在二者是同一个 `4.406250/4.406250`
  ⇒ 原来的召回分歧消失；
- `rewind_near_` 现在只剩写入，属待删死代码（连同 `kReuseRewindMinimum` / `kReuseRewindMaximum`）。

**剩余（第二半）**：用例改在下一层失败 —— `shared_a_second` 复用 647（decode 逐 token 检查点，
落在 chunk 网格之外），后缀首块是 `647+17`，而 from-scratch 是 `640+24`，仍会翻 token。修法：
**召回边界只允许落在 `prefill_chunk` 网格上**（647 → 640、71 → 0），用例断言随之改成 640/0。
特性价值不受损：跨会话共享的 system prompt 边界 512 本来就在网格上，舍掉的只是网格内 ≤1 块的
细粒度复用（代价是重算 ≤255 token）。

**实测结果（对齐已实现）**：召回边界加 `position % prefill_chunk == 0` 过滤后单跑一次：
`shared_c`（512，网格上）oracle 与召回都是 `chunk=512+128 top1=54 4.406250` ⇒ 精确一致；
`shared_a_second` 的 647 被降级到 512（oracle 的 640 device snapshot 同样被过滤，双方都落到 512）
⇒ 两侧 chunk 计划一致。代价与副作用：

- 复用量的损失**有界**（≤1 个 chunk = ≤255 token），因为对齐回退取的是「不晚于 frontier 的某个
  chunk 边界」检查点；
- 但当**连一个对齐检查点都不存在**时（例如 71 token 的小会话，`opening` 的 64 也非 256 倍数），
  复用会掉到 0 = 全量重算，仓库存档的「host slab 召回」用例因此不再覆盖该特性。
  用例当前失败点即此：`a recalled conversation reused 0 prompt tokens, expected 71`。

⇒ 更精确的规则（下一步实现）：**优先取「网格对齐且存在」的最深检查点；一个都没有时，回退到最深
的任意检查点**。这样 647→512（两侧对齐、精确），而 71 仍保留 71（小会话的 host slab 召回不被牺牲）。

**1c 已实现（两通道选择）**：先只在对齐候选里选最深的一个；若该通道一个都没选中（整条 lineage
都落在第一个 chunk 内），再退回「任意候选里最深的一个」。实测 plain 路线全绿：
`recall reused 71 prompt tokens bit-identically`，同时 512 的 cascade 场景保持精确一致。

**验收（已达成）**：plain 与 mtp 两条路线各自输出
`TP-2 session retention (plain|mtp) passed: recall reused 71 prompt tokens bit-identically; LRU eviction
forced a full prefill`；`NINFER_TP2_DEBUG_LOGITS`（含临时 `pc=` 字段）与 `NINFER_TP2_DEBUG_KV`
诊断代码、以及 `rewind_near_` / `kReuseRewindMinimum/Maximum` 死代码已全部删除（`tp2_generation_core.cpp`
残留引用为 0，用例仍全绿）。

**Round 20 补充结论（边界敏感性的定量）**：同一批 640-token prompt 跑 `prefill_chunk ∈
{128,256,384,512,640}`（该值按 `% 128` 量化），得到：

- **末块宽度本身不是变量**：末块 128（pc=128/256/512）、256（pc=384）、640（pc=640 的其中一个
  prompt）三种情况下 logits 逐位相同（`197@4.937500`、`54@4.406250`）；
- **真正的变量是「最后一块从哪里开始」**：唯一一次差异出现在 pc=640 的另一个 prompt —— 末块被截成
  `504+136` 而非 `512+128`，logit 由 6.218750 跳到 7.687500（Δ=1.47，远大于 bf16 步长 0.0625），
  但 argmax 不变、token 序列一字未变；
- **token 层面零翻转**：5 种分块 × 3 个 prompt × 8 轮 decode = 120 步，argmax 全部一致；同一 prompt
  在一次运行内的重复副本也逐位相同（跨进程可复现）；
- ⇒ 0.3~0.56 不是量化噪声，而是分块边界导致的末列归约顺序差异（量级可达 ~1.5），对 greedy 结果几乎
  无影响，但**精确并列**（`shared_c` 为 54 与 220 都 4.406250、gap=0）可能翻转 —— 这同时解释了非生产
  `prefill_chunk` 下 cascade 场景的 FAIL（网格变化改了重算跨度，不是缺陷）。用例的 256 已固定下来，
  并在注释里写明它与召回对齐网格绑定。

**host KV arena 可分页化（本轮附带完成）**：`HostBuffer`（`src/core/arena.{h,cu}`）默认分页
（`std::malloc`，可被 OS 换出），`HostPinning::PreferPinned` 才走 `cudaMallocHost`，且**锁页被拒
自动回退分页**；`HostKVArena` 日志打印真实层级（`host KV arena shard 0: 64.0 MiB pageable`）；CLI
新增 `--host-kv-pinned`；文档同步更新（`docs/serving.md`、`docs/tp2-dual-5060ti.md`，并更正原文
「逐 token 一致」的过强表述）。端到端验证（`ninfer-serve --devices 0,1 --host-kv-mib 4096`，7 个不同
system prompt 强制淘汰）：默认输出 `host KV arena shard 0/1: 2048.0 MiB pageable`，加
`--host-kv-pinned` 输出 `2048.0 MiB pinned`，两次均无 `refused`，停服后显存干净释放 ⇒ 4 GiB 池可建、
开关真实生效。⇒ `--host-kv-mib` 不再要求锁页常驻，也不会因锁页被拒而整体失败；提交量不变
（分页 ≠ 少占提交）。

**踩到并修掉的隐含陷阱**：`src/runtime/engine/model_instance.cpp` 的 TP-2 归一化用指定初始化器重建
`ContextCacheOptions`，只搬运它显式列出的字段 ⇒ 新字段会被静默丢弃（本次 `host_kv_pinned` 就这样失效
过一次，表现为 `--host-kv-pinned` 无效果、日志仍报 `pageable`）。给该结构加字段时必须同步这里。
worklog §36.1 记录过同一函数的同类事故（当年 `--spec` 也被丢过一次）⇒ 这是同一函数的第二次；修改
`ContextCacheOptions` 时必须连 `model_instance.cpp` 的指定初始化器一起改。

### 3.2 交付前收尾（归档 §7）

- [ ] 用推荐配置复测并发 C=1/2/4 与流式/stop 冒烟（Round 31/34 的结论基于 bf16 配置）
- [ ] 交付时按 AGENTS.md 移除本文件（PLAN.md）；`tools/tp_bootstrap/` 下约 300 个一次性诊断脚本
      需决定保留/清理范围（`build_r35.sh`、`serve_*`、`r37+r38+r39+r4x` 系列与
      `docs/tp2-dual-5060ti.md` 引用为可复现流程）

### 3.3 Windows 单卡 CLI tiny artifact 子任务（用户决定；前置已就绪）

目标：产出小到装进单张 16 GiB 卡的 `.ninfer`，用 Windows `ninfer.exe` 单卡跑通「加载 → 生成 → 采样」。
前置（Round 54.9）：CPU 版 torch 在 `build-win\torch-venv`（torch 2.14.0+cpu）；转换器接口已确认。
下一步：① 官方 recipe 直接跑 tiny checkpoint（`--recipe qwen3_8_27b_nvfp4 --components text`）；
② 不成功再写 tiny recipe/override；③ 生成脚本放 `tools/win_port/tiny_model.py`（合成 tiny config +
随机 bf16 权重，HF 命名）；④ 成功则写入 `docs/windows.md` 并提交。

### 3.4 既有缺陷（未修，低优先）

- TP-2 路线 `bf16` KV + `--spec mtp` 在第一次 prefill 就崩（`prompt.cu:63` `cudaErrorInvalidValue`）；
  fp8 + MTP 正常（推荐配置不受影响）。排查入口：bf16 prompt-attention 的 launch 参数。

### 3.5 暂缓 / 可选（未排期）

- Round 15 剩余：工具调用约束真实模型端到端复测（75 条探针）；WARN 诊断；路线 B（完整 GBNF）；
  TP-2 `validate_licensed_tokens` 守卫
- Round 14 候选（用户暂缓）：① system+tools 前缀末尾加锚点（修新会话/压缩后第一轮 46.9 s，可省约 11 s）；
  ② prefill 吞吐（suffix 0.95–1.13k tok/s、冷启 1.35–1.38k）；③ 每请求 ~0.3 s 固定开销
- 归档 §11.6：MTP priming 对图片列用占位 embedding（~100 行，无收益证据）；单图上限 8,192 硬编码
  （让出 shard 1 可重拿更大 envelope）；视觉编码期 shard 0 空闲（可接受）
- 归档 §6：KV dtype 扫描补全（nvfp4/k8v4 只验证了可启动与吞吐，无质量数据）

---

### 3.6 DFlash2 上 TP-2 的适配计划（未排期；结论来自 §7 调研）

**现状**：DFlash2 已完整实现、并在**单卡 5090** 路线合格（数学与状态见 `docs/maintainer/dflash.md`，实现见
`execution/draft.cpp`、`load/dflash{,2}.cpp`，性能与 corpus 见 `docs/performance/*`）；TP-2 路线原**有意拒绝**
`--spec dflash2`（`model_instance.cpp` 抛 `TP-2 generation supports --spec mtp only`，理由是 dual-shard
loader 不上电 draft 组件），限制被明确保留（worklog §36.1 `:713`、`:2438`）。**B5 已把该拒绝打开**（见下文 B5 结果），
`--spec dflash`（v1）仍拒绝。

**单卡收益证据（已备，成本＝读文档）**：仓库已发布单张 5090、C=1 的 DFlash2 K=7 对 MTP3 对比
（`docs/performance/qwen3.8-27b.md:275-307`）：

- 每请求 phase 速率：`nvfp4` 档 +64.5% / +21.1% / +19.2%（三个 AIME 长解码 fixture）、Code +36.6%、
  Translation +33.1%、Structured +62.3%、Story −3.8%；`groupwise-int` 档多数为负（Story −35.6%、
  AIME15 −11.1%）⇒ 收益**依赖权重格式**；
- 整语料（C=1）：`nvfp4` decode 速率 **+19.5%**、makespan **−22.7%**；`groupwise-int` −9.1% / +12.1%；
- 机制：DFlash2 每轮提议更多（tokens/round 3.5–6.5 vs MTP3 2.7–3.7），但接受率更低（35–78% vs 37–91%）；
- 文档自身的保留：两次 campaign 的 revision 与工件不同、输出长度随机不同，**不是隔离的 backend 对比**，
  且**未跑新的 MTP3 基线**、不代表答案正确性；groupwise-int 另有 AIME30 重复循环离群样本；DFlash2 K=15
  与并发 DFlash2 **未发布**；
- 本机做不了单卡对照：最小 27B 工件也有 17.4 GB（> 16 GiB），`qwen3_8_27b_w4a4_w8a8_dflash2.ninfer` 为
  24.8 GB ⇒ **只能以已发布 campaign 作为收益证据**。

**推论**：本机路线是 w4a4/w8a8（接近 `nvfp4` 档）⇒ 收益为正、方向支持适配。但 DFlash2 的收益来自「每轮
更多提议」，而双卡上 draft 每轮成本（5 层 × 两卡 + 特征汇聚）与 verify 成本都会上升 ⇒ 能否保住这 +19.5%
取决于 B1–B4 的实现效率，**这才是本项的真实风险**。归档对双卡的预期是 **DFlash2 90–180 tok/s**
（worklog `:60`），作为 B7 的验收目标。

**可直接复用的既有面（MTP 已铺好）**：分片放置机制（`tp_split_spec.cpp:87-94` + `tp_shard_views.cpp:42-61`
的「一卡持有、另一卡留空 view」，加上 `has_weight` 容忍缺失）；加载 seam（`load.cpp:63-85,125-155`）；**整条
验证回路**（`forward_tp2_window`、Verify 绑定、每列 logits/hidden、`run_verify_window`、RecordForReplay+fold、
`speculative_accept_sparse_drafts`）；独立 draft KV 几何先例（`state/decoder_state.h:20-29`）；以及 graph 分桶
`dflash_graph_profiles`（`graph_profiles.h:9`、`graph_profiles.cpp:91`，已用于 `graphs.cpp:375,399`）。

**必须新设计的地方（单卡假设的破除）**：
1. **条件特征跨卡**：提议条件是目标 5 个 block 的 residual 拼接投影（`draft.cpp:59-89` 的 `DFlashFeatureSink`），
   而 TP-2 把 64 层切在两卡 ⇒ 每步都要跨卡 handoff，代码里**尚无设计**；
2. **selector 与词表**：`candidate_selector_path` 直读**全词表** codebook（pred/succ 各 248320×256 BF16，合计约
   254 MiB），且 proposal head 与目标 lm_head 共用权重（`load.cpp:83`）⇒ TP-2 已按 vocab 切 text head 的做法
   会破坏 selector 的全词表 top-k ⇒ codebook 必须**整份复制**；
3. **提议回路**：MTP 的「右移一位 embedding + 末列设备覆盖 + host 串行 AR」是 MTP 特有（`tp2_generation_core.cpp
   :760-802,1004-1042`），DFlash2 是一次非因果 masked 块 + 稀疏拒绝采样，需重做（验证面不用）；
4. **每卡预算**：草稿按现成工件的实际编码**实测 2.07 GiB**（`qwen3_8_27b_w4a4_w8a8_dflash2.ninfer` 与同源无
   dflash2 工件的 payload 差 = 2,226,792,960 B；纯 BF16 则 3.59 GiB），另有 5 层 local ring K(BF16)+V(FP16)
   固定 40 MiB，无 per-token 增长（`dflash.md:274-289`）。

**决策点（先定再写码）**：
- **D1 归属**：整份复制还是按 head/FFN 切分？**实测结论：复制不可行** —— 满上下文配置（262144 tok / fp8 /
  MTP K=2）下 free 仅 697 / 1217 MiB（`docs/tp2-dual-5060ti.md:329-337`），而复制需 2.07 GiB/卡；DFlash2 与
  MTP 又是**互斥 backend**，选 DFlash2 释放 shard 0 的 MTP 权重 430 MiB + KV 516 MiB 后 free 也只到约
  1.6 GiB。切分后每卡约 1.04 GiB 可行，但 shard 1 仅余约 0.2 GiB。**评估阶段先用小上下文**：本机
  `--max-context 4096` 实测 free 4344 MiB（§3.1 验收记录），可先绕开这个瓶颈；
- **D2 量化**：草稿权重用何种格式（bf16 复制无余量 ⇒ 需量化或切分），需先看工件里 companion 权重的实际精度；
- **D3 特征 handoff**：目标侧按 block 捕获后 all-gather（每步 5×5120 BF16）还是让提议卡各持一半？
- **D4 提议位置**：沿用 MTP 的「只在 shard 0 提议」（则特征与 codebook 都要在 shard 0 齐备），还是两卡各提议一半。

**B1 结果（已完成）**：`dflash2/*` 在 split spec 中归入 shard-local（`shards=0x1`，与 `mtp/*` 同规则），
`tp_split_spec.h` 契约注释同步；顺带修掉一个真实缺陷：`parameters.cpp` 的 draft 分支缺 shard 存在性判断
（MTP/Vision 都有），在 shard 1 上用不存在的权重构造参数块 ⇒ 报 `projection weight must be a matrix`。
修复后 `ninfer_qwen3_5_tp2_load_test` 三例全绿（`--spec dflash2` 头复制、`--spec dflash2 --lm-head-draft`
头按 vocab 切、`--spec mtp --lm-head-draft` 无回归），用例另断言 shard 0 持有整份 draft（feature_projection
[5120,25600]、draft query [4096,5120]、predecessor codebook [248320,256]），shard 1 不付这份字节。加载
用例直接走 loader、不经过 Engine 选项归一化 ⇒ 生成路径的 `--spec mtp only` 门禁可保留到 B5 打通。

**B2/B6 设计要点（调研结论）**：

- 捕获 plumbing 很小：`run_layers_tp2` 每层在两次 all-reduce 之后已调用 `tap.capture_layer`，两个
  `forward_tp2_*` 入口只是硬编码 `NullTap` ⇒ 加可选 `DFlashFeatureSink*` 分支即可，层循环不用改；
- 两卡 residual **逐位相同**（三条 allreduce 都在本端做 BF16 local+peer 加法）⇒ 只捕 shard 0 就够；
- TP-2 chunk ≤ 1024 < S=2048 ⇒「只存最后活动窗口」不会触发，features 缓冲只需一个 chunk 大小；
- **必须新增的状态搬运**：draft 的 local context ring 目前不在 `state_backing`/host checkpoint/host KV slab
  中，而 prefix reuse 会跳过共享前缀（被跳过段拿不到目标残差）⇒ 不搬状态，召回/reuse 后 draft context
  必然缺失；checkpoint 前还须 flush pending features。

**可行性结论与潜在问题（调研汇总）**：功能可行（加载已实测），但有三处必须先解决或确认：

1. **reduced proposal head 的 vocab 切分与 DFlash2 单卡 top-k 冲突（最先、最确定）**：`proposal_head` 默认
   Optimized，`tp_split_spec.cpp:123-129` 会把它 ColumnParallel ⇒ shard 0 只有 131072/2 = 65536 行；而
   `propose_dflash2_batch`（`draft.cpp:356-370`）在 shard 0 上对整个 reduced 词表做 `linear_topk`，该算子按
   精确行数匹配 profile（248320 / 131072）⇒ 65536 直接抛 `unsupported head profile`，且 DFlash2 路径**没有**
   任何跨卡合并（MTP 是在 `text.cpp:782-825` 用 `merge_local_row_blocks` + allreduce 合并的）。**已修**：
   `load.cpp:137-145` 让 DFlash2 保留整份提议头（`split_proposal_head` 加 `!dflash2()`）；工作区容量按 shard
   本地参数计算（`planning/startup.cpp:631-636`），头变整份后会自动跟着变大；
2. **宽窗口 verify 掉出 tiny-T**：目标注意力的 tiny-T 内核只实现 T=1..6（全几何）/ T=7..8（仅 24 头），而 TP-2
   单卡是 12/2 几何、MTP 的 `draft_tokens` 本来就被 clamp 到 1..5 ⇒ W≤6 正落在快路径。DFlash2 允许 K=1..15
   ⇒ **W≥7 会走 ChunkedSmallT/Prompt（可跑，但是另一条内核/慢路径）** ⇒ 本机收益应先按 K≤5 估算
   （tokens/round 由 3.59 降到约 3.0 ⇒ 性能预期由 ~108 降到约 ~90 tok/s），W≥7 的实际代价必须实测；
3. **草案侧不缺宽窗口**：`sliding_window_attention` 覆盖 T=1..16、window 2048/4096，DFlash2 专用融合算子到
   T≤48 ⇒ 瓶颈只在目标 verify 那一侧；
4. **更正：`dflash_graph_profiles` 在 TP-2 上无意义** —— TP-2 强制 `use_cuda_graph=false`，`prepare_graphs`
   首行就 return，这些桶从不 capture；TP-2 用自己那套 `WindowGraph` + `select_window_graph`
   （`tp2_generation_core.cpp:868-914`，按 `visible_end` 分桶，eager 与 capture 共用同一 `forward_tp2_window`）
   ⇒ 阶段分解里「接 dflash_graph_profiles」这条删除。

**状态面与时序（调研汇总；最高风险区）**：

- **运行期状态面为零**：TP-2 core 没有 `StateImageDevicePool`、没有 draft local ring、没有 pending/prefill feature
  缓冲（设备侧只有 GDN 池 `tp2_generation_core.cpp:500-523`）；单卡侧有现成布局可移植（`state/state_image.h:26-30`、
  `state_image.cpp:117-131,159-164`、`startup.cpp:139-165`）。
- **要动的位置**：`build_shard` 建 ring 并计入 state arena slot；`snapshot_host_checkpoint`（`:2022-2057`，现只
  D2H `state_backing`）加 ring 与 frontier；会话 slab 注册（`session_ensure_host_slabs` `:1153-1273`，仿
  `host_mtp_kv`）；`SessionEntry`（`h:190-225`）加 draft 镜像与 frontier；四处拷贝点（store `:1296-1317`、
  restore `:1352-1369`、device snapshot `:2008-2013`、round scratch `:2496-2497`）；以及
  `forward_tp2_prefill/window` 新增 `DFlashFeatureSink*` 入口（`text.h:196-213` 目前没有）。
- **两个硬时序**：① checkpoint/发布前必须 flush pending features（单卡靠 `commit.cpp:305-319`），而 TP-2 的
  `session_store_active` 在请求边界只拷 text KV + GDN + MTP KV，既不 flush 也不搬 draft 面；② **prefix reuse
  与 draft context 天然冲突**：跳过共享前缀就不产生 `[0,reuse)` 的 target residual（`execute_walk` 从 reuse 起，
  `:2132`），draft 上下文出现空洞，现有镜像补不回来 ⇒ 要么把 draft context 纳入 reuse 边界状态，要么该路线
  放弃 reuse。
- **验收口径**：TP-2 没有 Verify 相位（MTP 当年退回 Prefill 并记为「非逐位一致」），所以 DFlash2 在 TP-2 上
  只能按「无退化 + 质量同档 + 有加速」验收，**不能**按逐位 parity 验收。
- **历史陷阱复现风险**：per-device smem opt-in 仍有四处进程级 static（`q8_dynamic_grouped_conv_add_materialized.cu:46`
  等），恰好覆盖 draft 的 finish 路径；Round 49 的「前几个 token 正常、随后 0 复读」就是同类状态错位的静默表征。

**阶段分解（每阶段独立验收）**：
- **B1 加载与放置**：放开拒绝 + `tp_split_spec` 增加 `dflash2/*` 规则（feature_projection/codebook 复制，context
  K/V 按 head 切）+ 每卡 draft 配置/state/plan（`tp2_generation_core.cpp:436-523`）⇒ 验收：两卡都物化成功、
  `ninfer_qwen3_5_tp2_load_test` 通过；
- **B2 上下文物化**：分两步。**B2a（已完成）**：两个 `forward_tp2_*` 入口接受可选
  `DFlashFeatureSink*`，prefill 里 `begin` → 层循环 → `capture_positions` → `consume_prefill_chunk`，
  window 里 `begin` → 层循环（batch 模式），`NullTap` 仍是默认。**B2b（已完成，`ae83d4a6`）**：shard 0
  按单卡持久布局分配 `prefill_features[hidden*L, chunk]` / `prefill_positions[chunk]` /
  `pending_features[hidden*L, K+1, 1]` 与零初始化的 draft local ring（`plan_cyclic_kv_cache`，lane=1），
  prefill 循环装配 sink：`layers` 取 `draft.target_layer_ids`，consumer 按 lane 0、整 chunk 走
  `dflash_append_context`（scratch 已 scope）。门禁未动 ⇒ 运行期仍恒走 `NullTap`，sessions 的 plain 与
  mtp 都通过。**四个未决项**：① ring 未纳入 `session_store_active`/host slab/checkpoint（B6），lane 固定 0，
  且 prefix reuse 会让 draft context 空洞；② workspace 峰值未实测（chunk=1024 时 consumer 约 60 MB）；
  ③ verify 的 batch 字段未接（B3/B4），`pending_features` 仅分配未使用；④ **验收（TP-2-only，已完成）**：
  `tests/models/qwen3_5/test_tp2_dflash_append.cpp` 经 loader-only 路径（`plan_load(DFlash2)` +
  `materialize_model_tp2`，不经 Engine、门禁不动）驱动 sink ⇒ 装/不装 sink 的 prefill 末列 logits 逐位相同、
  两 shard 之间也相同；consumer 1 次/轮、`captured_mask=0x1f`、positions 恰为 `[0,1024)`；ring 结构合理（live
  全写、nonfinite=0、容量余量全零）；直调 append 与 sink 路径逐位相同；1024 一次 vs 2×512 两次逐位相同；
  跨进程可复现（fnv1a=0x1a53fd824cd4e360）。arena 90.1 MiB、**workspace 峰值 113 MiB（59% of 192 MiB，
  装/不装 sink 相同）** ⇒ 未决项②结案。**仍未证明** ring 的数值正确性 —— 单卡 oracle 经三条证据确认不可构造
  （loader 无条件绑 text、`Parameters` 无条件准备整条 text 栈、整模型单卡放不下）；
- **B3 提议前向**：masked 块 5 层滑动 + 动态卷积按 D1 执行 ⇒ 验收：提议对同一 features 确定可复现，且**贪心解码下
  DFlash2 与 plain 的输出 token 序列完全一致**（投机只改速度、不改贪心结果，这条同时覆盖接受与折叠的正确性）；
  **B3 结果（本轮，TP-2-only；未提交）**：新增出口 `dflash_propose_batch`（`execution/draft.cpp`、`program/context.h`）
  复用生产 `propose_dflash2_batch`（不重写数学；masked 块 + `linear_topk` + `candidate_selector_path` 一起跑属复用生产代码，
  非新写 selector）。打通路上发现两个 TP-2 缺口：① `text/token_embedding` 是 RowParallel
  （`load/tp_split_spec.cpp:117-122`），而 masked draft 要整宽 embedding ⇒ 新增 `TextContext::embedding_full_width`
  （复用 `embedding_tp2`，与 MTP stem 同路径，`execution/text.cpp`），`propose_dflash2_batch` 在 `DFlashBatchContext.tp_card`
  非空时走它；② 跨卡 allreduce 结束后 `DevicePair::allreduce` 把 peer 设备留为 current（`core/tp/device_pair.cu:440,459`），
  使 selector 的 [256,5120] BF16 投影（需 opt-in 动态 smem）在错误设备实例上设置属性 ⇒ 提案在嵌入后须重绑本卡。测试
  `test_tp2_dflash_append` 扩为 prefill→append→proposal：K=7 产出 draft[7]、candidate[16,7]、proposal_q[16,7]、query
  positions `[1023,1031)`，每行 16 个候选互异且落在公共 token 域、draft ∈ 该行候选、贪心 proposal_q 为精确 one-hot；同
  ring 两次提议与全新 prefill→append→proposal 链逐位一致（fnv1a=0xbad27a494a9bc853，两个独立进程一致；K=5 为
  0xbee487264ca8ffb8）。每卡草稿权重实测为量化档（layers/feature_projection=q8_g32_fp16、codebook=bf16、tied output
  head=fp8_e4m3fn），shard 0 的 draft 块（含被共享的 tied head 1213 MiB）3655.4 MiB；proposal frame 4.4 MiB、瞬时
  workspace 峰值 4.4 MiB（192 MiB 的 2%）、事件计时 **7.95 ms/次（K=7）、7.93 ms/次（K=5）**。**本轮未做**：selector 的
  Engine 发布、verify/接受/折叠、会话/checkpoint 状态（B4–B6）；因此 plan 的「贪心 DFlash2 == plain」端到端验收无法在本轮
  建立，B5 前不得声称。
- **B4 组装轮次（已完成，`b4e84056`）**：selector 数学在 B3 已随生产 `propose_dflash2_batch` 跑通；本阶段把整轮收敛成生产组件
  `program/dflash_round.{h,cpp}`（draft 持久 context + 精确 B 的 decode frame + **独立 proposal arena** + 发布 K drafts /
  candidate ids / proposal q / query ids+positions），core 的六个 shard 字段收敛为单个 `dflash_round`，sink 工厂改为委托，
  单卡与双卡共用 `dflash2_proposal_workspace_bytes()`。**arena 归属坑已证明修好**：投毒 shard 0 常驻 `prefill_hidden` 后
  再次 propose ⇒ 图案逐位存活、shard workspace 计数不变、组件 arena 峰值 4.4 MiB（K=7）/3.3 MiB（K=5），与 planner 预算
  完全相同。验收：组装轮次逐位复现 B3 哈希（K=7 `0xbad27a494a9bc853`、K=5 `0xbee487264ca8ffb8`，跨二进制/跨进程），
  单次提议 7.95 ms；context 90.4 MiB（较 B3 多 0.34 MiB，因组件按生产 `lanes=K+1=8` 分配 `pending_features`，暂无人读）；
- **B5 结果（已完成，工作树未提交）**：DFlash2 的 masked draft 已在 TP-2 跑通「prefill sink → append_pending →
  masked 提议 → 目标 verify（带 feature sink）→ 稀疏拒绝采样 → GDN fold → terminal append」整轮，gate 在
  `model_instance.cpp` 打开（`--spec dflash2` 放行，`--spec dflash` 仍拒绝）。
  - 改动：`program/dflash_round.{h,cpp}` 新增 `make_verify_sink()`/`append_pending()`/`draft_window()`/
    `feature_lanes()`；`runtime/engine/tp2_generation_core.{h,cpp}` 的 `run_verify_window` 增加 sink 形参，构造期建立
    DFlash2 标志/草稿数/context frontier，`build_shard` 为 DFlash2 建 GDN records + `replay_fold`（width=drafts+1），
    decode 分支按单卡 `execution/draft.cpp:dflash_decode_batch_body` 的顺序实现，extent==0 回退 plain step；
    `src/runtime/engine/model_instance.cpp:97-112` 打开 gate 并为本路线关闭 context cache/host checkpoint/会话保留
    （draft ring 尚未进入 B6 的状态镜像）。
  - 关键缺陷（本轮修复）：verify window 的 ids/positions 是设备端产物，D2H 到 pinned buffer 后**未同步**就被
    `forward_tp2_window` 读到另一张卡的 stream 上（MTP 的窗口是 host 直接写的，所以既有路径不暴露此问题）⇒ 修复前
    输出退化成重复片段，加一次 `cudaStreamSynchronize` 后连贯。
  - 验收：K=7/K=5 `ninfer_qwen3_5_tp2_dflash_append_test` 哈希不变；`load_test` 三例、`sessions_test` 全绿；
    `ninfer-serve --devices 0,1 --max-context 4096 --kv-dtype fp8 --greedy --no-prefix-reuse` 上 5 个 prompt 全部连贯
    （最长同词重复=1），同一 prompt 重复请求逐字节一致（确定性）。
  - **「DFlash2 == plain 逐 token」不成立，且本引擎不可能成立**：窗口前向与单 token decode 是不同执行形状，逐列
    logits 有差（`docs/tp2-dual-5060ti.md:172-176` 已对 MTP 声明）。本轮对照实测（同工件、同 prompt、greedy 128
    token）：MTP K=2 与 plain 公共前缀 320 字符即分叉；DFlash2 **K=2 为 390 字符（优于 MTP K=2）**、K=7 为 207
    字符（随窗口变宽而变短）。因此 B5 的验收改为「不劣于 MTP 且不退化」，已满足；严格的 token 逐位一致作为
    **无法建立**项记录。
  - 实测（4096 context、fp8 KV、greedy、65 prompt token→128 输出 token）：plain 35.1 tok/s；MTP K=2 62.8 tok/s
    （56 轮/127 提交=2.27 token/轮、接受率 63.4%）；DFlash2 K=2 51.0 tok/s（58 轮、接受率 59.5%）；**DFlash2 K=7
    63.6 tok/s（41 轮/127 提交=3.10 token/轮、接受率 30.0%）**，为 plain 的 1.81×、与 MTP K=2 持平。未达 §3.6 的
    108 tok/s roofline 估算 —— DFlash2 的 verify 目前是 **eager**（feature sink 无法进 graph），同宽 verify 比 MTP
    多约 4 ms/轮，且提议约 6.4–6.8 ms/轮。内存：shard 0 `weights+ctx 13878.6 MiB`、free 1764 MiB（draft 3655.4 MiB）；
    两种 proposal head（默认 Full 与 `--lm-head-draft`）输出与速率一致；
- **B6 结果（已完成，工作树未提交；保留禁用保留）**：DFlash2 的 masked draft context（局部 cyclic K/V ring +
  其绝对 frontier）现在与目标 GDN/KV 一起走 TP-2 的全部状态通道，但**不重新打开**会话保留：保留验收要求的
  「召回逐 token 一致」对该路线在合成 prompt 上不成立（见下）。改动：
  - `program/dflash_round.{h,cpp}`：新增 `context_image_bytes()` 与
    `copy_context_to_host/from_host/to_device/from_device`——把 ring 按 layer-major（先 K 后 V、按 layer extent
    紧排、跳过 alignment pitch）打包成宿主镜像与设备快照通用的扁平镜像；
  - `runtime/engine/tp2_generation_core.{h,cpp}`：`HostCheckpoint` 增加 `dflash_buffer`+`dflash_frontier`，
    `Shard` 增加 `dflash_snapshots[0..1]`+arena，`SessionEntry` 增加 `host_dflash`/`host_dflash_prompt`/
    `host_dflash_shared`；`build_shard` 按实际分配计账；`session_ensure_host_slabs` 为拥有 draft 的 shard
    强制分配 `host_dflash`（失败即拒绝该会话）；`session_store_active`/`session_restore`/
    `session_capture_shared_state` 按 `RecallState`（Frontier/PromptEnd/Shared）成对搬运目标 GDN 与 draft ring；
    `snapshot_host_checkpoint` 记录 `dflash_frontier`；`execute_walk` 新增 `begin_dflash_state`（None→zero、
    LiveState→no-op、DeviceSnapshot/HostCheckpoint→按镜像恢复，HostCheckpoint 校验 `dflash_frontier == position`）；
    发布/取消/publish 前用 `flush_dflash_context` 提交 pending staging；`extent==0` 保持「真实轮次」（否则最后一列的
    draft 残差缺失、frontier 处留洞）；复用扫描对 DFlash2 打开（移除 `!dflash2_enabled_` 残留守卫）；
  - `tests/models/qwen3_5/test_tp2_sessions.cpp`：路由参数化 `plain|mtp|dflash2`（默认只跑前两条），DFlash2 用
    `draft_tokens=7` + fp8 KV。
  - 验收：`ninfer_qwen3_5_tp2_dflash_append_test` K=7 `0xbad27a494a9bc853`、K=5 `0xbee487264ca8ffb8` 不变；
    `ninfer_qwen3_5_tp2_load_test` 三例通过；`ninfer_qwen3_5_tp2_sessions_test` plain/mtp 全绿。
  - **召回边界诊断（本轮，先诊断后动手）**：DFlash2 的召回边界来自 `tp2_generation_core.cpp:1965-2045` 的复用扫描；
    对齐扫描只接受 `boundary % reuse_grid == 0`（`reuse_grid = min(max(prefill_chunk,64), maximum) = 256`），但
    `:2021-2045` 的兜底分支在 `reuse==0` 时**去掉 grid 限制**重扫同一批边界（live frontier / `cached_boundaries_` /
    host checkpoints），于是 prompt-end 召回把 reuse 定在 64（64 不是 256 的整数倍）。prefill chunk 循环
    （`:2455` 的 `for (t0 = reuse; …)` 与 `:2489` 的 `length = min(prefill_chunk, prompt_tokens - t0)`）**把 chunk
    锚在 reuse**，所以召回后缀的首块只有 24 列，而 from-scratch walk 的同一段是 88 列（一块）。fp8 目标 logits 在
    合成 prompt 的近似并列处翻转（同一对 token 220/198），贪心答案在 rerendered 分叉（`[2752 11 220 …]` vs
    `[2752 11 198 …]`）。列宽差异对 plain/MTP 不翻转，所以它们通过。三条诊断实测：
    - chunk 256 + 保留 ON：grid 对齐的 `shared_b`@512（DeviceSnapshot）与 oracle 逐轮一致并通过；失败在 prompt-end 的 64。
    - chunk 128（归一化后生效值）：连 grid 对齐的 `shared_b` 也在最后一个 token 翻转（198 vs 220）⇒ 边界是否在 grid 上
      **不是**唯一因素，chunk 宽度本身也参与。
    - 强制 DFlash2 全程 `extent=0`（只跑目标轮）：`shared_a_first` 即分叉（`got [197 197 92 198 198 198 1464 198]`
      vs `expected [197 197 92 198 695 197 197 92]`）⇒ 目标侧单独跑**也不**复现 oracle 的窗口轮次。
    **机制结论**：DFlash2 的生成 token 依赖每轮的 verify 窗口布局，而窗口布局由「提议 → 稀疏接受 → 提交前缀」驱动；
    召回点的 chunk 宽度改了后缀 walk 的窗口序列，接受计数随之改变，输出在近似并列处翻转。所以「把 chunk 锚在绝对边界、
    只在 reuse 点截断第一块」并不能让 prompt<chunk 的召回与从头一致（`:2489` 的 first block 仍是 `[reuse, prompt)`），
    也不能靠「draft 判失效」保证逐位一致（见上条 `extent=0` 诊断）。
  - **兜底契约（已实现，本轮选定 B）**：召回边界不在 prefill grid 上时，`tp2_generation_core.cpp:2044-2052` 设
    `dflash_draft_declined_`（`reuse != 0 && reuse % reuse_grid != 0`），该请求的每个 DFlash2 轮次 `extent=0`
    （`:2778-2788`），即只跑目标轮；目标侧 KV/GDN 复用**照常**（状态镜像仍按 `begin_dflash_state` 恢复）。契约通过
    `include/ninfer/types.h` 的 `GenerationResult::draft_context_declined` 暴露，由
    `test_tp2_sessions.cpp` 的 `draft_declined`/`compare_recall` 断言。理由：draft 只负责提议，每个 token 仍要过目标
    verify；draft 上下文无效只掉接受率，不影响目标自身 token 的合法性。
  - **未通过项（决定不重新打开保留）**：grid 对齐的 `shared_b`@512 召回与从头 walk 分叉，形态固定：
    engine `[1703 220 248046 198 248045 198 248045 198]` vs oracle `[1703 220 248046 198 248045 198 248045 220]`
    —— **前 7 个 token 与每个 verify 轮的 base/extent/licensed 逐项一致，只有最后一轮的 token 不同**；也就是说**召回路径
    本身是干净的**（状态、KV/GDN、边界、轮次序列都能复现），分叉落在 DFlash2 窗口的最后一个 **`extent=0` 钳位轮**
    （8 列全部钳到 anchor 的同一位置，`:2784-2826`）。该轮**在同一二进制上也不可复现**：`b6-acc2-dflash2.log`
    （FAIL）与 `b6-rep1/rep2.log`（PASS）是同一次构建的结果；同一二进制连跑 5 次（`build-win/b6-det-1..5.log`）得
    1 次通过（`b6-det-1`）、4 次失败（`b6-det-2..5`），4 次失败的 `got` **逐字符相同**
    （`[1703 220 248046 198 248045 198 248045 198]`，oracle 末位 220）⇒ 失败**有偏**而非纯随机。
    ⇒ DFlash2 的生成在本引擎里**不可复现**：钳位轮的近似并列处存在竞态/未同步读（同一输入翻转 198/220），
    与「DFlash2 窗口 ≠ 单 token decode」同属窗口形状这条根因，但**它是实现缺陷而非纯数值形状限制**。
  - **结论（一行）**：**不是召回边界/状态缺陷，而是 DFlash2 verify 窗口钳位轮（`extent=0`）的不可复现性** ——
    召回路径把状态/KV/GDN/边界/轮次序列都复现了（前 7 个 token 与每轮 base/extent/licensed 逐项一致），翻转只发生在
    窗口的最后一个钳位轮，且该轮在**同一二进制上也会随机翻转**（3 次运行 1 次失败）。按 §3.6 决策规则 DFlash2 未达到
    保留属性 ⇒ `model_instance.cpp:105-118` 的 DFlash2 保留禁用**保留**，该路线不声称「召回逐位一致」；且
    `NINFER_TEST_ROUTE=dflash2` 的契约断言在钳位轮翻转时仍会 FAIL（本路线不在默认路由集合内）。
  - **收口契约**：最终状态下 `NINFER_TEST_ROUTE=dflash2` 得到 `[mem] host sessions capacity 0 | retention disabled`
    并 **exit 0**（`route_keeps_retention` + `compare_recall`）：保留关闭 ⇒ 不发生跨会话召回，每条召回场景期望复用 0
    token（返回会话整段 prefill），唯一允许的非零复用是共享系统前缀经 device 快照的前缀复用（`shared_b` 512）；plain/mtp
    保持原有逐位复用断言。`draft_context_declined` 兜底契约保留（它是正确的改进：只损失接受率，不损失正确性）。
  - **后续若要打开保留，需要先解决**：DFlash2 verify 窗口 **`extent=0` 钳位轮**的逐位可复现性。本轮把该轮的 KV/注意力
    索引链查清，并**推翻初版猜测**（钳位列不是读到上一轮被拒提案的 KV）：
    - 钳位列的构造在 `src/ops/kernel/speculative_round.cuh:33-37`：`verify_ids[j>extent]=anchors[row]`、
      `positions[j>extent]=base_positions[row]+extent` ⇒ 钳位列是**第 `extent` 列的精确副本**（同 token、同位置）；
    - cache 槽位与 RoPE 位置绑到同一个钳位数组（`src/models/qwen3_5/execution/text.cpp:2193-2194`，`cache0`/`rope0`
      都用 `bind0.positions`），KV 槽位由该位置经页表得出 ⇒ 钳位列读的是**第 `extent` 列自己的槽位**，不是别处的陈旧行；
    - 写侧被掩码：`small_t_fp8.cuh:94-98,191`（`valid_tokens=min(valid_columns[batch]-column_begin,TokenTile)`，只有前
      `extent+1` 列写 KV）；读窗上界是钳位后的 `last_pos+1`（`:147-159`）；sink/append 同样被 `target_valid_columns`
      掩码（`execution/draft.cpp:578-660`、`program/dflash_round.cpp:339-385`）⇒ **钳位轮在目标窗口内确定且掩码正确**；
    - 于是跨运行变量只剩 TP-2 的**跨卡 allreduce 传输**：本机 `cudaDeviceCanAccessPeer(0↔1)==0`（探针
      `build-win/tmp_peer_probe.cu`），走的是 `src/core/tp/device_pair.cu:95-170` 的 in-kernel mapped-host
      arrival-token 握手（窗口 `[V,8]` logits 合并是 `ar_slices==5` 的多块路径，`:200-205,448-466`），这是该路线里唯一
      的异步跨卡机制。本轮试过 A/B：临时给 `NINFER_TP2_AR_STRATEGY` 加 `host` 值强制走 host staging（`:492-527`），
      但该路径**不是有效对照** —— 5 次全部 FAIL，且连第一个「fresh conversation」场景都产出乱码（`got [15 15 …]`、
      `expected [548 271 1919 5686 …]` 这类跨运行完全不同的输出），说明 host-staging 的 D2H 并未覆盖生产 kernel（该回退
      路径长期未用、已失效）；因此 A/B 结论为**无效**，跨卡传输嫌疑**未排除也未证实**。该临时开关已回退（`device_pair.cu`
      在最终工作树里无 diff），不留在源码里；
    - 判定落地前**不改钳位轮语义**（等于改兜底/接受语义），也不打开保留；在那之前不做 §3.6 的 B7 提速。
  - **本轮收口（试验 (b) 失败 ⇒ 路线回到构造期拒绝；取代上方 429–432 行与本条早期版本）**：按产品决策先实现了选项 (b)（把
    DFlash2 从**复用扫描本身**摘出去、构造期 `host_checkpoint_stride_=0`、路线放行、测试断言 `reused=0`），但**实测证明前缀复用
    不是成因**，故全部 (b) 改动已回退（`git checkout 61cd270d -- src/runtime/engine/tp2_generation_core.cpp/h
    src/runtime/engine/model_instance.cpp tests/models/qwen3_5/test_tp2_sessions.cpp docs/serving.md`），路线恢复为**构造期拒绝**，
    B1–B6 实现与测试照旧保留、`draft_context_declined` 契约与 B6 的 draft 状态搬运代码不动；拒绝消息与注释、`docs/serving.md`、
    测试头注释的理由已同步改为「同一配置下 verify 走法逐 run 不同」，不再归因前缀复用。
    **否定性证据**：`NINFER_TEST_ROUTE=dflash2` 连跑 5 次为 `P F P P F`（`build-win/b6c-dflash2-1..5.log`），失败仍是
    `shared_b`（`test_tp2_sessions.cpp:406`）的 `got [1703 220 248046 198 248045 198 248045 198]` vs `…220`——与复用开启时
    **逐字符相同**；而 :401 的复用期望（DFlash2 = 0）在失败run 里也通过，说明那几次确实没有复用。⇒ 上一轮「锁定到复用走法」的结论
    **被本次实验否定**，B5「DFlash2 不参与复用即确定」的前提同样不成立。
    **为什么这次能定**：把 DFlash2 从复用扫描里摘干净后（`reused=0`），坏值与复用开启时逐字符相同，失败发生在**两次全量
    prefill** 之间 ⇒ 复用不是成因。新的定位：同一进程、同一批选项下，**两个 Engine 实例的 DFlash2 走法会给出不同结果**——oracle
    实例（先构造）在 10 次运行里稳定 `220`，engine 实例（后构造）在 10 次里翻转 6 次；此前 10 次「有复用」运行同样只由 engine
    实例出错（`build-win/b6-det-1..5.log`、`b6-fin-1..5.log`）。plain/MTP 同场景 5+2 次全部逐位一致；钳位轮索引链与位精确
    allreduce 探针（`tests/test_tp_device_pair.cpp`：每规模 64 次换数据 + 200 深度排队，含 20 KiB / 1 MiB / 80 KiB 窗口层 /
    2433024 B 的 `[V,8]` 5-slice 合并，`DevicePair(0,1)` 5 规模全部逐位一致 PASS、3.5 s、`p2p_available=0`）都已排除 ⇒ 剩下的
    是「同一配置下逐 run 不同的那部分状态或时序」。
    **重新放行的前置条件（下一步实验）**：在失败run 里把 engine 与 oracle 的 `shared_b` 逐轮对齐——`tp2_generation_core.cpp`
    的 proposal/verify/argmax 段（:2813-2941）逐轮比较 licensed 列、verify 窗口 logits 与 KV/GDN 状态；并交换两个引擎的构造顺序
    （oracle 后建），看差异是否跟随「后构造的实例」，以区分分配布局相关与走法本身相关。修好后再把复用断言按路线恢复。
    其它验收（在 (b) 二进制上测得；回退后 plain/MTP 行为不变）：plain 2/2、mtp 2/2 逐位复用断言通过；append K=7
    `0xbad27a494a9bc853`、K=5 `0xbee487264ca8ffb8` 未变；load 通过；5 次 dflash2 日志去掉 `[mem]` 行后，通过run 互为逐字节
    相同（SHA256 `324835E9…`），失败run 互为相同（`8CF800…`）。
    性能：**未重测 decode tok/s**——本轮改动只落在选项归一化与复用扫描，未触及任何 decode/attention/head/kernel 文件，复用只
    影响 prefill，故 B5 记录的 TP-2 DFlash2 K=7 63.6 tok/s vs plain 35.1 tok/s（=1.81×；4096 context、单请求、greedy）不受影响。
    **单实例确定性地基实验（下一轮第 1 条，已做）**：新增探针 `tests/models/qwen3_5/test_tp2_dflash_solo.cpp`
    （目标 `ninfer_qwen3_5_tp2_dflash_solo_test`）——**单进程只构造一个 Engine**，用该引擎自己的答案续写并重放会话场景的请求序列
    （opening → other_a → a_continued → other_b/c/d → a_continued(evicted) → shared_a → shared_b），打印每次走法的 tokens 与 fnv1a 摘要。
    临时打开路线后**独立进程连跑 10 次**（`build-win/b6f-solo-1..10.log`），`shared_b`（reused=512）依次是：
    7 次 `[1703 220 248046 198 248045 198 248045 198]`（digest `0x4bcc3994a5efba7d`）、
    1 次 `… 248045 220]`（`0x4bcc3f94a5efc4af`）、1 次 `… 248045 248046]`（`0x49045194a1360b45`）。
    ⇒ **单实例、单进程的产品形态本身就不确定**，且错值不止一个 ⇒ 不是「两个 Engine 同进程并存」的 harness 假象，**确属产品缺陷**，
    也说明该 token 的 top-2/3 logits 只差 1 个 bf16 ulp 量级。门禁因此保持关闭；探针在路线被拒时按 SKIP(77) 处理，保留在树内，
    作为重新放行时的验收工具。
    **下一步（第 2–4 条）**：(2) 交换两个实例的构造顺序（oracle 后建）看差异是否跟随「后构造的实例」，并在失败run 里做
    engine/oracle 逐轮对齐（`tp2_generation_core.cpp:2813-2941` 的 proposal/verify/argmax 段）；(3) 投毒法逐个排查候选缓冲
    （draft ring / pending_features / round arena / GDN records / 快照槽 / paged KV），找出未被完全覆写的那一处；(4) 把 oracle 换成
    **实验 A（轮次序列切分，已做，决定性）**：把 DFlash2 每轮的 proposal 窗口与 target 许可前缀按轮打印（临时钩子 `NINFER_TP2_DIAG_ROUNDS`，
    已回退），solo 探针连跑 6 次（`build-win/b6h-round-1..6.log`）：
    - 每次都是 **48 轮，逐轮文本完全一致，只有最后一轮（`shared_b` 的第 5 轮、position=646）不同**；
    - 该轮 6 次运行的 **draft 窗口逐字节相同**：`window=[248045 248045 248045 248045 248045 248045 248045 248045] licensed=1`；
    - 只有 **target 许可/提交的 token 不同**：5 次 `step=[198]`、1 次 `step=[248046]`。
    ⇒ 按切分规则：**扰动在 target verify 侧，不在 proposal/draft 侧**；而且它只在这一个轮次上表现出来（其余 47 轮完全一致，说明整段状态轨迹
    逐位相同、连接受前缀都一样）。该轮窗口全为同一 id（草稿塌缩/钳位轮），`licensed=1` 表示目标自己换掉了草稿 ⇒ 表决的是**该轮第 0 列 logits
    的 argmax**，在 `198` 与 `248046` 之间以约 1 ulp 的差距翻转。
    **判断与剩余可能**：状态轨迹（KV/GDN/ring）逐位相同 ⇒ 不是状态搬运、不是复用、不是草案路径、不是跨卡 allreduce（位精确探针）；嫌疑落在
    该轮 target 窗口计算里**只影响末列 logits 的一处非确定读/归约**（未初始化或被上一轮残留污染的 workspace、verify 窗口缓冲、
    `verify_window_host_`、或那张全同窗口下被钳位的行）。下一步（本轮未做）：投毒法（B，候选缓冲写图案看谁没被覆写）、构造顺序交换与
    plain-oracle 控制组（C）、限时 `compute-sanitizer --tool initcheck`（D）；A 已经把范围压到「最后一轮第 0 列的 logits」，
    最快的下一步是在该轮 dump 该列 top-k logits 与全部中间量（`window_hidden` 哈希、KV 行哈希）逐 run 比对。
    **实验 1b（末轮 logits/hidden dump，两次尝试都未成立）**：加临时钩子 `NINFER_TP2_DIAG_LOGITS`（打印该轮第 0 列 top-5 与
    `target_hidden` 的哈希），6 次运行每次都在**第一轮**失败：`tp2_generation_core.cpp:2924: CUDA_CHECK(cudaMemcpyAsync(host_logits.data(),
    frame.target_logits.data, logit_count * sizeof(float), D2H, ...)) failed: cudaErrorInvalidValue`；先用视图指针（`window_logits.data`）、
    后用基张量指针（`frame.target_logits.data`）都一样 ⇒ `vocab * width * sizeof(float)` 与该张量的实际分配不符（很可能它不是
    `[vocab, width]` 的 fp32，而是本地半区 / bf16 / 只含末 token 的 merge 结果）。**要做「输入缓冲 vs 计算内部」的切分，先得拿到该张量自己的
    字节数或元素类型**（Tensor 的 size/bytes 字段，或逐步缩小拷贝长度试探）。日志：`build-win/b6j-logits-1..6.log`（视图指针版）、
    `b6k-logits-1..6.log`（基张量版）。
    **本轮按有界收口停止深挖，门禁保持关闭。**剩余可能（均未验证）：(i) 该轮 target 窗口里一处**未初始化、或被上一轮残留污染的 workspace 读**
    （尤其全同窗口、`licensed=1` 时与其它列不同的分支/钳位路径）；(ii) 两卡窗口前向里一处**跨 stream 的写读序缺口**（时序相关 ⇒ 低频、每轮都存在、
    只有末轮因 logits 恰好并列才显形）；(iii) 该轮 head/merge 归约的非确定顺序。区分它们需要的新手段：拿到 `frame.target_logits` /
    `frame.target_hidden` 的真实字节数后做跨 run 哈希；限时 `compute-sanitizer --tool initcheck` 跑 solo 探针；或对窗口前向做逐 kernel 的
    race 检查（nsys/sanitizer）。
    临时钩子已用 `git checkout -- src/runtime/engine/tp2_generation_core.cpp src/runtime/engine/model_instance.cpp` 干净回退（工作树对这些文件
    **诊断轮（1c 张量事实 + 工件对照）**：
    (i) **张量事实**（`src/models/qwen3_5/program/round_buffers.cpp:214-218`）：`target_logits` = **BF16** `{output_rows=248320, columns, batch=1}`、
    `target_hidden` = **BF16** `{hidden=5120, columns, batch}`；`tp2_generation_core.cpp:2844` 视图里的 `vocab` 是**全词表 248320**。
    前两次 dump 用 `vocab*width*4`（假设 fp32）是 2 倍超界，`cudaErrorInvalidValue` 由此而来；改成 **2 字节/元素后拷贝成功**。
    (ii) **但该 dump 的内容与 target 判定不一致**：host 端 top-5 的最大值只有 7.6–12.8、id 每次不同，而同一轮 `frame.target_argmax` 稳定是 `[198 …]`
    ⇒ 读到的字节**不是 argmax 读的那批行**（写出端布局/偏移问题）。所以「输入缓冲不同 vs 计算内部非确定」这条切分**仍未成立**，
    得到的 hash 不足以作结论（跨 run 变化只说明该 buffer 含未被写过的字节）。下一步需要的新手段：读 head 写出端
    （`project_head_tp2` / `merge_local_row_blocks`）确认 stamp 的行范围与布局，再只对**判定真正读到的行**做哈希。
    (iii) **工件对照**（同一二进制、同一场景、临时开门禁跑 solo 探针，每件 10 次；`build-win/b6n-{a,b,c}-1..10.log`）：
    a) `qwen3_8_27b_w4a4_w8a8_dflash2`（用户转换）：9× `[1703 220 248046 198 248045 198 248045 198]`（digest `0x4bcc3994a5efba7d`）
    + 1× 尾 token `6558`（`0x4bb38194a5c5b9d5`）⇒ **翻转**；
    b) `qwen3_8_27b_w4a4_dflash2`（同模型另一种 w4a4）：**10/10 完全相同**（`0xad284a4b774b1cc3`）⇒ 不翻转；
    c) **官方工件** `qwen3_8_27b_nvfp4`（README.md:18、docs/performance.md:22）：**10 次出现 4 种输出**（6× `2523`、1× `248046`、1× `3710`、2× `7734`）
    ⇒ **官方工件同样翻转**。按预设解释规则：**「同二进制同输入必同输出」的引擎缺陷被官方工件坐实，与用户转换无关**；b 件稳定只说明暴露面
    与该件的数值/并列位置相关。末轮候选累计出现过 7 个不同 token（198/220/248046/6558/2523/3710/7734）⇒ 该处 logits 间距在 bf16 量化最小刻度
    附近（本 dump 未能给出可信数值）。
    **诊断轮 2（head 写出端 + 有效行 dump）**：
    (a) **head 写出端事实**：`src/models/qwen3_5/execution/text.cpp:1608-1641`（`project_head_tp2`）——`vocab = dimension(config_.vocab_size)`（=248320）、
    `local = dimension(lm_head_->weight.n)`（=124160，每卡半区）；`local != vocab` 时每卡只把自己那半投到 `partial [local, columns]`，
    再 `merge_local_row_blocks(...)`（:1640）把两半按词表偏移 stamp/sum 成完整 `[V,T]`；`program/round_buffers.cpp:214-218` 把
    `target_logits` 定为 **BF16 {output_rows=248320, columns, batch}**。
    (b) **有效行**：`tp2_generation_core.cpp:1888-1892`——lm_head 按 `vocab` 行打包，但只有前 `public_tokens` 行真实（运行时实测 **248077**）；
    `src/ops/wrapper/argmax.cpp:37` 的 `argmax(logits,out,valid_rows,stream)` 要求 rank-2 `[vocab,T]`（:45）并用同一 valid_rows
    ⇒ :2911 的 argmax 只扫 **[0, 248077)**。上一轮扫 `[0,248320)` 读到 head 永不写的打包行，这就是 dump 与 `target_argmax` 不一致的原因。
    (c) **只扫有效行后仍不一致**（`b6p-logits-1..8.log`）：8 次 `pos=646 valid=248077 hidden=5120`，host top0 为
    `31074/217312/125935/217282/62082/…`（bf16 `0x40e3–0x4126`，`gap_ulps=2..18`），而 8 次末 token 全是 `198`、`target_argmax=[198 …]`；
    `logit_hash`/`hidden_hash` **每次运行都不同**。
    (d) **双拷贝判别**（`b6q-logits-1..4.log`）：同一轮连续拷两次并同步，4/4 `same=1`（hash 与 top0 完全一致）⇒ **不是并发写者覆写（非 race）**，
    而是我读到的地址/那一槽**不是 argmax 实际读的那块内存**。
    ⇒ 切分仍未成立。下一步必须先在模型侧确认窗口 verify 的 head 把 logits 写进哪个 tensor/arena 槽（`forward_tp2_window` 入口、
    `program/speculative/target_verification.cpp:18-41`、`round.frame()` 的槽位绑定），再用那个真实指针做跨 run 哈希；或限时
    `compute-sanitizer --tool initcheck`。钩子已回退，门禁保持关闭。
    **诊断轮 3（P1「同状态重跑同一轮」，未成立）**：在 :2911 argmax 之后立刻再跑一次 `round.make_verify_sink()` +
    `run_verify_window(...)` + `ops::argmax(...)`，比较两次的 `frame.target_argmax`（临时钩子 `NINFER_TP2_DIAG_RERUN`，已回退）。
    4 次运行（`build-win/b6t-rerun-1..4.log`）：所有 48 轮都 `equal=1`，**但探针本身改变了走法**——`shared_b` 输出从
    `[1703 220 248046 198 248045 198 248045 198]` 变成 `[1703 220 16 15 15 15 15 15]`，且走法在 `pos=643` 提前结束
    ⇒ 第二次 verify 不是「同状态重跑」：KV 写虽幂等，但 `make_verify_sink()` 的残差收集是**追加**语义，round 的 pending staging
    被写了两遍 ⇒ **`equal=1` 不能作为「计算内部确定」的证据**（诚实结论：本判据未成立）。
    ⇒ P1 要成立，先要有一个**覆盖 pending feature staging 的轮状态恢复原语**（现有 `snapshot_state(..., kRoundScratchSlot)` 只覆盖
    KV/GDN 一类），或能在第二次跑前重建 pending staging；这正是下一步需要的具体新手段。
    **诊断轮 4（P1b 同 sink 重跑与 initcheck，两者都未成立）**：
    (a) **P1b**：第二次 verify 复用 round 自己的 sink（不新建 `make_verify_sink()`），自检**不通过**——4 次运行（`build-win/b6w-rerun-1..4.log`）
    的 `shared_b` 仍是 `[1703 220 16 15 15 15 15 15]`（正确值 `[… 248045 198]`），轨迹照旧塌缩；`mismatch=20` 且 4 次完全相同（系统性而非随机）
    ⇒ 两次之间的**状态差异是探针自己造成的**（第二次 verify 改写了后续轮次真正读到的状态，复用 sink 不足以消除），所以 `mismatch`
    **不能**当作「计算内部非确定」的证据。要做出 P1，必须先有**覆盖 KV 记账/pending staging 的轮状态恢复原语**。
    (b) **initcheck 在本机不可用**：`compute-sanitizer --tool initcheck`（`build-win/b6v-initcheck.log`）报
    `Failed to initialize WDDM debugger interface. Please run EnableDebuggerInterface.bat as an administrator` 与 `Device not supported`，
    `ERROR SUMMARY: 4 errors` 全是这两类；探针本身仍正常 PASS（43 s、`shared_b` 原值、digest `0x4bcc3994a5efba7d`）⇒ P3 在本环境被排除
    （需管理员权限 + 设备支持）。⇒ **P1 与 P3 均不成立，门禁保持关闭**，按有界收口停手。
    **诊断轮 5（S1「彻底同步」实验，**假设成立**）**：只在 DFlash2 分支、在 `run_verify_window(...)`（`tp2_generation_core.cpp:2908`）之前插入一次性彻底同步
    **收口轮（S1 修复落地 + 最终二进制验收；结论：未完全稳定，门禁保持关闭）**：
    **收口轮 2（D 探因 + N=50 快速失败协议；未通过，门禁保持关闭）**：
    (1) **D（窗口前向再下一层）**：读 `src/models/qwen3_5/execution/text.cpp:2162-2222`——每卡用**自己的 arena** 绑窗口（`make_bind` → H2D
    `copy_i32(ids/positions)` 在该卡自己的 stream 上），紧接着 `embedding_tp2` + `run_layers_tp2`（配对层序列，peer 交换走
    `src/core/tp/device_pair.cu` 的核内 mapped-host 握手），**中间没有任何 host 侧 staging/事件**；两卡 stream 在进入窗口前已被每轮有界同步排空
    ⇒ **host 侧已无「最小必要依赖」可加**，残余不确定性在配对核内协议/归约里（需逐 kernel 查序，或在设备受支持/有管理员权限的环境跑 sanitizer）。
    (2) **N=50 快速失败协议**（`build-win/b8-solo-*.log`，脚本遇首个不一致即停）：**第 10 次失败**——digest `0x4bb38194a5c5b9d5`、
    tokens `[1703 220 248046 198 248045 198 248045 6558]`（其余 9 次均为 `0x4bcc3994a5efba7d`）。有界同步后三批汇总：`b6y`
    (12 次 0 失败) + `b7b` (12 次 1 失败) + `b8` (第 10 次失败) = **34 次 2 次失败 ≈ 5.9%**（95% Wilson 区间约 1.6%–19%），
    对照无同步基线约 20–30% ⇒ 该同步是**真实但不完整的修复**。
    **统计界限（如实）**：即使将来 0/50，95% 置信下也只能界定每 run 翻转率 ≲6%，不能说「已证明确定」；本轮实测 ~6% 已直接否证稳定性。
    (3) **未放行**：构造期拒绝保留，sessions/服务文档/refusal 文案/solo 转验收均未改；**K=7 decode tok/s 未测**。
    **下一步**：对 `run_layers_tp2`/`device_pair` 的核内握手做逐 kernel 读写序检查（谁读了 peer 尚未写完的数据），或换到受支持设备/管理员环境跑 initcheck；
    另可评估 (2) 的选项 B「近似并列保护」（语义新增，须单独声明触发比例/额外耗时/验收）。
    (1) **落地**：把 S1 的彻底同步（两卡 `cudaStreamSynchronize` + `cudaDeviceSynchronize`，只在 DFlash2 分支、`run_verify_window` 之前，
    `src/runtime/engine/tp2_generation_core.cpp`）作为**有界每轮同步**写进工作树（无 env 钩子；MTP/plain 不受影响）。
    (2) **最终二进制 12 次协议**（`build-win/b7b-solo-1..12.log`）：**11/12 逐字节一致**——`shared_b` 全为
    `[1703 220 248046 198 248045 198 248045 198]`、loghash `175A56CC3A333FA4`（11 次），**z-10 仍翻转为** `[… 248046]`
    （loghash `6F0194D3FCF00981`）。对照 S1 批次（`b6y-sync-1..12`）当时是 12/12 ⇒ 该同步把翻转率从基线（10 次 2–3 种）显著压低
    （两批合并 24 次仅 1 次），但**没有根除**：跨 stream / 跨卡写读序缺口是**主因之一**，残余来源仍在。
    (3) **同批其它验收（全部通过）**：sessions plain `recall reused 71 prompt tokens bit-identically … passed`、mtp `passed`、
    append K=7 `0xbad27a494a9bc853`、K=5 `0xbee487264ca8ffb8`、load `TP-2 dual-shard load passed devices=0,1 layers=64
    head=[124160,5120] vocabulary-parallel`、`ninfer_tp_device_pair_test` **PASS**。
    (4) **未放行**：构造期拒绝保留（`model_instance.cpp` 回到 HEAD），sessions dflash2 期望、`docs/serving.md`、refusal 文案、
    solo 探针转验收均未改；**K=7 decode tok/s 未测**（修复未通过验收，不进入性能验收）。
    **下一步**：对 `forward_tp2_window` 逐 kernel 查读写序（哪个 kernel 读了 peer 尚未写完的数据），或在有管理员权限/受支持设备的环境
    跑 `compute-sanitizer --tool initcheck`；12/12 稳定后再做 (4) 的放行清单。
    ——`shard_a_.device.bind_to_current_thread(); cudaStreamSynchronize(a); cudaDeviceSynchronize();`，对 `shard_b` 同样一次，再 bind 回 a——临时钩子
    `NINFER_TP2_DIAG_SYNC`（**已回退**）。solo 探针**连跑 12 次**（`build-win/b6y-sync-1..12.log`）：**12/12 逐字节一致**——
    `shared_b` 全为 `[1703 220 248046 198 248045 198 248045 198]`、digest 全为 `0x4bcc3994a5efba7d`、去掉 `[mem]` 行后整篇日志
    SHA256 前 16 位全为 `175A56CC3A333FA4`。对照（同一二进制/探针/工件、**无**同步）：`b6f`（10 次 3 种末 token）、`b6n-a`（10 次 2 种）。
    ⇒ **确认（至少是主因）是 DFlash2 verify 窗口前的跨 stream / 跨卡写读序缺口**，与 B5 修掉的 D2H 未同步属同一类
    （MTP 的窗口由 host 直接写，所以从不暴露）。
    **下一步（第 2 条，未做，需要新的一轮）**：把这次「彻底同步」换成**精确依赖**（事件，或 `pair_`/`adopt_peer` 已有的同步原语，只对 DFlash2 生效），
    然后按放行清单验收：solo 连跑 10 次逐字节一致、sessions dflash2 5 次全过、append K=7 `0xbad27a494a9bc853` / K=5 `0xbee487264ca8ffb8` 不变、
    load 通过、给一个 K=7 decode tok/s 数字（预期微秒级开销、无可测损失），全绿再撤构造期拒绝、改 serving.md/PLAN/refusal 文案并把 solo 探针转为验收用例。
    本轮的临时同步与开门禁改动都已回退，门禁保持关闭。
    回到 HEAD）；重建后 `NINFER_TEST_ROUTE=dflash2` refusal exit 0、solo 探针 exit 77。
    同进程的 plain engine 作控制组，判定是「两个 DFlash2 实例互相干扰」还是「任意第二个实例都受影响」。
    临时打开路线的改动已还原：`model_instance.cpp` 恢复构造期拒绝并重建验证（`dflash2 exit 0`、solo probe `exit 77`）。
- **B6 状态与保留（见上方 B6 结果）**：draft ring/pending features 已随会话召回保存恢复（复用 MTP 的 host slab
  先例），`dflash_graph_profiles` 已接；「会话切换后召回逐位一致」对本路线不成立，保留保持禁用；
- **B7 性能验收**：双卡吞吐对 90–180 tok/s 目标 + 与 MTP 的对比 + plain/mtp 无回归。
- **B7 前置侦察（已完成，未实施；路线放行后才可交付）**：DFlash2 verify 目前 eager 的原因**不是** capture-illegal（单卡路线已把同一个 `DFlashFeatureSink` 捕进 CUDA Graph，`draft.cpp:712-718` + `program/graphs.cpp:373-421`），而是 4 处未接线：
  ① 总开关只对 MTP 生效（`tp2_generation_core.cpp:374`）；② verify 桶只在 MTP 分支建（`:387-400`）；
  ③ `capture_verify_graph` 不收 sink（`:910-912`，**必须补**，否则图里没有 scatter 节点、`pending_features` 会静默缺列、只掉接受率）；
  ④ `:979-984` 有一条以「未固化 host capture 调用」为由的 `throw`，其前提已被单卡路线证伪。
  最大风险：建桶会让 **eager 分支也改用 bucket envelope**（`:960-971`），而 DFlash2 的输出对 verify 窗口布局敏感 ⇒ 必须先做「只切 envelope」的贪心逐字节 A/B，再做 graph on/off，否则同时改了两个变量。
  判据：graph arm 的 verify ≈ 30.6 ms（MTP 同宽实测）、比 eager 低 4~4.5 ms/轮、等输出吞吐约 +15%；硬前提 `pair_.in_kernel_allreduce()`；若否决 envelope 变化则捕获走不通且无等价替代（可动的只有 <0.1 ms 的 D2H+sync）。

**根因定案与修复（A 轮，2026-09-22；取代上方各诊断轮的"剩余可能"）**：钳位轮不可复现的根因已定位并修复——
**TP-2 的 `forward_tp2_window` 从未绑定 `active_valid_columns_`/`active_sequence_batch_`** ⇒ `attn_mix` 走非 batch 分支、KV append
完全无掩码（`small_t_fp8` 以 `Masked=false` 实例化 ⇒ `valid_tokens=TokenTile`，窗口每列都写 KV）。而 DFlash2 的预算钳位窗口把尾列钉在
最后有效列的**同一绝对位置**（`speculative_round.cuh:36` `positions[off]=base+(j<=extent?j:extent)`；TP-2 侧 `tp2_generation_core.cpp:2821-2826`
同义）⇒ 同一次窗口前向里多个 warp 并发写**同一个 paged-KV 槽**（FP8 code + scale 逐字节撕裂、last-writer-wins）⇒ 该槽内容运行间不确定 ⇒
本轮与后续所有读到该槽的 logits 抖动 ⇒ 近似并列处 `target_argmax[0]` 翻转。该机制完整解释全部已知事实：翻转只出现在钳位轮
（`extent<k`，即预算尾部轮次）；失败有偏（warp 交错有偏）；错值多变（8 列 K/V 皆可胜出）；构造顺序影响翻转率（时序/L2 状态）；
官方工件同样翻转；S1 同步只改交错统计不根除；单卡合格（`target_verify_batch_impl` 绑定掩码 ⇒ 尾列不写 KV）；MTP/plain 稳定
（MTP 窗口位置连续 `:3002-3008`、plain T=1，均无共享槽）；两个 Engine 实例翻转率不同。**本轮已排除**：`ar_inplace_bf16` 握手
（双缓冲奇偶 + 自旋发布门控，时序闭环）、`argmax`（固定全序 CAS，逐位确定）、`speculative_accept_sparse_drafts`
（`raw_greedy`/greedy-无惩罚直读 `target_tokens`；管线路径 `col>extent` 早退使 finalize 计数恰为 extent+1，extent=0 时读自己写的槽）、
split 归约器（固定顺序无 atomic）。**修复（已落地，验收中）**：① `attn_mix` else 分支与 batch 分支同式消费 `active_valid_columns_`；
② `forward_tp2_window`/`run_verify_window` 新增 `valid_columns`（= `target_valid_columns[0]` = extent+1，钳位列禁写 KV、其 logits 置零），
仅 DFlash2 调用传入；MTP 传 0（不绑掩码、原路由逐位不变）。
**A 轮验收（2026-09-22，全绿）**：① load ✓；② append 哈希**不变**（K=7 `0xbad27a494a9bc853`，prefill/proposal 侧未受修复影响，
无需重基线）；③ sessions plain/mtp ✓（recall 71 逐位，无回归）；④ solo 探针独立进程连跑 **10/10 逐位一致**（digest 全为
`0x4bcc3994a5efba7d`、9 条 walk tokens 全同；修复前 solo ~25%/run 翻转，10 连同概率仅 ~5%，证据充分）——其中
**召回逐 token 复现**的直接证据是探针内配对：`a_continued`（复用 71、draft declined）与 `a_continued_evicted`（从头 prefill）
8 token 全同；⑤ `NINFER_TEST_ROUTE=dflash2` 真场景连跑 **5/5 全过**（注意：当时三条 recall 均落 declined 边界，断言按 B6 中期
契约只钉首样本；shared_c@512——即 B6 翻转位——也只钉首样本）。B 轮已把 declined 边界与 shared_c@512 **收紧为全 token 对比**
作为放行回归；若全过，则「DFlash2 召回逐 token 复现 from-scratch」定案，B6「未通过项」根因即本缺陷。验证用临时改动（门禁放行、越过 sessions 绊线、route_keeps_retention 切 true）
均已回退，正式树保持门禁关闭。**留给 B 的升级点**：撤销门禁与绊线；`route_keeps_retention(DFlash2)` 可切 true（保留属性已实证）；
`draft_declined` 的 declined 边界目前仍只钉首样本，可尝试收紧为全 token 对比。

**B 轮（放行，2026-09-22）**：门禁与 sessions 绊线已撤（`model_instance.cpp` 只留 DFlash v1 拒绝，文案改「--spec mtp and
--spec dflash2」）；sessions 默认三路由、`route_keeps_retention` 收拢为全开（其三元期望分支一并删除）；solo 转验收用例（去拒绝
跳过，运行内加 recalled-vs-evicted 首样本断言）。**收紧实验（关键新证据）**：把 shared_c@512 与 declined 边界从首样本钉收紧为
全 token 对比后，shared_c@512（B6 翻转位）**全 token 通过**——A 轮修复在原分叉位实证；但 declined 边界的 rerender 场景在第 3 个
token **稳定分叉**（5/5 逐 token 完全相同：got `[2752 11 220 16 24 24 15 82]` vs oracle `[2752 11 198 220 220 16 15 15]`），非残余
抖动。机理：declined 请求跑 target-only 轮（1-valid 钳位窗）vs oracle 全 draft 轮（8-valid 窗）＝不同执行形状，注意力 KV 归约分片
随窗口尾位而变、写出的 KV 行永久携带 ulp 差，近并列轨迹按「draft pattern 改变即漂移」类各自稳定（与 MTP≠plain 及 tp2 文档「a
changed draft pattern shifts the trajectory on which near-ties are resolved」同类）。**结论**：declined 边界恢复首样本钉（边界穿越
logits＝保留真正携带的属性），其尾部可复现性由 solo digest 跨进程保证；aligned 边界保持全 token 对比。append K=7/K=5 与 ring
哈希同基线（`0xbad27a494a9bc853`/`0xbee487264ca8ffb8`/`0x1a53fd824cd4e360`）。文档已同步（tp2-dual-5060ti.md 新增 DFlash2
小节、Verification 表三行；serving.md TP-2 限制段重写）。
**B 轮验收（全绿）**：sessions 默认三路由 exit 0；`NINFER_TEST_ROUTE=dflash2` 5/5 全过；solo 10/10 全同 digest
`0x4bcc3994a5efba7d`；append K=7/K=5 哈希同基线；load ✓。**K=7 decode tok/s（bench_serve 合成负载）= 20.2–20.7 tok/s**，
同负载 MTP K=2 对照 66–72 tok/s；分解可见**每轮成本 ~48.8 ms/轮与 B5 口径一致（无回归）**，差异全在接受率（合成填充文本
~1.0 tok/轮 vs B5 口径 4096-context 贪心 3.10 tok/轮 = 63.6 tok/s）。另一发现：DFlash2 的 verify step 仍是 eager（MTP 已 graph，
日志 `[tp2-graph] verify step: eager`）——即 C（B7 图接线）的首要抓手。`serve.ps1` 增加 `-Spec mtp|dflash2` 使放行路线端到端可用。

**理论性能差距（单卡 5090 vs 双卡 TP-2，DFlash2 K=7；roofline 合成，非实测）**：

- **硬上限＝带宽比 2×**：两卡合计 896 GB/s = 一张 5090 的 1,792 GB/s 的一半（`docs/tp2-dual-5060ti.md:106-108`）。
  实测每 shard 每前向流 10.15 GB ⇒ 地板 10.15/448 ≈ **22.7 ms/轮**；5090 地板 ≈ 20.3/1792 ≈ 11.3 ms。
- **两条路线的效率实测相当**：TP-2 MTP K=2 是 38.2 ms/轮 ≈ 59% 地板；5090 DFlash2 192.5 tok/s、
  acc 37.0% ⇒ 3.59 token/轮 ⇒ 18.6 ms ≈ 61% 地板 ⇒ 差距主要就是带宽比。
- **DFlash2 特有的不利项**：draft（2.07 GiB）只在 shard 0 ⇒ 2.22/448 ≈ **5.0 ms/轮**（5090 上 1.24 ms，四倍）。
- **DFlash2 特有的有利项**：MTP 在 TP-2 上最大的单项开销是 **7–10 ms 的 host 串行 draft 链**
  （同文档 `:127-128`）；DFlash2 一次非因果掩码块前向并行产出全部提议 ⇒ 这条开销消失。
- **估算**：(22.7 verify + 5.0 draft + 2.9 启动间隙 + ≈2.5 allreduce) ≈ **33 ms/轮 ÷ 3.59 ≈ 108 tok/s**
  ⇒ 约 5090 的 **0.56×**，但对本机 MTP K=2 的 57 tok/s 是 **≈1.9×** —— 明显大于单卡上的 +19.5%，
  因为被省掉的那条链在 TP-2 轮次里占比最大。
- 对归档 **90–180 tok/s** 的修正：90 这一端与 roofline 相符；**180 不可达** —— 即使把启动间隙、allreduce、
  draft 全部归零，理论上限也只有 3.59/0.0227 ≈ **158 tok/s**。
- **D1 推论**：把 draft 切到两卡只能省约 1.5 ms/轮（≈5%），收益远小于复杂度 ⇒ **D1 只为内存做，不为速度做**。
  提速杠杆是恢复 CUDA graph（−2.9 ms）、减少/融合 allreduce、draft 权重降精度。

**C 轮（B7 verify CUDA Graph 接线，2026-09-22）**：B7 前置侦察的 4 处未接线全改：总开关 `:374` 对 `mtp||dflash2` 生效；
verify 桶在两分支共建（DFlash2 用 `dflash_graph_profiles`、目标包络 `{1, min(max_context, planned.max + draft_window + 1)}`）；
`capture_verify_graph` 收 `sink`/`valid_columns` 并转发进捕获体；`throw` 撤除。钳位 extent 走 pinned 缓冲区尾部 int
（`[ids(W), positions(W), valid(1)]`），捕获图以 memcpy 节点逐重放重读——**不能用 `set_i32_scalar`**：它把值烘进
`set_i32_scalar_kernel` 的启动参数，重放会永远钳在捕获时的 extent。分两段 A/B（只动一个变量）：①「只切 envelope」（eager +
桶包络）贪心逐字节 A/B 全绿（solo digest `0x4bcc3994a5efba7d` ×2 + sessions dflash2 过）⇒ 桶变量无害；② graph on/off 发现
**图路径 2/6 翻转**（shared_b 尾 token 取值 `248045 198/220/13962` 三元组恰为 A 轮未掩码竞争的指纹）而 eager 0/13+。
**C2 根因**：`capture_verify_graph` 签名先加了 `sink`/`valid_columns` 参数，但**捕获体里 `forward_tp2_window` 的调用没转发**——
默认参数 `nullptr/nullptr` 让编译静默通过 ⇒ 捕获序列＝「无掩码＋无 sink」的旧窗口 ⇒ A 轮修复的钳位 KV 追加竞争在图路由复活
（KV 撕裂 + `pending_features` 静默缺列）。修复＝捕获调用转发 `sink, valid_columns`（tp2_generation_core.cpp `capture_verify_graph`）。
**C4 验收（修复后）**：solo 图路径独立进程 **10/10 逐位一致**（全 `0x4bcc3994a5efba7d`，修复前 2/6 翻转）、`NINFER_TP2_VERIFY_GRAPH=0`
eager 同金值（graph≡eager 逐位）、sessions dflash2 4/4 过（recall 71 逐位）、sessions 默认三路由过（plain/mtp 无回归）、
append K=7/K=5/ring 三金值不变。**教训**：「签名先行、体转发后补」的分段编辑必须
成对落地；捕获路径上的默认参数是静默降级陷阱（错误形状合法、结果只在近并列处漂移）。
**C5 性能（bench_serve 合成负载，同构建 graph/eager A/B）**：等输出 128-token decode@8192 **6267→5627 ms（-10.2%，折算 -4.9 ms/轮）**、
20.3→22.6 tok/s；decode_short 20.6→23.0 tok/s（+11.7%）；prefill 三段持平；MTP K=2 对照 67.9/64.8 tok/s（B 基线带 66–72 内）无回归。
轮时差与 MTP 窗口图的 -4.3 ms 同量级（B7 判据 −4~4.5 ms/轮达标）；合成文本接受率 ~1 tok/轮 ⇒ 吞吐增益 +11%（判据 +15% 为 MTP 口径推算，
实测口径下合理）。

**E 轮（MTP draft 链图化，进行中）**：目标＝把 MTP 的 draft 链（K=2 实测 7–10 ms/轮，docs/tp2-dual-5060ti.md:127-128 的最大单笔开销）
整链捕获进 CUDA Graph（llama.cpp 同款：整轮含链全图）。选它的理由：纯启动方式变化、**不动数值契约**（C 轮已证 graph≡eager 逐位）、
复用 C 轮全部机制（WindowGraph/桶/pinned 输入/memcpy 节点）。步骤：① 侦察 `mtp_forward_batch`/`mtp_forward_ar_step`（text.cpp）的捕获面——
per-round 值（:1119/:1121 两个 `set_i32_scalar` 必须改 pinned 传递，C 轮教训）、host 同步点、采样/状态地址稳定性；② 整链一张图/桶
（K 步 + D2D hidden 接力 + `increment_i32_scalar` + 采样），per-round 输入走 pinned（anchor、position）+ 稳定设备张量
（`mtp_anchor_hidden`/采样状态/MTP KV slab），`drafts` 出设备张量 + 轮内一次 D2H；桶可复用 ordinary_graph_profiles（envelope 是路由提示、
宽度不改数值——C1 实证）；③ graph on/off 逐字节 A/B（`NINFER_TP2_MTP_CHAIN_GRAPH=0` 对照）+ sessions 三路由 + append 金值不变；
④ bench MTP K=2 对照（基线 38.2 ms/轮、67.9/64.8 tok/s）；⑤ 文档 + 提交。判定：链出 drafts 逐位不变 ⇒ 窗口/接受/轨迹全不动；
预期 38.2→~30 ms/轮、MTP decode +15–20%。
E2 进度：`tools/win_port/r52_ab.ps1`（r52 同五提示、greedy top_k=1、hash=sha256(content||reasoning)[:16]）已建，**pre-E 金值**（当前构建、
--spec mtp 131072 fp8 KV）：req0 `9f4abde908470928`(stop/130)、req1 `a956f217d6a6250c`(length/436)、req2 `847d706198441ac9`(length/608)、
req3 `49800780a7666b44`(stop/69)、req4 `46698bd3a4f115ee`(stop/634)。验收＝三臂一致（pre == eager(NINFER_TP2_MTP_CHAIN_GRAPH=0) == graph）。
捕获面结论：`mtp_forward_batch/ar_step`→`mtp_forward_core`→`stem/tail` 全设备算子无同步；per-round 值仅 :1119-:1121 三个 set_i32_scalar
（token/position/position+1）→ 改 pinned+copy_i32；envelope 是 host 结构进发射计划 → 桶化（内核 active-splits 从设备 positions 推导且
归约器同源 ⇒ 加宽按构造无害，C1 先例）；stem 的 embedding_tp2（列切）+ split-head proposal_argmax 走 pair ⇒ capture_group 双流。

**E 轮结果（验收完成，待 bench/文档/提交）**：实现＝pinned `[anchor, position, position+1, drafts(K)]` 缓冲 + memcpy 节点、`mtp_chain_body`
单一实现三路共用（eager×2 + capture）、`capture_mtp_chain_graph` capture_group 双流、watermark 复用 verify 窗口 select/reusable/launch；
三态开关 `NINFER_TP2_MTP_CHAIN_GRAPH`（unset=graph / `0`=eager(bucket) / `exact`=eager(exact)，与 `NINFER_TP2_DECODE_GRAPH` 同约定；
`DecodeStepMode` 更名 `StepLaunchMode` 两处共用）。**验收证据：graph / eager(bucket) / eager(exact) 三臂五条哈希逐位一致；且 git-stash
二分（HEAD 旧构建 vs 新构建）逐位一致 ⇒ 改动金值中性、图捕获逐位无损；exact 逐步 envelope == bucket 全宽 envelope ⇒ 链步数值对
envelope 中性（C1 同类再证）**。干净金值（复现 5 次、跨两构建）：req0 `5b4a978335a95cbb`(stop/186)、req1 `2072b4db667b861d`(length/449)、
req2 `3ce32871d0a13d34`(length/571)、req3 `56fd6e3da523651d`(stop/200)、req4 `cc73c73125c6f621`(stop/629)。
**教训（最初 pre 基线离群的根因）**：逗号优先级 bug 的首探针在同一服务器先发过一条拼接乱码长请求，其 prefill 以不同分块形状算出的
37-token 模板前缀 KV 被金样复用（「KV 行带归约形状 ulp」契约）⇒ 全程 near-tie 漂移；stash 二分自证改动无辜后定位到此。
**贪心金值必须在干净服务器上采集（此前零请求）**——已写入 `r52_ab.ps1` 头注释。
**E 轮回归补记（零字节推进陷阱）**：sessions 暴露——捕获体某侧 scratch 全是作用域内分配（净增量 0），重放侧 `alloc_bytes(0)` 被
`DeviceArena::alloc_bytes` 拒绝（"arena allocation must be nonzero"）；服务流净增量非零故金值全绿，会话流（recall/分块重 prefill）
净零、首个重放即炸。三处窗口/链重放推进点已按 `position_arena` 的 `target > used` 成例把 0 视为 no-op；sessions 三路由全过
（plain/mtp/dflash2，mtp 的 `2 layouts` 属性保持）。
**E 轮性能定论**（`NINFER_TP2_TIMING=1` 固定贪心轨迹 A/B，四次运行同轨迹 94 轮/180 提交）：链步 eager 3.73/3.69 → 图化 3.50/3.50 ms；
整轮 34.30/34.09 → 33.99/33.95 ms ⇒ 捕获消掉 ~0.2–0.3 ms/轮主机发射间隙（decode ≈ +0.7%），**不是** E5 预估的 +15–20%：旧"7–10 ms 链"
是更慢轮次的减法分解，实测链为 3.5–3.7 ms 真实 MTP 层 GPU 工作、发射间隙本就多被掩盖。bench_serve 跨运行 decode 率无法分辨（nonce
换内容 ⇒ 接受率/轮数漂移），以组件 A/B 为准。
- 不确定性：acceptance 与 token/轮由数学决定、跨卡可迁移（残差逐位相同），但本机 draft 是 w4a4 而非
  nvfp4 ⇒ 接受率会有小差；效率锚点取自 MTP 轮次；未计 prefill 与首轮抖动。

**DFlash2 22tok/s 根因（已实证，2026-09 诊断轮）**：非草稿质量问题，是**弃稿守卫被模板前缀缓存误触发**。
复用扫描先取网格对齐边界（`reuse_grid = clamp(prefill_chunk, 64, ·)`=256）；对齐找不到时的**非对齐回退**
（tp2_generation_core.cpp:2159-2184）取最深非对齐边界（聊天模板前缀 ~37-57 token 恒命中）⇒
`dflash_draft_declined_ = reuse % reuse_grid != 0`（:2193）⇒ 每轮 `extent=0`（:2923）⇒ 草稿一个都不进验收、
每轮只提交目标 1 token = 22-25 tok/s。实证（诊断 dump，同 K=7 服务）：冷缓存首请求 `extent=7`、每轮接受 0-3 个、
2.29 tok/轮（提案质量正常，与文档 story 档同级）；第 2 个请求（模板前缀已缓存）**每轮 `extent=0 count=1`** = 1.00 tok/轮。
守卫设计本意（:2186-2192 注释）：环属异构分块走法时其提案无法对从零走法的窗口许可。要害：**非对齐回退只覆盖
"共享前缀不足一个 chunk"的场景（clip<256 token），为省这点 prefill 牺牲整个草稿（decode ~2×）**；且 :2100-2102
设计注释自陈"向下取整最多多算一个 chunk"。修复方向（1+3）：clip≤1chunk 时向下取整保环规范（草稿在线），
深 clip 保留弃稿但**可见**（TP-2 接通 speculative stats + draft_context_declined 展示）。
"只有复读才命中"系误读：冷缓存下正常文本也命中（dump 实证），其余测量处于弃稿态（extent=0 无可接受）。
换工件无效 ✓ 与结论一致（弃稿是运行时策略）。

**修复 1+3 落地（本轮）**：
- **可见性（1）**：TP-2 现在填充 `GenerationResult::speculative`（轮循环前言按路线预置
  `accepted_per_position`/draft_window/backend，随后 dflash 与 MTP 两支各自累计 rounds/drafted/
  accepted/per-position，口径与单卡 decode.cpp:139/732 一致）⇒ 响应 `timings.draft_n`/
  `draft_n_accepted` 直接可用；弃稿时 stderr 打一行 `[tp2-draft] masked draft declined: …`。
  坑：TP-2 的 `Request` 不是 `RequestRecord`（无 `speculative_stats`），且该向量从未分配 ⇒ 直接
  `vec[i]+=1` 越界写导致 sessions 硬崩（输出全丢、exit 1）；改为累计进 `result.speculative` 并预置长度。
- **深修（3）**：非网格回退（复用扫描 fallback，唯一产生非对齐 reuse 的入口）对 DFlash2 加门：
  `position <= max(reuse_grid, 16×budget.remaining())` 时不取该边界，改用对齐扫描结果（通常 0）重放
  clip —— 环保持规范分块走法、草稿全程在线。成本模型：clip 重算 ~0.6 ms/token（prefill ~1.6K tok/s）
  vs 草稿省 ~19 ms/生成 token ⇒ 16× 余量下恒为净赚；超过才保留中块复用 + 弃稿（可见）。
- **证据**：`ninfer_qwen3_5_tp2_sessions_test` 三路全过（EXIT=0；测试镜像新策略：
  `draft_rounds_down()` + 浅 clip 断言 reuse==0 + `compare_recall` 第三类"从零重放"只钉边界首采样）；
  dflash2 服务 + `greedy_probe`（7 条共享模板前缀请求，K=7）实测每轮 3.17 committed
  （rounds=30/committed=95）、`draft_n` 187–276、accept 53–67、**无弃稿行**、prefill 188–388 ms
  ⇒ 由 ~22 tok/s 升至 ~71 tok/s（修复前为 1.00 tok/轮）。
- **契约说明**：从零重放的 recall 在热引擎中不保证与冷 oracle 逐 token 相同（`tools/win_port/r52_ab.ps1` 已
  记录：缓存 KV 行携带产生它的 prefill 归约形状，ulp 级）——可验证的新契约是"草稿保持在线 + 边界首采样一致"。

**工件现状（已查）**：本机 `D:/LLM/qwen3_8_27b_w4a4_w8a8.ninfer` **不含** DFlash2（组件仅 text/vision/mtp，
parameters=1422），但同目录已有 `qwen3_8_27b_w4a4_w8a8_dflash2.ninfer`（parameters=1513、objects=1218）⇒
评估**不需要重新转换**，直接换工件即可。

**待实测的未知量**：Windows 原生构建的真实 free（台账 `free` 取自 `cudaMemGetInfo`，WSL2 少报约 1 GiB）；
草稿按 head/row 切分是否数值等价（目前无对应 split 测试）；K=15 时 workspace 峰值；目标 5 层残差跨 shard 汇聚
的每步开销；草稿每层 allreduce 的延迟；稀疏拒绝采样在 TP-2 分片 logits 下的等价性。

## 4. 上游 cherry-pick 计划（第一梯队 + 第二梯队）

**背景**（2026-09-21 分析）：origin/master 领先本分支 20 个提交（merge base `e360c4c0`，
上游 HEAD `9e163eee`）。本地主负载是 NVFP4 TP-2；上游的 Q4/Q5 调优、sparse_moe、Q5 bench/test
提交与本机无关，不排期（完整逐提交判断见会话记录）。**新官方 artifact（`6cc95cc5`）与本地前端不兼容**：
新 artifact 内嵌维护版模板（sha256 `a497db9e…`），不在本地 `CompiledChatTemplate::resolve()` 的
白名单（`e84f32a2…`/`c3cf9e34…`）内，加载即抛 `unsupported frontend/chat_template.jinja`。
本地 nvfp4 共享代码的改动全部是 Windows 特有的 TMA 描述符暂存（`#if defined(_WIN32)`），与上游
性能改动正交但同文件。

**前置**

- [x] 处理工作树未提交改动（Round 18 会话恢复加固：`tp2_generation_core.{h,cpp}`、
      `test_tp2_sessions.cpp`，已实测未提交）——已由 `4890f554`（代码）与 `ad1336a1`（文档）提交
- [ ] 记录基线：`tools/win_port/bench_serve.ps1` decode/prefill + op 级 NVFP4 bench
      （WSL，a4 graph 模式，T=1024/1536/2048）

### 第一梯队：产品兼容（前端 jinja 链，按序摘取）

| # | 提交 | 内容 | 本地冲突预期 |
|---|---|---|---|
| 1 | `a2b7ed11` | 日志简化（3 文件） | 无 |
| 2 | `b9219f3f` | llama.cpp Jinja 引擎引入 `third_party/llama-jinja/`（~6,000 行）+ `ninfer_jinja` 目标 | 无（新目录） |
| 3 | `98dada0e` | 执行自定义 Jinja chat 模板（76 文件）：`chat_template.cpp` 摘要白名单 → Jinja 执行、新增 `prompt_layout`、CLI `--chat-template`、serve/translate 同步 | `chat_template/frontend/processor/tokenizer/prompt_layout` 本地未改、应干净应用；**`src/serve/*` 有本地 Windows 移植改动，预期冲突**（保留本地平台分支 + 采纳上游语义改动）；`include/ninfer/types.h`、`apps/cli/*` 需核对 |
| 4 | `8eaed538` | 模板 literal 内容保留（Refs #258） | 与 3 同文件，小 |
| 5 | `6cc95cc5` | 模型卡 SHA/manifest 更新为维护版模板 | 无（纯文档） |

**验收（每步）**

- 双平台（Windows + WSL）构建 `BUILD_EXIT=0`
- `ninfer_qwen3_5_frontend_test`（当前双平台既有失败——夹具模板被白名单拒绝——摘取后应转绿）、
  `ninfer_jinja_test`、`tests/text/test_chat_templates.py`（需 Python ≥3.11：Windows miniconda 3.13
  或 WSL python3）
- 端到端：新 artifact（新 SHA、模板 `a497db9e…`）在 8088/8099 加载；`reasoning_effort`、tools、
  多模态渲染正常；与旧 artifact 同 prompt greedy A/B（纯文本聊天应等价；工具调用格式可能不同，记录差异）

### 第二梯队：NVFP4 性能（按上游顺序；全部需在双 5060 Ti 重测）

| # | 提交 | 内容 | 本地影响 |
|---|---|---|---|
| 1 | `1d8587bc` | W4A4 激活 scale 平面按 tile 写、每 TMA 请求读一个 tile；`select_a4` 返回 `Nvfp4A4Route`（GEMM + 布局成对） | 影响本地 TMA 路由（linear_add T≥1024、swiglu TMA、attn/gdn input proj）；MMA 路由不受影响。上游 per-shape floor（34816x5120 从 256、其余从 1024）针对全几何 shape；本地半几何 shape 走 MMA，无需改 floor |
| 2 | `05507ab0` | 融合 SwiGLU TMA epilogue 用 `silu_approx`（2 文件：`math.cuh` + epilogue） | 直接作用于 MTP draft 路径（全几何 swiglu TMA）。**非逐位一致**（模型常用带内模糊度 ~1.3×，但消除尾部假零）⇒ greedy 输出可能在近似并列处翻转；验收判据是「无退化 + 质量同档」而非逐位 |
| 3 | `5f5fccab` | W4A4 TMA 路由接受非 256 整除 token 数（最后 M tile 可部分填充；scale 平面补零；epilogue clamp 填充 lane） | 本地 linear_add/swiglu TMA 现要求 `tokens % 256 == 0`，ragged prefill 尾块全落回 MMA；摘取后 ragged 宽度走 TMA。上游有意不动融合 SwiGLU 路由（依赖 silu_approx 的落地顺序） |

**冲突预期**：三条与本地 Windows TMA 描述符暂存（`Nvfp4TmaDescriptorStaging`、tensormap fence、
`#if defined(_WIN32)` 分支）同文件（`nvfp4_w4a4_tma.cuh`、`nvfp4_w4a4.cu`、
`nvfp4_linear_swiglu_w4a4_tma.cuh`）但正交——上游改布局/路由逻辑，本地改描述符传递方式，
合并时两者都保留。

**验收（每步）**

- 双平台构建；op 测试全过（含上游新增的路由边界用例：1023/1024/1025、255/256/257/512）
- 数值：`1d8587bc`/`5f5fccab` 是 exact 改动 ⇒ `NINFER_OP_REPORT_STATS=1` %.17g 两侧字节一致
  （上游基线 3285/3290 条记录）；`05507ab0` 非逐位 ⇒ 上游实测判据 + 本地 greedy A/B（无退化）
- 双 5060 Ti 重测：op 级 NVFP4 bench（a4 graph，T=1024/1025/1536/2048）+ `bench_serve.ps1`
  端到端，对比基线；上游 5090 数字（−8.7%/−10.8%/−17.5%）只作方向参考
- 三条全部完成后：双平台全量测试（`ninfer_qwen3_5_frontend_test` 既有问题由第一梯队解决）

**不排期（留档）**：Q4/Q5 A16 调优系列（`bb844c43`/`beedffa0`/`d3c125ed`/`b39de4d5`/`a9a0d10a`/
`5b4303c0`/`9e163eee`）、Q5 bench/test（`cb30e070`/`f76e19c0`）、`dc58675f`（sparse_moe，MoE 专用）、
`028eb61e`（Q8 cache policy，可选）、`f9c4a04b`（state cache bench 场景，可选）。

---

## 5. 进度

- [x] 前置（工作树处理 + 基线记录）——Round 18 已提交 `9123217c`；PLAN 归档 `9a1e5ae3`
- [x] 第一梯队-1 `a2b7ed11` → `322217ae`（干净）
- [x] 第一梯队-2 `b9219f3f` → `c6c49b2d`（干净，17 文件 +5966）
- [x] 第一梯队-3 `98dada0e` → `119e7f12`（7 冲突已解，75 文件）
- [x] 第一梯队-4 `8eaed538` → `dd985028`（干净，19 文件）
- [x] 第一梯队-5 `6cc95cc5` → `8a086f40`（干净，15 文件）
- [x] 第二梯队-1 `1d8587bc` → `1ed82861`（干净，22 文件；Windows staging 保留）
- [x] 第二梯队-2 `05507ab0` → `2a7ec6c5`（干净，2 文件）
- [x] 第二梯队-3 `5f5fccab` → `4b1e8507`（2 冲突已解，19 文件）
- [x] WSL 构建（BUILD_EXIT=0，212 目标全过；merge artifact 修复已提交 `75831a03`）
- [ ] WSL op 测试 + 字节一致 + 前端夹具测试（**WSL2 CUDA 驱动崩溃**：`cudaGetDeviceCount()` 内 PTX JIT segfault，GPU 被 Windows 服务占用；jinja 测试通过证明二进制无误，op 测试改在 Windows 侧跑）
- [x] Windows 构建（build-win2 全量 + BUILD_TESTING=ON，BUILD_EXIT=0，697 目标；`localtime_r`→`localtime_s` 修复 `474daf92`）
- [x] Windows op 测试（NVFP4 A4/A16、frontend、jinja 全过；`NINFER_OP_REPORT_STATS=1` 错误指标在容差内，268 条记录）
- [x] 端到端 A/B 验证（`tools/win_port/verify_cherry_pick.ps1`；旧 vs 新二进制同配置同脚本）

### 5.4 e2e A/B 验证结果（2026-09-21，双 5060 Ti，3456 服务）

同配置（启动脚本 MaxContext 204800、draft-tokens 3、KV fp8）同方法（bench_serve.ps1）：

| 项 | 旧二进制 | 新二进制 | Δ |
|---|---|---|---|
| prefill_2048 | 574.3 tok/s | 587.9 tok/s | +2.4% |
| prefill_8192 | 1,568.9 tok/s | 1,555.4 tok/s | −0.9% |
| prefill_32768 | 1,498.3 tok/s | 1,488.2 tok/s | −0.7% |
| decode_short | 71.6 tok/s | 72.5 tok/s | +1.3% |
| decode_at_8192 | 70.9 tok/s | 75.0 tok/s | +5.8% |

**结论**：cherry-pick 无退化。prefill 持平（噪声范围内），decode 略升（+1.3%~+5.8%）。decode token 数不同（旧 118/128 vs 新 84/83）因 jinja 模板渲染/采样随机性，吞吐非严格可比但方向持平或略升。正确性正常（两边都生成带 reasoning 的回复）。结果：`profiles/bench/upstream_cherry_pick/bench-{old,new}.json`。

### 5.1 `98dada0e` 冲突解决要点（本地 `--reasoning-effort` 特性接回上游 jinja 设计）

- `serve_options.h`：`enable_thinking`/`preserve_thinking` 改 `std::optional<bool>`（模板拥有默认值），保留本地 `default_reasoning_effort`。
- `engine.cpp`：采纳上游 `sampling_mode` 基于 `info.starts_in_reasoning`（渲染后判定），保留本地 `frontend_`/`capacity`；**重新加回** `Engine::prompt_capabilities()`（上游删除，本地 `--reasoning-effort` 启动校验需要）+ `engine.h` 声明。
- `translate.cpp`：合并上游 kwargs 合并机制（`merge_boolean` + 模板 `reasoning_effort`）与本地 `effective_reasoning_effort` 计算 + effort 校验 switch（只接受模板支持的 low/medium/xhigh/none）；签名保持 3 参（request, server, capabilities）。
- `translate.h`：`ResolvedPromptSemantics` 加回 `effective_reasoning_effort` 字段 + 3 参签名。
- `generation_service.cpp`/`request_log.cpp`/`test_serve_options.cpp`：保留本地 `prompt_capabilities_` + `default_reasoning_effort` 校验/日志；测试补回 `prompt_capabilities` 定义 + 全部改 3 参。
- `src/text/CMakeLists.txt`：本地 `UTF8PROC_STATIC`（Windows 修复）+ 上游 `ninfer_jinja` 目标两者都保留。

### 5.2 `5f5fccab` 冲突解决要点（Windows TMA 描述符暂存 × 上游 partial M tile）

- `nvfp4_w4a4_tma.cu`：采纳上游 ceiling 除法 `(tokens+kBlockM-1)/kBlockM` + 新 `tokens` 实参，保留本地 `#if defined(_WIN32)` staging 分支（`staged.get()` + `tokens`）。
- `nvfp4_w4a4_tma.cuh`：kernel 签名加上游 `int token_count` 参数，保留本地 `#if defined(_WIN32)` 描述符引用 + tensormap publish/acquire 块；kernel body 已用 `token_count` 边界（auto-merge）。

### 5.3 构建脚本修复

- `tools/tp_bootstrap/build_r35.sh` 的 rsync 列表漏了 `third_party`（`b9219f3f` 新增 `third_party/llama-jinja/`），导致 WSL 侧 `ninfer_jinja` 目标缺失、CMake 配置失败；已补入 `third_party`。

## 6. 官方 NVFP4 工件的 DFlash2 接受率更高：定位与"官方组件 + 本地 text"合成工件

### 6.1 目标

用户观察：官方工件 `D:/LLM/qwen3_8_27b_nvfp4.ninfer` 的 dflash2 接受率明显高于自转工件
`D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2.ninfer`，怀疑官方工件做了优化。要求把"官方工件的
mtp+dflash2+vision"与 `D:/LLM/W4A16/NVFP4/W4A4+W8A8` 的权重合成为一个新工件用于对比测试。

### 6.2 已确认事实（证据：直接解析两个工件的 v3 目录）

| 项目 | 官方 `qwen3_8_27b_nvfp4.ninfer` | 自转 `qwen3_8_27b_w4a4_w8a8_dflash2.ninfer` |
|---|---|---|
| metadata.name | `qwen3.8-27b` | `qwen3.8-27b-w4a4-w8a8` |
| recipe | 内置 `qwen3_8_27b_nvfp4` | `D:/LLM/w4a4_family_recipe.py` |
| dflash2 源 | 维护机 BF16 目录 `.../Qwen3.8-27B/dflash2` | `W4A16\NVFP4\W4A4+W8A8\DFlash2-FP8` |
| vision 绑定 | bf16 195 / q4 27 / q5 54 / q6 1 / q8 2（+162 非参对象） | 完全相同 |
| mtp 绑定 | bf16 7 / q8 3（+6） | 完全相同 |
| dflash2 绑定 | bf16 45 / q8 11（+35） | 完全相同 |
| dflash2 config | `target_layer_ids=[5,19,33,47,61]`、`selector_rank=256`、`selector_top_k=16`、`conv_kernel_size=2` | 逐字段相同 |
| text 绑定 | nvfp4 56 / fp8 74 | nvfp4 42 / fp8 88 |
| proposal | q4 1 + int32 1 | 相同 |

结论：**官方工件在 mtp/dflash2/vision 上没有任何"优化"**，三组件的格式、对象数、组件配置与自转
工件逐字段一致。真实差异只有两点：(a) dflash2 的**来源检查点**（BF16 原始 vs FP8 中间产物，
最终都存成 `q8_g32_fp16`，各 11 个 Q8 参数）；(b) text 的逐层表示（官方规则为"mlp 且
layer<56 用 nvfp4、其余 fp8"，自转按源检查点每层已存编码导入）。本地不存在 BF16 DFlash2
检查点（只有 `DFlash2-FP8`、`Qwen3.8-27B-DFlash2-EXL3-5.0bpw` 与若干 gguf），无法从源权重
复刻官方 draft；但官方工件自身携带该 draft 的 Q8 字节。

### 6.3 设计决策：逐对象字节移植，而不是重跑转换

新增 `tools/convert/graft_components.py`。v3 目录中每个逻辑绑定指向一个对象，对象的
`format/shape/layout/bytes` 由配方决定而与转换批次无关；实测两工件的 vision/mtp/dflash2
绑定对象**缺失 0、格式差异 0、字节差异 0**（vision 279、mtp 10、dflash2 56 个对象，共 1.46 GiB），
故可把 donor 对象字节直接覆盖到 base 载荷的对应位置（`entry_payload_start + object.offset`），
无需反量化、无需二次量化、无需重排载荷。工具同时写入 `metadata.graft` 与可选
`metadata.name`（JSON 区长度不变、空格补齐）、刷新 16 字节 `artifact_id`，并对每个移植对象做
SHA-256 双向校验。未采用"反量化成 safetensors 再用配方重转"的备选：它会在 Q8 上再加一次
量化往返，且 text 需重转 22 GB。

### 6.4 步骤与进度

1. [x] 解析两工件目录，确认组件一致性与可覆盖性（见 6.2）。
2. [x] 实现 `tools/convert/graft_components.py`。
3. [x] 生成 `D:/LLM/qwen3_8_27b_w4a4_w8a8_official_parts.ninfer`：411 个对象、2836.0 MiB
   （bf16 274 / q4 54 / q5 54 / q6 1 / q8 28），工具内 SHA-256 双向校验全部一致。
4. [x] 校验：Engine 正常加载（`engine ready` + `listening`，名字 `w4a4-w8a8-official-parts`）；移植对象
   411/411 等于 donor；未移植对象 807 个中抽样 410 个（4.0 GiB）等于 base；bindings/components 表不变。
5. [x] 接受率实测（同一提示、贪心、500 token 正文，dflash2 K=7，`--lm-head-draft`）：
   正文阶段 official 44.6% / user 52.9% / grafted 52.2%；思考阶段见 §6.6。
6. [x] 结论：合成工件跟随 text 而非 official 组件 ⇒ 接受率差异不来自 mtp/dflash2/vision。
   若要从源权重真正复刻官方 draft，仍需获取其 BF16 DFlash2 源检查点。

### 6.6 实测结果（同一提示、贪心、`--spec dflash2 --draft-tokens 7 --lm-head-draft`）

| 工件 | 阶段 | req1 | req2 | req3 | 均值 |
|---|---|---|---|---|---|
| official `qwen3_8_27b_nvfp4.ninfer` | 正文 | 42.3% | 44.5% | 47.1% | 44.6% |
| user `qwen3_8_27b_w4a4_w8a8_dflash2.ninfer` | 正文 | 50.7% | 54.0% | 53.9% | **52.9%** |
| grafted `..._official_parts.ninfer` | 正文 | 47.3% | 55.3% | 54.0% | **52.2%** |
| official | 思考（首轮 400 token 全思考） | 31.8% | 29.4% | 26.1% | 29.1% |

思考阶段三工件统一复测（`build-win/acc3-*.jsonl`）：

- official thinking-phase: 346/1066=32.5% | 347/1057=32.8% | 336/1137=29.6% => mean 31.6% | p1=111
- user thinking-phase: 351/1034=33.9% | 369/905=40.8% | 368/906=40.6% => mean 38.4% | p1=102
- grafted thinking-phase: 352/1027=34.3% | 368/912=40.4% | 368/906=40.6% => mean 38.4% | p1=102

采样默认值复测（`build-win/acc4-*.jsonl`，每个工件 4 次请求、800 token、thinking 开启、temperature 0.7 / top-k 20 / top-p 0.8）：

- official sampled: 39.2% | 37.5% | 29.0% | 42.9% => mean 37.2%
- user sampled: 39.8% | 34.9% | 31.4% | 40.8% => mean 36.7%
- grafted sampled: 45.5% | 38.1% | 47.1% | 42.3% => mean 43.2%

单次离散 29–47%，4 次采样下差值的标准误约 3–4 个百分点，因此这一组只能说明「合成件不劣于两者」，
不足以断言官方最高或最低；贪心两组（正文 52.2% vs 52.9%、思考 38.4% vs 38.4%）才是低方差证据。

结论（组件维度）：合成工件（官方 mtp/dflash2/vision + 自转 text）在正文贪心、思考贪心两个低方差设置下与
自转工件接受率基本一致，官方工件都最低 ⇒ **mtp/dflash2/vision 组件不是接受率差异的来源**，差异由 text（target）
权重与提示内容共同决定。要复现用户观察到的「官方更高」，需要用其真实系统提示、采样设置与更长预算复测。

观察：每个 serve 的 req#1 接受率都明显低于 req#2/req#3（如 grafted 47.3% vs 55.3%/54.0%），说明首个请求
受冷启动影响，比较时应看 req#2/req#3。贪心正文阶段下 official 最低、grafted 与 user 基本一致
（≈52–53%），即组件不是差异来源。用户观察到的官方更高很可能出现在以 reasoning 为主的真实会话里，
需按真实系统提示与更长预算复测。

## 7. TP-2 两卡显存不平衡：方案 A（dflash2 提案头切分）与方案 B（selector 下移）

### 7.1 问题与账目（实测，dflash2 K=7 + vision，--max-context 102400，已扣除账目里约 1 GiB 的驱动虚报）

| | GPU0（shard 0） | GPU2（shard 1） |
|---|---|---|
| 实占 | 14636 MiB | 13434 MiB |
| KV（fp8） | 1612.5 | 1612.5 |
| 非 KV 固定开销 | **13023.5** | **11821.5** |

KV 两卡必须同 token 数（按 head 切），上限由最重的卡决定：差 1202 MiB ≈ **76k token**；平衡后上限
从约 190k 提到约 228k（**+38k token**）。262144 时 GPU0 需 17151（超 840 ⇒ OOM）而 GPU2 只需 15949 ——
与"GPU0 OOM、GPU2 有余"的现象一致。KV 斜率 1612.5/102400 = 15.75 KiB/token/卡。

### 7.2 为什么"整份搬组件"不赚（算术结论）

固定开销里只有 dflash2 draft（工件 2123.6 MiB，实测 shard 0 多 1840）与 vision（826.5）是单卡独有，
两者合计 2746.5 超过每卡对称余量 1373：draft→shard1 且 vision→shard0 时上限不变（14195），只搬一层更差。
**唯一能赚的是"切开"某块**。

### 7.3 方案 A（进行中）：dflash2 路线按词表切分 `proposal/head`

- 收益：reduced proposal head Q4 [131072,5120] 340 MiB 两卡各半 ⇒ **每卡约 -170 MiB**，上下文 +~10.8k token；
  精度不损失（候选与整份头逐位一致），速度不损失（每轮多 KB 级交换）。
- 现状与门禁：`load.cpp:131-148` 的 `split_proposal_head` 排除 dflash2；`draft.cpp:362-385` 用融合
  `ops::linear_topk` 假设整份头。先例是 MTP 的 `proposal_argmax`（text.cpp:796-836）+ `merge_local_row_blocks`。
- 设计：新增小算子 `ops::merge_topk_candidates`（输入两列各 [16,U] 候选，按"分数降序、同分 global id 小者优先"
  合并，规则与 `linear_topk` 契约一致）；提案时两卡各跑自己半表的 `linear_topk`，用 `DevicePair::allreduce`
  把 hidden 广播到 peer、再把两列候选合成 [32,U] 交换，最后用新算子选 top-16。全局 top-16 ⊆ 两半 top-16 之并，
  同分规则一致 ⇒ 逐位精确。
- 待改文件：`load.cpp`（门禁+注释）、`execution/draft.cpp`（拆分分支）、`program/dflash_round.cpp`
  （`dflash2_proposal_workspace_bytes`）、必要时 `program/planning/startup.cpp`（peer 侧容量）、
  `load/tp_split_spec.{h,cpp}`（注释）、新算子与其 oracle 测试。

### 7.4 方案 A 的验收标准

1. 新算子 oracle 测试（host 朴素排序逐位比对）与 `ninfer_linear_tp2_split_grouped_head_test` 通过。
2. **摘要不变**：`ninfer_qwen3_5_tp2_dflash_append_test` 仍是 `0xbad27a494a9bc853`（K=7）/ `0xbee487264ca8ffb8`（K=5）；
   `ninfer_qwen3_5_tp2_dflash_solo_test` 仍是 `0x4bcc3994a5efba7d`。
3. 账目：`proposal/head` 改动前在两卡都是复制的，所以**两卡各减半 170 MiB**（见 §7.7 实测）。
4. 抽测贪心 dflash2：与改动前同命令同提示的输出逐字节一致（见 §7.7）。

### 7.7 方案 A 实施结果（已完成）

**改动**：`ops::merge_topk_candidates`（新算子 + oracle 测试）、`linear_topk` 支持 65536 行半表 profile、
`TextContext::proposal_topk_tp2`、`draft.cpp` 拆分分支、`startup.cpp` 文本工作区预留、`load.cpp` 门禁开放、
注释同步（`tp_split_spec.{h,cpp}`）。

**账目 A/B（同一工件 `qwen3_8_27b_w4a4_w8a8_dflash2.ninfer`、同一命令 `--spec dflash2 --draft-tokens 7
--lm-head-draft --vision --devices 0,1 --max-context 102400`，唯一变量是门禁开/关）**：

| 项 | 门禁关（复制） | 门禁开（半表） | 差 |
|---|---|---|---|
| shard 0 `weights+ctx` | 13878.6 | 13708.6 | **−170.0** |
| shard 1 `weights+ctx` | 12036.6 | 11866.6 | **−170.0** |
| shard 0 `free` | 136.0 | 308.0 | +172.0 |
| shard 1 `free` | 1338.0 | 1508.0 | +170.0 |
| nvidia-smi GPU0 / GPU2 | 15158 / 13954 | 14986 / 13784 | −172 / −170 |

上限由 shard 0 决定：+170 MiB ÷ 16.13 KiB/token ≈ **+10.8k token**（与设计预期一致）。
`dflash2 proposal` 轮工作区顺带 7.4 → 4.4 MiB（半表 `linear_topk` 暂存更小）。

**抽测（贪心，temperature 0，max_tokens 500，同提示）**：

| 项 | 门禁关 | 门禁开 |
|---|---|---|
| completion_tokens | 385（stop token） | 385（stop token） |
| dflash2 accepted | 252/931 (27.1%) | 252/931 (27.1%) |
| decode tok/s | 66.3 | 66.9 |
| 输出文本 sha256(前 32) | `2765879ab69653a592f330a335025ff9` | `2765879ab69653a592f330a335025ff9` |

输出逐字节一致 + 接受计数一致 ⇒ 拆分路径的候选与整份头逐位等价（贪心确定性）。

**踩到的坑（已修）**：`linear_topk` 的候选载荷是**按列连续**的（`merge.cu:82` 写 `column * 16 + rank`，
`merge.cu:22` 同类），新算子最初按 `rank * columns + column` 索引，U=7 时整列错位，抽测接受率掉到 7.4%、
tok/s 减半。修正后恢复到与改动前完全一致，并把该布局写进算子契约与 oracle 测试。

### 7.8 已提交代码中的一个隐患（工作项 B 一并修）

`DevicePair::allreduce` 是两卡之间**唯一**的集合通信，其语义是 **BF16 逐元素相加**
（`src/core/tp/device_pair.cu:21,95` 的 `add_bf16x8`/`__hadd`），并要求字节数为 16 的倍数
（`device_pair.cu:209`）。方案 A 的 `proposal_topk_tp2` 用它搬运 **I32 候选 id 与 FP32 分数**：
一侧 memset 0 后相加在数值上等于拷贝，但按 BF16 解释时，任何落在 signaling-NaN 模式
（0x7C01–0x7FFF / 0xFC01–0xFFFF）的 16 位 lane 会被静默化而改变位模式——id（<248077）的低 16 位、
FP32 分数的任一半字都可能命中，属低概率但真实的**静默错误**。本轮 E2E 逐字节一致只能说明这次没命中。

**修法**（不碰 `allreduce` 本体）：`TextContext` 自持 pinned host staging（2 MiB）+ 事件，
按 `cudaMemcpyAsync` D2H → `cudaEventSynchronize` → H2D 做**逐字节**跨卡搬运；A 的 union 交换与
B 的 selector 搬运都走它。代价是每轮几次 host 事件同步（µs 级，一轮约 47 ms，占比 <0.1%），
`DevicePair` 与 MTP/单设备路线完全不动。

**待办（本轮不修，验证成本高）**：MTP 路线在 `text.cpp:486` 附近同样用 allreduce 搬 I32 proposal ids，
属同一类 sNaN 位型风险。

**B 期实测（selector 在 shard 1，host staging 仍 5 次同步）**：提案 7.946 → 8.418 ms（+5.9%）；
K=7 `0xbad27a494a9bc853`、K=5 `0xbee487264ca8ffb8` 均不变；shard 0 draft 块 3655.4 → 3410.4 MiB（−245.0）。
hidden（71.7 KB BF16）改走 in-kernel allreduce 后：同步 5→4 次、staging 73 KB→1.4 KB；余下 4 次需
`DevicePair` 的逐字节 device 端拷贝变体才能消除。

## 8. 工作项 B：把 DFlash2 selector 移到 shard 1（已完成，验收见 §8.9）

目标：`dflash2/candidate_selector/*`（codebook 各 121.25 MiB + hidden_projection 2.5 MiB ≈ 245 MiB）
从 shard 0 移到 shard 1 ⇒ shard 0 少 245 MiB ⇒ 上限 +~15.6k token；精度零损失是硬门槛。

设计：
1. `tp_split_spec.cpp`：在 `mtp/|dflash2/|vision/` 规则**之前**加 `dflash2/candidate_selector/` → `shards=0x2`；其余 dflash2 不动。
2. 参数归属：selector 从 `DraftParameters::selector` 提升为顶层 `Parameters::dflash_selector`，按
   `source.has_weight(...)` 决定本卡是否存在（vision/MTP 的 shard-local 先例）⇒ shard 0 无、shard 1 有、
   单设备有（单设备继续本地跑）。
3. `TextContext::dflash_selector_tp2`：hidden/candidates/scores/anchors/frontiers 送 peer，在 peer 上跑
   `project` + `candidate_selector_path`，drafts/proposal_q 带回；sampling 用 host 侧
   `host_ingress.sampling` 直接 H2D 到 peer（不经 BF16 相加）。
4. 工作区：文本计划新增 `dflash_selector_peer` 预留（peer 侧缓冲 + selector 自己的
   `candidate_selector_path_workspace_capacity_bytes`），按"本卡有 selector、无 draft"判定。
5. 不改 selector 算子语义/算术。

验收：append K=7 `0xbad27a494a9bc853` / K=5 `0xbee487264ca8ffb8`、solo EXIT=0、账目 A/B
（shard 0 −245、shard 1 +245）、吞吐/接受率 A/B（>2% 回退如实上报）。

### 7.5 方案 B（待办）：把 `dflash2/candidate_selector`（245 MiB）移到 shard 1

selector 每 draft 轮只需一次（输入 hidden ~KB 级），搬到 shard 1 可再平衡约 245 MiB（+~15.6k token），
代价是每轮一次 PCIe 往返，需先做微基准量化；`dflash2/feature_projection`（132.8 MiB）同理，但它服务 prefill
（长提示要传特征），优先级更低。

### 7.6 其他选项（未采纳，留档）

- 不均匀 TP（KV head 1:3 + FFN 3:1）能一次吃掉 1202 MiB（+38k token），但 `tp_split_spec.cpp:138-143`
  的 part 目前硬编码二等分、merge/collective 也按半块，属明确的产品改动。
- `--spec mtp`（draft 仅 148 MiB）零代码即可开 262144（仓库实测行），代价是换 draft 后端。
- KV 量化类（nvfp4/k8v4/int8）损精度，排除；关 `--vision` 只省轻卡，dflash2 下不提高上限。


### 6.5 验收

- 新工具仅依赖 Python 标准库；`git diff --check` 干净。
- 合成工件可加载并服务，且组件字节归属如上。
- 接受率对比给出可复现的三组数字（同一提示与采样设置）。


### 8.9 落地结果（sendrecv 与 B 的验收，全部实测）

- **新增 `DevicePair::sendrecv`**（`core/tp/device_pair.{h,cu}`）：把既有 in-kernel 内核模板化为
  `ar_exchange<kAdd>`，`kAdd=false` 时 phase 3 把 peer 的 staged 字节**逐位写入** `recv`；staging、
  arrival token、slice、双缓冲机制全部复用 ⇒ 稳态**零主机同步**、可入 CUDA Graph、与 lockstep 兼容。
  单测 `ninfer_tp_device_pair_test` 新增对抗性用例（signaling-NaN 位型 lane、I32/FP32 位型、960B/64B/1MiB
  三档）**PASS**；本机 `p2p_available=0` ⇒ 走的正是 in-kernel 路径。
- **A 的隐患已修**：`proposal_topk_tp2` 的候选 union 交换改走 `sendrecv`（I32 id / FP32 分数不再经 BF16 相加）。
- **B 落地**：`tp_split_spec.cpp` 把 `dflash2/candidate_selector/*` 判给 shard 1；`dflash_selector_tp2`
  用两次**打包**交换（正向 draft 状态、反向 selector 输出）替代原主机同步通道；`peer_stage_copy` 与其
  pinned/event 状态已删除；工作区预留同步补齐。
- **验收数字**：账目 shard 0 `weights+ctx` 13708.6 → **13462.6（−246.0）**、shard 1 11866.6 →
  **12112.6（+246.0）**（两卡之和不变）；贪心同一提示输出 sha256 `2765879ab69653a592f330a335025ff9`
  与 selector 在 shard 0 时**完全一致**，接受计数同为 `252/931 (27.1%)`；decode **66.7 tok/s**
  （A 期基线 66.3–66.9，回退 <0.5%）；append 摘要 K=7 `0xbad27a494a9bc853`、K=5 `0xbee487264ca8ffb8`
  不变；提案耗时 8.097/8.037 ms（A 基线 7.946/7.962，+1.3~1.7%，端到端不可测）。
- 收益：shard 0 余量 +246 MiB ⇒ 上下文上限约 **+15.6k token**（16.13 KiB/token/卡）。

## 9. TP-2 双卡自旋挂死排查（2026-09-23）：根因定案并修复——token 计数器跨卡假共享

- **现象**（用户多次实遇 + r56 finish5 A/B 臂两次）：`ninfer-serve` 存活（/health 200）、无错误日志、两卡
  SM 100%、宿主轮循环不再推进；日志冻结在新请求 `[tp2-time] rounds=1` 之前 ⇒ **挂死位于新请求第一个
  DFlash2 轮内**（propose 的 6 次合算或 verify 图重放处）。与 Round 34 的并发打乱同症状、不同根因。
- **复现与取证**：op 纯队列压力 40 轮（~3.2 万合算、深队列、大小路径混跑）零命中；**图重放压力**
  （新增 `check_graph_queue`：捕获队列 20 次重放 + 急切合算交错，位精确校验）40 轮内 run 30 命中
  （`build-win/r57d/hang-stress-30.err` 现场）；serve 级 40 次臂（base/finish5 交替、每臂 7 请求）零命中
  ⇒ 放大器是**双卡 token bump 的并发窗口**（图重放把两次计数自增压到最近；生产 verify 图每轮 128 连发
  fuse bump 同理）。
- **现场签名**（`NINFER_TP2_AR_WATCHDOG=1` stall-dump，平时零 mapped 访问）：
  `calls=944 last=2097152 tok=[1549,1548] arrA=[1549×4 …] arrB=[1548×4 …]`。调用序列经全量审计为
  **构造性成对匹配**（13 处合算调用点均单主机决策、双流成对启动；decode step/verify/mtp chain 三图单元均
  `capture_group` 双流成组；DFlash2 轮走 `dflash_propose_batch` 急切路径），却出现**一侧计数丢一次
  bump**——丢更新位点即两卡并发 RMW 的同一缓存行。
- **根因**：`token_host_` 为 8 字节单分配、`token_b_ = token_a_ + 1`：两卡每次合算各自 RMW 自己的 int
  （小路径 fuse bump 内核自增、大路径 `bump_ar_token`），mapped host 内存跨设备不一致，**任一侧的整行
  写回会抹掉另一侧刚落的增量**（观测形状 `tok=[N+1, N]` 即后者被抹）。计数错一档后**每次**合算双方
  `arrival_other[b] != token` 永久互旋（decode 小路径单 block、无 order 链、自旋点唯一），全程无错误
  可见——宿主线程停在轮末 `cudaStreamSynchronize`，永远走不到下一个 CUDA_CHECK。llama.cpp 的
  host-staging token 一贯 64B 间隔防假共享（llamacpp-notes:156）；本实现 arrival/order/staging 均每卡
  独立分配（无跨卡写共享，安全），唯 token 对违反该纪律，且它还是全设计中唯一的跨卡写共享缓存行。
- **修复**：`kArTokenStrideBytes = 64`，两计数器各占独立缓存行（`core/tp/device_pair.cu`），数据通路
  零改动；watchdog 读取按 stride 适配。
- **验收（2026-09-23，2×RTX 5060 Ti、sm_120a、Windows 原生）**：① `ninfer_tp_device_pair_test` PASS
  （含 `check_graph_queue` 三种形态位精确）；② 修复后图压力 **60 轮 0 挂死**（修复前同电池 40 轮命中 1）；
  ③ serve 位一致对照臂 `drafted=2843 accepted=699 (24.59%)`、逐 prompt 344/109、399/102、337/110、
  396/100、495/87、379/104、493/87 与 r55/r56 基线**逐项一致**，吞吐 56.6 tok/s（基线 56.4，无回归）。
- **保留工具**：`DevicePair::start_ar_watchdog`（`NINFER_TP2_AR_WATCHDOG=1`，调用计数停滞才 dump
  tok/arr/order，签名解码：tok 不等＝计数失配；tok 相等且 arr 均发布＝可见性；槽新旧混杂＝协议竞态）、
  `tests/test_tp_device_pair.cpp::check_graph_queue`、`tools/tp_bootstrap/r57_spin_stress.ps1`
  （op 级压力循环）、`tools/tp_bootstrap/r57_spin_repro.ps1`（serve 级复现循环、工件交替、挂死自动取证）。

---

# PLAN.md archive (archived 2026-09-24, post-TP2-protocol-refactor state)

Archived verbatim from the working PLAN.md when it was reduced to key information and remaining work, after the TP-2 rendezvous protocol refactor (2026-09-23), the DFlash2 K sweeps and prefill cost attribution (2026-09-23), the DFlash2 draft quantization experiments r54-r57 (2026-09-23), the DSH/agent-loop prefix-reuse fixes and the TP-2 first-token publish fix (2026-09-24), and the DFlash2 draft re-quantization campaign r62-r66 (2026-09-24). Historical record only: the current state and the remaining work live in `PLAN.md` at the repository root, product behavior and measurements in [tp2-dual-5060ti.md](tp2-dual-5060ti.md).

# NInfer TP-2 计划（2× RTX 5060 Ti · Qwen3.8-27B NVFP4）

> **唯一的活动计划**，也是跨上下文压缩的持久记忆。整理：2026-09-22（上游 cherry-pick 与 DFlash2 各轮完成后
> 重写为交付前收尾版）。
> - 完整历史记录：`docs/tp2-dual-5060ti-worklog.md` —— 含两份 PLAN.md 全文逐字归档（2026-09-21 上游
>   cherry-pick 重规划版；2026-09-22 本版整理前的全文）。
> - 交付说明、推荐配置与实测数据：`docs/tp2-dual-5060ti.md`；Windows 原生移植：`docs/windows.md`。

---

## 1. 关键信息

- **分支**：`feat/windows-native-port`（Windows 原生 + WSL2 双树）。origin/master 领先的 20 个提交已全部
  cherry-pick 完成（前端 jinja 链 + NVFP4 性能三项），端到端 A/B 无退化。
- **硬件**：2× RTX 5060 Ti 16G（SYS 拓扑、无 P2P、448 GB/s/卡）。WSL2 构建树 `/home/zhuojun/ninfer`；
  Windows 工作树 `D:\Documents\workbench\ninfer`（构建树 `build-win`）。
- **主负载**：Qwen3.8-27B NVFP4，TP-2（权重按 shard 切半，本地自建半几何 shape）。草稿后端：`--spec mtp`
  （K=2、`--lm-head-draft`）或 `--spec dflash2`（K=7）；`--spec dflash`（v1）在构造期拒绝。
- **现状**：交付目标 ①–⑤ 全部完成；vision、Windows 移植、多会话 KV 池、decode graph + AR 策略均已落地实测；
  DFlash2 已在 TP-2 放行（B1–B6、verify CUDA Graph、MTP draft 链图化、弃稿守卫修复）。
- **关键数字**（262,144 配置，WSL）：prefill 1,588 tok/s；decode 59.0 tok/s（MTP K=2）；Windows 131,072 配置
  decode 56.9 tok/s（与 Linux 持平）。DFlash2 K=7 在 4096 上下文贪心档约 71 tok/s（与 MTP K=2 持平）；
  roofline 修正上限 ≈108 tok/s（原归档 90–180 的 180 不可达）。

**推荐运行配置**（WSL 8088 / Windows 8099 同配方，Windows 侧 `--max-context 131072`）：

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh` 是进程级默认思考强度（请求体优先）；`--chat-template` 随第一梯队摘取
可用（模板由内嵌 Jinja 执行）。

### 环境与运维要点

| 项 | 值 |
|---|---|
| 构建（WSL） | `bash tools/tp_bootstrap/r55_build.sh`（rsync + `cmake --build build_dyn -j 8`，成功标记 `BUILD_EXIT=0`，约 2–3 分钟；WSL 构建树现为 `build_dyn`（Ninja），`build_r35.sh` 仍指向已删除的 `build/`，不可用） |
| 构建（Windows） | `tools/win_port/configure.bat` + `build.bat`（VS2022 + CUDA 13.3，`-DCMAKE_CUDA_ARCHITECTURES=120a`） |
| 服务 | WSL 8088（`serve_supervise.sh`）；Windows 8099（`tools/win_port/serve.ps1`，默认前台；自测服务必须用 harness 后台 job，`Start-Process` 起的进程会随工具调用结束被杀） |
| 运行 PATH（Windows） | FFmpeg（`D:\ffmpeg-dev\…\bin`）与 libcurl（`D:\curl-dev\…\bin`）必须在 PATH，否则 `STATUS_DLL_NOT_FOUND` |
| artifact | `D:\LLM\qwen3_8_27b_nvfp4.ninfer`（23.7 GB，旧 artifact，模板 `c3cf9e34…`）；新官方 artifact（模板 `a497db9e…`）随第一梯队可用 |
| 进程纪律 | 全机同一时刻只有一个模型进程（WSL 8088 与 Windows 8099 互斥）；Windows 重新链接 exe 前先停服务（LNK1104） |
| 工具链 | 嵌套 `pwsh` → `wsl -e bash -lc` 吞 `$var` 与重定向 ⇒ 命令写成脚本文件放 `tools/tp_bootstrap/` 再执行；单次阻塞调用上限 600 s ⇒ 长任务用后台作业 |
| GPU/WSL 状态（2026-09-23 起） | 机器可见 3 卡：nvidia-smi 序 0/2＝5060 Ti、1＝Tesla T10；`--devices 0,1` 账本两 shard 均 16310.6 MiB ⇒ CUDA 序数取到两张 5060 Ti（与 T10 的 nvidia-smi 序错位），TP-2 配对不受影响。WSL 侧 CUDA 当日全线段错误（cast/q4/q6/q8/q5 未改动测试全挂，疑似 T10 可见后 CUDA init 崩）⇒ 测试改走 Windows `tools/win_port/test.ps1`，WSL build_dyn 只作编译验证 |

---

## 2. 已完成摘要（细节见归档）

- ① MTP 一致性：根因＝分片未写 replay 记录（fold 消费空日志），已修复；判据为「无退化 + 质量同档 + 有加速」。
- ② 基准：prefill 615→1,588 tok/s；decode plain 31.4 / MTP K=2 50.5–54.0 → 59.0 tok/s。
- ③④ KV / 上下文：五种 KV dtype 可用；65,536 → 131,072 → 262,144（词表/隐层并行切分，−1.2 GiB/卡）。
- ⑤ 文档：`docs/tp2-dual-5060ti.md` + `docs/windows.md`。
- Vision：静态分片分工（MTP→shard 0、vision→shard 1），单图上限 8,192 token，无新增配置。
- Windows 原生移植：VS2022 + CUDA 13.3 构建；CLI/serve/TP-2 全部跑通，性能与 Linux 持平；TMA 描述符
  mapped-pinned 修复。
- 优化：exact-batch verify graph（−4.4 ms/轮）、plain decode graph（+12.6%）、AR 按 payload 切传输（+0.63%）、
  MTP draft 链图化（−0.2~0.3 ms/轮）；AR∥MMA 子块流水实测慢 5%，已否决回退。
- Round 19/20：chunk 计划与 ring rewind 解耦（1a/1b/1c），同一 prompt 的结果不再依赖引擎历史；host KV arena
  默认可分页（`--host-kv-pinned` 才锁页，锁页被拒自动回退）。
- 多会话 KV 池：host KV 换入换出 + 5 会话 LRU + 共享前缀镜像；召回 KV 与 device 逐字节一致。
- 上游 cherry-pick：第一梯队 5 项（jinja 前端链 + `--chat-template`）、第二梯队 3 项（NVFP4 TMA 路由、
  SwiGLU epilogue、非 256 整除 token）全部落地；e2e A/B 无退化（decode +1.3%~+5.8%）。
- 官方 NVFP4 工件接受率调查：合成工件（官方 mtp/dflash2/vision + 自转 text）与自转件一致、官方件最低
  ⇒ 差异来自 text 权重，与组件无关。
- TP-2 显存不平衡：方案 A（reduced proposal head 按词表切分，−170 MiB/卡）与方案 B（selector 移至 shard 1，
  −246/+246 MiB）均已落地，输出逐字节一致。

---

## 3. 未完成事项

### 3.1 交付前收尾

- [ ] 用推荐配置复测并发 C=1/2/4 与流式/stop 冒烟（Round 31/34 的结论基于 bf16 配置）。
- [ ] 交付时按 AGENTS.md 移除本文件（PLAN.md）；决定 `tools/tp_bootstrap/` 下约 300 个一次性诊断脚本的
      保留/清理范围（`build_r35.sh`、`serve_*`、`r37+r38+r39+r4x` 系列与 `docs/tp2-dual-5060ti.md` 引用为
      可复现流程）。

### 3.2 Windows 单卡 CLI tiny artifact 子任务（用户决定；前置已就绪）

目标：产出小到装进单张 16 GiB 卡的 `.ninfer`，用 Windows `ninfer.exe` 单卡跑通「加载 → 生成 → 采样」。
前置（Round 54.9）：CPU 版 torch 在 `build-win\torch-venv`（torch 2.14.0+cpu）；转换器接口已确认。
下一步：① 官方 recipe 直接跑 tiny checkpoint（`--recipe qwen3_8_27b_nvfp4 --components text`）；
② 不成功再写 tiny recipe/override；③ 生成脚本放 `tools/win_port/tiny_model.py`（合成 tiny config +
随机 bf16 权重，HF 命名）；④ 成功则写入 `docs/windows.md` 并提交。

### 3.3 未修缺陷与已知局限

- TP-2 路线 `bf16` KV + `--spec mtp` 在第一次 prefill 就崩（`prompt.cu:63` `cudaErrorInvalidValue`）；
  fp8 + MTP 正常（推荐配置不受影响）。排查入口：bf16 prompt-attention 的 launch 参数。
- MTP 路线在 `execution/text.cpp:486` 附近同样用 BF16 相加的 allreduce 搬 I32 proposal ids ⇒ 落在
  signaling-NaN 位型时会被静默改写（DFlash2 的等价交换已改 `DevicePair::sendrecv` 逐字节；MTP 未改，
  验证成本高）。
- MTP3 ≥ 70 tok/s 门槛未达（纯 decode 上限 K=2 50.5–54.0）。
- TP-2 路线未跑 perplexity 评测（质量证据为同提示多采样 A/B）。
- per-shard arena 的 `memory_summary()` 仍报 `pages 0/0`（仅显示口径问题）。

### 3.4 暂缓 / 可选（未排期）

- Round 15 剩余：工具调用约束真实模型端到端复测（75 条探针）；WARN 诊断；路线 B（完整 GBNF）；
  TP-2 `validate_licensed_tokens` 守卫。
- Round 14 候选（用户暂缓）：① system+tools 前缀末尾加锚点（修新会话/压缩后第一轮 46.9 s，可省约 11 s）；
  ② prefill 吞吐（suffix 0.95–1.13k tok/s、冷启 1.35–1.38k）；③ 每请求 ~0.3 s 固定开销。
- 归档 §11.6：MTP priming 对图片列用占位 embedding（~100 行，无收益证据）；单图上限 8,192 硬编码
  （让出 shard 1 可重拿更大 envelope）；视觉编码期 shard 0 空闲（可接受）。
- 归档 §6：KV dtype 扫描补全（nvfp4/k8v4 只验证了可启动与吞吐，无质量数据）。
- 若要从源权重真正复刻官方工件的 DFlash2 draft，需先获取其 BF16 源检查点（本机只有 FP8/EXL3/GGUF）。

### 3.5 环境受阻的未做项（已用替代路径覆盖）

- [ ] WSL 侧 op 测试 + 字节一致 + 前端夹具测试：**WSL2 CUDA 驱动崩溃**（`cudaGetDeviceCount()` 内 PTX JIT
      segfault，GPU 被 Windows 服务占用）；jinja 测试通过证明二进制无误，op 测试改在 Windows 侧跑。
      Windows 侧已过（NVFP4 A4/A16、frontend、jinja；`NINFER_OP_REPORT_STATS=1` 错误指标在容差内）。

### 3.6 DFlash2 上 TP-2（主体已完成；余项待排期）

**状态**：`--spec dflash2` 已在 TP-2 放行（`--spec dflash` v1 仍构造期拒绝）。已完成：加载/分片（B1）、
feature sink 上下文物化（B2a/B2b）、masked 提议前向（B3）、组装轮次 `program/dflash_round`（B4）、整轮接通
（B5）、draft ring 随会话/检查点搬运（B6）、verify 窗口 CUDA Graph（C 轮）、MTP draft 链图化（E 轮）、
reduced proposal head 按词表切分（方案 A）、selector 移至 shard 1（方案 B）。新增 `DevicePair::sendrecv`
做逐字节跨卡交换，修掉方案 A 用 BF16-add allreduce 搬 I32/FP32 的 sNaN 位型风险。金值：append K=7
`0xbad27a494a9bc853` / K=5 `0xbee487264ca8ffb8`；ring `0x1a53fd824cd4e360`；solo digest
`0x4bcc3994a5efba7d`。

**根因定案（A 轮，2026-09-22）**：`forward_tp2_window` 从未绑定 `active_valid_columns_` /
`active_sequence_batch_` ⇒ 窗口每列都写 KV；而 DFlash2 的预算钳位窗口把尾列钉在同一绝对位置 ⇒ 同一次前向
多 warp 并发写同一 paged-KV 槽，内容运行间不确定 ⇒ 近似并列处 argmax 翻转（此前的 S1「彻底同步」只是压低
概率）。修复：窗口接受 `valid_columns`（钳位列禁写 KV、其 logits 置零），仅 DFlash2 传入，MTP/plain 路由
逐位不变。修复后召回逐 token 复现 from-scratch，保留属性已实证并开放（见归档 A/B/C 轮证据）。

**召回契约**：网格对齐边界（`reuse % reuse_grid == 0`）做全 token 对比；非对齐回退（declined 边界）只钉
边界首采样，尾部可复现性由 solo digest 跨进程保证；弃稿状态经 `GenerationResult::draft_context_declined`
暴露。

**弃稿守卫修复**：非网格回退（模板前缀 ~37–57 token 恒命中）曾让 `dflash_draft_declined_` 每轮置位 ⇒
`extent=0` ⇒ 每轮只提交 1 token（~22 tok/s）。现对 DFlash2 加门：`position <= max(reuse_grid, 16×remaining)`
时不取非对齐边界、改用对齐扫描重放 clip ⇒ 草稿全程在线；实测每轮 3.17 committed、约 71 tok/s。归档里
20–25 tok/s 的合成负载数字早于该修复。

**余项**：

- [ ] B7 性能验收：双卡吞吐对目标（roofline 修正 ≈108 tok/s，90 这端相符、180 不可达）+ 与 MTP 对比 +
      plain/mtp 无回归，并记录 `--spec dflash2` 的推荐 K。
- [ ] 待实测未知量：Windows 原生构建的真实 free（台账 `free` 取自 `cudaMemGetInfo`，WSL2 少报约 1 GiB）；
      草稿按 head/row 切分的数值等价性；K=15 的 workspace 峰值；目标 5 层残差跨 shard 汇聚的每步开销；
      草稿每层 allreduce 的延迟；稀疏拒绝采样在 TP-2 分片 logits 下的等价性。

**设计约束（勿重复调研）**：整份复制 draft 不可行——满上下文（262144 / fp8 / MTP K=2）下 free 仅
697/1217 MiB，而草稿 2.07 GiB/卡；切分后每卡约 1.04 GiB。提议条件是目标 5 个 block 的 residual 拼接投影，
TP-2 把 64 层切两卡 ⇒ 需跨卡 handoff（两卡 residual 逐位相同，只捕 shard 0 即可）。候选 selector 直读全词表
codebook（pred/succ 各 248320×256 BF16，约 254 MiB），必须整份。DFlash2 与 MTP 互斥：选 DFlash2 可释放
shard 0 的 MTP 权重 430 MiB + KV 516 MiB。每轮成本构成、D1–D4 决策与各轮否定结论见归档。

### 3.7 DFlash2 draft 量化实验（2026-09-23；已决定：保持 opt-in）

**目标**：验证「只改转换 recipe、不动算子」能把 TP-2 DFlash2 draft 变小多少、接受率掉多少。

**结论**：能，但只能动一块——draft 的 fused gate/up SwiGLU 可在 Q4 下运行，**shard 0 省 450 MiB**，K=7 接受率
不变（24.77% vs 24.59%），K=2 约 −2.5pp（1.6σ，样本不足以定论），探针吞吐持平。`mlp/down` 与
`attention/output` 不能改：draft 的 fused dynamic grouped-conv finish 硬编码要求 Q8（见下）。

**决策（2026-09-23，用户确认）**：保持为 **opt-in override**，不改官方 recipe（`official_recipes.py:35-46`
仍把 mtp/dflash/dflash2 钉在 Q8）；新 artifact 只作实验件，产品路线继续用 Q8 draft。

**分布一致性结论**：本引擎的投机解码是分布精确的——贪心分支发出的每个 token 都是 target 自己的 argmax
（`src/ops/kernel/speculative_round.cuh:120-142`），采样分支是带掩码残差校正的拒绝采样（`:144-214`，算法说明
`:216-228`），钳位列不写 KV（`docs/tp2-dual-5060ti.md:251-256`）⇒ **draft 权重（含其量化）只改变提议分布 q 与
接受率，不改变模型输出分布**。但「输出轨迹逐位一致」被打破：窗口形状差异在近似并列处翻转（既有性质，先例
`docs/tp2-dual-5060ti-worklog.md:1364-1366`），实测两个 K 的贪心文本长度均已变化。

**脚本与产物**：

- override：`tools/tp_bootstrap/r54_draft_mlp_override.py`（`dflash2/layers/*/mlp/gate|up` → `q4_g64_fp16`，
  `grouped_absmax`）；配方仍为 `D:/LLM/w4a4_family_recipe.py`，draft 源 `W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8`。
- 转换：`tools/tp_bootstrap/r54_convert_mlp4.ps1`（332 s，1218 objects，1 file）。
- 新 artifact：`out/qwen3_8_27b_w4a4_w8a8_dflash2_gateup4.ninfer`（24,338,661,380 B，比基线 −451.6 MiB）。
- A/B 臂：`tools/tp_bootstrap/r54_dflash2_arm.ps1`（起服 → greedy_probe 7 提示 × 160 token → 汇总 → 停服）；
  检查工具 `tools/tp_bootstrap/r54_draft_quant_inspect.py`、dry-run `tools/tp_bootstrap/r54_draft_quant_dryrun.py`。

**账本（`--max-context 131072`、DFlash2 K=7 + vision + host KV 32 GiB，同一二进制）**：

| arm | shard 0 `weights+ctx` | shard 0 `free` | shard 1 |
|---|---:|---:|---:|
| 基线（gate/up q8） | 13462.6 | 102.0 | 12112.6 |
| gate/up q4 | **13012.6** | **552.0** | 12112.6 |

⇒ shard 0 上限 +450 MiB ≈ **+28.5k token**（16.125 KiB/token/card）。

**接受率（greedy，每臂独立冷启服）**：

| arm | K=7 drafted/accepted | K=7 acc | K=7 tok/s | K=2 drafted/accepted | K=2 acc | K=2 tok/s |
|---|---|---:|---:|---|---:|---:|
| 基线 | 2843 / 699 | 24.59% | 54.9 | 1012 / 603 | **59.58%** | 52.0 |
| gate/up q4 | 2826 / 700 | **24.77%** | 57.2 | 1035 / 591 | 57.10% | 51.8 |

（tok/s 为 7×160 token 的探针用时折算，含 prefill，±5% 噪声。）

**失败记录（重要）**：第一版把 `mlp/down` 一并降到 Q5，加载正常但 warmup 即报
`FATAL warmup failed | linear dynamic grouped conv add: invalid projection_weight`。根因：
`src/ops/wrapper/dynamic_grouped_conv.cpp:61-73` 的 `require_finish_projection_weight` 硬性要求
`Q8_G32_FP16 / RowSplit / group 32`——该 finish 把 down/output 与动态卷积 delta 融合，draft 的
`mlp/down`（5120×17408）与 `attention/output`（5120×4096）没有 Q5/Q4 变体。该中间 artifact 已删除。

**后续杠杆（未排期；按性价比排序）**：

- [x] **fused dynamic-conv finish 的 Q5 变体**（放开 `dynamic_grouped_conv.cpp` 的 Q8-only 校验）：
      已落地（r56，2026-09-23）。`mlp/down` −172 MiB、`attention/output` −42 MiB（5 层 draft 合计）。
      实现为「Q5 GEMM 物化 + 共享 finish kernel」路线：**复用已合规的 `select_q5_a16_launch` Q5 A16
      GEMM，无新 GEMM kernel**（原"需要新 kernel"预估偏高）——新增 q5 形状 `n5120_k4096`
      （`src/ops/linear/q5/shapes/n5120_k4096.cu`，selector 镜像 n5120_k17408）+ dispatch 注册；
      新增 `src/ops/dynamic_grouped_conv/q5/`（plan/materialized，路线名
      `dynamic_grouped_conv_add.q5.*.materialized_bf16`）；finish kernel 从 q8 materialized 抽出为共享
      `src/ops/dynamic_grouped_conv/dynamic_conv_finish.cu`（q8/q5 共用，算术逐位不变）。wrapper
      校验与非重叠检查按 qtype（Q8_G32_FP16 / Q5_G64_FP16 RowSplit，Q5 含 qhigh 高位面检查）分派；
      容量 API `linear_dynamic_grouped_conv_add_workspace_capacity_bytes` 增加 qtype 首参（对齐
      `linear_workspace_capacity_bytes` 惯例，`draft.cpp`/`dflash_round.cpp` 调用方跟进）；公共契约
      （`include/ninfer/ops/dynamic_grouped_conv.h`）同步双 codec 语义；bench 增加 `--qtype q8|q5`。
      **验证**：op oracle `ninfer_linear_dynamic_grouped_conv_add_test` 通过（4.38 s：Q8/Q5 ×
      C∈{4096,17408} 全 W/B 域 120 形状 × 图重放，FP64 oracle 按存储 scale 独立解码）；
      `ninfer_linear_q5_a16_test` 通过（3.11 s，新增 5120×4096 全路由边界 vs FP64 oracle）；
      WSL build_dyn 与 Windows build-win 编译通过。
      **实验件已转换**（2026-09-23，CONVERT_EXIT=0，285.8 s，1218 objects）：
      `D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_finish5.ninfer`（23,449.4 MiB，**−213.3 MiB** vs 基线
      23,662.7，与精确计算一致；恰 10 对象 `q5_g64_fp16`：`dflash2/layers/*/attention/output`
      [5120,4096] ×5、`dflash2/layers/*/mlp/down` [5120,17408] ×5，其余 draft tensor 不变）。
      override `tools/tp_bootstrap/r56_draft_finish_override.py`，转换脚本
      `tools/tp_bootstrap/r56_convert_finish5.ps1`（r54 配方，`--device cpu`）。
      **r56 实测**（2026-09-23，Windows 8099，K=7 贪心探针 7×160，2×5060 Ti，日志 `build-win/r56/`）：
      账本 shard 0 free **102.0 → 314.0 MiB**（weights+ctx 13462.6 → 13250.6 = −212.0 MiB 设备侧，
      工件 −213.3，1.3 为分配粒度吸收），shard 1 不变 810.0 ✓ 本地性成立；
      接受率 base **24.59%**（699/2843）→ finish5 **25.13%**（703/2798），**+0.54pp**（p0/p2/p3 与
      base 逐轮全同，p1/p4/p5/p6 文本轨迹漂移使分母变小；无下降迹象）；吞吐 56.4 → 56.2 tok/s（−0.4%）。
      确定性：base 臂跨重建逐字节复现 r55 基线（2843/699、文本 len 全同）——共享 finish 重构后
      Q8 路线零漂移。判定：**维持 opt-in**，升契约特性前仍欠 §3.7 末尾三件套验证。
      **已知 flake（既有问题，用户确认偶发多次遇到）**：首次 finish5 臂在 req#2 流中途概率性挂死
      （两卡 SM 100% 自旋等 handoff、/health 200、日志无报错；桌面进程已排除），重跑即过；
      挂死机理待专项排查（用户指示暂缓，日志留存 `build-win/r56/serve-finish5-hang1.log`）；**该专项已由 §3.8 接手**（2026-09-23 第二次复现后定案为设备侧自旋，处置见 §3.8）。
- [x] **`feature_projection` 新 profile**（5120×25600，132.8 MiB）：已落地（r55，2026-09-23）。
      更正：该 tensor 走 plain `linear` op（`project` → `ops::linear`），不是 `linear_add`。
      新增 `src/ops/linear/q5/shapes/n5120_k25600.cu`（selector 镜像 n5120_k17408：T=1 simt_r8_c4、
      T=2–6 ksplit、T≤24 simt_r8_c8、其余 mma_r64_c128）+ dispatch 表注册 + 合规测试用例
      （`tests/ops/linear/test_q5_a16.cpp` 5120×25600，seed 183U，FP64 oracle）。opt-in override
      `tools/tp_bootstrap/r55_draft_feature_proj_override.py`（`dflash2/feature_projection` → Q5）。
      收益精确值：该对象为 shard 0 本地全量（tp_split_spec 的 dflash2/* Replicated shards=0x1），
      Q8 132.81 MiB → Q5 82.03 MiB（row_split_k128_v1：400 组/行 × 42 B + scale），**−50.8 MiB**
      （原估 −63 偏高）。转换脚本 `tools/tp_bootstrap/r55_convert_featproj5.ps1`（r54 配方 +
      r55 override，`--device cpu`，无需 GPU）。WSL build_dyn 编译通过（BUILD_EXIT=0）。
      **实验件已转换**（2026-09-23，CONVERT_EXIT=0，299 s，1218 objects）：
      `D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_featproj5.ninfer`（23,611.9 MiB，
      实测 −50.8 MiB vs 基线，与精确计算一致；`dflash2/feature_projection` 已确认
      `q5_g64_fp16:[5120,25600]`，其余 draft tensor 保持 Q8）。A/B 基线：
      `D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2.ninfer`（23,662.7 MiB）。
      **r55 实测**（2026-09-23，Windows 8099，K=7 贪心探针 7×160，2×5060 Ti，日志 `build-win/r55/`）：
      op 合规 `ninfer_linear_q5_a16_test` 通过（5120×25600 全路由边界 vs FP64 oracle，2.9 s）；
      账本 shard 0 free **102.0 → 152.0 MiB**（weights+ctx 13462.6 → 13412.6，设备侧 −50.0 MiB，
      分配粒度吃掉 0.8），shard 1 不变 810.0 ✓ 本地性成立；接受率 base **24.59%**（699/2843）→
      featproj5 **24.40%**（696/2853），**−0.19pp**（与 r54 gate/up Q4 的 ±0.2pp 同带宽 ⇒ 不变）；
      吞吐 56.7 → 56.4 tok/s（−0.5%）。探针确定性：base 跨重建逐字节复现（2843/699、文本 len 全同）；
      改 draft 必然微漂文本（FP 平局随轮切移动，r54 gateup4 同现象：834→837、795→784；
      featproj5 漂 4/7 条），目标贪心流仍是验证 oracle。判定：**维持 opt-in**，
      升契约特性前仍欠 §3.7 末尾三件套验证。
- [ ] **selector codebook 量化**（242.5 MiB BF16，非 GEMM）：需要新的 selector 路径（codebook codec + 保持 top-k 域）。
- [ ] **全 draft NVFP4**（约 −0.95 GiB）：先给转换器加 BF16/FP8→NVFP4 方法（现只有 `import_encoded` 能产 NVFP4），
      再补上面三条的 NVFP4 profile；`linear_pair` 与 draft 版 `attn_input_proj` 仍是 Q8-only，全量化必须动它们。

**组合实验件（r57，2026-09-23）**：三个已验证量化合并为最终实验件
`D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer`（22,947.1 MiB，**−715.6 MiB** vs 基线
23,662.7，= 451.6+50.8+213.3 精确求和；21 对象：gate/up ×10 `q4_g64_fp16`、
feature_projection/down/output ×11 `q5_g64_fp16`；`attention/{query,key,value,context_*}`
[6144,5120]、draft 版 `attn_input_proj`/`linear_pair` 等仍 Q8——属杠杆#4 范围）。
override `tools/tp_bootstrap/r57_draft_all_override.py`（含计数校验）+
`tools/tp_bootstrap/r57_convert_draftall.ps1`（CONVERT_EXIT=0，290 s）。
**r57 实测**（Windows 8099，K=7 贪心探针 7×160，日志 `build-win/r56/`）：账本 shard 0
weights+ctx 13462.6 → **12748.6**（−714.0 MiB，粒度吸收 1.6；三杠杆单独测量之和 712.0 的 2 MiB
差为各自粒度取整方式不同），free 102.0 → **816.0 MiB**（≈ +45k token 上限）；shard 1 不变 810.0 ✓；
接受率 **25.3%**（706/2791）vs 基线 24.59%（无下降）；吞吐 56.9 tok/s（+0.9%）。
命名 `qwen3.8-27b-w4a4-w8a8-draftall`。

**升级为受契约保护的特性前必须补的验证（本次未做）**：

- [ ] DFlash2 三件套针对新 artifact 重基线：`ninfer_qwen3_5_tp2_dflash_append_test`（K=7/K=5 金值）、
      `ninfer_qwen3_5_tp2_dflash_solo_test`（digest）、`ninfer_qwen3_5_tp2_sessions_test`（保留断言）。
- [ ] 采样模式（temperature>0）接受率/吞吐 A/B：分布精确性由算法保证，但 q 变化会影响采样下的接受率。
- [ ] Q4 gate_up 在该形状上的 op oracle，以及「按存储 scale 独立解码」检查（AGENTS.md 的数值契约）。

---

### 3.8 TP-2 会合协议重构（2026-09-23；事故驱动，进行中）

**事故（第二次复现，且已定位到设备侧自旋）**：2026-09-23 21:02:56，Windows 3456，`qwen3.8-27b-w4a4-draftall`
（245760 / k8v4 / DFlash2 K=5 / 4 会话 host KV）。req#34 在 prefill 2,189 tok + decode 158 tok 后冻结：
两卡 SM 100%、显存 12,962/12,558 MiB 恒定、`/health` 200、host 恰好 1 个核连续跑满、日志零输出；
req#35–#39（客户端约每 5 分钟重试）被接受但永不推进 ⇒ 6 条请求被静默吞掉（`--max-pending-requests 16`）。
日志留档 `build-win/serve-hang-2026-09-23.log`（30,559 B），现场快照 `build-win/live-serve-snapshot.log`。
**杀进程后两卡仍报 100% / 2805 MHz / 22–25 W / 0 MiB 且无 compute 进程** ⇒ 卡住的是设备侧自旋内核
（`__nanosleep` 轮询的低功耗签名）。进程终止不会立刻回收：21:45 杀进程，21:47 两卡仍 100%/2805 MHz/~25 W，21:49 自行回落到 180 MHz/0%/4–7 W ⇒ 设备侧自旋的又一佐证；测量前先确认两卡回到 0%（本次已确认，无需重启）。

**与下午修复的关系（结论：不是漏打补丁）**：部署件 `C:\ninfer\ninfer-serve.exe` = `build-win/apps/ninfer-serve.exe`
（202,497,024 B，14:23:32），二进制内含 `NINFER_TP2_AR_WATCHDOG` 与 `graph queue[` 字面量 ⇒ 14:50–14:51
那批提交（`12760cb8` 看门狗 → `f8d33f13` cache-line 修复 → `fe96c13d` 图重放测试）在链接时已在树内；
14:24–14:30 的 r57e（图压力 60/60 PASS）与 r57f（serve 复检）跑的就是它。⇒ 本次是**同类现象的另一条路径**：
`src/core/tp/device_pair.cu:191-197` 只堵掉「两计数器同 cache line 互相写回覆盖」这一条漏 bump。

**三条设计缺陷（根因判断）**：

- **D1 会合标识＝两卡各自 RMW 的自增计数器**（`device_pair.cu:208`、`:116-119`）：把"两卡永久锁步"当成
  不可验证、不可恢复的分布式不变量。且本平台无法用原子修——`atomicAdd_system()` 需要
  `hostNativeAtomicSupported`，PCIe 消费卡没有（llama.cpp `ggml/src/ggml-cuda/allreduce.cu:56-58`）。
- **D2 等待严格相等且无上界**（`:153-158`）：唯一失败模式是活锁，且能带走整个服务。
- **D3 会合落在最不可控的一致性域**（两块 GPU 写同一块 mapped host memory），每 token 约 128 次。

**关键决策（对照 llama.cpp ar3-opt 与上游 `allreduce.cu`；细节见 `docs/tp2-dual-5060ti-llamacpp-notes.md`）**：

1. **借"谁来发号"，不借架构**。采用 host 权威 id，但**不**采用 llama.cpp 的图级 meta 架构
   （每 decode step 约 81 段 host lockstep）：同机 3 卡实测 44.5 tok/s vs 本引擎 2 卡约 100 tok/s，
   且违反本仓库 "Models own finite execution composition" 的 ownership。
2. **id 送达复用本引擎已有机制**：host 每轮写 pinned `base` → 图内 memcpy node 拷进 device 标量 →
   内核算 `id = base + call_index`（`call_index` 为捕获参数，跨 replay 恒定）。与 `valid_columns` 同路
   （`src/models/qwen3_5/execution/text.h:203-216`、`tp2_generation_core.cpp:998-1006`）。
   **必须走 memcpy node，不能按值当 kernel 实参**——llama.cpp 的形态一旦入图会重放冻结值，自旋条件被
   上一轮残留值满足 ⇒ 不等待直接读，静默算错（比挂死更难查）。
3. **大 payload 事件路径只作 A/B 候选**，不预设替换：sliced 已在链路地板（2.70 vs 2.66 ms）；上游
   copy-engine 固定开销约 80 µs 对 SM kernel stage A 约 30 µs，故其 1 MB 阈值。
4. **补两边都没有的**：有界自旋 + 失败语义 + 退化路径。partial 合并（`ggml-backend-meta.cpp:2126-2180`，
   已核实）BF16 下非结合、与逐位摘要契约冲突，本轮不做。

**改动清单**：

**Phase 1 —— 止血（不动协议，可独立上线）**
- [ ] `device_pair.cu`：自旋加 `clock64()` 上界（env `NINFER_TP2_AR_TIMEOUT_MS`，默认 2000，0=关），超时后
      `__threadfence_system()` + 置 mapped `stalled` 标志并让所有 block 退出；扩展现有 `ArWatchState`
      （`:236-244`）承载该标志。
- [ ] `allreduce`/`sendrecv` 返回状态；超时后 host 排空两条流、复位 arrival/order/parity，失败该请求并
      **毒化该会话上下文**（要求重新 prefill，避免用半写 KV 继续），服务保持存活。
- [ ] host 侧每轮不变式检查：`token_a_ == token_b_`（两个 mapped 计数器 host 可直读），不等即 dump 并失败。
- [ ] 验收：故障注入（人为让一侧少一次合算／丢一次发布）必须表现为"该请求 5xx + 服务存活 + 下一请求正常"；
      真实 agent 流量连续 ≥1 h 无永久挂死；prefill/decode A/B 回退 ≤1%。

**Phase 2 —— id 权威化（删掉整类 desync）** ✅ 已完成（2026-09-23）
- [x] `device_pair.{h,cu}`：删除 `token_host_/token_a_/token_b_`、`bump_ar_token`、`fuse_bump` 与
      `ar_size_keyed_`/`NINFER_TP2_AR_STRATEGY`。落地形态比原计划更严，三处偏离都是有意的：
      - **每个捕获图一条 id 通道**（`create_ar_channel()`，通道 k 占 `[(k+1)<<32, …)`），而不是"每设备一格 pinned cell"：
        每轮三个 window 各 launch 一次，共用一格会在上一个 launch 的 memcpy node 执行前就被下一次写掉而串值。
        每通道 = 一格 pinned cell + 每设备一个 device 标量（图内 memcpy node 的源与目标）。
      - **id 全程 64 位**（arrival/order 槽改 `unsigned long long`）：32 位截断会让不同通道的 id 段撞值，
        那样陈旧槽又能满足未来的自旋——正是要根除的那一类。
      - **eager 走 64 位 kernel 实参**（从 `1<<62` 起、独立区间），不用 mapped cell：eager 调用会背靠背入队，
        共用一格同样会读到后一个 id。内核只做 `token = base_dev ? *base_dev + call_index : value`，
      `call_index` 是捕获期常量，烘进实参是安全的（烘 id 不安全）。
      - 钩子为 `begin_capture(channel)/end_capture()/arm_round(channel)`：`end_capture` 自记 `calls`（少一处真相）；
        `ar_token_skew()` 删除（计数器概念消失）→ `ar_last_id()` 供错误消息与看门狗；
        `clear_ar_stall()` 只清 trip 标志——**id 永不重复 ⇒ 陈旧槽不可能满足未来自旋 ⇒ 停滞那一轮原地可恢复**，
        这是 Phase 2 相对 Phase 1 的实质收益。
- [x] `src/core/decode_graph.{h,cpp}`：**未改**（钩子由调用方显式调，未使用 `cudaStreamIsCapturing` 隐式判定）。
- [x] `src/runtime/engine/tp2_generation_core.{h,cpp}`：`WindowGraph` 增 `ar_channel`；三个捕获点接钩子
      （`capture_verify_graph`/`capture_decode_graph`/`capture_mtp_chain_graph`）；`launch_window_graph` 先 `arm_round`。
      9 个模型侧 `allreduce/sendrecv` 调用点确实**一行未改**。
- [x] 验收（2026-09-23）：
      `ninfer_tp_device_pair_test` PASS——含改造后的 `check_graph_queue`：**逐次重放换新操作数**（否则 id 复用根本测不出来，
      旧操作数会让陈旧槽照样"通过"）+ 每轮 `arm_round`；
      `tp2_dflash_solo`(solo digest)、`tp2_dflash_append`(K=7/K=5)、3× `linear_tp2_split` 全 PASS；
      `tp2_sessions` 仍以**改动前逐字节相同**的序列失败（既有 ulp 非契约，未被扰动）；
      服务级注入 3000：`HTTP 503 | TP-2 allreduce stalled at rendezvous id 4294968076`
      （`= (1<<32) + 1548` ⇒ 正是通道 1 的段，证明 base 确实经图内 memcpy node 送达）⇒ 下一条请求 HTTP 200；
      同轮 `decode 120.4 tok/s / dflash2 accepted 782/1,200 (65.2%)` 与 Phase 1 的 120.0/65.2% 完全一致（无性能回退）；
      看门狗 dump 同时出现 `12884914887`(通道 3) 与 `4611686018427390359`(eager `1<<62`+) ⇒ 多通道与双区间并存互不干扰。

**Phase 3 —— 大 payload 事件路径（A/B 决定）** ❌ 已否决（2026-09-23，实测定案）
- [x] 决定测量：**不写完整实现**，因为候选路径的**地板**已经超过现路径的**全部成本**。事件路径与 sliced 路径搬的是同样的字节
      （每设备 D2H 本地 delta + H2D 对端 delta，都过同一条 PCIe），所以先量 copy engine 在该 payload 形态下的天花板即可判定。
      探针：`NINFER_TP2_AR_COPY_BENCH=1 build-win/tests/ninfer_tp_device_pair_test.exe`
      （`check_copy_engine_ceiling`：按计划的 `chunk = clamp(nbytes/4, 512 KiB, 2 MiB)` 分块，**每设备两条 copy stream 让双向同时在飞**；
      串行单流版本实测 3.315 vs 双向 3.316 ms ⇒ copy engine 不会因双向并发变快，故该数字是可靠地板）。
      实测：**10 MiB → 3.32 ms/次（两向合计 6.33 GB/s）；24 MiB → 8.86 ms（5.68 GB/s）**。
      现 sliced 路径同一 10 MiB delta 实测 **2.70 ms**（≈7.4 GB/s，链路地板 2.66 ms 的 98.5%）。
      ⇒ 候选路径尚未计入 event 栅栏（约 80 µs/次固定）与 add 输入所需的额外设备往返，就已慢约 23%：
      "prefill 吞吐 ≥ 现 sliced 路径"不可能满足。第二个条件（删除 8-slice write-order 链）随之不成立——write-order 链正是现路径
      吃到链路地板的原因，删掉它只会更慢。
      范围说明：prefill 每层 payload 被 `--prefill-chunk 1024` 钉在 10 MiB，与提示长度无关，所以按 payload 形态做微观测量即可判定
      （现路径已在链路地板 1.5% 以内 ⇒ 端到端不可能有 >1.5% 的收益，而候选在传输层就差 23%）。
- [x] 结论：照计划 **"否则不做"**——不实现事件路径，不动 sliced 路径与 write-order 链。若日后换平台（PCIe 双向带宽更不对称、
      或 SM 访问 mapped host 更慢的机型）可重跑该探针复核。

**Phase 4 —— partial 合并可行性审计** ✅ 已审结（2026-09-23）：**形态不存在，门槛未触发，不做**
- [x] 审计范围：9 个模型侧调用点（`text.cpp:409/475/822/893/950/952/2288`、`text.h:496/509`）。判据：是否存在
      **两个独立 partial 各自归约后送进同一个 elementwise ADD**——只有这种形态才能压成"本地两 partial 先相加 → 一次 collective → 一次 add"。
      逐点分类：
      | 调用点 | 实际形态 | 可合并 |
      |---|---|---|
      | `text.h:496` mixer delta → residual_add | 行并行 partial → 一次 allreduce → `x += delta` | **否**：紧随的 MLP（`text.h:505`）读的是**就地更新后**的 `x`，两次 add 严格依赖 |
      | `text.h:509` MLP delta → residual_add | 同上 | **否**：同上；下一层 mixer 又依赖本层结果 |
      | `text.cpp:409 / 475 / 822 / 893 / 2288` | 一侧**零填充**、另一侧为本地数据拷贝 ⇒ 实为 replicate/gather（add-over-zeros） | **否**：不存在两个独立 partial；且 `__hadd(x,0)==x` 逐位成立，这里本就不是部分和归约点 |
      | `text.cpp:950 / 952` | 背靠背两次 `sendrecv`（ids + scores） | **否**：结果是 top-k **选择式合并**，不是 elementwise ADD |
- [x] 结论：**形态不存在** ⇒ 按门槛"存在才评估"不进入评估，**FP64 oracle 不触发**，无实现改动。附带记录（审计副产品，均判定不做）：
      - replicate 类 5 处改为 `sendrecv` 逐位等价（对端全零），但成本由搬运字节主导、内核内 add 几乎免费 ⇒ 无可测收益。
      - `950/952` 合并为一次需打包 I32/FP32 两种 dtype，省下的只是一次握手（每轮几十 µs / 37 ms 轮时 ⇒ ~0.1%）。
      - 推论：每 decode round 约 128 次 collective 里，96 次是 48 层 × 2 次**真**部分和归约，且每个结果都有依赖它的消费者
        ⇒ **在不改数学的前提下不可约减**；这也解释了为什么 transport 的优化空间只在每次 collective 的成本上（Phase 1–3 已封口）。

**进度**

- [x] **Phase 1a 传输层（2026-09-23）**：`device_pair.{h,cu}` 加 `%globaltimer` deadline（`NINFER_TP2_AR_TIMEOUT_MS`，默认 2000，0=关）、每侧 mapped `stall` 标志、
      协作式退出（thread 0 经 shared 发布、全 block 一起 return，避免 `__syncthreads` 死锁）、`ar_stalled()/ar_token_skew()/clear_ar_stall()`
      （复位=两计数器取大值 + 清零 arrival/order 槽）、以及测试用故障注入 `set_ar_fault_skip_peer_call()`。
      新用例 `tests/test_tp_device_pair.cpp::check_ar_timeout`（跳过一次对端启动）实测：**gave up after 2000 ms, token skew 1 →
      复位后逐位正确**，`ninfer_tp_device_pair_test` PASS（其余 allreduce/sendrecv/图重放/move 用例全过）。
      ⚠️ **1a 不能单独上线**：超时内核在 phase 3 之前返回，调用方只拿到本地偏和 ⇒ 会把"响亮挂死"变成"静默算错"；已与 1b 同批完成。
      另修一个 1a 自身的严重缺陷：内核入口先读**两侧** trip 标志，已 trip 立即退出——否则首次超时后 token 永久错位，同一轮后续 ~128 次合算会各烧 2 s（≈4 分钟/轮）才等到 host 检测。
- [x] **Phase 1b 引擎层（2026-09-23）**：`TP2GenerationCore::abort_if_ar_stalled()`（`tp2_generation_core.cpp`，声明在 `tp2_generation_core.h`）在**每轮收敛点**
      （`request.budget.remaining()`/output policy 之前、任何 token 送达客户端之前）与 prefill 首个 token 采样后检测；触发条件**只用 trip 标志**——
      plain decode 路径只排空 shard A，镜像 shard 落后 1~2 次合算属正常，用 token 偏差判会误报 503，偏差只写进错误消息当取证；
      触发后排空两卡 → `clear_ar_stall()` → `invalidate_host_checkpoints()` + `session_invalidate_active()` → `RequestError(Unavailable)` ⇒ **HTTP 503 + 下一请求重新 prefill**。
      故障注入入口：`NINFER_TP2_AR_FAULT_SKIP_PEER_CALL=<n>`（env，构造期读；诊断用）。
      **服务级验收**（新件已部署 `C:\ninfer\ninfer-serve.exe`，验收时 hash `001446D4…`；注入 3000）：`[ar-watch] calls=3114 … tok=[34054,34053]`（偏差 1）
      → `WARN req#2 failed during generation | HTTP 503 | service unavailable` → 错误体 `TP-2 allreduce stalled (token skew 1)…`
      → 复位后 dump `tok=[34054,34054]`、arrival/order 全 0 → **后续两条请求 1 s 内 HTTP 200**。Phase 1 完成。
      记录教训：本次验收原文只留在本节与当次会话——`C:\ninfer\serve-win.log` 被随后的干净重启覆盖；后续验收先 `--request-log-jsonl` 或先复制日志。
      **回归矩阵**（artifact `D:/LLM/qwen3_8_27b_w4a4_dflash2_draftall.ninfer`；每项都做了"改动后 / git-stash 改动前 HEAD"两次）：
      `ninfer_tp_device_pair_test`（含新注入用例）**PASS**；`ninfer_qwen3_5_tp2_dflash_solo_test`（solo digest）**PASS**；`ninfer_qwen3_5_tp2_dflash_append_test`（K=7/K=5 哈希）**PASS**；3× `ninfer_linear_tp2_split_*` **PASS**；
      `tp2_sessions_test`(plain recall 序列分叉)、`dflash2_real_test`(1.7 s cudaMalloc OOM)、`dflash_real_test`(0xc0000409 快败) 三项**在改动前 HEAD 上逐字节/同码复现** ⇒ 既有问题与本次无关
      （sessions 即 worklog 记的"热引擎从零重放不保证与冷 oracle 逐 token 相同"的 ulp 非契约；另两项为 artifact/环境类，docs 未收录这两个用例名）。
- [x] **Phase 2（2026-09-23）**：见上方 Phase 2 清单（id 权威化完成，已提交 `de74d966`/`b9c5d11c`/`634df875`）。
- [x] **Phase 3（2026-09-23）**：见上方 Phase 3 清单——**实测否决，不做**（copy engine 地板 3.32 ms vs 现路径 2.70 ms）。
- [x] **Phase 4（2026-09-23）**：见上方 Phase 4 清单——**形态不存在，门槛未触发，不做**（9 个调用点逐点分类，FP64 oracle 未触发）。
      **部署注意**：`build-win/apps/ninfer-serve.exe` **不会**自动同步到 `C:\ninfer\`，需手动复制；PE 时间戳使每次链接的 hash 都不同，比对以"部署件与 build-win 产出 `Get-FileHash` 相等"为准（本轮已同步，`108AD767…`）。
- [x] 停掉卡死进程并留档日志（2026-09-23；显存立即清零；残留自旋内核约 2 分钟后自行回落）
- [x] 测量前置确认：两卡空闲占用已归零（2026-09-23 21:49，无需重启）
### 3.9 DFlash2 DraftTokens（K）扫描（2026-09-23；结论：**维持 K=5**）

**问题**：`qwen3_8_27b_w4a4_dflash2_draftall.ninfer` 在 MTP + DFlash2 路线下 `--draft-tokens` 取多少最优。
**方法**：`tools/tp_bootstrap/r59_draft_tokens_sweep.ps1`——每个 K 用**完全相同的生产参数**启动 ninfer-serve
（245760 / k8v4 / `--spec dflash2` / `--lm-head-draft` / temp 0.7 / top-k 20 / top-p 0.8 / vision / preserve-thinking），
只改 `--draft-tokens`；每 K 先 warmup 一次，再对 3 类 prompt（reason 推理链 / prose 描述文 / code 代码）各 5 次、`max_tokens=256`。
指标取自请求日志的 `decode X tok/s` 与 `dflash2 accepted a/b`；tokens/round = output ÷ (drafted/K)，rounds/s = decode ÷ tokens/round。

| K | decode tok/s | 接受率 | tokens/round | rounds/s | reason | prose | code |
|---|---|---|---|---|---|---|---|
| 3 | 83.8 | 58.4% | 2.78 | 30.2 | 93.4 | 76.0 | 82.2 |
| **5** | **96.4** | 48.5% | 3.45 | 27.9 | 113.9 | 81.6 | **93.6** |
| 7 | 96.1 | 36.9% | 3.62 | 26.6 | **117.8** | **83.2** | 87.4 |
| 9 | 89.1 | 30.1% | 3.74 | 23.8 | 113.1 | 72.9 | 81.3 |

（粗扫 8 点的其余值：K=1 → 60.3、K=11 → 84.6、K=13 → 85.4、K=15 → 81.8 tok/s，均低于 K=3~7 的平台。）

**K=6 补测 + 交错复核**（5/6/7 轮流启动 3 轮、每 K 共 18 样本；同一轮内比较可消掉机器状态漂移——K=5/7 都随机器转热从 c1 升到 c3）：

| K | 池化 decode | 接受率 | tokens/round | rounds/s（三轮） | 推理链 | 描述文 | 代码 |
|---|---|---|---|---|---|---|---|
| 5 | 93.3 | 46.3% | 3.34 | **27.9 / 28.0 / 27.9** | 112.1 | 77.4 | **90.4** |
| 6 | 89.7 | 40.9% | 3.48 | **25.7 / 25.7 / 25.7** | 111.8 | 75.7 | 81.6 |
| 7 | **96.9** | 37.4% | 3.65 | 26.6 / 26.4 / 26.6 | **123.2** | **78.2** | 89.4 |

同轮配对（c1/c2/c3）：K=5 88.1 / 93.8 / 98.0，K=6 87.6 / 94.0 / 87.5，K=7 94.4 / 96.8 / 99.6 ⇒ **K=7 三轮均不低于 K=5（+7% / +3% / +2%，差距随机器转热收窄）**，K=6 三轮均低于 K=7。

**K=6 是一个可复现的每轮成本凹陷**：它的 rounds/s 三轮都是 25.7（±0.0），既低于窗口更窄的 K=5（27.9）也低于窗口更宽的 K=7（26.6）；接受率与 tokens/round 都正常插在两者之间 ⇒ 不是噪声，也不在启动台账里（`[mem]` 的 `record` 随 K 单调 5.1/6.0/6.8 MiB，`[tp2-graph]` 三个 K 相同）。疑似按窗口宽度分桶/对齐导致的核函数边界效应，**待查**（线索：`NINFER_TP2_TIMING=1` 取逐相位每轮耗时）。

**机制**：K 增大提高 tokens/round（2.78 → 3.74），但接受率按位置快速衰减（58% → 30%），同时窗口变宽使每轮更贵
（rounds/s 30.2 → 23.8）⇒ 净收益在 K=5~7 到顶。
**结论**：**K=5 与 K=7 相差只在几个百分点内，K=7 略优**；**不要用 K=6**（被两侧同时支配）；K≤3 与 K≥9 明显更差。
- 交错配对里 K=7 三轮均 ≥ K=5（+7% / +3% / +2%，热机后收窄到 ~+1.6%），驱动项是**推理链 +10%**（123.2 vs 112.1）；
  code/tool-call 类反过来 K=5 略高（90.4 vs 89.4），描述文几乎相同。
- 单轮粗测时两者是 96.4 vs 96.1 的平局 ⇒ 差异 ≤ 几个百分点，在样本波动内；要拍板需要更长的配对测量。
- 倾向：客户流量以思考为主 ⇒ **建议把 `--draft-tokens` 从 5 调到 7**（也正是二进制对 dflash2 的默认值 7，与 `append K=7` 金值一致）；
  工具调用占比若上升，5 与 7 的差距会进一步缩小。
**边界**：单机单 artifact、3 类 prompt、temp 0.7、256-token 生成；接受率与内容强相关，换语料结论可能移动。
显存不是区分项（K=3→9 每卡 free 仅差 ~10 MiB；K=15 在 245760 下也能正常启动）。
### 3.10 纯 MTP 后端（`--spec mtp`）的 K 扫描（2026-09-23；结论：**K=3/4 平局，K=5 明显差；仍用 DFlash2**）

**方法**：同 §3.9 的生产配方与交错方法（3 轮 × K=3/4/5 × 每 K 18 样本），只把 `--spec` 换成 `mtp`：

| K | decode tok/s | 接受率 | tokens/round | rounds/s（三轮） | 推理链 | 描述文 | 代码 |
|---|---|---|---|---|---|---|---|
| 3 | 79.4 | 52.7% | 2.58 | **30.8 / 30.8 / 30.8** | 88.5 | **73.8** | **75.9** |
| 4 | **79.7** | 45.0% | 2.80 | 28.5 / 28.5 / 28.5 | **92.4** | 71.0 | 75.7 |
| 5 | 74.5 | 37.2% | 2.86 | 26.1 / 26.1 / 26.1 | 88.3 | 67.7 | 67.5 |

**结论**：MTP 的最优落在 **K=3~4（79.4 vs 79.7，统计平局）**，K=5 低 6.5%；分类看 K=4 优于长推理链（92.4 vs 88.5），K=3 优于描述文（73.8 vs 71.0），代码打平。
但**同 K 对比 DFlash2 全面落后**：K=5 时 74.5 vs 93.3（-20%），且 K 越大差距越大（MTP 的 tokens/round 2.58→2.86 只微增，DFlash2 是 3.34→3.65），
加上 MTP 上限 K=5 无法再换收益 ⇒ **继续用 `--spec dflash2`**。
**顺带澄清**：启动脚本第 60-61 行注释"TP-2 只支持 mtp（dflash/dflash2 会被拒绝）"**已过期**——本轮与 §3.9 都在 TP-2 上实测跑通了 dflash2。
**方法论旁证**：两个后端的 rounds/s 在各自三轮里都只变 ±0.1（MTP 30.8 / 28.5 / 26.1；DFlash2 27.9 / 25.7 / 26.6），
说明"每轮成本"是 K 的确定性函数，decode 的轮间波动来自接受率/内容，交错配对是这类比较的正确做法。

### 3.11 prefill 对比：DFlash2 K=7 / MTP K=3 / 无投机（2026-09-23；结论：**投机几乎不付 prefill 代价，DFlash2 慢 ~3-4%**）

**方法**：`tools/tp_bootstrap/r60_prefill_spec_compare.ps1`——三个配置用**同一套生产参数**，只差 spec 相关开关
（`--spec dflash2 --draft-tokens 7 --lm-head-draft` / `--spec mtp --draft-tokens 3 --lm-head-draft` / 三者都去掉）；
每个配置一个**新进程**（不跨配置复用），两个长度用**不同文本**，日志确认 `cache 0%`（6481 那 0.6% 是共用的 chat 模板前缀 ~39 tok）。
为排除"同一轮里越晚跑越热"的顺序伪影，除 2 轮正序（dflash2→mtp→plain）外又跑 1 轮**反序**（plain→mtp→dflash2）。

| prompt | DFlash2 K=7 | MTP K=3 | 无投机 | TTFT（dflash2 / mtp / plain） |
|---|---|---|---|---|
| ~1,673 tok | 1,473 tok/s | **1,593** | 1,567 | 1.10 / 1.05 / 1.10 s |
| ~6,481 tok | 1,743 tok/s | 1,807 | **1,817** | 3.70 / 3.60 / 3.55 s |

（每组 3 次测量，配置内重复性 ±1%；反序轮里 dflash2 排在最后仍最低 ⇒ 顺序伪影已排除。prompt 变长速率上升 = 每请求固定开销被摊薄。）

**结论**：开投机**几乎不付 prefill 代价**（1% 内），MTP 与无投机基本打平；**DFlash2 稳定慢 3-4%**（扣掉 0.6% 的前缀命中差后仍 ~3%）。
整请求上可忽略：decode 侧 DFlash2 领先 MTP ~20%（§3.10），远超这 3-4% ⇒ **生产继续 DFlash2（K=5 或 7）**。
**未定位**：DFlash2 这 3-4% 的确切来源（未做逐相位归因；要查可用 `NINFER_TP2_TIMING=1` 拆 prefill 每相位耗时）。

### 3.12 DFlash2 K=7 的 prefill 深度曲线（0 → 239k；2026-09-23）

**方法**：`tools/tp_bootstrap/r61_prefill_depth_curve.ps1`——单进程、单一逐步加长的文档（每次请求发送 1..i 段），
靠前缀复用让每步**只 prefill 新增段**，于是日志里 `prefill X tok/s (N tok)` 的 N 就是"新增 token 数"（已验证 `N = prompt − cache`），
即**该深度区间内的瞬时速率**。K=7 / dflash2 / 生产参数，共 17 步、每步 ~14.1k token，max_tokens=1 以隔离 prefill。

| 深度区间 | 速率 | 拟合 | 累计耗时 |
|---|---|---|---|
| 0-14,116 | 1,810 | 1,785 | 7.8 s |
| 13,312-28,180 | 1,610 | 1,590 | 17.0 s |
| 27,648-42,244 | 1,440 | 1,428 | 27.1 s |
| 41,984-56,308 | 1,310 | 1,297 | 38.1 s |
| 55,296-70,372 | 1,200 | 1,191 | 50.7 s |
| 69,632-84,436 | 1,090 | 1,098 | 64.3 s |
| 83,968-98,500 | 1,000 | 1,018 | 78.8 s |
| 98,304-112,564 | 931 | 949 | 94.1 s |
| 111,616-126,628 | 908 | 891 | 110.6 s |
| 125,952-140,693 | 822 | 838 | 128.5 s |
| 140,288-154,758 | 797 | 791 | 146.7 s |
| 154,624-168,823 | 738 | 749 | 166.0 s |
| 167,936-182,888 | 700 | 712 | 187.4 s |
| 182,272-196,953 | 684 | 678 | 208.9 s |
| 196,608-211,018 | 653 | 647 | 231.0 s |
| 210,944-225,083 | 622 | 618 | 253.7 s |
| 224,256-239,148 | 597 | 593 | 278.7 s |

**拟合**：每 token 成本 `1/rate = 5.25e-4 + 5.01e-9 · depth` 秒（17 点，全部误差 <3%）
⇒ 深度 0 外推 **1,905 tok/s**、245,760 处 **569 tok/s**；**满上下文时 attention 已占每 token 成本的 70%**（浅层由固定开销主导）。
**实用结论**：灌满 239k 上下文实际耗时 **278.7 s**（其中 248,072 token 被真正计算——比 239,148 多 3.7%，是 1024-chunk 检查点粒度导致的尾部重算），
有效速率 890 tok/s（首段 1,810 → 末段 597，降 3.0×）。
**复现与旁证**：独立重跑的前 16 点逐点吻合（±1%）；`cache` 数字显示前缀复用在 239k 深度仍然近乎完美（缓存边界 = 上一 prompt 减 ~1 个 chunk），
即上下文缓存环在满深度可用。

### 3.13 深上下文 prefill 的成本归因与优化空间（2026-09-23；分析，未改代码）

**问题**：§3.12 测到 prefill 速率 1,905 → 569 tok/s（3.3×），是否有优化空间。

**架构事实**（`D:/LLM/W4A16/NVFP4/W4A4/config.json`）：Qwen3.5 是**混合注意力**——64 层里只有 **16 层 full attention**
（24 Q / 4 KV / head_dim 256；TP-2 单卡 12 Q / 2 KV），另 **48 层 linear attention（GDN）成本与深度无关**。
⇒ 整条斜率只来自 16/64 层；若 64 层全是 full attention，同一深度处 ≈185 tok/s（当前 569，混合结构已买到 3.1×）。

**实测分解**（§3.12 拟合）：每 token 成本 = **0.525 ms（深度无关，1,905 tok/s 上限）+ 5.012e-9·d**；
在 245,760 处两者分别占 **30% / 70%** ⇒ 这 70% 就是"任何 attention 侧优化"的**收益预算**：完美内核最多 3.3×，现实约 1.3–1.8×。

**生产 prefill 走哪个内核**（已核实）：`causal_softmax_attention.cpp:382-385` 对 `(q_heads==12||16) && width<=16` 才给 ChunkedSmallT，
宽度 1024 的 prefill 因而落到 **Prompt** 路线 ⇒ `causal_attention_prompt_k8v4_kernel`（flash 风格：Br=64 行、Bc=64 KV 列、16 warps、
FP8 tensor core 做 QK^T、FP16 做 PV），但 **grid = (tokens/64, 12 Q 头)**：**每块只处理一个 Q 头**，而单卡只有 2 个 KV 头
⇒ 6 个 Q 头共享 1 个 KV 头却各自独立遍历同一段 KV（请求级 6× 冗余；L2 可能吸收一部分，需 profile 判定）。

**推算的交通量与效率**（以引擎自己的 `[mem]` 账本为锚：kv 3015 MiB/shard @245760 ⇒ 12.87 KB/token/shard，
其中 full-attention 部分 16 层×2 KV 头×256 dim×1.5 B = 12,288 B ✓）：
245,760 深度、1024-token chunk 下 KV 流量 ≈ 16 行 tile × 12 Q 头 × 384 B × 245,760 ≈ **290 GB/chunk/卡**；
实测 attention 时间 ≈1.26 s/chunk ⇒ **~230 GB/s（峰值的 51%）**；attention 计算 ≈4.95e13 FLOP/chunk ⇒ **~39 TFLOP/s（FP8 roofline 的 ~40%）**。
⇒ 两个 roofline 都未饱和，且存在 6× 的请求级冗余 ⇒ 确有空间。

**优化清单（按收益排序；均为估算，需 profile 定夺）**：
1. **GQA 感知的 prompt 内核**：让共享同一 KV 头的 Q 头共用一个 KV 数据流（如 6 头×16 行/块），流量 290 GB→48 GB/chunk；
   若转为带宽受限（~400 GB/s）⇒ attention 0.12 ms/token，245k 处端到端可达 **~1,550 tok/s（2.7×）**；
   若被 FP8 计算下限拦住（~60–70% roofline）⇒ 更现实 **~830–1,000 tok/s（1.5–1.8×）**。50k 处约 1.3×，10k 处约 1.05×。
2. **提高 Prompt 内核已达到的带宽/占用**（KV tile 双缓冲、prefetch、occupancy）：对 attention 项最多 ~2×。
3. **通用性说明**：收益集中在长上下文；**短 prompt 的瓶颈是 0.525 ms/token 的固定项**（48 层线性注意力 + MLP + W4A4 GEMM），那是另一条线。
4. **不是杠杆**：更小的 KV dtype（已 1.5 B/元素，且有质量代价）、TP 切分、GQA 比例（已 4 KV 头）、chunk 上限（`kPrefillChunkMaximum=1024` 是 TP-2 既定设计）。

**下一步（建议先做）**：`ncu` 抓 `causal_attention_prompt_k8v4_kernel`（深度 ≥100k 时）看 **DRAM 吞吐 / L2 命中率 / tensor pipe 利用率**
——据此判定 6× 冗余是否真落到 DRAM（决定做"减流量"还是"提占用"）。注意 worklog 已修过同族的 12 头路由漏洞（仅覆盖 width≤16 的 verify 窗口），
大 chunk 这条是遗留。

**补：PCIe 拓扑（Gen5 x8 + Gen4 x4）已考虑，且是"深度无关项"的主因（2026-09-23 交叉验证）**

worklog Round 35b/35c（`docs/tp2-dual-5060ti-worklog.md:636-691`）早已实测本机拓扑并定位：

- 卡 0（`pci 01:00.0`）**Gen5 x8，实测 ~20 GB/s**；卡 1（CUDA device 1，`pci 08:00.0`）**芯片组 Gen4 x4，实测 ~7 GB/s**（`nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max` 复现：device 0 = 5/x8，device 2 = 4/x4）。
- nsys 归因：`ar_inplace_bf16` 占内核总时间 **48%**（1024 宽块内约 60%）；每层 2 次 allreduce、载荷 [hidden 5120, T] bf16、写+读往返 = 4·hidden·T 字节/层/卡
  ⇒ **2.6 MB/token**，在 7 GB/s 上 = **0.368 ms/token ≈ 2.67k tok/s 的 AR 硬件上限**（实测边际斜率正好贴上、与块宽无关）。
- 关键否证：这条 Gen4 x4（走芯片组）链路**双向合计只有 ~7.9 GB/s**（不是每方向）；21 MB 往返物理下限 2.66 ms，8 切片有序流水已达 **2.704 ms = 98.5%**。
  （本机 Phase 3 独立复现同一数字：10 MiB 载荷 2.70 ms ✓）⇒ **AR 侧软件已无空间**；多 block AR 无差异、copy-engine 会合更慢且结果不一致，均已实测否决。

**与本节拟合的交叉验证**：我测的深度无关项 `a = 0.525 ms/token`（1024 块 = 538 ms）与 worklog 在 ~1.3k 深度的 nsys 归因（AR ~350 + MMA ~188 + 其它 ~35 = 573 ms）**一致**
⇒ 该固定项的构成是 **AR ~65% + MMA ~34% + 其它 ~6%**。注意 worklog 那次深度只有 ~1.3k，所以其归因自然只覆盖深度无关部分，与本节 attention 项不冲突。

**由此修正 §3.13 的措辞**：固定项**不是**"另一条可优化的计算线"，而是**已经贴死链路地板的 PCIe 开销**。
浅层上限 1,905 tok/s = 1/(0.34 + 0.18 + 0.03) ms，其中 0.34 ms/token 已无软件空间。两条杠杆：

| 杠杆 | 作用域 | 收益 | 代价/风险 |
|---|---|---|---|
| 卡 1 移到 CPU 直连 Gen5 x8 槽 | 硬件 | 浅/中上下文 **1,905 → ~3,390 tok/s（1.78×）**；245k 处 569→655（1.15×）；14k 处 1,810→2,740（1.51×） | 零代码；需主板有空槽 |
| AR 与 MMA 重叠（环形 staging + 事件同步 + 子块流水） | 软件 | 上限 ~1.5×（AR 隐藏后 ~4,760 tok/s 浅层上限） | 大重构，跨卡 rendezvous 死锁风险 |
| GQA 感知 prompt 内核（见上） | 软件 | 深上下文 1.5–1.8× | 内核工程 |

三者互补：硬件换浅层、内核换深层。P2P 不可用的根因也是这个槽位——`canAccessPeer=0`（worklog:200）、`nvidia-smi topo` 显示两卡为 **SYS**（不同 root complex），所以 peer 流量必须过主机桥，这也是 host-staging 设计的由来（`device_pair.h:114`）。

### 3.14 待实施的两个软件杠杆（2026-09-23；L1 已开工）

按 §3.13 的账，prefill 两个成本项各有杠杆，且**互补**（一个作用于 `b·d`，一个作用于 `a`）：

| # | 杠杆 | 作用项 | 预估收益 | 前置/风险 |
|---|---|---|---|---|
| **L1** | **GQA 感知 prompt 内核**：让共享同一 KV 头的 6 个 Q 头共用一份 KV 数据流（现在 `grid=(tokens/64, QHeads=12)`，每块只处理 1 个 Q 头，单卡 2 个 KV 头被各读 6 遍） | `b·d`：深上下文 attention，245k 处占 **70%** | 深上下文 **1.5–1.8×**（245k: 569 → ~830–1,000 tok/s） | 先过测量门确认 6× 冗余是否真落到 DRAM；只改调度与数据复用，**不动物理累加顺序 ⇒ 可与现内核逐位一致**。**（2026-09-23 测量门否证 ⇒ 已取消，见下）** |
| **L2** | **AR 与 MMA 重叠**（环形 staging + 事件同步 + 子块流水） | `a`：深度无关项，其中 AR ≈ 65% | 浅层上限 **~1.5×**（1,905 → ~4,760 tok/s 上限） | 跨卡 rendezvous 死锁风险，属大重构（worklog Round 35c 已评估） |

> 硬件杠杆（卡 1 从芯片组 Gen4 x4 槽移到 CPU 直连 Gen5 x8）不在表内：零代码但需主板槽位，收益 1.78×（浅/中上下文），与 L1/L2 可叠加。

**L1 执行顺序**：
1. **测量门（先做，未过则不动代码）**：ncu 抓 `causal_attention_prompt_k8v4_kernel`（生产几何 12q/2kv/d256/k8v4，KV 深度 ≥100k），看
   `dram__bytes_read.sum`、`lts__t_sector_hit_rate.pct`、tensor pipe 利用率、achieved occupancy。
   - L2 命中率高且 DRAM 已近峰值 ⇒ 6× 冗余已被 L2 吸收 ⇒ 杠杆变小，改走"提占用/降延迟"；
   - DRAM 流量 ≈ 6× 唯一 KV 字节且吞吐近峰值 ⇒ 冗余真实 ⇒ 进入第 2 步。
2. **设计**：块的行维度由「1 头 × 64 行」改为「6 头 × R 行」（R 由 smem 预算反推，Q/K/V/P/stats 各项已在 `prompt_k8v4.cuh` 里算过），
   KV tile 仍按 Bc=64 顺序流过；**每行看到的 KV tile 顺序与累加顺序不变** ⇒ 目标是与现内核逐位一致。
3. **实现**：只改 `prompt_k8v4`（生产 dtype）；其余四种 dtype 的 prompt 内核暂保持现路径（同构，后续同步）。
4. **验证**：OP 级 FP32/FP64 oracle（12/2/d256 深包络）+ 同步 `causal_attention_workspace_capacity_bytes` + 模型级金值（`tp2_dflash_solo`/`append` 摘要）+ 复跑 §3.12 深度曲线作性能对照。

**L1 测量门结论（2026-09-23，已否证 ⇒ 不改内核）**：用现成的 op 级基准 `bench/ops/causal_softmax_attention_bench`（本次为它补上生产分片几何 `d256-h12-kv2`——op 与测试本就支持，bench 落后）在 W=1024（Prompt 路线）、h12-kv2 下实测，245,760 深度单层：

| KV dtype | 耗时 | 达成 math | 唯一 KV 字节 |
|---|---|---|---|
| bf16 | 82,927 µs | 37.4 TFLOP/s | **482.0 MB** |
| int8 | 83,207 µs | 37.2 | 248.5 MB |
| fp8 | **73,782 µs** | 42.0 | 242.9 MB |
| nvfp4 | 82,270 µs | 37.7 | 135.6 MB |
| k8v4（生产） | 73,910 µs | 41.9 | 189.2 MB |

**KV 字节跨 3.5×，耗时只差 13%**：fp8 比 k8v4 多 28% 字节却同耗时；nvfp4 少 28% 字节反而慢 11%；bf16 字节 2.55× 只慢 12%
⇒ **内核不是 KV 带宽受限，而是计算/ALU 受限** ⇒ 去掉 6× 请求冗余不会有时间收益，**L1 按原设计取消**（内核代码未动，符合门禁约定）。

**顺带确定的天花板**：同机 T=1024 下 `ninfer_fp8_linear_add_bench`（A8, K=17408）**85.9 TFLOP/s**、`ninfer_q4_linear_swiglu_bench`（W4A4）**44.1 TFLOP/s**；
attention 的 QK^T 与 PV 各约 21 TFLOP/s（合计 41.9）。与纯 FP8 GEMM 比每个 attention GEMM 只到 24%，
但 k8v4 与 fp8 同耗时说明**差距不只在 MMA dtype**，而在内核结构（k8v4 的 PV 走 FP16、Hadamard 旋转/行缩放、softmax、smem 流量）。
⇒ 若仍要攻 attention，唯一形态是**计算路径效率**（FP8 PV + 减少 ALU 遍数），且**必须先有 profiler**：本机**未安装 Nsight Compute（ncu）**，黑盒计时已到极限。
预期：attention 完全消除 ⇒ 245k 处 569 → ~1,900 tok/s（3.3× 上限）；现实减半 ⇒ ~1.4×。

**基线留档**（W=1024、h12-kv2、k8v4、单层 median）：8,192 → 2,637 µs；32,768 → 9,924；65,536 → 19,630；131,072 → 39,602；245,760 → 73,880。
线性度极好，且 ×16 层 = 1.18 s/chunk，与 §3.12 模型级 1.26 s/chunk 只差 6% ⇒ 两条独立测量互证。
复跑：`ninfer_causal_softmax_attention_bench.exe --entry cached --geometry d256-h12-kv2 --kv-dtype k8v4 --tokens 1024 --context 8192,32768,65536,131072,245760 --execution eager --cache cold`。
**构建注意**：`-DNINFER_BUILD_BENCHMARKS=ON` 在 Windows 上会让默认目标失败（`bench/context_cost/model_context_fixture.cpp:400` 用了 MSVC 不支持的 `__int128`），
本次只按目标构建（`cmake --build build-win --target ninfer_causal_softmax_attention_bench`）；该缓存开关已复位为 OFF，不影响常规构建。

**L1 后续（2026-09-23）：profiler 已就绪，但被驱动权限挡住**

- 已安装 Nsight Compute 2025.4.1（winget `Nvidia.Nsight.Compute`），`ncu --version` 正常；但非管理员进程读性能计数器被拒：
  `ERR_NVGPUCTRPERM`（target device 0）。解锁二选一：① 用**管理员** PowerShell 运行 ncu；
  ② NVIDIA 控制面板 → Desktop → Enable Developer Settings → Developer → Manage GPU Performance Counters → 允许所有用户访问 → 应用（持久生效）。
- 无计数器可得的归因（已完成）：
  * **W 扫描**（深度 65,536、k8v4、h12-kv2）：W=256 → 31.7、512 → 41.8、1024 → 41.8、2048 → 45.4 TFLOP/s
    ⇒ **W=1024 已在平台区**；把 prefill chunk 上限从 1024 提到 2048 只值 ~9%，不是杠杆。
  * **同机 GEMM 天花板**（T=1024）：FP8 A8 **85.9**、W4A4 **44.1** TFLOP/s；attention 平台约 45（QK+PV 合计），
    而 QK^T / PV 各约 21–23 ⇒ QK^T(FP8) ≈ 其 dtype 天花板的 25%，PV(FP16) ≈ 50%。
  * 4-bit K 路径（nvfp4-K vs fp8-K）贵 +11%；bf16（完全无反量化）反而慢 12% ⇒ **MMA dtype 比反量化更主导**。
- **候选改动（待 profile 确认后再动，遵守门禁）**：把 prompt 内核的 KV tile 宽度 `Bc` 从 64 提到 128
  （QK^T 的 N 维翻倍、KV 循环次数减半；smem 由 ~99 KB 增至 ~165 KB，仍可单块容纳）。

**L1 实验记录（2026-09-23）：两条低成本路径均被硬件墙挡死，内核已回退**

ncu 对 `causal_attention_prompt_k8v4_kernel`（W=1024、65,536 深度、k8v4、h12-kv2）的实测：

| 指标 | 值 | 含义 |
|---|---|---|
| Compute (SM) Throughput | **63.54%** | 计算为主 |
| Tensor 管线 | **63.5%（最高）** | 张量管线是最忙资源 |
| **DRAM Throughput** | **1.09%** | KV 流量基本全被 L2 吸收 |
| L2 / L1 吞吐 | 10.10% / 40.43% | 都不是瓶颈 |
| Issue Slots Busy / IPC | 30.89% / 1.39 | 不是发射受限 |
| 最大 stall | 等数学管线 4.0/11.5 周期（34.4%） | 张量管线排队 |
| Occupancy | 理论=实际 **33.33%**（16/48 warp） | 1 block/SM |
| Block Limit | 寄存器 1 / smem 1 | 双重限制 |
| 寄存器/线程 · 动态 smem | **100** · 85.12 KB（配置 102.40 KB） | — |

⇒ **"减 6× KV 冗余"彻底证伪**（DRAM 1%）；真瓶颈是张量管线 + 33% 占用。

两次实验（均先过 oracle 正确性）：
1. **Warps 16→32**（同 smem 内翻倍 warp，寄存器上限随之压到 64）：正确性 **PASS**，性能 **−12%**（65k: 19.63→22.17 ms；245k: 73.88→82.43 ms）。
   原因结构性：1024 线程的块被 64K 寄存器文件硬顶在 ≤64 寄存器/线程，而内核需 ~100 ⇒ spilling。**死路**。
2. **Bc 64→128**（加宽 KV tile 改善 MMA 形状，smem 85,120→151,808 B）：**无法启动**，
   `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` 返回 `cudaErrorInvalidValue` ⇒ 消费级 Blackwell **单块动态 smem 上限 ~100 KB**。**死路**。

**结论**：该内核已贴住本机硬件边界（张量管线最忙、DRAM 空闲、占用被"寄存器文件 + 100 KB smem"同时锁死）。
剩余可做的都更大且收益有限：Br=32/256 线程的 tile 重构争取 2–4 block/SM（受寄存器支配）、PV 改 FP8、压缩非张量 FP32 指令（占指令 11%，ncu 提示 ~3.6%）。
上限：attention 项约 1.2–1.4× ⇒ 245k 端到端 ~1.1–1.3×、50k ~1.05–1.1×。
⇒ **性价比排序更明确：硬件移槽（浅/中 1.78×，零代码）> L2（AR/MMA 重叠，浅层 ~1.5×）> attention 重构（深上下文，收益最小、风险最大）。**

**顺带修复**：`bench/context_cost/model_context_fixture.cpp` 的 `__int128` 改为等价的 uint64 溢出检查（两个累加项各自不溢出，只需检查其和），
`-DNINFER_BUILD_BENCHMARKS=ON` 现在在 Windows 可正常构建（已用默认目标 `cmake --build build-win` 验证通过）。

---

## 4. DSH/agent-loop 前缀复用修复（2026-09-24，进行中）

**问题**：DSH 接入 ninfer 的 agent loop 里，某轮生成大 tool call（写大文件，24k token）后，
下一轮必须把这 24k token 重新 prefill（实测 req#3：prompt 34,050 / cache 9,216 / prefill 24,834 tok / TTFT 17.0s）。

**已确证（NINFER_TP2_REUSE_TRACE=1 / NINFER_TP2_SESSION_TRACE=1 实测）**：
1. 引擎保存了上一轮完整历史（`cached` = prompt+generated），但 `shared` 等于上一轮 **prompt** 长度
   （332/331/315 三例），即**第一个生成 token 就不匹配** → 只能从上一轮 prompt 末尾重启。
2. 用 `/v1/chat/completions` 原样回放 `reasoning_content` 复现同样结果 → 不是 DSH 特有。
3. artifact 内嵌模板对 `reasoning_content` 做 `|trim`，把模型生成的首个空格吃掉（实测
   `"ABCD"` 与 `"  ABCD  "` 渲染出的 prompt 长度完全相同）；而模型 reasoning 首个 token 带前导空格，
   因此任何文本回放都无法逐 token 复现。
4. TP-2 路线自建 frontend 时**漏传 `chat_template_path`**（`tp2_generation_core.cpp:472-480`），
   所以 `--chat-template` 在该路线被静默忽略（实测 developer 角色仍报 `artifact:chat_template.jinja`）。
5. 复用扫描里"对齐边界优先"会遮蔽更深的未对齐 live frontier，只有在 `reuse == 0` 时才走兜底；
   而 DFlash2 的兜底带代价门控（放弃 draft ≈ 3× 解码变慢 vs 省下的 prefill）。

**结果（2026-09-24）**：
- [x] A `tp2_generation_core.cpp`：TP-2 frontend 透传 `chat_template_path`。实测：部署新件并带
      `--chat-template` 重启后，`developer` 角色由 400 变 200（覆盖真正生效）。
- [x] C `tp2_generation_core.cpp`：兜底边界扫描改为无条件第二轮，门控改用"边际节省"
      （`position - reuse`），对齐边界仍是最低优先级。`dflash2`/`mtp` 两路 `ninfer_qwen3_5_tp2_sessions_test` PASS。
- [x] D `tests/models/qwen3_5/test_tp2_sessions.cpp`：新增常驻会话连续两轮用例（1000 token prompt；
      二轮期望 plain/mtp = 1007、dflash2 = 768）。没有 C 时 plain/mtp 会停在 768，用例能抓住该改动。
- [x] E 构建（`nm`/ninja，测试目标 + `ninfer-serve` 并同步 `C:\ninfer\ninfer-serve.exe`）+ 测试 +
      在线复测；证据：`profiles/sessions-{dflash2,mtp,plain-trace,all}.log`、`profiles/reuse-trace*.log`。
- [x] F 文档：`docs/tp2-dual-5060ti.md` 增补常驻连续两轮、门控与 `--chat-template` 说明。
- [~] B 模板 `|trim` → `rstrip('\n')`：**已回退（实测否定）**。覆盖为维护模板后原样回放
      `reasoning_content` 仍得 `shared == 上一轮 prompt 长度`，与 artifact 模板结果一致。

**残留根因（新发现，未修）**：文本协议回放无法逐 token 复现。`max_tokens=1` 时实测
`completion_tokens=1 / reasoning_tokens=0 / reasoning_content 为空`；k=2 得 `" need"`、k=3 得
`" need answer"`，而回放一律停在 `shared=72`（上一轮 prompt 长度）。即**第一个生成 token 不在已发布的
reasoning 文本里**（或回放重分词与采样边界不同），因此 DFlash2 上"复用上一轮回答尾部"对任何文本客户端
都不可达。再叠加门控（放弃 draft ≈ 3× 解码 vs 省下 prefill），32k 输出预算下重算 24.8k token（17 s）
是代价模型下的**理性选择**。

**可选杠杆**：客户端输出预算（`remaining` 需 < 节省量/16 ≈ 1.5k token 才触发复用）；让 DFlash2 在
未对齐复用下保持 draft（需要先解决 `extent=0` 钳位轮的可复现性）；或走携带 token id 的续写协议。

**既有失败（与本次改动无关）**：`NINFER_TEST_ROUTE=plain` 在 `shared_a_first` 处 `reuse=0/src=none`
的从零 walk 与冷 oracle 首 token 不同（`[2752 13 198 …]` vs `[365 2798 349 …]`）。trace 证明该场景及
其之前所有复用决策（0/71/512）与测试期望一致，新扫描路径在 `shared_prefix=0` 时不可能取任何边界。

## 5. TP-2 首个生成 token 不发布（2026-09-24，已修）

**现象**：`max_tokens=1` 时 `completion_tokens=1 / reasoning_tokens=0 / 无文本`；`max_tokens=k` 只发布 k−1 个 token
的文本；逐字节回放上一轮回答（chat-completions 原样回传 `reasoning_content`+`content`，带/不带 tools、换模板都一样）时
`shared` 永远等于上一轮 prompt 长度（实测 67 / 306 / 72）。于是 TP-2 上"复用上一轮回答尾部"对任何文本客户端都不可达，
`cache` 在服务日志里恒为 `floor(上一轮 prompt/1024)×1024`。

**根因**：`src/runtime/engine/tp2_generation_core.cpp` 的 prefill 末尾用 `request.generated.push_back(first)` 直接落账，
**没有经过输出会话**（`preview_model`/`commit_preview`）。单卡路线会把 prefill 采样的首 token 经
`commit_pending`→`preview_model` 提交（`src/runtime/engine/engine_core.h`），TP-2 是唯一漏掉的路线；因此
`generated`（KV/状态/会话目录索引的序列）比"已发布文本"多一个 token。

**修复**：prefill 采样后 `preview_model({first}, budget.remaining(), limit_reason())` → `budget.commit` →
`publish_preview(false)`；若该决策已终止请求（stop token 或预算用尽）则记入 `first_token_finish` 并跳过 decode 循环。
这也顺带修掉"首 token 是 stop token 时引擎仍继续生成"和首 token 未受语法约束（前者已修，后者记录为独立遗留项）。

**验证**（证据：`C:\ninfer\serve-win.log` 的 `[tp2-reuse]` trace、`profiles/fixtok-*.log`）：
- `max_tokens=1` → `reasoning_tokens=1`、`R="We"`（修复前 0 / 空）。
- 回放 40 token 回答：`shared=107`（= 67 prompt + 40 生成；修复前 67）。
- 回放 1400 token 回答：`reuse=1470 src=live` = 整个上一轮回答（DFlash2 按门控对该 1-token 请求弃稿，代价可忽略）。
- `ninfer_qwen3_5_tp2_sessions_test`：dflash2 / mtp 均 PASS（既有断言无回归）。
- 新增用例 `check_first_token_published`：thinking 开、输出预算 1 token 时 `reasoning_tokens` 必须为 1。

**残留（未修）**：门控仍按 `max(reuse_grid, 16 × 剩余预算)` 决定是否取未对齐边界，所以 DSH 的 32,768 预算下长回答仍会整段重算
（代价模型下的理性选择）；prefill 的首个 sample 仍未应用工具语法掩码（TP-2 prefill 与 decode 路径的不对称，独立遗留项）。

**附带观察（未修，独立）**：在服务实例上以 `temperature: 0` 连发同一请求 3 次，前 4 个 token 出现两种结果
（`"The user wants me"` / `"We need to respond"`）。`src/serve/translate.cpp:38-88` 会把请求里的 temperature 覆盖到服务默认值上，
所以温度确实被应用；DFlash2 的稀疏接受（`speculative_accept_sparse_drafts`）配合每次请求随机 seed（`translate.cpp:51-57`）
使解码在温度 0 下也不保证逐位 argmax——这正是 `--greedy`（"force temperature 0 (exact argmax)"）存在的原因。
需要可复现 A/B 时应加 `--greedy`；这条独立于本次修复，未改动。

## 6. 复用门控改按"会话自身轮次"定价（2026-09-24，已改）

**问题**：门控 `position - reuse <= max(reuse_grid, 16 × request.budget.remaining())` 里的 budget 是客户端设的上限
（DSH 固定 32,768）。预算 ≥ 16K 时 `16 × budget >= max_context`，未对齐回退分支**在数学上不可达**——而它是尾部复用
的唯一入口（网格上没有任何 decode 检查点），也就是说该机器生产配置下永远不会触发，即使节省量是整个上下文。

**改动**：
- `SessionEntry` 增 `generated_tokens_total` / `generated_turns`；每轮结束后按 `request.generated.size()` 累加
  （只统计生成 >=1 token 的轮次，保证均值非零）。
- 门控改用 `expected = min(budget, total/turns)`；没有已完成轮次（首轮请求、或关闭 retention）退回 budget（原最坏情况）。
- 语义上只放宽（`min(budget, mean) <= budget`），任何原先接受的边界仍被接受。

**验证**：
- 测试 `ninfer_qwen3_5_tp2_sessions_test`：dflash2 / mtp EXIT 0。新增场景 = 5 短轮 + 1 长轮（60 token）+ 1024 预算
  续轮，期望复用直达 frontier；另加两条场景自检，若某天不再落在 take 分支会带数字明确失败。证据 `profiles/gate-{dflash2,mtp}.log`。
- 在线（serve + trace）：20 短轮（均值 2）+ 1 个 2000-token 长轮 → 续轮 **budget 32,768** 下 `reuse=2442 src=live`、
  `cache 2,442 (99.3%)`、**TTFT 57.9 ms**（prefill 仅 17 token）；旧规则阈值 16 x 32,768 = 524,288 必然跳过。
  同一请求按门控弃稿，decode 22.5 tok/s、输出 37 token ⇒ 代价约 1.2 s < 省下约 1.6 s，本注是净赚。

**已知边界（故意保留）**：定价用的均值包含突发轮，所以"突发之后紧接的短轮"能否复用取决于该会话的均值——
均值 > 节省量/16 时仍然跳过（风险中性结论，不是 bug）。当前 DSH 会话均值约 1,735，其 24.8k 尾部仍会被跳过；
要翻转只能把比率 16 做成可配选项，或走第 3 步（未对齐复用下保住 masked draft）。

## 7. 未对齐复用不再弃稿：masked draft 全程在线（2026-09-24，已改）

**前提被实测推翻**：第 2 步的门控建立在"未对齐复用必须弃掉 masked draft"之上。实测这个前提不成立——
保留 draft 后 ring 只有 ulp 级差异，接受率几乎不变。

**实测（同一 artifact、serve 配方 + trace；20 短轮均值 2 + 一个 2000-token 长轮，续轮 budget=32768）**：
- 复用结果相同：`reuse=2442 src=live`、`cache 2,442 (99.3%)`、TTFT 58 ms、prefill 仅 17 token；
- 弃稿（旧行为）：`decode 22.5 tok/s`，输出 37 token；
- 保留 draft（新行为）：`decode 87.1 tok/s`、`dflash2 accepted 28/63 (44.4%)`。
弃稿保护的从来不是正确性（target verify 始终为每个 token 发牌），只是"逐位复现 from-scratch walk"这一契约。

**改动**：
- `extent` 不再因边界未对齐而归零；删除 `dflash_draft_declined_`、`GenerationResult::draft_context_declined`、
  stderr 提示与 solo 测试里的打印。
- 复用扫描合并为单趟：网格过滤与门控删除，**最深边界获胜**；`reuse_grid` 随之删除。
- 第 2 步的 `SessionEntry::generated_tokens_*` 与会话均值定价一并删除（前提消失，门控失去意义）。

**契约（严格版实验定界）**：边界在网格上 ⇒ 与 oracle 逐 token 一致；边界在 chunk 内 ⇒ 只保证 boundary crossing。
把 `compare_recall` 临时改成"处处要求逐 token 相等"实跑，失败于 `a conversation behind a re-rendered answer`：
`got [2752 11 220 16 24 24 15 82]` vs `expected [2752 11 198 220 220 16 15 15]` —— 前两个 token 相同、第三个分叉，
与旧注释预测的"第三 token 必然分叉"一致，说明分叉来自未对齐的 prefill 分块（窗口切分不同），不是 draft 模式。
保留 draft 反而把分叉来源从两个（模式 + 分块）减到一个（分块）。

**验证**：
- `ninfer_qwen3_5_tp2_sessions_test`：dflash2 / mtp EXIT 0；plain 在本机失败于既有场景 "a conversation behind a
  shared system prompt"（`got [2752 13 198 ...]` / `expected [365 2798 349 ...]`），与上一会话记录的同一失败逐位一致。
  本次提交对 plain/mtp 是**行为等价重构**（旧门控只在 `dflash2_enabled_` 时生效，两者早已取最深边界），故与该失败无关。
- `ninfer_qwen3_5_tp2_dflash_solo_test`（重建后）：两个独立进程 digest 均为 `0x4bcc3994a5efba7d`、PASS，与文档记录值一致
  ⇒ 网格对齐路径未被扰动（solo 场景本身不含未对齐复用）。
- served 路由（step-3 二进制）：`reuse=2442 src=live`、`cache 99.4%`、TTFT 51-57 ms、prefill 14 token、无弃稿提示。
  同一二进制独立重启两次结果逐位相同（`output 52`、`accepted 31/147 (21.1%)`、`decode 53.9 tok/s`）⇒ 该二进制确定性，
  21% 不是 run 间抖动（早先"temperature 0 非确定"的结论只适用于当时那个场景）。
- **接受率是文本属性，不是本次改动的属性**：同一复用边界、同一二进制，只改续写内容 —— prose 53.9 tok/s / 21.1%，
  数字列表 175.8 tok/s / 99.6%（446/448）。因此 `44.4%`（实验那次，数字型、output 37）与 `21.1%`（终结版，prose、output 52）
  是两段不同文本的度量，不能当 A/B；真正同 walk 的 A/B 是"弃稿 22.5 tok/s vs 保留 87.1 tok/s / 44.4%"（同一 conversation、
  同一 scan，只切换 extent）。
- 两次构建为何会造出不同文本：门控删除也改变了**早期短轮**的复用边界（旧门控要求 `saving > max(1024, 16×expected)`，
  短轮 prompt 的 off-grid 节省量远小于 1024，旧代码退到对齐边界；新代码取 frontier）⇒ 前缀不同 ⇒ 2-token 短答案在 tie 处可能不同
  ⇒ 内容级联分叉（观测：旧 2,459/37 vs 新 2,456/52）。两者都是合法 walk。

## 8. DFlash2 draft 二次量化 r62–r66（2026-09-24；立项，未开工）

**背景**：§3.7（r54–r57）把 draft 降到「当时消费 op 支持的最低 qtype」，此后每个 draft 组件都坐在
op 级下限上。2026-09-24 对剩余空间做了逐 op 支持性审计（证据链：消费 op 的 qtype 校验器），结论是
**当前构建没有一项可以「只改配方」落地**——剩余空间全部需要新 op 变体。本战役做其中性价比最高的四项
（codebook 与环 FP8 明确缓做，见下）。

**审计结论（每项的精度锁点）**：

| 组件 | 当前 | 锁点（消费 op） | 判定 |
|---|---|---|---|
| mlp/gate_up [34816,5120] | q4 | `linear_swiglu` q4 plan 形状精确闭合（`q4_linear_swiglu_plan.cpp:33`） | 已在底，不动 |
| mlp/down [5120,17408]、attention/output [5120,4096] | q5 | 融合 `linear_dynamic_grouped_conv_add` 校验器只放行 Q5_G64/Q8_G32（`wrapper/dynamic_grouped_conv.cpp:60-95`） | **r63**：新 q4 变体 |
| feature_projection [5120,25600] | q5 | 普通 `ops::linear`；q4 注册表无 {5120,25600}（`q4_dispatch.cpp:13-25`） | **r62**：新 q4 shape |
| context_key [6144,5120] | q8 | decode：`attn_input_proj` 三输出重载硬断言 `require_q8_rowsplit`（`wrapper/attn_input_proj.cpp:243-261`）；prefill：`context_kv_materialize` 硬断言 Q8_G32 [1024,5120]（`context_kv_materialize.cpp:37-52`） | **r65**：两处新 q4 变体 |
| kernel_projection [1280,5120] | bf16 | `rmsnorm_dynamic_grouped_conv_prepare` 硬断言 BF16+Contiguous+无 scale（`wrapper/dynamic_grouped_conv.cpp:39-52`） | **r64**：新 q4 prepare |
| codebook [248320,256]×2（shard 1） | bf16 | `candidate_selector_path` 唯一 bf16 变体（`wrapper/candidate_selector.cpp:110-113`）；且参数是裸 `Tensor` 非 `Weight`（`parameters.h:113`）⇒ 还要改 Tensor→Weight 管线 + TP-2 peer 路径 | **缓做**（改动最深 + 接受率敏感度最高：256 维内积选择） |
| 环 K=BF16/V=FP16 | — | 三处锁死：`CyclicKVCache` 布局、SWA 校验（`sliding_window_attention.cpp:60-62`）、`context_kv_materialize` 的 `validate_cache`（:98-99）；窗口只注册 {2048, 4096} | **缓做**（~55 MiB，工程量最大、性价比最差） |

**目标收益**（shard 0，精确值；GPU0 free 607 → ~860 MiB ≈ +16k token KV 上限）：

| 项 | 对象 | 当前 → 目标 | 节省 |
|---|---|---|---:|
| r62 | feature_projection | q5 82.03 → q4 66.4 | **15.6** |
| r63 | attention/output ×5 + mlp/down ×5 | q5 344.5 → q4 278.9 | **65.6** |
| r64 | kernel_projection ×10 | bf16 125.0 → q4 33.2 | **91.8** |
| r65 | context_key ×5 | q8 159.4 → q4 79.7 | **79.7** |
| 合计 | | | **252.7** |

**决策（沿用 §3.7）**：四项均为 opt-in override，不改官方配方（`official_recipes.py:35-46` 继续把
mtp/dflash/dflash2 钉在 Q8、kernel_projection/codebook/hidden_projection 排除在量化外）；artifact 为实验件。
升级为受契约保护的特性前必须补 §3.7 末尾三件套（append K=7/K=5 金值、solo digest、sessions 保留断言）
+ 采样模式 A/B + Q4 op oracle 的「按存储 scale 独立解码」检查。

### 8.1 逐项改动清单

**r62 — feature_projection → q4（最小面，先打通全链路）**
- 新增 `src/ops/linear/q4/shapes/n5120_k25600.cu`（selector 镜像 `src/ops/linear/q5/shapes/n5120_k25600.cu`：
  T=1 simt_r8_c4、T=2–6 ksplit、T≤24 simt_r8_c8、其余 mma_r64_c128；draft prefill 宽度 ≤2048 × batch）。
- 改动：`q4_shapes.h` 声明 + `q4_dispatch.cpp` 注册表加 {5120,25600} + `src/ops/linear/q4/sources.cmake`。
- 测试：`tests/ops/linear/test_q4_a16.cpp` 加 5120×25600（独立 seed、FP64 oracle、全路由边界）。
- override `tools/tp_bootstrap/r62_draft_feature_q4.py`（`dflash2/feature_projection` → q4_g64_fp16，
  `grouped_absmax`）+ dry-run + 转换脚本（r57 配方 + `--device cpu`）。

**r63 — attention/output + mlp/down → q4（融合 conv-add 的 q4 变体）**
- 关键结构事实（审计发现）：q5 变体的 GEMM 部分委托普通 q5 注册表（`select_q5_a16_launch`），
  卷积+残差尾是 qtype 无关的共享 `dynamic_conv_finish_launch`（`q5_dynamic_grouped_conv_add_materialized.cu:11-20`）
  ⇒ **q4 变体不需要新融合 kernel，只需要两个新普通 q4 GEMM 形状 + q4 plan 目录**（比原「新融合 kernel 一族」预估小）。
- 新增 `src/ops/linear/q4/shapes/n5120_k4096.cu`、`n5120_k17408.cu`（镜像 q5 同名 shape）。
- 新增 `src/ops/dynamic_grouped_conv/q4/`（plan/materialized，镜像 `q5_dynamic_grouped_conv_add_plan.cpp`：
  投影委托 `select_q4_a16_launch(5120, input_rows, tokens)`，尾走共享 finish；路线名
  `dynamic_grouped_conv_add.q4.*.materialized_bf16`）。
- 改动：`wrapper/dynamic_grouped_conv.cpp`（`require_finish_projection_weight`/`projection_planes`
  放行 Q4_G64 group 64、无 qhigh；`linear_dynamic_grouped_conv_add` 加 q4 分派分支）；
  `include/ninfer/ops/dynamic_grouped_conv.h` 三 codec 语义同步；容量 API 已有 qtype 首参（r56 加过），只补 q4 分支。
- 测试：`tests/ops/test_linear_dynamic_grouped_conv_add.cpp` 加 Q4 × C∈{4096,17408} 全 W/B 域（2..128 tokens）
  × 图重放，FP64 oracle 按存储 scale 独立解码（r56 先例：120 形状 4.38 s）。
- override `tools/tp_bootstrap/r63_draft_finish_q4.py`（`/mlp/down`、`/attention/output` → q4）+ 计数校验（各 5）。

**r64 — kernel_projection → q4（prepare 的 q4 变体）**
- 新增 q4 prepare partial kernel（[1280,5120] q4 反量化 GEMM，split-K）+ q4 plan（镜像
  `bf16_dynamic_grouped_conv_prepare_plan.cpp` 的路线：tokens≤48 走 R16C{8,16,32,48}S8，余 R32C{32,64}S4）。
- 改动：`wrapper/dynamic_grouped_conv.cpp`（`require_kernel_projection_weight` 放行 Q4_G64；
  `rmsnorm_dynamic_grouped_conv_prepare` 加 q4 分派）+ `sources.cmake`。
- 测试：`tests/ops/test_dynamic_grouped_conv_prepare.cpp` 加 q4 臂：prepare = rmsnorm∘GEMM∘(base+delta)
  独立 FP64 参考实现，全 T 域（2..128）。
- **数值性质警示**：该 GEMM 的产物是动态卷积权重本身（权重空间误差，非激活空间误差），敏感度可能高于
  r62/r63 的普通 GEMM ⇒ 接受率门禁必须过；**回退 = 保持 bf16**（本项可选，失败不阻塞战役，总收益降为 160.9）。
- override `tools/tp_bootstrap/r64_draft_kernel_q4.py`（`/attention_conv/kernel_projection`、
  `/mlp_conv/kernel_projection` → q4；官方配方目前对这两个名字是 `continue` 排除，dry-run 确认命中 10 对象）。

**r65 — context_key → q4（两处 op，最大项，最后做）**
- decode 路径：新增 `src/ops/attn_input_proj/q4/` plan + 6 个 dflash2 schedule（镜像 `kDFlash2Routes`：
  SmallT ≤48、MmaR16C64K128、MmaR32C32K128、MmaR32C64K128、MmaR32C64、MmaR64C128；draft decode T=(K+1)×B ≤64，
  目录仍按闭合惯例覆盖到 kAnyCols）。
- prefill 路径：新增 `context_kv_materialize` 的 q4 变体（7 条路线 KSplit16/KSplit24/Mma32/Mma80/Mma96/Fused64/Mma64
  全要——prefill 宽度 1..2048 都会触达；q4 反量化 = 2 code/byte + scale 每 64，对照现 q8 的每 32；
  新文件 `materialize_q4.cu` + launch 按 qtype 分派）。
- 改动：`wrapper/attn_input_proj.cpp`（三输出重载加 Q4 分支：`require_q4_rowsplit` [6144, hidden]）；
  `context_kv_materialize.cpp`（`require_weight` 放行 Q4_G64 group 64、scale 字节 rows×80×2；dispatch 按 qtype）。
- 测试：`tests/ops/test_attn_input_proj.cpp` 加 dflash2 三输出 q4 路由边界；materialize 测试：5 层 ×
  全 W/B profile，环内容对朴素参考（q4 GEMM → key_norm rmsnorm → RoPE → 环形写入）逐元素比对。
- **recipe 注意**：官方配方 `recipe.share(prefix+"context_key", prefix+"key")`（`official_recipes.py:52-54`）
  使 context_key/context_value 与 query/key/value 绑定共享同一 [6144,5120] 对象 ⇒ override 对共享对象赋值，
  dry-run 必须确认「恰好 5 个对象变 q4_g64_fp16」（仿 r57 计数校验）。
- **接受率风险**：本批最高（K 投影误差在 2048 窗口环内逐 token 累积，影响窗口内每个后续 decode 步）
  ⇒ 门禁必须过；失败回退 = 保持 q8。
- override `tools/tp_bootstrap/r65_draft_contextkey_q4.py`。

**r66 — 合并实验件**：r62–r65 四项合一（override `r66_draft_q4_all.py` 含 26 对象计数校验：
feature 1 + down 5 + output 5 + kernel 10 + context_key 5），转换后全量验收。

### 8.2 每步验收标准（门禁）

1. **op oracle**（AGENTS.md 数值契约）：每个新形状/路线 vs 独立 FP32/FP64 朴素参考；packed 输入按
   **存储 scale 独立解码**（不信任转换器输出的 scale 语义）；路由边界（T 分界点 ±1）全覆盖；图重放覆盖。
2. **接受率 A/B**（沿用 §3.7 方法）：greedy 探针 7×160、K=5 与 K=7、每臂独立冷启服；
   基线臂 = 现役 r57 件（`D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer`，r57 实测 acc K=7 25.3% /
   56.9 tok/s）；**通过线 |Δacc| ≤ 1.0pp 且 Δtok/s ≥ −3%**（r54–r57 观测带 ±0.5pp / ±1%）。
   每步另加一个组合中间臂（前序已通过项 + 本项）以隔离交互效应。
3. **确定性**：基线臂跨重建逐字节复现（r57 先例：2843/699、文本 len 全同）；新臂接受率必须解释到
   探针文本漂移（tie 翻转）而非系统性下降。
4. **显存核对**：`nvidia-smi` + 启动 `[mem]` 台账，设备侧节省与精确 MiB 计算差 ≤ 分配粒度（先例 ~1.6 MiB）。
5. **升级门禁（r66 后）**：DFlash2 三件套对新件重基线（`ninfer_qwen3_5_tp2_dflash_append_test` K=7/K=5 金值、
   `ninfer_qwen3_5_tp2_dflash_solo_test` digest、`ninfer_qwen3_5_tp2_sessions_test` 保留断言）+
   采样模式（temp 0.7）接受率/吞吐 A/B（r54 先例：采样下 K=2 曾 −2.5pp 1.6σ）。

### 8.3 回退点

- **C++ 侧**：全部为加法（新 shape 文件、新 q4 plan 目录、新校验分支）；既有 q5/q8/bf16 路线位型不变
  （r56 先例：共享 finish 重构后 Q8 路线零漂移）⇒ 回退 = git revert，无 artifact 格式变化。
- **artifact 侧**：每项独立 override，官方配方不动 ⇒ 回退 = 用前一组 override 重转（~300 s）；
  旧 artifact 永远可被新引擎运行（引擎是超集）。
- **项级回退**：r64 或 r65 接受率不过 ⇒ 该项保持现精度，战役继续（r64 失败总收益 160.9；
  r65 失败 220.5；两者都失败 140.0）。

### 8.4 执行顺序与依赖

0. 基线复测：现役 r57 件 K=5/7 接受率 + 吞吐（确认与 §3.7 记录的 25.3%/56.9 一致，环境无漂移）。
1. **r62**（最小面：打通 shape→dispatch→oracle→override→转换→A/B 全链路，验证方法论）。
2. **r63**（复用 r62 的 dispatch 基建 + r56 的 q5 plan 模板）。
3. **r64**（独立 op；可与 r63 并行开发，A/B 顺序执行——进程纪律：全机同一时刻只有一个模型进程）。
4. **r65**（最大项；此时 r62–r64 的接受率数据在手里，若已出现系统性下降趋势可提前止损）。
5. **r66**：合并件 + 采样 A/B + 三件套重基线 + 显存核对（GPU0 free 607 → ~860；GPU2 不变）+
   决定是否申请升契约特性（当前决策：保持 opt-in）。

**构建/运维约定**（沿用 §1 环境表）：Windows build-win（WSL CUDA 已死，只做编译验证）；测试按目标构建
（`cmake --build build-win --target ninfer_linear_q4_a16_test` 等）；转换 `--device cpu`（~300 s，无需 GPU）；
serve 8099 走 harness 后台 job；重新链接前先停服务，exe 手动同步 `C:\ninfer\` 并以 `Get-FileHash` 比对。

### 8.5 进度

**Step 0 基线复测（2026-09-24，完成）**：现役 r57 件（`D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer`）、
当前 HEAD 二进制、`--max-context 131072`、贪心探针 7×160、每臂独立冷启服
（`tools/tp_bootstrap/r62_baseline_arms.ps1`，日志与 JSONL 在 `build-win/r62/`）：

| K | drafted / accepted | acc | tok/s |
|---|---|---:|---:|
| 7 | 2578 / 736 | 28.55% | 61.1 |
| 5 | 1968 / 711 | 36.13% | 61.0 |

**与 §3.7 归档（K=7 25.3% / 56.9）不一致，已归因**：§3.7 之后有两次改数值/改 walk 的提交——
`ea6204fa fix(ops): restore accurate silu in nvfp4 swiglu`（改模型数值 ⇒ 文本轨迹变）与 §4–§7 的 TP-2 首 token
发布 + 复用扫描改动；接受率是文本属性（§7 已论证），绝对值随二进制漂移。⇒ **本战役一律以本次复测的 r57 件为
基线**，§3.7 的数字仅作历史。副产物：本机当前二进制 K=5 与 K=7 吞吐持平（61.0 vs 61.1）。

**Step 1 r62 feature_projection q4（完成；结论与门禁见 §8.6）**

- [x] **op 变体**：`src/ops/linear/q4/shapes/n5120_k25600.cu` + `q4_shapes.h` / `q4_dispatch.cpp` /
      `src/ops/linear/q4/sources.cmake` 注册。**偏差（有意）**：选择器用 q4 的粗桶（`T=1 → simt_r8_c4`、
      `T≤8 → ksplit<5120,25600,8>`、`T≤24 → simt_r8_c8`、其余 `mma_r64_c128`），不是 §8.1 写的逐 T 容量桶
      （2..6）。理由：q4 的 `Capacity` 只是编译期**掩码列上界**，tile 宽度由 `(Capacity+7)/8*8` 统一到 8，
      逐 T 实例化只改一个 staging 循环上界（掩码路径下是死代码）⇒ 纯代码膨胀；既有 q4 shape 的惯例就是粗桶。
- [x] **oracle**：`ninfer_linear_q4_a16_test` **PASS**（12.1 s）。新增 5120×25600 两组：`Comparison::Full`
      （T∈{1..7,24,25}，逐元素）与 `Comparison::Sampled`（T∈{1..9,15,16,17,23,24,25,26,32,33,63,64,65,127,128,129,2048}），
      两组都含图重放；FP64 朴素参考按存储 scale 独立解码。
- [x] **override + dry-run**：`tools/tp_bootstrap/r62_draft_feature_q4.py`（**累积式**：r57 三杠杆 +
      feature_projection→q4；断言 gate/up/down/output 各 5）+ `tools/tp_bootstrap/r62_draft_q4_dryrun.py`。
      dry-run 解析：`q4=11`（10 gate/up + 1 feature）、`q5=10`（down + output）、`q8=25`、`bf16=45`
      ⇒ 与 r57 件只差 feature_projection 一个对象。
      **解释（对 §8.1 的偏离）**：§8.1 把每个 override 描述成「只改本项」，但 §8.2 要求基线臂 = r57 件
      ⇒ 每步的臂必须是**累积件（前序已通过项 + 本项）**，否则会退回官方 Q8 配方、测的不是产品路径。
      累积式还让「本步 vs 上一步」隔离出该项的增量效果。
- [ ] **转换**：`tools/tp_bootstrap/r62_convert_feature4.ps1`（r57 配方 + 本 override，`--device cpu`）
      → `D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_featproj4.ninfer`（进行中）。
- [x] **A/B + 确定性**：重建后基线臂逐字节复现（本步 C++ 全为加法）+ r62 臂门禁 —— 结论与数字见 §8.6。

**后续步骤（未开工）**

- [ ] 2 r63 finish 投影 q4（两 shape + q4 plan + wrapper + oracle + override + 转换 + A/B）
- [ ] 3 r64 kernel_projection q4（prepare kernel + wrapper + oracle + override + 转换 + A/B）
- [ ] 4 r65 context_key q4（attn_input_proj q4 + materialize q4 + 两处 wrapper + oracle + override + 转换 + A/B）
- [ ] 5 r66 合并件 + 采样 A/B + 三件套重基线 + 显存核对 + 升级决策

### 8.6 执行记录（2026-09-24；跨上下文压缩的进度锚）

**方法论修正（重要，后续步骤沿用）**：§8.2 的「贪心 7×160 探针 |Δacc| ≤ 1.0pp」被判为**内容混淆**——draft 权重一变，
target 的 tie 翻转就改文本，接受率随文本走（§7 已给 21% vs 99.6% 的极端例）。r62 的同一对工件：

| 指标 | 基线（r57 件） | r62 件 | Δ |
|---|---:|---:|---:|
| 贪心 K=7 acc | 28.55% | 28.64% | +0.09pp |
| 贪心 K=5 acc | 36.13% | 34.02% | **−2.11pp（超门禁）** |
| 采样 K=5 acc（3 类 × 6 次 × 256 token，pooled） | 44.12% | 45.66% | **+1.54pp** |
| 采样 K=7 acc（后来补测基线：15 次 × 256 token） | 33.90% | 36.58%（r63 件） | +2.68pp |

采样分项（reason / prose / code）：+4.3 / +0.1 / +1.5 pp ⇒ 两个指标符号相反 ⇒ 贪心差值是文本漂移，不是 q 退化。
**新门禁方法**：`tools/tp_bootstrap/r62_sampling_ab.ps1`（采样接受率，内容平均、对机器漂移不敏感）作为接受率裁决；
贪心探针只用于「基线跨重建逐字节复现」的确定性检查；显存用 `[mem]` 账本核对。
**确定性**：r62 件两次独立冷启服逐字节相同；重建（r62+r63+r64 代码）后的基线臂 K=7 JSONL 与改动前二进制的
基线 SHA256 完全相同（`D61C1112…`）⇒ 引擎与工件都确定。

**Step 1 r62 feature_projection q4（完成：通过）**

- op：`src/ops/linear/q4/shapes/n5120_k25600.cu` + `q4_shapes.h` / `q4_dispatch.cpp` / `sources.cmake` 注册。
  选择器用 q4 **粗桶**（T=1 → simt_r8_c4、T≤8 → ksplit<5120,25600,8>、T≤24 → simt_r8_c8、其余 mma_r64_c128），
  不是 §8.1 写的逐 T 容量桶（2..6）：q4 的 `Capacity` 只是编译期掩码列上界，tile 由 `(Capacity+7)/8*8` 统一到 8，
  逐 T 实例化只改一个 staging 循环上界（掩码路径下是死代码）⇒ 纯代码膨胀。
- oracle：`ninfer_linear_q4_a16_test` PASS（12.1–14.7 s；Full T∈{1..7,24,25} + Sampled 到 2048，含图重放）。
- override `r62_draft_feature_q4.py`（**累积式**：r57 三杠杆 + feature→q4）+ dry-run `r62_draft_q4_dryrun.py`：
  q4=11（10 gate/up + 1 feature）/ q5=10 / q8=25 / bf16=45。
- 转换 `r62_convert_feature4.ps1` → `..._dflash2_featproj4.ninfer`（375 s，1218 对象）；文件 −16,384,000 B。
- 显存：shard 0 `weights+ctx` 12750.6 → 12734.6 MiB，free 816 → 832 MiB。

**关于 override 的累积式设计**：§8.1 把每个 override 描述成「只改本项」，但 §8.2 的基线是 r57 件 ⇒ 每步的臂必须是
累积件（前序已通过项 + 本项），否则会退回官方 Q8 配方、测的不是产品路径。r62→r63→r64 的 override 都是累积式，
因此 **r64 件本身就是 r62+r63+r64 的合并候选**（r66 只是加计数校验的正式件）。

**Step 2 r63 attention/output + mlp/down q4（完成：通过）**

- 结构发现：q5 变体的 GEMM 委托普通 q5 注册表，卷积+残差尾是 qtype 无关的共享 `dynamic_conv_finish_launch`
  ⇒ q4 变体**不需要新融合 kernel**，只要两个新 q4 shape + 一个 q4 plan 目录（比 §8.1 的预估小）。
- 新增：`src/ops/linear/q4/shapes/n5120_k4096.cu`、`n5120_k17408.cu`；`src/ops/dynamic_grouped_conv/q4/` 四个文件；
  wrapper 的 `projection_planes` / `require_finish_projection_weight` / 容量分派 / 执行分派加 q4 分支；
  `bench/ops/linear_dynamic_grouped_conv_add_bench.cu` 支持 `--qtype q4`（路线名 `…q4.ksplit_exact…` 已实测）。
- oracle：`ninfer_linear_q4_a16_test` PASS（12.7 s）；`ninfer_linear_dynamic_grouped_conv_add_test` PASS
  （7.9 s：Q8/Q5/Q4 × C∈{4096,17408} 全 W/B 域 + 图重放）。
- override `r63_draft_finish_q4.py` dry-run：q4=21（1 feature + 10 gate/up + 5 down + 5 output）/ q8=25 / bf16=45。
- 转换 `r63_convert_finish4.ps1` → `..._dflash2_finish4.ninfer`（378 s，1218 对象）；文件 22,947.1 → 22,865.8 MiB
  = **−81.25 MiB**（= r62 15.625 + r63 65.625，精确一致）。
- A/B：`tools/tp_bootstrap/r62_step_arms.ps1 -Tag finish4`（贪心 K=7/K=5 + 采样 K=7/K=5）——**通过**：
  贪心 K=7 28.55%→27.03%（−1.52pp，仍属内容混淆）、贪心 K=5 36.13%→35.80%（−0.33pp ✓）、
  采样 K=7 33.90%→36.58%（**+2.68pp**）、采样 K=5 44.12%→46.21%（**+2.09pp**）⇒ 两个采样都明显上涨。
  为补齐采样基线，补测了基线件 K=7 采样（15 次，33.90%）。
- 显存（shard 0 `[mem]`）：`weights+ctx` 12750.6 → **12668.6 MiB**，free 816 → 898/904 MiB；
  实测 −82.0 MiB vs 预测 −81.625 MiB（差 0.375 MiB = artifact 分配粒度，与 r62 步同样的常数偏移）。

**Step 3 r64 kernel_projection q4（op 完成，待转换/A-B）**

- 实现选择（对 §8.1 的偏差）：不写「q4 prepare partial kernel」，而是**复用已过 oracle 的普通 q4 GEMM**——
  新增 `src/ops/linear/q4/shapes/n1280_k5120.cu`；`…/q4/q4_dynamic_grouped_conv_prepare_plan.cpp` 做
  rmsnorm → 普通 q4 linear（把 [1280,tokens] 系数矩阵写进 workspace）→ 新 reduce
  （`q4_dynamic_grouped_conv_prepare_reduce.cu`：把 bf16 reduce 的 FP32 split-K 部分和输入换成 BF16 系数矩阵）。
- 代价与理由：系数矩阵以 BF16 materialize（bf16 路线保留 FP32 部分和），多约半个 BF16 ulp；该中间量不是可观测边界
  （AGENTS.md），且远小于 q4 权重本身的量化误差。测试因此给 Q4 臂单独 criterion（prepared 预算 1.5× bf16），
  实测最坏比值 rel_l2 0.62× / gross 0.75× of budget，finish 用共享预算 0.74× / 0.68×。
- 容量：prepare 的公开容量 API 没有 qtype，q4 路线预留与查询相同的量（bf16 split-K 部分和恒大于 q4 的 BF16 系数），
  保持 `peak_used == query` 不变量。
- oracle：`ninfer_linear_q4_a16_test` PASS；`ninfer_dynamic_grouped_conv_prepare_test` PASS（bf16 + q4 双臂，
  label 前缀已加 codec 便于定位）。
- 转换：**并入 r66 合并件**（`r64_convert_kernel4.ps1` 不再单独跑：累积式 r66 件 = r62+r63+r64，
  与单独跑 r64 会得到逐字节相同的文件，省一次 6 分钟转换与 23 GB 中间件）；r64 的门禁即在 r66 件上测。

**Step 4 r65 context_key q4（决定：本次缓做）**

- 事实核对（比 §8.1 的审计更准确）：draft 的 `attention/{query,key,value,context_key,context_value}` 在工件里
  **全部别名同一个 [6144,5120] 对象**（每层 1 个 ×5 层），所以「context_key ×5 = 79.7 MiB」实际是把 5 个融合 QKV
  对象整体降到 q4，而且必须**同时**改两处消费方才能加载：
  - decode：`attn_input_proj` 三输出重载与 `prepare_attn_input_proj_weights` 都硬要求 Q8（`weight_input.cpp:176`）；
  - prefill：`context_kv_materialize` 的融合 kernel（`materialize.cu` 482 行）以**每 32 元素 1 字节有符号码 + FP16 scale**
    手写 staging/MMA/分数写回，并融合 key 的 norm+rope+环形缓存写入与 value 的 fp16 写回。
- 结论：r65 不是「再加一个 shape」，而是要写一族 4-bit 融合材质化 kernel（7 路线）+ 3 输出 q4 变体 + 两套 oracle。
  相对 79.7 MiB 收益，本次战役的剩余预算与风险不划算；§8.3 本身也允许项级回退（「该项保持现精度」）并把它排在最后。
- 因此本次交付 = **r62 + r63 + r64（173 MiB / 252.7 MiB = 68%）** + r66 合并件；r65 的审计与路线图保留。

**Step 5 r66（完成：通过）**

- override `tools/tp_bootstrap/r66_draft_q4_all.py`（单文件累积式，计数校验 31 个 selection：feature 1 + gate/up 10 +
  down 5 + output 5 + kernel 10；gate/up 每层共享一个存储对象 ⇒ 提升 26 个对象）。
- dry-run：q4 31 / q8 25 / bf16 35（r57 件的 bf16 45 中有 10 个 kernel 投影转入 q4）。
- 转换 `r66_convert_q4all.ps1` → `D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_q4all.ninfer`（326 s，1218 对象）。
- **文件级核账（精确相等）**：24,061,771,780 → 23,880,318,980 B = **−181,452,800 B = −173.05 MiB**
  = 16,384,000 (r62) + 68,812,800 (r63) + 96,256,000 (r64)，逐字节预测一致。
- 对象格式：`q4_g64_fp16` 81（r57 件 71 + 10）、`bf16` 569、`q5_g64_fp16` 54、`q8_g32_fp16` 12。
- A/B（`r62_step_arms.ps1 -Tag q4all`）——**通过**：

  | 指标 | 基线 r57 件 | r66 q4all 件 | Δ | 门禁 |
  |---|---:|---:|---:|---|
  | 贪心 K=7 acc | 28.55% | 26.95% | −1.60pp | 超线（内容混淆，见 §8.6 方法修正） |
  | 贪心 K=5 acc | 36.13% | 35.96% | −0.17pp | ✓ |
  | 采样 K=7 acc（15×256） | 33.90% | 35.05% | **+1.15pp** | ✓ |
  | 采样 K=5 acc（15×256） | 44.12% | 47.84% | **+3.72pp** | ✓ |

  **贪心探针为何不可跨臂比较（直接证据，非推断）**：`greedy_probe.ps1` 记录了完整文本，逐条 SHA256 比对基线件与
  q4all 件的 K=7 探针：**7 条 prompt 里 6 条文本不同**（长度 779/788、768/789、877/890、741/778、804/785、804/798），
  只有 idx 5 逐字节相同。机制是 §7 记录过的：draft 提案长度改变 verify 批次形状 ⇒ target 数值路径变 ⇒ 贪心近
  平局翻转 ⇒ 文本改变 ⇒ 接受率的分母换了内容。所以贪心 7×160 的差值是「不同文本上的两个数」。
  **诚实的保留**：唯一文本稳定的 idx 5 上 K=7 接受率仍降 3.75pp（672 drafted），说明 K=7 的 q4 draft 在该内容上
  可能略弱；但 pooled 采样（K=7 共 7711 drafted）是更可靠的统计量且上涨 +1.15pp。

  两个采样门禁都通过；贪心 K=7 的超线在 r62/r63/r64 每步都出现（−1.5 ~ −2.1pp）而采样同时上涨，
  是 §8.6 记录的内容混淆，不是单调退化（若为退化，采样应同步下降）。
- **显存核对（shard 0 `[mem]`）**：`weights+ctx` 12750.6 → **12576.6 MiB**（−174.0 MiB，预测文件级 −173.05 MiB，
  差 0.95 MiB ≤ 分配粒度 1.6 MiB）；free 816 → **990 MiB**（K=7）/ 996 MiB（K=5）；shard 1 不变（810 → 810 MiB @K=5）。
  §8.4 写的「GPU0 free 607 → ~860」是 §3.7 旧归档的绝对数（不同 capacity/KV 配置），本次实测的增量与预测一致。
- 三件套：`tools/tp_bootstrap/r66_suites.ps1`。**运行前必须把 FFmpeg 与 libcurl 的 `bin` 加进 PATH**，
  否则 `solo`/`sessions` 在加载期直接 `0xC0000135`（STATUS_DLL_NOT_FOUND）——`append` 不依赖媒体路径，所以只有它不受影响。
  三件套实测（`NINFER_TEST_ARTIFACT` 指向 q4all 件，除注明外）：

  | 套件 | 结果 | 证据 |
  |---|---|---|
  | `append` K=7 | PASS (37.9 s) | ring 重放逐字节相同；proposal 确定 `fnv1a=0xda91572dd83980bd`；workspace peak 113.3/192 MiB |
  | `append` K=5 | PASS (37.5 s) | 同上，`fnv1a=0x5a3629fb79be1cd3` |
  | `solo` 两进程 | PASS (41.1 / 41.8 s) | 两个独立进程 digest 相同 `0x4bcc3994a5efba7d` |
  | `sessions` | **FAIL（plain 路线，先于本战役存在）** | 见下 |

  `sessions` 的失败：`FAIL (plain): a conversation behind a shared system prompt diverged from the oracle
  on its first sample: got [2752 13 …] expected [365 2798 …]`。**用未改动的基线件（r57 `draftall`）在同一二进制上
  复现出逐 token 相同的 `got`/`expected`** ⇒ 该失败与本次 q4 杠杆无关（plain 路线根本不加载 draft）。
  它挡不住本次交付（draft q4 由 append 两 K + solo digest + op oracle + 采样 A/B 覆盖），但**挡住了「升级为契约特性」**：
  升级门禁要求三件套全绿，而其中一件存在与本战役无关的既有失败 ⇒ 结论是保持 opt-in（见下）。

  **被改动路线自己的证据（补测）**：`NINFER_TEST_ROUTE=dflash2` 定向跑 sessions，baseline 与 q4all **都 PASS**
  （各 164.8 s，`TP-2 session retention (dflash2) passed: recall reused 71 prompt tokens bit-identically;
  LRU eviction forced a full prefill`）⇒ 失败只存在于 plain 路线，且与基线件逐 token 相同。

**升级决策（§8.4 第 5 步）**：**保持 opt-in，不申请升契约特性**。理由两条：
1. 三件套里 `sessions` 的 plain 路线有一个**先于本战役存在**的失败（基线件逐 token 复现），升级门禁要求三件套全绿，
   在该失败修好之前不具备升级条件；
2. §8.2 的字面门禁「贪心 |Δacc| ≤ 1.0pp」在 K=7 未过（−1.60pp）——虽然已证明该指标跨臂不可比（6/7 条 prompt 文本不同），
   但把门禁改成采样主导属于「门禁修订」，应当与「升级」分开决策。
   q4 件继续以显式 override + 专门 artifact 形式提供（`…_q4all.ninfer`），引擎与官方配方都不动。
**W4A4 家族的同款最终件（用户 2026-09-24 追加要求）**

- 脚本 `tools/tp_bootstrap/r66_convert_q4all_w4a4.ps1` → `D:/LLM/qwen3_8_27b_w4a4_dflash2_q4all.ninfer`
  （`--model`/`--source quantized` 换成 `W4A16/NVFP4/W4A4`，draft 仍取 `W4A4+W8A8/DFlash2-FP8`；
  这与既有 `…_w4a4_dflash2_draftall.ninfer` 的 provenance 一致——两家族只差 text/vision/MTP 源）。
- dry-run `r62_draft_q4_dryrun.py --base D:/LLM/W4A16/NVFP4/W4A4 --draft D:/LLM/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8`：
  q4 31 / q8 25 / bf16 35，与 w8a8 家族同解（两个家族的 r57 基线都由 `r57_draft_all_override.py` 生成，draft 源相同）。
- **实测**：18,874,128,644 → 18,692,675,844 B（−181,452,800 B，与 w8a8 家族逐字节同额）；
  报告 `name=qwen3.8-27b-w4a4-q4all`、1590 对象、格式计数与 w8a8 件完全相同（q4 81 / bf16 569 / q5 54 / q8 12）。
- **跨家族核验（对象级 SHA256）**：两个 q4all 件的 **66 个 dflash2 对象全部逐字节相同**（含 31 个 q4 selection 的存储对象）
  ⇒ draft 质量与已过 A/B 的 w8a8 件一致，接受率证据直接可移用；`text/layers/0/*` 16 个对象里 8 个不同（两家族确实只差
  text/vision/MTP 源）。
