# PLAN：TP-2 多并发（真批处理）实施计划

> **本文件是 TP-2 多并发项目的唯一活动计划**，也是跨上下文压缩的持久记忆；上下文被压缩或会话恢复后，先重读本文件再继续。
>
> - **基线**：分支 `feat/tp2-concurrency`，自 `feat/windows-native-port` @ `d1a9b57bb1ee780d0c395d4d106f835405357ff1` 切出。Windows 工作树 `D:/Documents/workbench/ninfer`（构建 `build-win`，VS2022 + CUDA 13.3）；WSL 树只作编译验证。
> - **来源**：本文件由源码级可行性评估固化（C=1 四层证据链、逐层判定、改动点清单、风险表、分阶段步骤）。
> - **相邻权威**：`docs/tp2-dual-5060ti.md`（交付与实测）、`docs/tp2-decisions.md`（已定案，重复调研前必读）、`PLAN.md`（§3.7 架构级隐患）、`docs/maintainer/engine-architecture.md`（单卡执行/调度契约）、`docs/maintainer/paged-kv-cache.md`（KV 物理模型与 execution row）、`docs/maintainer/resource-scheduling-and-context-cache.md`（准入与保留语义）。
> - **语义**：本项目的多并发＝**真批处理（紧凑 decode batch）**：B 个 decode-ready 请求合成一个 round，共享一次权重读取。不含抢占、priority/QoS、大规模 continuous batching。

---

## 0. 待产品决策（阻塞 Phase 1 动手）

| # | 决策 | 建议 | 影响面 |
|---|---|---|---|
| D1 | 语义确认：A 排队串行（已具备）／B 真批处理（本项目）／C 时间片轮转 | B | 整个项目是否成立 |
| D2 | 并发档位与上下文档位：C 上限、`--max-context`/KV 档位、spec 后端 | 先 C≤4、`--max-context 131072` + fp8、MTP K=2 或 plain | 内存预算、收益门禁 |
| D3 | 契约变更：`docs/serving.md` 的 one request at a time、`AGENTS.md` 产品执行模型段 | 同步更新 | 对外行为 |
| D4 | 批内失败语义：一个 lane 的 allreduce give-up／异常 | 整批失败，必要时整机（对齐 engine-architecture §7.4） | 失败半径、可用性 |
| D5 | 会话缓存：Phase 1 禁用跨会话 retention 是否可接受 | 可接受，文档化 | TTFT 回退 |
| D6 | 选型：方案甲（TP2GenerationCore 内自建 worker/lane）还是方案乙（TP-2 变 Program 双设备后端） | Phase 1 甲、Phase 3 视情况乙 | 工作量与长期漂移风险 |

> **状态（2026-09-30，P0.1，已接受）**：D1–D6 按「建议」列接受，作为 Phase 1 的前置输入（用户指示「按你建议执行」）。即 D1=B；D2=先 C≤4、131072、共享 KV 池、plain/MTP 或 DFlash2；D3=同步更新 `docs/serving.md`/`AGENTS.md` 契约；D4=整批失败（必要时整机）；D5=Phase 1 禁用跨会话 retention 并文档化；D6=Phase 1 走方案甲、Phase 3 复评方案乙。其中 D3/D5 属对外可感知变更，落地时再向用户做一次显式确认。

---

## 1. 目标与非目标

### 1.1 目标
- 在 `--devices a,b` 的 TP-2 路线上支持**启动期固定的并发 C（1..8，首期 ≤4）**：B 个 decode-ready 请求组成一个紧凑 decode batch，一个 round 内两卡 lockstep 前向一次。
- 聚合 decode 吞吐随 B 近似线性（暂定门禁：B=2 时聚合 ≥1.6×、B=4 时 ≥2.5×，相对同配置 C=1）。
- 每条 lane 的输出、预算、stop、取消与终态语义与单卡 EngineCore 契约一致。
- 单请求行为（C=1）与既有 TP-2 实测结果逐位/按既定容忍不回归。

### 1.2 非目标（首期不做）
- 抢占式调度、priority/QoS、请求优先级反转处理。
- 大规模 continuous batching、跨 Engine context store、多 GPU placement 泛化（仍限两卡）。
- prefill 批处理：单卡契约本就「同时最多一个 staged prefill」，prefill 继续单 lane（chunk 之间插 decode round）。
- 多并发模式下的跨会话 retention 与稳定块复用（Phase 2 再补）。
- 多模态请求并发（Phase 1 串行化多模态 lane）。

---

## 2. 现状：C=1 被写死在四层（证据链）

| 层 | 位置 | 现状 |
|---|---|---|
| 引擎归一化 | `src/runtime/engine/model_instance.cpp:85-104` | TP-2 分支强制 `max_concurrency=1`、`max_pending_requests=1`；`options.kv_capacity` 被覆盖为 `explicit(max_context)`，用户 `--kv-capacity` 实际被忽略。 |
| 服务层 | `src/serve/serve_options.cpp:429-434` | `effective_request_capacity` 对 `device_b>=0` 返回 `{1,1}`，注释明写 TP-2 一次一个请求加一个排队（P6 修复后服务层与引擎同源）。 |
| 调用模型 | `src/runtime/engine/tp2_generation_core.h:353`、`.cpp:1458-1473` | `submit()` 只建 `Submission`；`Submission::wait()` 在调用者（HTTP）线程持 `execution_mutex_` 内联跑完整个 `execute()`。无 worker、无 lane、无请求状态机。 |
| 执行组合 | `src/models/qwen3_5/execution/text.cpp:2054/2156/2279/2724`、`text.h:179-250` | `forward_tp2*` 全部单行：一个 `positions`、`kv_table_rows[1]`、state slot 标量（见 `text.cpp:2337-2364`）。 |
| 资源规划 | `src/runtime/engine/tp2_generation_core.cpp:567-595/609-632/841-887` | `kv_table_rows=1`；GDN `slot_count=1`、state arena `(2+kReuseSnapshotCount)×state_bytes`；KV/MTP 页启动时一次性 reserve+materialize 到 row 0 并永久 pin。 |
| Workspace | `tp2_generation_core.cpp:151/691` | 单卡 192 MiB 单 arena，按 max_concurrency=1 规划。 |
| Round/图 | `tp2_generation_core.cpp:900`、`.h:368-388/437-451`、`.cpp:983-992/523` | `RoundStateSpec.batch_capacity=1`；图只按 visible extent 分桶（不含 B）；三类 step 各一份 pinned ingress；`reserve_ar_channels` 按桶数预留（现日志 24）；图复用时要求 arena watermark 精确等于 capture 的 `arena_begin`。 |
| 会话/复用 | `tp2_generation_core.h:215-280/519-578` | `sessions_` 只允许一个 resident lineage；`active_session_`、`anchor_session_`、`cached_prompt_tokens_`、`cached_state_valid_`、`host_checkpoint_*`、`block_anchor_*`、`reuse_source_`、`dflash_context_frontier_`、`mtp_previous_draft_` 全是核心级全局状态。 |
| 多模态 | `tp2_generation_core.cpp:706-747` | 一份 `VisionWorkspacePlan` + 一个 arena，注释明写 one session per multimodal request owns the encoding on top of them。 |

历史佐证：Round 34 的并发挂死根因是两个线程共用同一 workspace arena 且打乱 AR token 递增（`docs/tp2-dual-5060ti-worklog.md:516-529`）；修复只是加 `execution_mutex_` 串行化，并未引入并发执行。

---

## 3. 收益模型与可行性判定

### 3.1 为什么只有真批处理有意义
- batch-1 decode 是**权重流带宽受限**：轮时地板＝每 shard 权重字节 / 448 GB/s；实测 38.2 ms ≈ 地板的 59%（`docs/tp2-dual-5060ti.md:125-131`），ninfer 每 shard 每 forward 流 10.15 GB。
- 时间片轮转不摊薄权重读取 ⇒ 无吞吐收益；真批处理让 B 个 lane 共享一次权重流 ⇒ 聚合约 B×。
- batch 带来的额外成本：attention/GEMM 的 T=B 放大、allreduce 载荷 ×B（hidden 5120×B×2 B，B=8 也仅 80 KiB，可忽略）、host 侧 per-lane accept/fold/session/sampling。

### 3.2 逐层可行性

| 层 | 现状 | 判定 |
|---|---|---|
| 量化 GEMM／attention／GDN／采样 | `ops::sample` 原生 `[physical_rows,B]` + `SamplingConfig[B]`（`include/ninfer/ops/sampling.h:36-48`）；`propose_dflash2_batch` frame 为 `[...,B]`（`execution/draft.cpp:277-300`） | 无需改 |
| Paged KV 多 row | `KVExecutionTablePool::row_count/acquire/publish`、`DeviceKVPagePool::reserve/materialize`（`src/core/paged_kv_cache.h:217-232/349-351`） | 原语齐备 |
| GDN state 多 slot | 单卡按 `C + device_state_slots` 规划（`program/planning/startup.cpp:109-149`） | 原语齐备 |
| CUDA graph exact-B 分桶 | 单卡 `topology_class*max_concurrency+(B-1)`（`program/graphs.cpp:328-329/365-366`） | 有参照，TP-2 侧要重做 |
| 批 decode 模型执行 | 单卡有 `target_verify_batch`／`mtp_forward_decode_batch`／`mtp_propose_batch` + `[.,.,B]` frame（`program/speculative/mtp.cpp:70-200`） | 需新增 TP-2 batch 变体 |
| TP-2 执行组合 | 全单行 | 要新写 |
| 调度／准入／请求状态机 | 无 | 要新写或改造 Program |
| 会话缓存／前缀复用 | 单 lineage 全局状态 | 要 per-lane 化 |

**关键收窄**：单卡契约「同时最多一个 staged prefill」（`engine-architecture.md:360-367`）使 prefill 可继续单 lane，只有 decode/verify/MTP/DFlash2 的 window 需要 batch 变体，`text.cpp` 改造面约减半。

### 3.3 内存预算（首期硬约束）

**权威台账**（`docs/tp2-dual-5060ti.md:489-521`，262,144 / fp8 KV / MTP K=2 出厂配置；int8 KV 在同口径下**大 48 MiB**，因 64 元素 scale group）：

| 组件 | shard 0 | shard 1 | 并发增量 |
|---|---:|---:|---|
| Weights + CUDA context | 11,494.6 | 11,494.6 | 0 |
| Text KV（16.125 KiB/token/shard；fp8 262144 → 4,128.0） | 4,128.0 | 4,128.0 | **0（共享池：Σ lane 活跃 token ≤ capacity）** |
| MTP layer KV（2 KiB/token，仅 shard 0） | 516.2 | 0.0 | 0 |
| Linear-attention state arena（2 live + 2 reuse snapshots = 4×73.4） | 293.6 | 293.6 | **+73.4 MiB/shard/lane**；关掉 retention 时 lane 只占 1 plane ⇒ **C≤4 恰好塞进现有 4 plane，零增量**；C=8 需 +293.6。**⚠ 2026-09-30 实测修正并已定案（P1.2b = 方向 c）**：P1.2a 把 `slot_count` 改为 `lanes` 会让 `state_bytes` 自身 ×lanes，而 arena 仍是 `(2+kReuseSnapshotCount)×state_bytes` ⇒ 真实代价是 **+293.6 MiB/shard/lane**（C=4 时 state 293.6 → 1174.4 MiB/shard，+880.8），**不是零增量**；改按实测值编预算。配套的 D5 由 `lanes_ > 1` 机械强制（见 §11.1）。 |
| Program workspace arena（prefill chunk 1024 峰值 ≈140） | 192.0 | 192.0 | 待测：batch window T=B 抬高，暂按 +0…+96 预算 |
| MTP replay records / round anchor | 2.6 | 2.6 | 随 B 小幅增长（数 MiB 内） |
| DFlash2 draft ring | ≈40 | 0 | per-lane 则 +40/lane（shard 0）；**实现选共享 ⇒ 0** |
| **整卡占用（nvidia-smi，C=1）** | **15,614** | **15,094** | 16,311 MiB/卡 |

**判定**：
- 出厂 262,144 配置 C=1 余 **697 / 1,217 MiB**；本机 2026-09-30 活服务器（245760 / int8 / DFlash2 K=5）实测 15,162 / 14,834 ⇒ 余 **1,149 / 1,477 MiB**。
- **共享 KV 是硬前提而非优化**：若每 lane 各占全上下文，C=4 @131072 需 4×2,064 = 8,256 MiB/shard，直接不可行。
- C=4 + retention off + 共享 KV/ring：增量 ≈0（state 复用现有 4 plane）⇒ 余量充足。C=8：+293.6（state）再叠加 workspace/ring 增长，262,144 下会挤到 <150 MiB ⇒ **不可接受**。
- 结论：首期 **131072 + 共享 KV + 关闭 retention，C≤4**；262,144 上放 C>2 需先释放 MTP/draft 或降 KV dtype。

### 3.4 Phase 0 实测记录（2026-09-30，复用本机活服务器）

**环境**：`C:\ninfer\ninfer-serve.exe D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer --devices 0,1 --max-context 245760 --port 3456 --kv-dtype int8 --max-concurrency 1 --max-pending-requests 16 --spec dflash2 --draft-tokens 5 --lm-head-draft --reasoning-effort xhigh --vision --vision-item-tokens 8192 --preserve-thinking`。两卡 RTX 5060 Ti 16,311 MiB。**未重启、未改配置**。

**P0.2 排队并发现场复测**（`_temp/tp2_p0_smoke.ps1`，结果 `_temp/tp2_p0_smoke.csv`；固定 nonce 前缀避免复用，max_tokens=96）：

| C | 批墙时 ms | HTTP 200 | 运行/排队 | 备注 |
|---:|---:|---:|---|---|
| 1 | 1,135 | 1 | 1/0 | prompt 363 ms + decode 647 ms / 96 tok |
| 2 | 1,511 | 2 | 1/1 | 先跑者 790 ms 完成，另一者等锁到 1,511 ms；790+711 ≈ 1,501 ⇒ **引擎真串行** |
| 4 | 1,548 | 2 | 1/1 | 第 3、4 个 **HTTP 429** |
| 8 | 1,544 | 2 | 1/1 | 第 3–8 个全部 429 |

429 响应体：`{"error":{"code":"server_overloaded","message":"inference request queue is full","param":null,"type":"rate_limit_error"}}`。

⇒ **语义 A 的真实边界**：TP-2 并发容量是 **1 运行 + 1 排队 = 2**（`src/serve/serve_options.cpp:429-434` 的 `{1,1}`），第 3 个并发**立即 429，不排队**；`--max-pending-requests 16` 对 TP-2 无效。`PLAN.md:170` 的「C=1/2/4 冒烟」由此关闭：C=4 **不会**排队，只会被拒。

**P0.4 基线可复现性**：`temperature:0` 在 dflash2 路线下**不逐位可复现**——同 prompt 连跑 4 次，两两公共前缀仅 15–71 字符 / 约 350 字符；且 `cache_n` 随冷/热在 0/1 间摆动。与 `PLAN.md:193-194`（稀疏接受 + 每请求随机 seed，`translate.cpp:51-57`）一致。因此：
- 冻结的是**软基线**：配置 + 吞吐画像（纯 decode ≈150 tok/s；96 tok 请求 draft 接受 79/96；prompt 70–360 ms，进程首请求曾见 903 ms）+ 一次运行文本（`_temp/tp2_p0_baseline_run1.txt`、`run2.txt`）。
- 硬 oracle 必须离线（`--greedy` 或固定 seed 的 C++ TP-2 套件）；HTTP 端到端只做吞吐/行为/终止语义 A/B，文本比较改用一致率容忍（见 §8）。

---

## 4. 架构方案

### 4.1 选型
- **方案甲（推荐 Phase 1）**：在 `TP2GenerationCore` 内自建 worker 线程 + lane 槽位 + FIFO 准入 + round 循环，沿用现有 lockstep 执行与 session/checkpoint 机制（Phase 1 禁用跨会话 retention）。自包含、可增量交付；代价是继续复制 EngineCore 的调度语义（`PLAN.md:271-274` 的漂移风险）。
- **方案乙（Phase 3 评估）**：把 TP-2 收敛为 `Program` 的双设备执行后端，调度／准入／资源／缓存全复用 EngineCore，直接消除双核心漂移。跨 `Program`／`SequencePlanner`／KV／StateImage／workspace／graph 规划全层，工作量大。

### 4.2 Phase 1 最小可用形态
- `max_concurrency = C`、`max_pending_requests = P` 由 CLI 进入归一化（不再强制 1）。
- KV：共享物理页池 + 每 lane 一条 execution row + 每 lane reservation；`--kv-capacity` 语义生效（总量预算）。
- GDN state：`slot_count = C`（每 lane 一个 live slot + 每 lane 一个 round scratch），Phase 1 不做 per-lane 快照。
- Round：`batch_capacity = C`；decode round 成员＝当前 decode-ready 的全部 lane（紧凑，不 padding）。
- 执行：统一为 eager（`EagerBucket`），暂不做 `(bucket,B)` CUDA graph。
- 调度：FIFO 准入 + 单 staged prefill + decode 不饿死；无 backfill 保护、无抢占。
- 多模态：拿全局 vision 锁串行化。
- 输出：每 lane 独立 `OutputSink`／预算／finish reason，round 提交后按 lane 发布。

### 4.3 接口草图（Phase 1 目标形态）
```
// 执行组合层：新增 batch window 变体（单行版本保留给 C=1 与 prefill）
void forward_tp2_window_batch(TextContext& peer, tp::DevicePair& pair,
                              const std::int32_t* ids,      // [B * width] 行主序
                              const std::int32_t* positions,// [B * width]
                              const Tensor& kv_table_rows,  // I32 [B]
                              const Tensor& state_src, const Tensor& state_dst, // I32 [B]
                              ops::CausalAttentionExecutionEnvelope envelope,   // 批内最宽
                              Tensor& logits_columns,      // [V, B*width]
                              Tensor* hidden_columns, DFlashFeatureSink* sink,
                              const std::int32_t* valid_columns);
```
- `RoundStateSpec.batch_capacity = C`；ingress 结构（`OrdinaryDecodeIngress`／`MtpDecodeIngress`／`DFlashDecodeIngress`）的 `text_kv_table_rows[B]`、`state_*_slots[B]`、`positions[B]` 按行填充，参照 `program/decode.cpp:45/285/401/559/759`。
- `round_base` 改为**启动时统一规划**（per-lane 持久 scratch 之上），成为常量；否则 `reusable_window_graph` 的水位条件（`tp2_generation_core.cpp:987/1103/1200`）无法满足。

---

## 5. 改动点清单

### A. 执行组合层（`TextContext`）
- A1 新增 batch 变体：`forward_tp2_decode_window`／`forward_tp2_window`（`text.h:179-250`、`text.cpp:2054-2900`），接受 `[B]` positions、`kv_table_rows[B]`、state slots `[B]`、per-row envelope/valid_columns；`forward_tp2` 的 logits 由 `[V,1]` 扩展为 `[V,B]`。参照 `target_verify_batch` 与单卡 ordinary decode batch。
- A2 `mtp_chain_body`（`tp2_generation_core.cpp:1212-1330`）改为 batch 或按 lane 循环，per-lane pinned ingress；参照 `program/speculative/mtp.cpp:70-200` 的 `mtp_forward_decode_batch`／`mtp_propose_batch`。
- A3 DFlash2：`plan_dflash2_round`／`dflash2_proposal_workspace_bytes` 的 batch 由 1 改为 B（`tp2_generation_core.cpp:767-774`），`DFlash2RoundSpec.batch_capacity`（`program/dflash_round.h:58`）与 per-lane ring/frame。
- A4 allreduce 载荷随 T×B 变宽：确认 `in_kernel_allreduce_bytes()`（`device_pair.h:48-52`）与 `prefill_chunk_width`（`tp2_generation_core.cpp:1362`）的 clamp 仍成立。

### B. 核心与调度
- B1 以 worker + lane 槽位 + 事件替代 `wait()` 内联执行（`tp2_generation_core.h:353`、`.cpp:1458-1473`）。
- B2 `execute_walk`（`tp2_generation_core.cpp:2431` 起）拆成可恢复阶段：materialize → prefill chunk（单 lane）→ first sample → decode rounds（batch）→ finish；decode round 循环上移到核心。
- B3 核心级全局状态 per-lane 化（`tp2_generation_core.h:519-578` 列出的全部字段）。
- B4 取消：round 边界压缩 batch 成员；被取消 lane 的 KV/state 释放不得影响同批其他 lane。

### C. 资源、KV 与 state
- C1 `kv_table_rows = 1 -> C`（`tp2_generation_core.cpp:592`）；启动时 pin 全量页到 row 0 改为共享池 + 每 lane reservation/materialize（`:841-887`）。警惕 worklog:380 记录的每请求 reserve/materialize 泄漏族缺陷。
- C2 GDN `slot_count = 1 -> C`（`:616`），`kRoundScratchSlot` 每 lane 一份。
- C3 `RoundStateSpec.batch_capacity = 1 -> C`（`:900`）；MTP KV 页同样多 row。
- C4 准入/容量：Phase 1 用**静态或简化分区**（每 lane 固定 page group 上限），不移植 ResourceManager 的 backfill proof；`--kv-capacity` 变为真实总预算。

### D. CUDA graph 与传输
- D1 图键从 visible extent 扩为 `(bucket, B)`；三类 step 的 pinned ingress 变 `[B*width]`。
- D2 `reserve_ar_channels` 计数 ×C（`tp2_generation_core.cpp:523`，现 24 → 约 192），必须在首个 `create_ar_channel` 前算对（P1 修复的正是触顶）。
- D3 `round_base` 启动时统一规划（见 4.3）。
- D4 Phase 1 可先只上 eager（省掉 D1/D2 的图矩阵），代价是单卡实测的个位数百分比（DFlash2 −10.2%、MTP 34.2→34.0 ms）。

### E. 会话与缓存
- E1 单 resident → C resident，或 Phase 1 在多 lane 模式下禁用跨会话 retention 与 host checkpoint ring（`tp2_generation_core.h:215-280`）。
- E2 Phase 2 按 lane 重写 catalog/eviction/recall/anchor 语义，对齐单卡 context cache。

### F. 多模态
- F1 单 `vision_workspace_`／`vision_arena_`（`tp2_generation_core.cpp:706-747`）加全局锁串行化；Phase 2 再 per-lane 化。

### G. 服务与契约
- G1 `effective_request_capacity` 放开为 `{C, P}`（`src/serve/serve_options.cpp:429-434`），线程池与请求寿命容量同源。
- G2 文档：`docs/serving.md:68-69`、`docs/tp2-dual-5060ti.md:1-5/487`、`model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md:219`、`AGENTS.md` 产品执行模型段。

### H. 测试
- H1 TP-2 现有一律单请求：`tests/models/qwen3_5/test_tp2_{forward,sessions,dflash_append,dflash_solo,load}.cpp`。
- H2 新增：批内两两 oracle（同 lane 单跑 vs 批跑）、B 与 lane 组合、取消/批量失败、`tools/bench/ttft` 并发档。

---

## 6. 风险表

| # | 风险 | 证据/机理 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | 双核心漂移加剧 | `PLAN.md:271-274`；并发要么再抄 EngineCore 调度，要么重构 Program | 长期正确性/维护成本 | 优先评估方案乙；甲方案严格复用 EngineCore stage 语义 |
| R2 | 图复用 workspace watermark | `round_base` 现按请求设置；多 lane 不一致 ⇒ `logic_error`（`tp2_generation_core.cpp:987/1103/1200`） | 功能不可用 | 启动时统一规划 per-lane 持久 scratch，`round_base` 变量化常量 |
| R3 | KV/state 生命周期回归 | worklog:380 的每请求 reserve/materialize 泄漏 | 第二个请求起 500/OOM | 真正的 reservation 契约 + 定向测试 |
| R4 | AR give-up 爆炸半径 | `PLAN.md:253-267`；批内共享一次 round | 一个 lane 污染整批 | 明确 D4 失败语义；round 边界不允许半提交 |
| R5 | 内存不足 | free 697/1217 MiB @262144；GDN 73.4 MiB/shard/lane | C>2 全上下文不可行 | 共享 KV 池 + 131072 档位；释放 MTP/draft 预案 |
| R6 | 聚合吞吐不达线性 | 轮时非纯权重地板（59%），batch 放大 compute 与 host per-lane | 收益低于预期 | B=2/4 实测门禁先行 |
| R7 | DFlash2 的 T=(K+1)×B 上限 | NVFP4 A16 路由仅 T≤16（`docs/tp2-dflash2-draft-nvfp4.md:112-113`）；现产 Q4/Q8 路由上限 131072 且需 %4 对齐 | 未来量化路线与并发组合被卡 | 保持 Q4 draft；量化时把 (K+1)×B 纳入标定 |
| R8 | 会话缓存语义退化 | 单 resident lineage 与多 lane 不兼容 | TTFT 回退 | Phase 1 禁用并文档化；Phase 2 重写 |
| R9 | 多模态争用 | 单 plan/arena/单 VisionPrefillSession | 结果不确定/崩溃 | Phase 1 串行化 |
| R10 | 产品契约变更 | `serving.md:68-69`、`AGENTS.md` | 需显式决策 | 先落契约与档位 |

---

## 7. 分阶段有序步骤

### Phase 0：决策与基线（不写产品代码）
- [x] P0.1 确认 D1–D6（尤其 D1 语义、D2 档位、D4 失败语义）。→ 2026-09-30 按建议接受，见 §0。
- [x] P0.2 复测排队串行并发（C=1/2/4 冒烟，PLAN.md:170 遗留项），把语义 A 的边界钉死。→ 边界 = 1 运行 + 1 排队，第 3 个 429，见 §3.4。
- [x] P0.3 产出并发内存预算表：每 lane GDN/draft/scratch + 共享 KV 总量 → C 与 `--max-context` 可行组合。→ 见 §3.3。
- [x] P0.4 冻结 C=1 回归基线（现有 TP-2 套件 + 一段 262144/131072 贪心文本），供批处理 A/B。→ 实测 temperature 0 不可逐位复现，改冻结**软基线** + 离线硬 oracle，见 §3.4。

### Phase 1：可用真并发（建议 C≤4、eager、共享 KV 池、禁用跨会话 retention）
- [x] P1.1 归一化放开：`max_concurrency` 收敛到 `kTp2GenerationMaxConcurrency`、`max_pending_requests` 直通（`model_instance.cpp:96-97`、`serve_options.cpp:429`、新头 `include/ninfer/tp2_capacity.h`）。构建绿、两个 option 测试通过（2026-09-30）。
- [x] P1.2 资源层：共享 KV 池多 row + 每 lane reservation；`slot_count=C`；`batch_capacity=C`（C1–C3）。**施工图见 §11.1。**
  - [x] P1.2a 资源宽度参数化（2026-09-30）：`build_shard` 里写死的宽度全部改为 `lanes`（见 §11.1 顶端），`lanes = options_.max_concurrency`。门禁常量仍为 1 ⇒ 布局逐字节等价。构建绿；`tp2|tp_device|engine_options|serve_options` 11 项测试全通过（5 个 `ninfer_qwen3_5_tp2_*` 因需 `--artifact` 且走自带 `ShardState` 而 Skipped）。
  - [x] P1.2b state arena 乘数与 retention 决策（2026-09-30，方向 c）+ 运行前硬化：新增 `lanes_` 成员（`tp2_generation_core.h:556` 一带），构造函数写入；`lanes_ > 1` 时**整条 pinned host checkpoint ring 与 session catalog 都不分配**（D5 机械强制，见 §11.1）。门禁仍为 1 ⇒ 逐字节等价，构建/测试绿。
  - [x] P1.2c 每 lane KV execution row（2026-09-30）：`build_shard` 的 KV 物化改为**静态分区** —— `pages = pages_for_tokens(capacity)` 按 lane 均分（`per_lane = pages / lanes`，最后一 lane 吃余数，保证每页恰好分配一次；`per_lane == 0` 直接抛 `TP-2 KV budget cannot give every lane a page`），逐 lane `pool.reserve(count)` → `pool.materialize` → `tables.acquire(lane)` → `tables.publish(lane, 0, handles, stream)`。`Shard` 的 `kv_pages`/`kv_page_handles`/`kv_row` 换成 `kv_lane_pages`/`kv_lane_handles`/`kv_rows`（`tp2_generation_core.h:175` 一带；MTP 的三个成员仍单 row），retention 的导出/恢复两处固定读 lane 0（`tp2_generation_core.cpp:1870`/`:1973`，D5 下单 lane 设施）。构造函数新增守卫：`lanes_ > 1 && (mtp_enabled_ || dflash2_enabled_)` 抛 `invalid_argument`（verify 窗口、MTP chain、DFlash2 round 仍是单 row，多 lane 会把所有 lane 灌进 row 0）。`lanes=1` 时与原实现逐字节等价。

- [x] P1.3 执行组合：batch decode window 变体（A1）。`forward_tp2_decode_window_batch` + `gdn_mix` 的 shard per-lane conv/递推分支已落地（2026-09-30，见 §11.2）。构建绿；`tp2|tp_device|engine_options|serve_options` 11 项测试全通过（5 个 `ninfer_qwen3_5_tp2_*` Skipped）。verify/MTP/DFlash2 的批量变体仍待做。
- [x] P1.4 核心：FIFO 准入 + lane 绑定 + 共享批量 decode round（B1–B2），不做逐 lane 复用。**施工图见 §11.3。**
  - [x] P1.4a 多 lane 准入与 FIFO 队列（leader-driver，2026-09-30）：`PendingRequest` / `lane_queue_` / `wait_lanes` / `drive_lane_queue` / `execute_lane` 落地。**不引入独立 worker 线程**：提交线程把请求压入 `lane_queue_`，若 `lane_driver_active_ == false` 则该线程在取得 `execution_mutex_` 后成为驱动者执行 `drive_lane_queue()`，否则在 `lane_queue_cv_` 上等自己的 `complete`。
  - [x] P1.4c 共享批量 decode round（2026-09-30）：新增专用 `TP2GenerationCore::execute_plain_batch(...)`，plain-only、单线程；**不重构 `execute_walk`**。详见 §9。
  - [x] P1.4d 门禁实测（2026-09-30）：**B=2 1.913×、B=4 3.143×**（当时的客户端恰好把 4 条请求压进 3 ms 成批窗口），批内逐字节 oracle 通过。**该数字不可复现**：到达抖动大于成批窗口时 4 并发会拆成 2×2。以 P1.8 的**同时性客户端**复测为准（B=2 1.76–1.91×、B=4 2.80×）。详见 §9。
- [x] P1.5 采样/输出：per-lane `SamplingConfig[B]`、`token_counts`、OutputSink 与终态。批量路径在 P1.4c 的 `execute_plain_batch` 内落地并验证（含 2 条并发 SSE 流，见 §9）。留到 Phase 2 的是非 plain 路线（MTP/DFlash2）的复用。
- [x] P1.6 取消与失败：per-lane 退休（round 边界压缩，退役 lane 的 config 被紧密重排后不再参与 `forward_tp2_decode_window_batch`）；D4 语义在 `drive_lane_queue` 的 `catch (...)` 里落地为**整批同生共死**（同一 `std::exception_ptr` 发给批内全部成员 + `session_invalidate_active()`）。端到端取消冒烟通过，见 §9。
- [x] P1.7 服务层容量放开（G1）：`effective_request_capacity` 经 `tp2_generation_concurrency` 放开到上限 4（`src/serve/serve_options.cpp:430`）；`--max-concurrency 1`（默认）逐字节不变。产品契约文档已同步：`docs/serving.md`、`AGENTS.md`、`docs/tp2-dual-5060ti.md`、`README.md`、两个 model card。
- [x] P1.8 验收（2026-09-30）：批内 oracle 全部逐字节等于 solo（CONC2 两种配对 + CONC4×3）；聚合门禁 B=2 1.76–1.91×、B=4 **2.80×**；单请求比单 lane（CUDA Graph）路线慢约 24%，属 §4.2 的预期取舍（**该 24% 已由 P2.2b 消除**，见 §9 P2.2b 行）。首轮失败及根因见 §9。**口径收窄（2026-10-01）**：「逐字节」是观测结果而非保证——批内所有 lane 共享一个 attention envelope（由最长 lane 决定），比最长 lane 短的 lane 其 reduction tree 与单跑不同 ⇒ ulp 级漂移，可能翻转低 gap token（判据 A）。2026-10-01 在 plain 路线 C=4、4 条长度 24/22/20/20 的 prompt 上首次观测到 lane2/lane3 与单跑不同，详见 §9 与 §12.5。

### Phase 2：后端与缓存补齐
- [ ] P2.1 MTP batch chain（A2）与 DFlash2 per-lane/batch（A3）。**已拆解：P2.1a/§12.1（执行层 verify-window 批量变体）→ P2.1b/§12.2（MTP）→ P2.1c/§12.3（DFlash2）**。**P2.1a/P2.1b/P2.1c 均已完成（2026-09-30，见 §12.1/§12.2/§12.3 状态行）**。
- [x] P2.2 `(bucket,B)` CUDA graph + AR channel 预算 + `round_base` 常量（D1–D3）。**已完成（2026-10-01）**：P2.2a `(bucket,B)` 图键（`WindowGraph.batch`/`WindowGraph.lane`）；P2.2b plain 批量 decode CUDA Graph；P2.2c MTP/DFlash2 批量 verify CUDA Graph + 三处崩溃/别名修复；P2.2d AR channel 预算静态复核 + C=1..4 live 计数。测量与实现记录见 §12.5，验收行见 §9。
- [x] P2.3 跨会话 retention per-lane 化（E2）。**已完成并验证（2026-10-01）**：设计见 §12.6，Stage 2c 快照方向 bug 的发现与修复见 §12.8，`a recalled conversation` 已修复，门禁 11/11、artifact 用例 exit 0、Stage 2b 四个活体探针 `PROBE_EXIT=0`（含真实 `src=host` restore）。单 lane 路线零变化（`tp2_sessions` plain 推进到与 HEAD 逐字相同的既有失败）。遗留：lanes>1 时 `copy_lane_state` 的 2D 拷贝跨 slot 步进（§12.8 末行）。
- [x] P2.4 多模态 per-lane（F1）。**已完成并实机验证（2026-10-01，见 §12.9）**：媒体请求进批（`drive_lane_queue` 删除 media 串行拒绝），两个 batch executor 各自 per-lane 持有 Vision 会话；双图批 `batch=2 path=batched` 答案正确，混合批（media+text）实测成批。施工中另发现并修复了一个 **HEAD 既存**的「不同图片互相复用 KV」正确性缺陷（每张图合并成相同 placeholder token id ⇒ token-only 复用判定误判）——新增 `MediaSpan` 精确 item 比较并接入 `scan_lane_reuse`/`session_recall`/`execute_walk`/两个 batch executor。
- [x] P2.5 运维文档与并发档位定稿（G2）。**已完成（2026-10-01，见 §12.10）**：四处文档同步（`docs/serving.md`、`docs/tp2-dual-5060ti.md`、`model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md`；`AGENTS.md:35-39` 已是最新），并发档位定稿为 `--max-concurrency 1..4` + 131072/共享 KV/per-lane retention，并删除临时 `NINFER_TP2_LOGITS_PROBE` 探针。

### Phase 3（可选）：架构收敛
- [ ] P3.1 评估方案乙：TP-2 变 Program 双设备后端，删除 `TP2GenerationCore` 的调度/资源/缓存重复实现，落地 R1 缓解。

---

## 8. 验证方案
- **数值 oracle**：批内每 lane 与同 lane 单跑比较；精确输出用逐位，概率路径按既有容忍（温度 0 不保证逐位 argmax，需 `--greedy`，见 `PLAN.md:193-194`）。
- **可复现性前提（P0.4 实测）**：dflash2 路线 `temperature:0` 也不逐位复现（同 prompt 4 次公共前缀仅 15–71/350 字符）⇒ 硬 oracle 必须在离线 `--greedy`/固定 seed 下做；HTTP A/B 用前缀一致率/token 一致率容忍，或整个 A/B 都在 `--greedy` 服务器上跑。
- **行为**：round 成员与 B 变化、取消、批量失败、终态与 finish reason、流式分块顺序。
- **性能**：`tools/bench/ttft` 并发档 + `tools/win_port/bench_serve.ps1`；报告聚合 tok/s、轮时、per-draft 接受率、与 C=1 的比值。
- **内存**：每 shard `[mem]` ledger + `nvidia-smi` 实测 free；C 与上下文档位组合。
- **回归**：`tools/win_port/test.ps1` 的 TP-2 套件（`ninfer_qwen3_5_tp2_*`、`ninfer_tp_device_pair_test`、`ninfer_engine_options_test`、`ninfer_serve_options_test`）全绿。
- **禁用项与契约须显式验证**：Phase 1 的两个「禁用项」在 Phase 2 均已解除——P2.3 把跨会话 retention 做成 per-lane（单 lane 行为零变化，见 §12.6/§12.8），P2.4 让多模态请求进批（媒体在批次内按 lane 各建 Vision 会话，见 §12.9）。因此验证改为：①多模态 + 文本混合批实测成批且答案正确；②媒体身份参与前缀复用判定，不同图片不再互相复用 KV（`MediaSpan` / `media_prefix_cap`，见 §12.9）；③单 lane 路线逐字节不回归（`tp2_sessions` plain 与 HEAD 逐字相同）。

---

## 9. 进度
- [x] 建立分支 `feat/tp2-concurrency` 与本文档（基线 `d1a9b57b`）。
- [x] Phase 0（P0.1–P0.4）：决策接受、C=1/2/4/8 并发复测、内存预算表、C=1 软基线（2026-09-30）。
- [x] P1.1 归一化放开：代码 + 构建 + `ninfer_engine_options_test`/`ninfer_serve_options_test` 全绿（2026-09-30）。
- [x] Phase 1（P1.2–P1.8）：**全部完成**（2026-09-30）；P1.2/P1.3/P1.4 施工图见 §11。验收证据：批内 oracle 逐字节、B=2 1.76–1.91×（门禁 ≥1.6×）、B=4 2.80×（门禁 ≥2.5×）、C=1 走未改动的单 lane 路线（`forward_tp2_decode_window`，不经过 `_batch`）故逐字节不回归（最终二进制实测：`--max-concurrency 1 --max-context 131072 --kv-dtype int8`，health 21 s 就绪、单请求 wall 1440 ms / prompt 100.8 ms / predicted 1324.1 ms for 51 token，内容正确）、`test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 11/11。
- [x] P1.2a 资源宽度参数化（2026-09-30）：`tp2_generation_core.cpp::build_shard` 的 5 处宽度 + 2 处 hidden 列数改用 `lanes`；`lanes` 取自 `options_.max_concurrency`（已由 `model_instance.cpp:96-97` 收敛到 `kTp2GenerationMaxConcurrency`，且 `engine.cpp:167` 在 `:173` 构造 TP-2 core 之前完成归一化 ⇒ 今天恒为 1，逐字节等价）。
  构建：`build.ps1 -Jobs 16` → exit 0（25 s，增量）。测试：`test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` → 11/11 通过。
  双卡端到端冒烟（2026-09-30 20:15，`build-win/apps/ninfer-serve.exe`，`--devices 0,1 --max-context 131072 --kv-dtype int8 --spec dflash2 --draft-tokens 5 --max-concurrency 2 --max-pending-requests 16 --host-kv-mib 20480`）：engine ready 41.6 s；C=1 请求 HTTP 200、内容 `pong`、finish=stop、prompt 19 tok 311 ms、decode 2 tok、dflash2 接受 1/5；**C=2 与 C=4 并发全部 200**（P0.2 时 C=4 是 2×200 + 2×429）⇒ 第 3+ 并发现在进 FIFO 而不是被拒，P1.1 的队列深度放开端到端生效；`--max-concurrency 2` 被正确收敛到 1（仍单请求执行）。
  实测显存账本（131072/int8/dflash2，lanes=1）：shard0 `weights+ctx 11528.6 | kv 2112.0 | state 293.6 | record 5.1 | draft-snap 80.0 | workspace 192.0 | free 1996.0` MiB；shard1 `weights+ctx 10686.6 | kv 2112.0 | state 293.6 | record 5.1 | free 3018.0` MiB。
  注意：`ninfer_qwen3_5_tp2_{load,forward,sessions,dflash_solo,dflash_append}_test` 会被 ctest **Skipped**（`test_tp2_forward.cpp:158` 需显式 `--artifact <path>`；):163 需两卡），且它们自带 `build_shard_state`（`test_tp2_forward.cpp:120-129`）**不走** `TP2GenerationCore::build_shard`，所以这些测试无法验证本次改动；真正的回归门是双卡 `ninfer-serve --devices 0,1` 冒烟。
- [x] P1.2b/P1.2c（2026-09-30）：state 乘数按方向 (c) 接受、D5 由 `lanes_ == 1` 机械强制（`host_checkpoint_stride_ = 0` + session catalog 空）；`build_shard` 的 KV 改为每 lane 一条 execution row 的静态分区，`Shard` 三个 KV 成员 per-lane 化，retention 固定走 lane 0；构造函数拒绝 `lanes_ > 1` 与 mtp/dflash2 的组合。构建 exit 0；`test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` **11/11 通过（71.1 s）**。基线 test 复验（全量一致重编后）同为 11/11（71.1 s）。
  门禁临时改 2 的启动冒烟（2026-09-30 20:51，`--spec` 省略 ⇒ plain，`--devices 0,1 --max-concurrency 2 --max-context 131072 --kv-dtype int8 --max-pending-requests 16 --host-state-slots 32 --host-kv-mib 20480 --port 8099`）：`[mem] host checkpoint ring off: the 2-lane route would pin one state image per lane per slot (D5)`、`[mem] host sessions capacity 0 | host KV 0.0 MiB/shard | retention disabled`、`[mem] shard N capacity 131072 lanes 2 | weights+ctx 10448.6 | kv 2112.0 | state 587.2 | record 0.0 | draft-snap 0.0 | workspace 192.0 | vision 0.0 | free 2968.0 of 16310.6 MiB`（两 shard 各一行）、`[mem] shard N KV pages 2048 over 2 lanes = 1024 pages (65536 tokens) per lane`；engine ready 40.4 s；两次**串行**请求均 200 / 内容 pong（prompt 236 ms、42 ms，各 2 tok）。并发冒烟会踩坏 lane 0（P1.4 未做，所有 lane 都走 `ingress.text_kv_table_rows[0]=0`），故只做串行；冒烟后门禁已改回 1 并重编。
- [x] P1.4a 多 lane 准入与 FIFO 队列（leader-driver，2026-09-30）：`PendingRequest` / `lane_queue_` / `lane_queue_mutex_` / `lane_queue_cv_` / `lane_driver_active_` / `active_lane_` 落地于 `tp2_generation_core.h:303-314` 与 `:556` 一带；`wait_lanes` / `drive_lane_queue` / `execute_lane` 在 `tp2_generation_core.cpp:1559-1671`。**不引入独立 worker 线程**：提交线程把请求压入 `lane_queue_`，若 `lane_driver_active_ == false` 则该线程在取得 `execution_mutex_` 后成为驱动者执行 `drive_lane_queue()`，否则在 `lane_queue_cv_` 上等自己的 `complete`。`execute_lane(lane)` 里 `active_lane_ = lane` + 两 shard `device.bind_to_current_thread()` + `context->set_linear_state_slots(lane, lane)`。
  门禁临时改 4、`--max-concurrency 2/4` 的双卡冒烟：2/4/8 并发全部 HTTP 200 且内容逐字正确；lane 隔离 oracle（48 token，同一 prompt 先单跑再并发）逐字节一致 ⇒ KV row / GDN slot 绑定正确。但每 tok 速率与单跑相同（≈33 tok/s）⇒ P1.4a 只给准入与隔离、**无吞吐收益**，与 §3.1 的判断一致；收益必须来自共享 decode round。
- [x] P1.4c 共享批量 decode round（2026-09-30）：新增**专用** `void TP2GenerationCore::execute_plain_batch(const std::vector<std::shared_ptr<PendingRequest>>& batch)`，plain-only、单线程、`execute_walk` 与 `lanes_ == 1` 路径**未改**。`drive_lane_queue` 里 `bool batchable = lanes_ > 1 && !mtp_enabled_ && !dflash2_enabled_;` 并逐成员检查 `qwen::PreparedPromptAccess::view(...).has_media()` 与 `tool_call_output != nullptr`（拒绝原因 route/media/grammar 记进 trace），不满足则退回逐 lane 串行 `execute_lane`。
  批内流程：一批开始时两 shard 各 `cudaMemsetAsync(state_backing, 0, bytes)` 清整池（D5 下无跨请求状态，所有 lane 从零开始）→ 逐 lane 串行 prefill（各自 chunk scope + `forward_tp2_prefill(..., active_lane_)`，末 chunk 采首 token）→ **共享 decode 循环**：每轮 `position_arena(ws_a/ws_b, round_base, round_base)` 复位 → 组装 host arrays → 一次 `forward_tp2_decode_window_batch(...)` → 一次 `ops::sample([V,B], out[B], per-lane configs)` → D2H `columns` 个 token → 逐 lane 走 tail（budget / preview_model / publish_preview）→ 批结束 `timing.report("plain-batch")`。
  **关键修复（P1.4c 实测暴露）**：最初驱动线程一入队就立刻成批，并发请求各自成 1-lane 批（B=2 只有 **1.073×**，trace 连续两次 `batch=1`）。加入 3 ms 成批窗口（`constexpr std::chrono::microseconds kBatchFormationWindow{3000}` + `lane_queue_cv_.wait_for(queue, kBatchFormationWindow, [&]{ return lane_queue_.size() >= lanes_; })`，并在 `push_back` 后补 `lane_queue_cv_.notify_all()`）后，trace 显示 `batch=2 capacity=2 path=batched`。
  `submit` 新增护栏：`if (lanes_ > 1 && summary.prompt_tokens > context_window) throw RequestError(RequestErrorKind::ContextLengthExceeded, "prompt exceeds this lane's context capacity");`（多 lane 下 `context_window - prompt_tokens` 会无符号下溢）。
- [x] P1.4d 门禁实测（2026-09-30，plain 路线、`temperature 0`、`reasoning_effort none`、48 token、prompt 为「Count from 1 to 80…」；plain + temperature 0 已实测**逐位确定**，solo 与并发输出完全一致，硬 oracle 可用）：
  - **B=2：串行 3098 ms vs 并发 1619 ms = 1.913×**（门禁 ≥1.6× 通过），两条输出与 solo 逐字节相同。
  - **B=4（门禁 4）：串行 5814 ms vs 并发 1850 ms = 3.143×**（门禁 ≥2.5× 通过），四条输出与 solo 逐字节相同；trace `batch=4 capacity=4 path=batched`。
  - 单请求对比（48 token）：lanes>1（eager）predicted_ms 中位 ≈1560（≈31 tok/s）；**lanes=1（CUDA graph 路径）predicted_ms 中位 1259（≈38 tok/s）** ⇒ 多 lane 档位下单请求慢约 24%，代价来自 `lanes_ > 1` 强制 `EagerExact`（`tp2_generation_core.cpp:532`；启动日志 `[tp2-graph] plain decode step: graph` vs `eager(exact)`）。属 §4.2 / P2.2 已预期的取舍；`--max-concurrency 1` 时路径逐字节不变、**无回归**。**2026-10-01 更正（P2.2b 后）**：多 lane 档位的单请求已不再走 eager 批量 decode，而是 replay 批量 decode CUDA Graph，差距落在测量噪声内——plain C=4 单跑 `predicted_ms 1233` vs 本行 C=1 单跑中位 1259；MTP C=2 单跑 711 ms vs C=1 oracle 810 ms。
  - 流式：2 条并发 SSE（不同 prompt）各自 delta 正确（32 / 13 个 delta），per-lane OutputSink 与终态正常。
  - 内存账（131072 / int8 / plain）：lanes=1 每 shard `state 293.6 | free 3262.0`；lanes=2 `state 587.2 | free 2968.0`；lanes=4 `state 1174.5 | free 2380.0` MiB（`weights+ctx 10448.6 | kv 2112.0 | workspace 192.0` 固定），`KV pages 2048 over C lanes = 2048/C pages 每 lane`，并出现 `host checkpoint ring off ... (D5)` 与 `host sessions capacity 0 | retention disabled`。
  - 日志：`_temp/p14c_serve.log`、`_temp/p14c_serve2.log`（含 trace）、`_temp/p14c_serve4.log`、`_temp/p14c_serve1.log`（lanes=1 基线）。
- [x] P1.5/P1.6 行为验证（2026-09-30，capacity 4、plain）：**流式** —— 2 条并发 SSE（不同 prompt）分别产出 32 / 13 个 delta，per-lane `OutputSink` 分块顺序与终态（finish reason / usage）均正确。**取消** —— 同时发起 A（1.2 s 后用 `AbortController` 断开连接）与 B（48 token oracle 请求），trace 为 `batch=2 capacity=4 path=batched`；A 在批内退役，日志 `req#2 done | openai-chat | cancelled | prompt 27 | output 35 | cache 0 (0.0%) | TTFT 48.9 ms | total 1.2s | prefill 561.8 tok/s (27 tok) | decode 29.5 tok/s`，同批 B 正常返回 200 且内容与 solo 逐字节一致（1595 ms vs solo 1456 ms）⇒ **取消不污染同批其余 lane**。取消来源链路：`src/serve/http_transport.cpp:36 client_disconnected()` → `src/serve/openai_chat_http.cpp:51/75` → `src/serve/generation_service.cpp:419-433` 的 `CancellationView`。日志 `_temp/p16_serve.log`。
- [x] P1.6 D4 失败语义核对 + P1.7 服务层容量（已在 P1.1 完成，仅文档待 D3）：`drive_lane_queue` 的 `catch (...)` 取 `std::current_exception()` 后 `session_invalidate_active()`，并把同一个 `exception_ptr` 写到批内**每一个** `PendingRequest::failure`（`tp2_generation_core.cpp` 的 `drive_lane_queue` 内）⇒ 批内不存在半提交；`GenerationResult` 与取消路径的逐 lane 职责在 `execute_plain_batch:1996-2019`。
- [x] 回归测试修复（P1.6 冒烟暴露，2026-09-30）：首次全跑 `test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 出现 1 例失败 —— `ninfer_serve_options_test`「TP-2 route did not clamp the service concurrency to the core capability」。根因是 `tests/test_serve_options.cpp:417-421` 写死 `"--max-concurrency", "3"` 却断言 `tp2_capacity.max_concurrency == ninfer::kTp2GenerationMaxConcurrency`（`min(3,1)=1` 才成立；门禁抬到 4 后 `min(3,4)=3` ≠ 4）。改为 `std::to_string(ninfer::kTp2GenerationMaxConcurrency + 1)` 后测试对常量取值无关。重跑 → **11/11 通过，TEST_EXIT=0，69.14 s**。
- [!] **施工中发现的构建系统缺陷（非本任务引入，但会让所有验证失效）**：`build-win/CMakeFiles/rules.ninja` 的 `msvc_deps_prefix` 是 CMake 把 cl 的中文 UTF-8 提示按 CP936 误解码后再编码的乱码，与 cl 实际输出（`注意: 包含文件:  `，两个尾随空格）字节不符 ⇒ ninja 一条都匹配不上，`.cpp/.cc` 的依赖记录全为 `#deps 0`，**改头文件不触发任何重编**。实测影响：`src/runtime/engine/engine.cpp` 在 `:164` / `:173` 需要 `TP2GenerationCore` 的完整类型，而它的 obj 停在 9/29 22:28（早于 P1.2a 的 `lanes_` 与 P1.2c 的成员更换）⇒ 已链接二进制里按旧布局分配该类。修复：`tools/win_port/build.ps1` 新增 `Repair-MsvcDepsPrefix`（构建前用 `cl /nologo /showIncludes` 探针取真实前缀并回写 rules.ninja；前缀变化时同时删除 `.ninja_deps` 并 touch 全部 `src`/`tests` 下的 C++ 源 —— 旧前缀记录出的路径是 include 行的尾巴，重新加载会让 ninja 去 stat `:\\Documents\\...` 并在开工前直接失败）。修复后 `tp2_generation_core.cpp.obj` / `engine.cpp.obj` 的 `#deps` = 197/198，改头文件可正确触发重编。
- [x] P1.8 验收（2026-09-30）—— 首轮失败，定位到一个**真实缺陷**并修复。首轮用 `Start-ThreadJob` 并发 4 条不同 prompt（`temperature 0`、`reasoning_effort none`、48 token）：B=4 只有 1.974×（门禁 ≥2.5×），且**批内 oracle 失败** —— pair (count80, alphabet) 三次里两次让 count80 在第 25 个 token 提前 stop。
- [x] **根因与修复（P1.8）**：`src/models/qwen3_5/execution/text.cpp` 的 GDN mixer 两条批量分支都带 `active_sequence_batch_ > 1` 条件 —— `:1732-1733` `const bool per_lane_conv = shard_config_ != nullptr && ph == Phase::Verify && active_sequence_batch_ > 1 && !recording;` 与 `:1866` `else if (shard_config_ != nullptr && ph == Phase::Verify && active_sequence_batch_ > 1)`。条件为假时 `:1751-1757` / `:1894-1898` 退回标量路径，读的是 `TextContext::set_linear_state_slots()`（`text.cpp:375-382`，写 `linear_state_source_slot_` / `linear_state_destination_slot_`）发布的**标量** slot，而不是批量入口绑定的 `active_linear_state_source_slots_` / `destination_slots_` 设备向量。`execute_plain_batch` 的 prefill 循环逐 lane 调 `set_linear_state_slots(slot, slot)`（`tp2_generation_core.cpp:1914`），最后一次停在**最后一条 lane**；于是当某轮 `columns == 1`（批内其余 lane 已退休）时，幸存 lane 若 slot==0 就读写了 slot==1 的 recurrence，立刻产出垃圾 token（幸存者恰好 slot==1 时不出错；(0,0)/(2,2) 这类同 prompt 批永远看不到）。逐 lane 串行路径 `execute_lane` 每轮显式重设 slot（`tp2_generation_core.cpp:1713`）所以无此问题。**修复**：在 `forward_tp2_decode_window_batch` 的 envelope 校验之后插入 `if (batch == 1) { const std::int32_t slot = state_slots[0]; set_linear_state_slots(slot, slot); peer.set_linear_state_slots(slot, slot); }`。选择在模型层入口修而不是把 `> 1` 放宽成 `>= 1`：后者会让单 lane 的 `forward_tp2_decode_window` 也切到批量 kernel，改动主路线行为与性能。attention 一路走 `active_sequence_batch_ != 0`（`text.cpp:1514`），batch==1 也用 `active_kv_table_rows_` 设备向量，所以 KV row 一直是对的，只有 GDN state 受影响。
- [x] P1.8 复测（2026-09-30，plain、`--max-concurrency 4 --max-context 131072 --kv-dtype int8`）：改用**同一进程内 `System.Net.Http.HttpClient` + `Task.WaitAll` 同时发出**（`Start-ThreadJob` 每个 job 启动约 100 ms，到达抖动远大于 3 ms 成批窗口，4 并发会被拆成 2×2，那是客户端噪声不是引擎问题）。`_temp/p18_fix.log` trace：`batch=4 capacity=4 path=batched` ×3、`batch=2 capacity=4 path=batched` ×2、`batch=1` ×10（solo）。**CONC4 三次 1806/1805/1808 ms vs SERIAL4 5056 ms ⇒ 2.80×（≥2.5× 通过）**；CONC2（count100..130 + squares）2861/1622 = **1.76×**（≥1.6× 通过）；CONC2（count80 + alphabet）2195/1560 = 1.41×，因为两条 lane 输出长度差一倍（48 vs 24），批轮成本由最长 lane 决定 —— 首期不做批内补位（连续批处理），该组合收益天然偏低。**批内 oracle：CONC2 两种配对 + CONC4×3 全部与各自 solo 逐字节一致**，solo 自身两次重复亦逐字节稳定。脚本 `_temp/p18_accept2.ps1`，输出 `_temp/p18_accept2.out`。
- [x] P1.8 收尾：删除了 `execute_plain_batch` 内的**临时**逐轮/逐 lane trace（`[tp2-lane] round columns=...` / `lane done ...`，含 `finalize` lambda 的捕获回退为 `[]`）；保留 `drive_lane_queue` 里低频的 `[tp2-lane] batch=... capacity=... path=...`（env `NINFER_TP2_LANE_TRACE=1` 才打印）。
- [x] P1.8 回归测试（2026-09-30）：把 `tests/models/qwen3_5/test_tp2_forward.cpp` 从单 lane 扩到两 lane —— 新增 `constexpr std::uint32_t kTestLanes = 2;` 与 `kTestPoolPages = kTestCachePages * kTestLanes`（=64）；`DecoderStateSpec.kv_table_rows = kTestLanes`、`.text_physical_page_groups = kTestPoolPages`；`LinearAttentionStatePoolSpec.slot_count = kTestLanes`；`ShardState` 的 `KVExecutionRowLease kv_row` 换成 `std::vector<KVExecutionRowLease> kv_rows`；`publish_kv_rows` 逐 lane `tables.acquire(lane)` + `tables.publish(row, 0, std::span<const DeviceKVPageHandle>(kv_handles.data() + lane * kTestCachePages, kTestCachePages), stream)`（每 lane 32 个物理页）。新增第 5 个用例：lane0/lane1 用**不同** token（`prompt_ids[0..8)` 与 `prompt_ids[150..158)`；同 token 时两条 recurrence 逐位相同，读错 slot 也看不出来），先 `set_linear_state_slots(lane, lane)` 再 `forward_tp2_prefill(..., lane)`，然后用 `forward_tp2_decode_window_batch(..., batch=1, slots={0}, envelope{1,9})` 走一次单列批，要求与「只 prefill lane0」的对照**逐位相同**（`max_logit_diff == 0` 且 argmax 相同）。
- [x] 回归测试双向验证（2026-09-30，artifact `D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer`，直接跑 `build-win/tests/ninfer_qwen3_5_tp2_forward_test.exe --artifact ...`，PATH 前置 ffmpeg/curl bin）：修复在位 `argmax_alone=92 argmax_shrunk=92 max_logit_diff=0`、TEST_EXIT=0；把 `src/models/qwen3_5/execution/text.cpp` 的修复暂时改成 `if (false && batch == 1)` 后重建 → `argmax_alone=92 argmax_shrunk=198 max_logit_diff=11.5`、TEST_EXIT=1、`FAIL: a one-column batch did not run its own lane's state` ⇒ 用例确实能抓住该缺陷；已还原修复（`text.cpp:2329` 为 `if (batch == 1) {`）并重建 exit 0。日志 `_temp/tp2_regress_fixed.log` / `_temp/tp2_regress_broken.log`。
- [x] 文档数字同步：`docs/serving.md:75-81` 的 `1.91x / 3.14x` 改为 `1.76-1.91x at --max-concurrency 2 and 2.80x at 4`，并补一句说明 batch round 的成本由最长成员决定（输出长度差一倍的配对实测 1.41x）。
- [!] 测试环境注意：`serve.ps1 -Stop` 之后**立刻**跑全套会让 `ninfer_tp_device_pair_test` 假失败（报 `44 failures`、`graph queue[...] replay N element 0: expected X, got 0`、`[mem] TP-2 in-kernel allreduce staging 24.0 MiB of 48.0 (larger rungs did not map)`）—— 退出服务后显存尚未完全释放；等 `nvidia-smi` 两卡归 0 MiB 后重跑即 11/11（69.28 s）。

- 构建/测试口径：`pwsh -NoProfile -File tools/win_port/build.ps1 -Jobs 16`（exit 0；日志 `build-win/build.log`）；测试必须走 `tools/win_port/test.ps1`（`test.ps1:30` 把 ffmpeg/libcurl 的 bin 加进 `PATH`），直接跑 `build-win/tests/*.exe` 会因缺 DLL 报 `0xC0000135`。
- **基线（d1a9b57b）本就存在的两个陈旧断言**，已一并修正（与 P1.1 无关，但挡着 §8 的「全绿」门禁）：
  1. `tests/test_engine_options.cpp:49` 断言 TP-2 `host_kv_capacity_bytes == 0`，与 `model_instance.cpp:131-142`「TP-2 保留 host KV 预算供 retention 用」及 `types.h:33`（默认 8 GiB）矛盾；改为断言保留 `kDefaultHostKvCapacityBytes`。
  2. `tests/test_serve_options.cpp:282-285` 在思考关闭时断言 medium effort 已解析，与 `translate.cpp:194-202`（只在 `result.enable_thinking == true` 时算 `effective_reasoning_effort`）矛盾；改为用 `thinking_request.enable_thinking = true` 驱动。
- [x] P2.1a 执行层批量 verify window（2026-09-30）：`forward_tp2_window_batch`（`text.h`/`text.cpp`）、分片 record per-lane 化、引擎侧 `replay_views`（每 batch 一份 record/fold 视图）+ `run_verify_window_batch` + `fold_verify_window` 全部落地。验收见 §12.1 状态行；判据放宽为「硬判据①复制 lane 逐位 ②batch=1 vs C=1 逐位；放宽判据 per-lane 每列 argmax 等价（近邻并列豁免）+ 漂移 ≤2.5」（用户 2026-09-30 决策 A），副作用：并发下 token 依赖本轮活跃 lane 数（占用相关、非逐位确定）。
- [x] P2.1b MTP 批量链（2026-09-30）：`execute_spec_batch` 的 MTP 段（批量链 + 批量 verify window + greedy accept + per-column commit）、MTP K/V 池按 lane 切分、`lane_context_window()` 上下文界修正、MTP 路由 `dflash_round == nullptr` 的空引用崩溃修复全部落地。验收见 §12.2 状态行：B=2 `same=True/True` 1.48×、B=4 `same=4/4` 2.26×。
- [x] P2.1c DFlash2 批量 round（2026-09-30）：见 §12.3 状态行（B=2 `same=True/True` 1.50×、B=4 `same=4/4` 2.07×）。
- [x] P2.2a `(bucket,B)` CUDA Graph 图键（2026-10-01）：`WindowGraph` 增 `batch`（本批活跃 lane 数）与 `lane`（单列捕获时烘进的 state slot；设备绑定时为 −1）；`select_window_graph(std::vector<WindowGraph>&, visible_end, batch, lane)`（`tp2_generation_core.cpp:1171`）先按 `batch` 过滤、再在 `lane != -1` 时要求 `graph.lane == lane`，`reusable_window_graph`（`:1184-1193`）额外要求 `captured && round_base[0] >= shard_a_.round_base && round_base[1] >= shard_b_.round_base`。width==1 的条目逐 lane 各捕获一份（`lane` = 该 lane 的 state slot），width≥2 的条目 `lane = -1`。
- [x] P2.2b plain 批量 decode CUDA Graph（2026-10-01）：新增 `decode_batch_graphs_`（8 桶 × `2*lanes_-1` 条）、`capture_decode_batch_graph`（`:1464`）与 `run_plain_decode_step_batch`（`:1502`），`execute_plain_batch` 每轮 decode 改走该入口。验收（`_temp/20261001-0158_p22b_plain_smoke.ps1`）：graph 与 exact 的 solo/并发全部逐字节相同（`graph solo == oracle p0=True p1=True`、3 组并发 `True/True`）；2-lane solo `graph 1293 ms` vs `exact 1457 ms`（graph 快 11.3%）；并发 wall `graph 1394/1385/1394` vs `exact 1607/1604/1604`（快约 15%）；聚合 `graph 1.85×` vs `exact 1.60×`；单请求 solo 1312–1377 ms 与 C=1 基线同档。
- [x] P2.2c MTP/DFlash2 批量 verify CUDA Graph + 三处修复（2026-10-01）：`verify_batch_graphs_`（MTP 12 桶 / DFlash2 6 桶，每桶 `2*lanes_-1` 条）、`capture_verify_batch_graph`（`:1218`）、`run_verify_window_batch` 的 `batch_family` 路由（bucket envelope 恒取自 `verify_graphs_`，否则默认 `batch=1,lane=-1` 会跳过全部批量条目并误报覆盖不全）。修复 ① MTP 链的 `positions` 分配移出 `max_extent` 步循环（`tp2_generation_core.cpp:3026-3030`；否则每轮活跃 lane 数不同 ⇒ verify 调用点 `used()` 漂移 ⇒ `TP-2 batched verify CUDA Graph replay found a different workspace layout`）；② `gdn_mix` 的 per-lane conv staging 改设备端 `ops::fill_i32_positions(base_slots, slots, window, s)`（`text.cpp:1757-1762`，`ops::fill_i32_positions` 增 `stride` 参数；原先用函数局部 host vector 做 H2D，捕获后 replay 读失效栈地址 ⇒ DFlash2 并发 `cudaErrorIllegalAddress`，且该 H2D 会同步）；③ spec 路线 pinned verify staging 的五段偏移（`batch_window_section(section, stride) = base + section*stride` 让 `valid/kv_rows/slots` 覆写到 `spec_ids` 上，width=6/`lanes_=2` 时 ids [0,12) 与 valid [4,6) 重叠）改为 `batch_window_base()` + 显式 `ids → positions → valid → kv_rows → slots`。
- [x] P2.2c 验收（三臂 smoke `_temp/20261001-0500_p22c2b_smoke.ps1`：oracle=C1 / graph=C2 缺省 / exact=C2 + `NINFER_TP2_VERIFY_BATCH_GRAPH=exact`；`--max-context 131072 --kv-dtype int8`、temperature 0、48 token）：MTP 三臂 solo 与三次并发全 `graph==oracle`/`exact==oracle`/`graph==exact` True/True，wall graph `884/856/851` vs exact `955/954/951`（≈10% 快）；DFlash2 全 True/True，wall graph `504/478/481` vs exact `550/545/545`（≈12% 快），`draft=39/39`，无 `cudaErrorIllegalAddress`；2-lane MTP solo 711 ms vs 1-lane 810 ms（无单请求回归）。构建 exit 0；`test.ps1 -Filter 'position|tp2|tp_device|engine_options|serve_options'` 12/12；artifact `tp2_forward`/`tp2_load` 均 exit 0。
- [x] P2.2d AR channel 预算复核（2026-10-01，静态 + live）：全仓 `create_ar_channel` 仅 5 处（`tp2_generation_core.cpp:1274/1321/1532/1626/1765`），分别对应 `verify_graphs_`/`verify_batch_graphs_`/`decode_graphs_`/`decode_batch_graphs_`/`mtp_chain_graphs_` 各一个元素，且每处都被 `if (graph.ar_channel == tp::DevicePair::kNoArChannel)` 守护 ⇒ 同一 entry 因 `round_base` 回退被重新捕获也不取新 channel；ctor `:659` 在首个 capture 之前按五个 vector 的 size 之和 `reserve_ar_channels(...)` ⇒ `create_ar_channel` 的触顶分支不可达。计数公式：桶数 `ordinary_graph_profiles`=8 / `mtp_graph_profiles`=12 / `dflash_graph_profiles(DFlash2,…)`=6，每条批量家族每桶 `2*lanes_-1` 条，`decode_graphs_` 仅 `lanes_==1` 8 条。live `[mem] TP-2 rendezvous id channels reserved`：MTP C=1 32 / C=2 48(exact)·84(graph) / C=3 124 / C=4 164；DFlash2 C=2 30·48 / C=3 76 / C=4 104 —— 全部等于公式值，无 throw、health ok。同轮四 prompt 逐 lane oracle：DFlash2 C=3 1.45×、C=4 全 True 1.60×；MTP C=3/C=4 命中下一条的翻转现象。探针 `_temp/20261001-0600_p22d_ar_check.ps1`。
- [!] **B≥3 异质 prompt 的低 gap token 翻转（既有现象，非 P2.2 引入，判据 A 范围）**：批内所有 lane 共用一个 attention envelope（由批内最长 lane 的 visible extent 选 split policy；`tp2_generation_core.cpp:3057-3059` 注释「envelope steers split policy only … the widest lane bounds every lane's window」）⇒ 比最长 lane 短的 lane 其 reduction tree 与单跑不同 ⇒ ulp 级 logits 漂移 ⇒ 可能在序列后段（实测约第 35/48 token）翻转一个低 gap argmax。复现（`_temp/20261001-0700_p22div_mtp.ps1`）：plain（无 `--spec`）、C=4、4 条长度 24/22/20/20 的 prompt，`idx=0,1,2,3` 时 lane2/lane3 与单跑不同，而 `idx=0,1,0,1`（位置差 2）逐字节相同；MTP 的 graph 臂与全 eager 臂（`NINFER_TP2_VERIFY_BATCH_GRAPH=exact` + `NINFER_TP2_DECODE_BATCH_GRAPH=exact`）都复现，同四 prompt 的 DFlash2 C=3/C=4 不翻转。⇒ §12.2/§12.3 的「输出 token 仍逐字相同」与 P1.8/P2.1b/P2.1c 的 `same=4/4` 都只是该漂移未触发时的观测，不是保证。
- [x] Phase 2 其余（P2.3–P2.5）**全部完成（2026-10-01）**：P2.3 跨会话 retention per-lane 化（§12.6 设计、§12.8 Stage 2c 快照方向 bug 修复）；P2.4 多模态 per-lane + 媒体身份前缀复用修复（§12.9）；P2.5 运维文档同步与并发档位定稿（§12.10）。验收：门禁 11/11、两个 artifact 用例 exit 0、`tp2_sessions` 三路线只剩 HEAD 既有失败、双图批与混合批实机成批、缺陷用例 load_04→Purple / load_05→Brown。
- [ ] Phase 3（P3.1，可选）。

---

## 10. 未决问题与已知限制
- §3.1 的吞吐预期与 R6 仍需 Phase 1 真批处理实测；Phase 0 只补了**串行语义边界**与**内存预算**（§3.4，复用活服务器，未改配置）。
- R4 的 give-up 安全语义（`PLAN.md:253-267`、§3.6(b)）仍未定，批处理会放大其影响，D4 必须先决。
- 方案甲会加重 `PLAN.md:271-274` 的双核心漂移，Phase 3 需重新评估方案乙。
- DFlash2 与 MTP 互斥的既有约束保持不变；并发首期建议只用 plain/MTP。

---

## 11. Phase 1 施工图（代码级，file:line 为 2026-09-30 的 `feat/tp2-concurrency` 工作树）

### 11.1 P1.2 资源层
**P1.2a 已落地（2026-09-30）**：`build_shard`（`tp2_generation_core.cpp:545-929`）新增 `const std::int32_t lanes = static_cast<std::int32_t>(options_.max_concurrency == 0 ? 1U : options_.max_concurrency);`，并把下列写死宽度改为 `lanes`：`:598` `DecoderStateSpec.kv_table_rows`、`:622` `LinearAttentionStatePoolSpec.slot_count`、`:674` `GdnReplayRecordSpec.record_capacity`、`:701`/`:706` `prefill_hidden`/`mtp_anchor_hidden` 的列数、`:908` `RoundStateSpec.batch_capacity`（该字段是 `std::uint32_t`，需 `static_cast`）。`state_arena` 公式不变（`state_bytes` 自身已含 `lanes` 个槽）。类型陷阱：`kv_table_rows`/`slot_count`/`record_capacity` 是 `std::int32_t`，`batch_capacity` 是 `std::uint32_t`，`Tensor` 形状是 `std::int32_t` —— 直接传 `std::uint32_t` 会触发 C2397/C2398。

**state arena 乘数（2026-09-30 实测发现；P1.2b 决策 = 方向 (c)，接受乘数）**：`state_bytes` 是**整个 live 池**（`slot_count` 个槽）的大小，而 `state_arena = (2 + kReuseSnapshotCount) * state_bytes` 是「2 live + 2 snapshot」四份**整池**拷贝。因此 `slot_count = lanes` 的效果是 state 显存 ×lanes ×4：lanes=1 时 shard0 state 293.6 MiB（实测），lanes=4 时 1174.4 MiB（+880.8 MiB/shard）；此时 shard0 free 会从 1996.0 掉到约 1115 MiB。§3.3 表格里「C≤4 关掉 retention 即零增量」的说法只有在下述前提下才成立，否则要按 +293.6 MiB/shard/lane 重算预算。三个可选方向（按 Phase 1 取舍）：(a) 保留 `slot_count = lanes`，但把 snapshot/round-scratch 平面从「整池拷贝」改为 **每 lane 一份或整批共享一份**，把 arena 从 `4 x state_bytes` 降到约 `(lanes + 2) / lanes` 倍；(b) retention 关闭时直接不分配两个 reuse snapshot 平面（`execute_walk:2589-2591` 的 device-snapshot 分支与 `:2961/:2969` 的复用扫描在 D5 下本来就不走），只留 base + round scratch；(c) 接受 +880.8 MiB/shard 并把 §3.3 预算表改成实测值。**无论哪条，都要先重新确认 `kRoundScratchSlot = kReuseSnapshotCount = 2` 在每 lane 一份之后是否还成立。**

**同源发现：pinned host checkpoint 也 ×lanes**。`tp2_generation_core.cpp:641-657` 在 `host_checkpoint_stride_ != 0`（即 retention 开）时为每个槽分配 `PinnedHostBuffer(shard.state_backing.bytes, true)`，而 `state_backing.bytes == state_bytes`；槽数 = `host_state_slots + divergence + block`（本机实测 34）。lanes=1 时 34×73.4 = 2495.6 MiB/shard pinned；**lanes=4 时 34×293.6 = 9.98 GiB/shard（两卡合计约 20 GiB pinned host 内存）**，且这是启动期 eager pin。⇒ D5「Phase 1 禁用跨会话 retention」不只省 state 平面，还会**整条 ring 消失**（`host_checkpoint_stride_ == 0` 时不分配），这是 Phase 1 必须走 D5 的第二个理由。选 (c) 时必须同时确认 `--host-kv-mib 0` 或等价开关下 `host_checkpoint_stride_ == 0`。

**P1.2b 已落地（2026-09-30，方向 c + D5 机械强制）**：不重定义 snapshot 平面（(a)/(b) 都要求把 `kRoundScratchSlot`/`kReuseSnapshotCount` 的语义从「整池」改成「每 lane」，而这两处又是 verify 折叠与前缀复用的核心不变式，Phase 1 不动），改为**接受 ×lanes 的 state 乘数并按实测值编预算**，同时用 D5 把随 lane 线性放大的两个 host 设施整条关掉：
- `src/runtime/engine/tp2_generation_core.h:556` 一带新增成员 `std::uint32_t lanes_ = 1;`；构造函数在 `validate_tp2_devices(...)` 之后写入 `lanes_ = (options_.max_concurrency == 0 ? 1U : options_.max_concurrency);`；`build_shard` 开头改用 `const std::int32_t lanes = static_cast<std::int32_t>(lanes_);`（原来在 `build_shard` 里独立算一遍，现在单一来源）。
- 构造函数里 pinned ring 的守卫 `if (host_slots != 0)` 改为 `if (host_slots != 0 && lanes_ == 1)`，并加 `else if (host_slots != 0)` 打印 `[mem] host checkpoint ring off: the %u-lane route would pin one state image per lane per slot (D5)`。⇒ lanes>1 时 `host_checkpoint_stride_ == 0`，`tp2_generation_core.cpp:654` 的 34 个 `PinnedHostBuffer(..., true)` 一个都不分配（省 ~20 GiB pinned @lanes=4）。
- session catalog 的守卫 `if (host_kv_bytes / 2 != 0 && sessions > 1)` 改为 `if (lanes_ == 1 && host_kv_bytes / 2 != 0 && sessions > 1)`（session image 同样载着 per-lane state，所以 catalog 也是单 lane 设施）。⇒ lanes>1 时 `session_capacity_ == 0`。
- `[mem] shard` 账本行改为 `[mem] shard %d capacity %u lanes %u | ...`，运维可直接看到生效的 lane 数。
- 语义代价（D5 已记录）：多 lane 时前缀复用只剩设备侧 `state_snapshots[0]`（每次 prefill 完成时无条件写入，`tp2_generation_core.cpp:3333-3336`），深 rewind（slot 1，:3110-3130/:3170-3190，本就被 `host_checkpoint_stride_ != 0` 守卫）与跨会话 recall 全部不可用。
- 因 `kTp2GenerationMaxConcurrency` 仍为 1，`lanes_ == 1` ⇒ 全部走原路径 ⇒ C=1 逐字节等价。

**KV 布局铁律**：每 shard 只保留**一个** `pages_for_tokens(options_.max_context)` 的物理池（与今天同尺寸）；C 条 lane 各映射该池里**互不相交**的页子集。`C x capacity` 的 per-lane 全量池在 131072/2 卡上不可行（每 lane 2064 MiB/shard，见 §3.3）。因此 `--kv-capacity` 是**总预算**，准入必须保证批内 lane 的最坏上下文之和 <= 池容量。

`src/runtime/engine/tp2_generation_core.cpp::build_shard`（:545-929）逐点改：
- `:592 .kv_table_rows = 1` -> `lanes`；`:871` MTP 的 logical pages 不变但 `.kv_table_rows` 同步为 `lanes`。
- `:616 .slot_count = 1` -> `lanes`（每条 lane 一个 live 槽）；`:626 state_arena` 由 `(2 + kReuseSnapshotCount) * state_bytes` 改为容纳 `lanes` 个 live 槽 + round scratch/reuse 槽。`Shard::state_snapshots`（`tp2_generation_core.h:138`）语义要重定义：Phase 1 `kRoundScratchSlot` 必须**每 lane 一份**（verify 折叠从轮前快照回放）。
- `:666 .record_capacity = 1` -> `lanes`；`:685 replay_fold` 的 target 从 `state->all_layers_view()` 的单槽改为 per-lane 槽。
- `:900 .batch_capacity = 1` -> `lanes`（MTP round state，shard 0）。
- `:692 prefill_hidden {2*hidden,1}`、`:697 mtp_anchor_hidden {hidden,1}` 需按 lane 各一份（或加 lane 维）。
- `:755 plan_dflash2_round(...)`：`DFlash2RoundSpec.batch_capacity`（`src/models/qwen3_5/program/dflash_round.h:58`）已有，Phase 1 可先只支持 B=1 的 draft、把 DFlash2 并发留到 Phase 2。
- `:847-862 / :866-887` 页定型：现在 `reserve(pages)` 一次 + 全部 pin 到 row 0。改为一次 `reserve(total_pages)`（lease 常驻），按 lane 页区间 `tables.acquire(i)` + `publish(row, logical_begin, slice, stream)`。**释放必须走 `DeviceKVPageLease::release()`/`DeviceKVPageReservation::release()`**，否则复现 `docs/tp2-dual-5060ti-worklog.md:380` 的容量泄漏（`reserved_pages_` 累加，第 2 个请求 reservation failed）。API 见 `src/core/paged_kv_cache.h`（`kPagedKVPageSize = 64`）。
- 图相关不必动：归一化已把 TP-2 的 `use_cuda_graph` 置 false，Phase 1 走 eager（D4）。但 `round_base`（`:3392-3393`）与 `reusable_window_graph`（`:987`、`:1103-1107`、`:1200-1203`）的「watermark 必须精确等于 arena_begin」约束意味着**一旦开图就必须统一规划 per-lane 持久 scratch**（R2）。
- 验收：`[mem]` ledger（`:808-834`）对照 §3.3 预算表；`ninfer_qwen3_5_tp2_*` + `ninfer_tp_device_pair_test` 全绿。

### 11.2 P1.3 执行组合层
新的批量 forward（`src/models/qwen3_5/execution/text.h:190-192` 的 `forward_tp2_decode_window` 目前只吃单 `token`/`position` 指针）：
```
void forward_tp2_decode_window_batch(TextContext& peer, tp::DevicePair& pair,
    const std::int32_t* tokens, const std::int32_t* positions, std::int32_t count,
    const Tensor& kv_table_rows,      // [B] I32，每条 lane 的执行表行
    const Tensor& linear_state_slots, // [B] I32，每条 lane 的 GDN live 槽
    ops::CausalAttentionExecutionEnvelope envelope, Tensor& logits /* [V, B] */);
```
- 现有可直接复用的批量机制：`active_kv_table_rows_`/`active_backend_kv_table_rows_` 覆盖（`text.cpp:1512-1513`、:1268、:1322、:1403）；批量入口已有 `require_tensor_shape(kv_table_rows, I32, {batch})`（`:1252`/`:1307`/`:1399`）。
- 采样已经是批形状：`ops::sample` 建议 `include/ninfer/ops/sampling.h:36-48`（`[physical_rows, B]` + `SamplingConfig[B]`）；单卡侧参考 `src/models/qwen3_5/program/speculative/mtp.cpp:70-200`。
- **TP-2 core 现在把行号写死**：`tp2_generation_core.cpp:3491-3492` `ingress.text_kv_table_rows[0] = 0; ingress.dflash_kv_table_rows[0] = 0;` —— 批变体必须把 lane 行号写进 ingress（ingress 数组形状已是 `std::array<std::int32_t, kMaximumConcurrency>`，见 `src/models/qwen3_5/program/round_buffers.h:35/55/56/85/86`）。
- Phase 1 只上 plain（无 MTP/DFlash2）：`--spec none`。MTP batch chain 与 DFlash2 per-lane 留 Phase 2。
- 采样/输出侧：`tp2_generation_core.cpp:2768-2811` 一带现在是单请求 `ops::sample` + 单 `tool_mask`；改 per-lane `SamplingConfig[B]`、`token_counts` 与 per-lane `OutputSink`。

### 11.3 P1.4 核心调度
- 现状：`Submission::wait`（`:1458-1473`）在调用者 HTTP 线程内持 `execution_mutex_` 内联跑完 `execute()`；`execute_walk` 是 1550 行单体（`:2431-3987`），全程读写 core 成员态。
- Phase 1 形态：保留 `execution_mutex_` 作为「一个 round 只被一个线程驱动」的锁，新增 `AdmissionQueue`（FIFO，深度 = `max_pending_requests`）+ 批量成形（取队首至多 C 条）→ `execute_batch`。批内 **prefill 仍逐 lane 串行**（沿用现有 prefill 路径，只把 KV row/GDN 槽换成该 lane 的），decode 才进入共享 round 循环。
- **per-lane 化的全局态清单**（`tp2_generation_core.h:215-280` 与 core 成员）：`cached_prompt_tokens_`、`cached_state_valid_`、`cached_boundaries_`、`active_session_`/`anchor_session_`、`live_state_valid_`、`host_checkpoint_*`、`block_anchor_*`、`reuse_source_`、`dflash_context_frontier_`、`mtp_previous_draft_`。Phase 1 按 D5 **整体禁用跨会话 retention**（前缀复用扫描 `:2538-2596` 在多 lane 模式下不走 session 分支）。
- 失败语义 D4：`abort_if_ar_stalled`（`:1495`）在批内升级为整批失败（必要时整机），不允许半提交。
- 多模态：`vision_arena_`/`VisionWorkspacePlan`（`:706-747`）仍全局唯一 —— Phase 1 用全局锁把多模态请求串行化（F1）。

---

## 12. Phase 2 施工图（P2.1，file:line 为 2026-09-30 工作树）

### 12.0 施工顺序与理由
- P2.1 拆三步：**P2.1a** 执行层 verify-window 批量变体（width×batch）→ **P2.1b** MTP batch chain → **P2.1c** DFlash2 batch。
- 理由：两个后端共享同一前置——目标 verify 窗口的「每 lane T 列」批量执行层入口。Phase 1 的 `forward_tp2_decode_window_batch` 只支持 width=1（每 lane 一列）。
- P2.1 只走 eager：`decode_step_mode_` 在 lanes>1 时已是 `EagerExact`（`tp2_generation_core.cpp:524-545`），图矩阵留 P2.2。
- **顺序修正（2026-09-30，源码复核见 §12.4）**：b 与 c 的先后可调。鉴于生产默认路线是 `--spec dflash2`，且 `DFlash2Round`/ingress 数组已按 `kMaximumConcurrency` 批量就绪，P2.1 的实现顺序改为 **a（已完成）→ c（DFlash2）→ b（MTP）**：先在 DFlash2 上把 `execute_spec_batch` 骨架、三段式批量 round、per-lane accept/fold/流式/预算全部跑通并用单 lane oracle 验收，MTP 复用同一骨架（只需替换 draft 段与 greedy accept）。

### 12.1 P2.1a 执行层：批量 verify window（含 head-split 分片的 GDN record）
- **状态：已完成（2026-09-30）。** 验收证据：① `ninfer_qwen3_5_tp2_forward_test`（`--artifact D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer`）exit 0，第 6 组即下文放宽判据；② `--spec dflash2 --max-concurrency 1` 服务在 `verify step: graph` 与 `NINFER_TP2_VERIFY_GRAPH=0`（`verify step: eager`）两种模式下，对同一 `temperature 0 / reasoning_effort none / max_tokens 48` 请求产出**完全相同**的 token 文本（pn=48，两者 `[1, 2, 3, …, 14, 1]`）⇒ 单 lane eager 路收敛到 `run_verify_window_batch(batch=1)` + `fold_verify_window` 后行为不变；③ 构建 `tools/win_port/build.ps1` exit 0。
- **修正（2026-09-30 源码复核，取代本节初稿）**：TP-2 的投机 verify 不走 `forward_tp2_decode_window_batch`，而是 `run_verify_window`（`tp2_generation_core.cpp:1133-1198`）→ `forward_tp2_window`（单 lane、T=K+1 列），并要求 `set_gdn_state_action(GdnStateAction::RecordForReplay)`（`:1104-1107`/`:1162-1167`）+ 事后 fold（`:1196-1197` 复位 `UpdateInPlace`）。故 P2.1a 的目标是 `forward_tp2_window` 的批变体。
- 阻塞点（源码级已核实）：
  - `text.cpp:1611-1616`：`record_rows = active_sequence_batch_ > 0 ? active_sequence_batch_ : 1`，且 `recording && shard_config_ != nullptr && record_rows != 1` 抛 `"Replay-record GDN on a shard requires one record row"`。
  - `text.cpp:1724-1758`：`per_lane_conv = shard_config_ != nullptr && ph == Phase::Verify && active_sequence_batch_ > 1 && !recording`；该分支要求 `active_sequence_width_ == 1`（`:1735-1736` 抛 `"the GDN shard window supports one column per lane"`）。recording 时落标量 `conv_state_in/out`（`:1752-1757`），只有一个 slot 对。
  - `text.cpp:1837-1865`：分片 record 分支**已**按 `record_rows` 张成 `[..., width, record_rows]`（`:1844-1858`），但 `initial_slots`（`:1859-1860`）用 `set_i32_scalar(initial_slots, linear_state_source_slot_)` 把所有行填成同一 slot —— 单 lane 假设。
  - `text.cpp:1596-1598`：`batched_verify = (ph == Phase::Verify) && (batch > 1 || width > 1) && shard_config_ == nullptr` —— head-split 分片永远不走该融合路。
  - `text.cpp:1866-1874`：分片 batch verify（width 必须为 1）走 `gated_delta_net_batch_update`；`:1628-1630` 对 `UpdateInPlace && width != 1` 抛错 ⇒ 投机 verify 在分片上只能走 record 路线。
- P2.1a 最小改动面：
  1. 分片 recording 的 conv 侧 per-lane 化：改用 `ops::causal_conv1d_silu_snapshot`（`:1743-1746` 已在使用）配 per-lane source/destination slot 向量，并把原始窗口按 `[conv_channels, T, B]` 写入 record 平面（`:1763` 现有 `records.conv.bytes() != qkv.bytes()` 检查是单行尺寸）。
  2. 分片 record 的 recurrence 侧 per-lane 化：`initial_slots`（`:1859`）改由 per-lane I32 向量填充，替换 `set_i32_scalar`。
  3. 放开 `:1613-1616` 的 `record_rows != 1` 守卫。
  4. fold/replay：核对 `GdnReplayRecordLayer` 的 conv/key/value/gate 平面按 `batch_capacity` 张成且 replay 逐 lane 回放（`RoundStateSpec.batch_capacity` 已按 lanes 传入，`tp2_generation_core.cpp:979`；record 消费方 = fold 调用点）。
  5. 执行层入口：新增 `forward_tp2_window_batch(peer, pair, ids, positions, kv_table_rows, state_slots, width, batch, envelope, logits /* [V,width*batch] */, hidden, sink, valid_columns)`，与 `forward_tp2_window` 共享主体；列数 = width*batch，bindings 设 `active_sequence_width_ = width`、`active_sequence_batch_ = batch`。
- 布局：`attn_mix`（`text.cpp:1516-1540`）要求 `width * batch == T` 且 `cache_positions.view({width, batch})` ⇒ 逐列数组按 **lane 最快** 排列（index = col*batch + lane）；每个 lane 的 T 列重复同一 `kv_table_rows`/`state_slots` 值。
- 风险：Phase 2 数值风险最高的改动（首次在 head-split 分片上以 batch>1 走 record+fold）。验收必须用「同 lane 单跑（现有单人 verify）vs 批跑」逐位 oracle。
- 决策：P2.1b/P2.1c 均以本步为前置。若本步不通，则 TP-2 上 MTP/DFlash2 多并发不可用（R1 的具体表现），应评估上调 P2.2/P2.3 或回到 P3.1 方案乙。
- **实测结论（2026-09-30，P2.1a 判别实验；取代本节的「逐位 oracle」要求）**：
  - 证据① per-lane 绑定正确：复制 lane 对照（两 lane 装同一请求）`lane_gap=0`；`window_batch_dup` 的 lane0 各列与 `window_batch`（lane1 装不同请求）的 lane0 各列 `other_lane1_gap=0`（逐位）⇒ 一个 lane 的窗口输出与同批其他 lane 的**内容完全无关**，无跨 lane 串扰、无共享量化块、无共享 tile。
  - 证据② batch=1 入口与历史 C=1 路由逐位一致：`legacy_vs_lane0_diff=0`（`forward_tp2_window` vs `forward_tp2_window_batch(batch=1)`）。
  - 证据③ 差异是引擎的 M/列数敏感性，不是批处理缺陷：**单 lane、batch=1、8 列窗口**（同一 lane、同一 KV 执行行、同一 state slot、同一代码路径）与 4 列窗口相比 `max_logit_diff=1.43164`（逐列 0.6875 / 0.9375 / 1.1875 / 1.43164，argmax 全对）；两 lane 批（T=8）对单 lane 单跑（T=4）`max_logit_diff=2.03125`、`argmax_mismatches=1`（唯一翻转列 lane1 col2：solo=4400 / batch=3049，该列 top1-top2 margin 0.3125→0.125，属近邻并列）。
  - 证据④ GDN record 平面定位：四个 record 平面（conv/key/value/gate）在 compact layer 0 **逐位相同**，自 layer 1 起全元素发散（conv `differing=937362 max=1.625 first_layer=1`、key `max=0.1875`、value `max=0.158203`、gate `max=3.93931`；conv 平面 `mean_abs=0.874619`，首差异元素 solo=-0.208984 / batch=-0.214844）⇒ 发散发生在 global layer 0 的 GDN 尾部（recurrence 输出 → 输出投影 → 残差 → FFN → layer 1 pre-mixer norm）。层型序 64 = 16×[linear,linear,linear,full]，compact layer 0/1 = global layer 0/1，中间无 attention。
  - 结论：批 verify window **可用**（绑定正确、batch=1 逐位等价 C=1、无耦合），但「批窗口 vs 单跑窗口」不可能逐位相等；该 M 敏感性是**既存**属性（单 lane 8 列 vs 4 列即复现），修它等于改 C=1 的 verify 窗口数值，超出 P2.1a 范围。同类既存有界差异先例（同一次运行输出）：`chunking 300 -> 256+44: top5_gap=0.375 max_logit_diff=1.07031 (bounded)`、`batched prefill vs sequential walk (A4 vs A16): 1.53125`、`batched prefill route parity T=7: 0.96875`；而 `128+172` / `64+236` 分块逐位一致。
  - 具体核未定位：需在 `run_layers_tp2` 加 env-gated 逐层 residual dump 才能把 M 敏感性钉到某个 op（候选：GDN scan 分块、GEMM 的 M tiling）。已放弃追查（既存属性且修它会改 C=1）。
  - ⇒ **验收判据改为（待用户确认）**：硬判据 = ①复制 lane `lane_gap==0`；②batch=1 入口 vs 历史 C=1 路由逐位；③每 lane 每列 argmax 相等，或差值落在该列并行前的 top1−top2 margin 之内（近邻并列豁免）；④每 lane 每列 logits 漂移 ≤ 2.5（有界）。不再要求 batch>1 与单跑逐位相等。
  - 布局勘误：本节 `:401` 的「index = col*batch + lane」与实测不符；`attn_mix` 的 `cache_positions.view({width, batch})`（lane 最慢）与 host 的 lane-major 排布都是 `t = column + width*lane`，`gdn_replay_records` 的 dim0 最快 / lane 为最慢维亦然（复制 lane 逐位一致即为其正确性证据）。
  - 临时诊断代码（`tests/models/qwen3_5/test_tp2_forward.cpp` 的 `compare_record_plane`、`download_float_plane`、`top_two_margin`、`run_window_solo_wide`、`build_wide_records`/`record_action_wide`/`wide_arena0..1`/`wide_records0..1`）待判据定稿后删除；当前该测试仍以 `FAIL: the batched verify window diverged from its per-lane solo window`（exit 1）结束。

#### 12.1.1 record/fold 的既有批能力与第一步落地
- record 平面布局已经支持批：`plan_gdn_replay_records`（`src/core/gdn_replay_records.cpp:96-113`）给出 `conv [conv_channels, width, outer]`、`key/value [dim, heads, width, outer]`、`gate [2, heads, width, outer]`，`GdnReplayRecords::layer(layer, rows)`（`:125-140`）切出 `[..., width, rows]`；dim0 最快 ⇒ 一个 lane 的窗口是连续 `width` 列、lane 是最慢维，与 `attn_mix` 的 `[width, batch]` 绑定一致（index = lane*width + column）。因此 rows=batch 时 `records.conv.bytes() == qkv.bytes()` 自然成立，`text.cpp:1766` 的整块 memcpy 无需改形。
- fold 本来就是逐行的：`ops::GdnReplayFoldRow{source_state_slot, destination_state_slot, commit_columns}` 的 span（DFlash2 `tp2_generation_core.cpp:4484-4492`、MTP `:4533-4540`），多 lane 只是传 B 行、每行自己的 slot 与 commit。fold 前恢复的是**整池**快照（`:4478-4483`、`:4528-4532`），所以 verify 期间是否推进 conv/recurrent 状态无关紧要 ⇒ 每 lane 的 source/destination slot 可以直接写成自己的 slot（in-place）。
- 分片 record 分支给 recurrent record op 传 `Tensor{}` 作为 valid（`text.cpp:1864`）⇒ 尾部未提交列由 fold 的 `commit_columns` 排除；valid 只影响 attention 的 KV 追加。P2.1 用「批内各 lane valid 计数的最小值」作单一标量（沿用今天 `[1]` tensor 形状），避免改 attention op。
- **已落地（第一步）**：`forward_tp2_window_batch`（声明 `text.h`，实现 `text.cpp`，`forward_tp2_window` 改为 batch=1 薄封装）。batch>1 时按 lane 展开成 `[width*batch]` 的 kv row / state slot 向量，并绑定 `active_sequence_batch_=batch`、`active_sequence_width_=width`；batch==1 保持标量绑定（值 0），逐位不变。
- **第二步（已落地，2026-09-30）**：① 分片 record 的 per-lane 化 —— `text.cpp:1611-1617` 放开 `record_rows != 1` 守卫（`record_rows = active_sequence_batch_ > 0 ? active_sequence_batch_ : 1`）、`:1724-1768` recording 的 conv 改 `ops::causal_conv1d_silu_snapshot`（整池 D2D stage + per-lane `snapshot_base_slots`，`per_lane_conv = shard && (Verify || recording) && batch > 1`）、`:1837-1865` 分片递推 record 的 `width = T / record_rows` 且 `initial_slots` 由 `active_linear_state_source_slots_`（per-lane I32 `[rows]`）填充；② 执行层入口 `forward_tp2_window_batch`（`text.h`/`text.cpp`，batch>1 展开 `[width*batch]` 绑定，`forward_tp2_window` 为 batch=1 薄封装）；③ 引擎侧 —— `tp2_generation_core.h/.cpp` 的 `Shard` 改为 `record_arena` + `std::vector<ReplayViews> replay_views`（`replay_views[batch-1]`，访问器 `records_for(batch)` / `fold_for(batch)`），`build_shard` 对 batch=1..lanes 各规划一份 record 布局（`layer(layer,rows)` 切片只有 `rows == record_capacity` 才连续），新增 `run_verify_window_batch(ids, positions, kv_table_rows, state_slots, width, batch, first_position, ...)` 与 `fold_verify_window(state_slots, commit_columns, batch)`；单 lane eager 路收敛到同一 batch 入口（batch=1）；DFlash2/MTP 的 fold 调用点改为 `fold_verify_window(slots={0}, columns={committed}, 1)`。

### 12.2 P2.1b MTP batch chain
**状态：已落地（2026-09-30，双并发与四并发活体 smoke 均通过）**
- 实现（`src/runtime/engine/tp2_generation_core.cpp` 的 `execute_spec_batch`，MTP 分支）：`mtp_pack`（7×columns host 打包 anchors/bases/extents/lengths/rows/valid/selectors）一次 H2D → 逐 step `ctx_a.mtp_forward_decode_batch(ids, in, positions, positions, mtp_valid, mtp_rows, envelope, out)` + `ctx_a.mtp_propose_batch(out.view({hidden, columns}), chain_logits, step_drafts)`；step 0 用 `mtp_anchors`/`chain_anchor`（从 `shard_a_.mtp_anchor_hidden` 的第 `lane.slot` 列 gather），step≥1 用上一步草稿的双缓冲 hidden（chain_a/chain_b 交替）。anchor/window 语义与单 lane 权威链（`mtp_chain_body`）逐项相同：每步 cache position = `lane.position - 1 + step`，envelope visible = `max_position + step`。
- 三个真 bug（前两个是 dim-0 最密语义）：
  1. `Tensor::is_contiguous`（`src/core/tensor.cpp:85-97`）要求 `nb[i] == elem*prod(ne[0..i-1])`（**dim 0 最密**），`view()` 只接受连续张量。`Tensor{7,columns}` 沿 dim 0 切 `slice(0,i,1).view({columns})` 在 columns>1 时抛 `view requires a contiguous tensor` ⇒ 改为 rank-1 `{7*columns}` + `slice(0, i*columns, columns)`。
  2. `mtp_forward_stem`（`src/models/qwen3_5/execution/text.cpp:538-540`）对 ids 做 `ids.view({T})`，而 `mtp_drafts.slice(0, step-1, 1)` 的 `{1,columns}` 切片不连续 ⇒ 新增 rank-1 `step_ids` 暂存，用 `cudaMemcpy2DAsync`（每 K 个 int32 取一个）gather。
  3. **草稿镜像的 request-major 布局**：`mtp_drafts = mtp_drafts_flat.view({K, columns})` 的元素 (i,b) 在 `flat[i + K*b]`；ops 按 request-major 索引（`src/ops/kernel/speculative_round.cuh:20-39` 写 `drafts[row*k + j-1]`、`:109,127` 比 `drafts[row*k + lane]`）⇒ 每步落盘必须 `cudaMemcpy2DAsync(mtp_drafts_flat.data + step*4, K*4, step_drafts, 4, 4, columns, D2D)`，不能按下标连续的 step-major 写。columns==1 时两种布局重合（故 batch=1 一直正常）；columns≥2 曾把接受率从 solo 的 72% 打到 27%/16%，而提交前缀仍正确（column 0 是 anchor）⇒ 表现为「输出对、性能反降」。
- 引擎侧另外三处：① MTP K/V 池按 lane 切分（`build_shard`：`logical_pages/lanes` 页写进每 lane 自己的 `KVExecutionRowLease` 并 `publish(row.handle(), 0, handles, stream)`，`mtp_lane_handles`/`mtp_rows`/`mtp_views`；多出的 `mtp_physical_pages` 仍是死容量，无需改动）；② `lane_context_window()` 修掉 `submit` 的上下文界 off-by-one（多 lane 时每 lane 的上下文界是 `lane_token_capacity_`，再留 `mtp_drafts_+2` 的尾部余量，否则最后一个 lane 会写到下一个 lane 的物理页 0）；③ `auto* round = shard_a_.dflash_round.get()` + `if (tail_count > 0 && dflash2_enabled_)` —— MTP 路由下 `dflash_round == nullptr`，原先的空引用在「预算耗尽且仍有尾段列」时以 0xC0000005 崩溃。
- 验收（`_temp/p21b_mtp_smoke.ps1` B=2、`_temp/p21b_mtp_batch4.ps1` B=4；`--devices 0,1 --max-context 131072 --kv-dtype int8 --spec mtp --draft-tokens 5`，temperature 0 / max_tokens 48）：
  - B=2：solo wall 721/722 ms，CONC2 wall 976/972/969 ms ⇒ **1.48×**，三次全 `same=True/True`；
  - B=4：solo wall 725/730/374/731（serial 2560 ms），CONC4 wall 1133/1130/1132 ms ⇒ **2.259/2.265/2.261×**，三次全 `same=4/4`；
  - `[tp2-time] spec-batch` 每轮 = mtp 19.6 + verify 39.2（B=1）→ 11.1 + 24.2（B=2）→ 6.9 + 16.1（B=4）⇒ 每 lane 每轮成本 0.61×（B=2）/ 0.39×（B=4）；
  - 内存（B=4）：`[mem] shard 0 capacity 131072 lanes 4 ... state 1174.5 ... free 1058.0 of 16310.6 MiB`、`KV pages 2048 over 4 lanes = 512 pages (32768 tokens) per lane`、`MTP KV pages 2048 over 4 lanes = 512 pages per lane`。
- 与 DFlash2 对照（同机同件）：B=2 DFlash2 1.50× / MTP 1.48×，B=4 DFlash2 2.07× / MTP 2.26×（MTP 的 draft 段在批下更便宜）。**两者都低于 Phase 1 的 1.6×/2.5× 门槛（§9 的 plain 路线验收，1.76×/2.80×）**，因为那套门槛按无投机的 plain 路线标的：投机路线单请求已把权重流按 K+1 列摊薄，批处理只能再摊 lane 维。剩余差距的主因是 **prefill 仍逐 lane 串行**（约 120 ms/972 ms 在 B=2、200 ms/1133 ms 在 B=4）；若 prefill 也批量，B=2→1.68×、B=4→2.72×。列为后续项（不在 P2.1 范围）。
- 已知有界副作用（沿用 P2.1a 判据 A）：非最长 lane 的链 envelope 被抬高到批内 `max_position + step`（batch 级单值，不能逐 lane），该 lane 的 MTP 头数值与单跑略有差异、接受率下降（B=2/B=4 live 中 lane1 恒为 `34/62`，solo 为 `36/52`；lane0/lane3 与 solo 完全相同），**输出 token 仍逐字相同**（**注：2026-10-01 口径收窄——该结论只在批内 prompt 位置差 ≤2 时成立；位置差 ≥4 且 B≥3 时该漂移可能翻转一个低 gap token，见 §9 P2.2d 后的 `[!]` 与 §12.5 末条**），且三次运行逐位可复现（不是竞态）。
- **前置修正（见 §12.4）**：本节列的「`mtp_chain_body` 换批量形态」只是批量投机 round 的第 1 段（draft 段）；真正的施工主体是新增 `execute_spec_batch`，因为现有投机 walk 是单请求单 lane 的流式循环。
- 现状 `tp2_generation_core.cpp:1291-1342` `mtp_chain_body`：`ar_hidden [hidden,1]`、`drafts [K]`、pins `[anchor, position, position+1, drafts[K]]`（`:1384-1388`）、`ctx.mtp_forward_batch(..., logits_column=0, ...)`（`:1326`）、`mtp_forward_ar_step` 循环（`:1328-1337`）、D2H drafts（`:1340`）。
- 单卡批量参考（已存在且已产品化）：`src/models/qwen3_5/program/speculative/mtp.cpp:70-212` `mtp_decode_batch_body`（batch 1..kMaximumConcurrency）、`:214` capture / `:221` `mtp_decode_batch`；用 `card.mtp_forward_decode_batch`（`text.h:380-384`：`ids/hidden/cache_positions/rope_positions/valid_columns/kv_table_rows/envelope → mtp_hidden`）与 `card.mtp_propose_batch`（`text.h:385`）。
- 改动要点：`mtp_chain_body` 换批量形态（`ar_hidden [hidden,B]`、`drafts [K,B]`、pins 的 anchor/position 变 per-lane 数组、harvest 走 `mtp_forward_decode_batch`+`mtp_propose_batch`）；`mtp_chain_host_`（`:507`）容量 ×B；调用点 `mtp_propose_window`（`:1377-1427`）与 `:4320`（`shard_a_.mtp_anchor_hidden`）需 per-lane anchor/hidden。
- 资源层已就位：`RoundStateSpec.batch_capacity = lanes`（`:979`）；`round_buffers.h` 的 `MtpDecodeStateLayout`（`:118`）/ `MtpDecodeState`（`:216`）/ `MtpDecodeIngress`（`:47`）/ `MtpDecodeEgress`（`:63`）已按 `batch_capacity` 校验与切分（`round_buffers.cpp:235`/`:310`，上限 kMaximumConcurrency）；MTP K/V 多 row 在 `build_shard` 的 `mtp_shard` 分支（`:589`/`:700`）。
- 采样已是批形状：`ops::sample`（`include/ninfer/ops/sampling.h:36-48`，`[physical_rows, B]` + `SamplingConfig[B]`）。

### 12.3 P2.1c DFlash2 batch
**状态：已落地（2026-09-30，双并发与四并发活体 smoke 均通过）**
- DFlash2 round 层（`src/models/qwen3_5/program/dflash_round.h` / `dflash_round.cpp`）：`plan_dflash2_round(..., std::uint32_t batch_capacity = 1)` 把容量写进 `DFlash2RoundSpec.batch_capacity`；ctor 按它张 `pending_features [F,L,B]`、`plan_cyclic_kv_cache(..., B)`、`RoundStateSpec.batch_capacity`、`continuation_hidden_ [hidden,B]`；`make_prefill_sink(ExecutionCore, std::int32_t lane = 0)`（lane 选择 draft ring slot）；`make_verify_sink(std::int32_t batch = 1)` 新增 `verify_lanes_` / `verify_valid_` 两个 slice（`scatter_bf16_batch` 要求 lanes/valid 恰好 B 项，而 frame 行数按容量张成）；新增 `append_pending_batch(execution, slots, starts, ends, batch)`（4 路 H2D + `ops::prepare_ragged_prefix` + `dflash_append_context`，全零批次早退）；`propose(..., batch)` 走 `dflash_propose_batch(context, batch, k, envelopes)`。
- 引擎层：新增 `void TP2GenerationCore::execute_spec_batch(const std::vector<std::shared_ptr<PendingRequest>>& batch)`（`tp2_generation_core.cpp:2223-2800`，`.h:347` 声明），复用 `execute_plain_batch` 的 LaneState / 流式 / 预算骨架，三段式 = ①批量 draft（ingress 填 B 行 + `round.propose`）②P2.1a 的 `run_verify_window_batch` ③per-lane accept + 一次 `fold_verify_window(slots, columns, B)`；`drive_lane_queue` 分派改为 `dflash2_enabled_ ? execute_spec_batch(batch) : execute_plain_batch(batch)`（`batchable = lanes_ > 1`，media/grammar 仍逐个 lane 串行）。
- 批次失败诊断（永久性改进）：`drive_lane_queue` 的 `catch (...)` 在 `session_invalidate_active()` 前 rethrow 取 `what()` 打印 `[tp2-lane] batch of %zu failed: %s`（此前批次失败只回 `HTTP 500 internal error`）。
- 放开多 lane：`include/ninfer/tp2_capacity.h:32-36` 只让「非 `None` 且非 `DFlash2`」坍缩到 1（DFlash2 得 `min(requested, 4)`）；`tp2_generation_core.cpp:343-361` ctor 守卫同步改为 `lanes_ > 1U && !dflash2_enabled_ && backend != None`；`model_instance.cpp:99-112` 注释与警告条件同步；`tests/test_engine_options.cpp:53-69`、`tests/test_serve_options.cpp:426-441` 期望已更新（DFlash2 得 2 / MTP 得 1）。D5 仍生效（`[mem] host checkpoint ring off: the 4-lane route would pin one state image per lane per slot (D5)`），`verify_graph_enabled_ = (env != "0") && lanes_ == 1U`（多 lane 一律 eager）。
- 施工中修的两个真 bug：① `ops::argmax` 的 `out` 必须是 rank-1（`ne[1..3]` 全 1）——`frame.target_argmax.slice(1, 0, columns)` 在 B>1 时是 `[width,B]`（B=1 的 `[width,1]` 恰好退化成 rank-1 才通过），改为 `Tensor target_argmax_flat = frame_target_argmax.view({width * columns})` 的扁平别名（与 accept 用的 3-D `[V,width,B]` 共享同一块内存，索引 `k + width*lane` 一致）；② pinned 窗口按**容量**跨度分配但退役 lane 后一轮只有 live 列，`frame_verify_ids` / `frame_verify_positions` / `frame_licensed` 三处 D2H 的拷贝大小改为 `window_live = width * columns`（`licensed_counts` 仍 `columns`）。
- 验收（`_temp/p21c_batch_smoke.ps1` B=2、`_temp/p21c_batch4.ps1` B=4；`--max-context 131072 --kv-dtype int8 --spec dflash2 --draft-tokens 5 --lm-head-draft`）：B=2 三次全 `same=True/True`；B=4 三次全 `same=4/4`（每个 lane 的 text 与同一请求单独跑逐字相同，`draft=39/39` / `20/20`）；B=4 聚合比 `2.068/2.071/2.068×`（serial 1524 ms vs 737 ms）；`[tp2-time] spec-batch` 的 verify/round = 38.2 ms(B=1) → 22.6(B=2) → 14.7(B=4)；`[mem] shard 0 capacity 131072 lanes 4 ... state 1174.5 ... free 726.0 of 16310.6 MiB`，KV 每 lane 512 pages = 32768 tokens。
- 测试回归（artifact 门禁，`--artifact D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer`）：`tp2_forward` exit 0（68.7 s）、`tp2_dflash_append` exit 0（48.6 s，`K=7 proposal deterministic fnv1a=0x2b86112ce3724e5f`）、`tp2_load` exit 0（42.8 s）；`tp2_dflash_solo` exit 1 —— **既有失败，非本次引入**（`docs/tp2-dual-5060ti-worklog.md:6549-6551` 已记录同一 artifact 上 `a_continued` 首采样 `[59399 475 327 363 …]` vs from-scratch `[365 4577 62 16 …]`），不要重复排查；该件上 `tp2_sessions` 见下。
- R7 标定：**K=5 / B=4 ⇒ T=(K+1)×B=24 正常跑通**（无几何或对齐拒绝），所以「T ≤ 16」是 NVFP4 A16 draft 路由的界，不是当前 Q4/Q8 draft 路由的硬界；但 B=4 把每 lane 上下文压到 32768 tokens（131072/4），R5 仍限制「大 context + C>2」。
- **前置修正（见 §12.4）**：同 §12.2；DFlash2 的优势是其 `DFlash2Round`/ingress 数组已按 `kMaximumConcurrency` 张成（`round_buffers.h:35/55/56/85/86`），故 draft 段只需把 ingress 填 B 行，但 verify/accept/fold/流式仍要 per-lane。
- 唯一硬编码：`src/models/qwen3_5/program/dflash_round.cpp:55` `spec.batch_capacity = 1;`。`plan_dflash2_round`（`dflash_round.h:66-69`）需加 `std::uint32_t batch`（默认 1），`dflash_round.h:58` 的默认值保持 1（单卡与 TP-2 默认不变）。
- round 组件已完全支持 batch：`dflash_round.cpp:222`（proposal workspace × batch_capacity）、`:226`（hidden frame `{hidden, batch_capacity}`）、`:354-358`（`batch_features`/`batch_lanes`/`batch_valid_columns`/`batch_size`）、`:418` `dflash_propose_batch(context, batch_capacity, k, envelopes)`；proposal 输出形状已是 `[K,B]`/`[candidate_k,K,B]`/`[K+1,B]`（`dflash_round.h:82-88`）。
- TP-2 接线点：`tp2_generation_core.cpp:817` `plan_dflash2_round(...)`、`:820-821` `DFlash2Round` 构造、`:979` `RoundStateSpec.batch_capacity = lanes`、`:4138` `ingress.dflash_kv_table_rows[0] = 0;`（数组形状已是 `std::array<..., kMaximumConcurrency>`，见 `round_buffers.h:35/55/56/85/86`）、`:4149` `round.propose(...)`、整段 `:4082-4160`。
- 上限 R7：T=(K+1)×B ≤ 16（NVFP4 A16 路由，`docs/tp2-dflash2-draft-nvfp4.md:112-113`）；现产 Q4/Q8 draft 路由另需 %4 对齐。⇒ DFlash2 建议 B≤2（K=5 → T=12）或 K=3/B=4（T=16），P2.1c 必须标定后才定档。

### 12.4 施工修正（2026-09-30 源码复核）：P2.1b/P2.1c 的真实主体是「批量投机 round」
- 结论：§12.2/§12.3 只写了「把 draft chain 换成批量形态」，但真正的前置是**投机 round 本身的 per-lane 化**。现有投机 walk 是「一个请求一条线」的流式循环，硬编码单 lane：
  - `src/runtime/engine/tp2_generation_core.cpp:4124-4619` 的 `while (!finished)` 主体持有全部单 lane 状态：`current`（anchor token）、`position`、`logical_pos_a`、`mtp_anchor`/`mtp_position`（`:4100-4101`）、`dflash_context_frontier_`、`sampling_a`、`request.budget`/`request.generated`/`request.output`（流式 prefix）、`tool_constraint`/`constraint_live()`/`tool_mask_columns`、`timing`（合成本请求的 round 时延）、`host_checkpoint_stride_` 的 checkpoint 写点（`:4631-4639`）、`abort_if_ar_stalled()`（`:4522`）。
  - DFlash2 分支（`:4149-4353`）把 ingress 写死在 lane 0：`ingress.anchors[0]`、`execution_frontiers[0]`、`context_frontiers[0]`、`proposal_extents[0]`、`proposal_valid_columns[0]`、`target_valid_columns[0]`、`text_kv_table_rows[0]`、`dflash_kv_table_rows[0]`、`active_lanes[0]`、`state_source_slots[0]`/`state_destination_slots[0]`、`sampling[0]`（`:4192-4211`）；`dflash_context_frontier_` 是单个成员（`:4157-4191`）。verify 用 host 窗口 `window_ids`（`:4231-4239`）与 `run_verify_window(window_ids, position, window_logits, window_hidden, &verify_sink, extent+1)`（`:4270-4272`）。
  - MTP 分支（`:4384-4519`）用 `mtp_propose_window(shard_a_, shard_a_.mtp_anchor_hidden, mtp_anchor, mtp_position-1, ws_a)`（`:4388-4390`）、host 拼单窗口（`:4395-4403`）、`run_verify_window(window_ids, mtp_position, ...)`（`:4448-4449`）、`mtp_anchor_hidden` 单个 `[hidden,1]`（`:4605-4611`）。
  - accept/licensing 也是单 lane：`ops::speculative_accept_greedy_drafts`（MTP，`:4483-4486`）与 `ops::speculative_accept_sparse_drafts`（DFlash2，`:4313-4318`）都按 `[width,1]` 形状跑；`licensed_counts` 是标量 `[1]`（`:4426`/`:4489`）。
- 设计（与 Phase 1 `execute_plain_batch` 同构，**不矢量改写 `execute_walk`**）：新增 `void TP2GenerationCore::execute_spec_batch(const std::vector<std::shared_ptr<PendingRequest>>& batch)`，复用 `execute_plain_batch`（`:1774-2110` 一带）的 `LaneState` 模式，逐 lane 持有 `current`/`position`/`mtp_anchor`/`mtp_position`/`logical_pos`/`sampling`/`budget`/`output`/`tool_constraint`/MTP anchor hidden 行/DFlash2 lane 行。`drive_lane_queue` 的 `batchable` 条件改为：plain → `execute_plain_batch`；`mtp_enabled_ || dflash2_enabled_` → `execute_spec_batch`；否则退回逐 lane `execute_lane`。
- 批量 round 三段式（每段整体跑一次，而不是每 lane 一次）：
  1. **draft 段**：B 条 lane 的 propose 合成一次批量调用（DFlash2：`round.ingress()` 填 B 行 + `round.propose(...)`；MTP：`mtp_chain_body` 批量形态）。这一段才是 §12.2/§12.3 的原始内容。
  2. **verify 段**：`run_verify_window_batch(window_ids, window_positions, kv_rows, state_slots, width, B, first_position, logits_columns /* [V,width*B] */, hidden_columns, sink, valid_columns)` —— **P2.1a 已落地**（`:1151-1249`）。
  3. **accept/fold 段**：**per-lane 循环**调用现有单 lane 核（logits 切成 `window_logits.slice(1, lane*width, width)` 保持 `[V,width]` 形状），fold 用 `fold_verify_window(slots, columns, B)` 一次（P2.1a 已落地）；每 lane 的 `committed` 从 `licensed_counts[lane]` 读回。
- 该设计把「批量核」限制在 draft 段与 verify 段（真正的算力所在），accept/流式/预算/checkpoint 仍是逐 lane 的既有代码路径 ⇒ 数值 oracle 继续用「单 lane 单跑 vs 批内该 lane」的既有判据（P2.1a 判据 A），风险集中在 draft 段与 verify 段的绑定。
- per-lane 退出/预算语义沿用 `execute_plain_batch` 的 retirement/budget 段（`:2000-2110`）：某 lane 结束后从批里摘除，剩余 lane 用更小的 batch 继续（record/fold 视图已按 batch 1..B 预建，P2.1a），不需要连续批处理。
- 缓存/会话：spec 路线的 `lanes_ > 1` 目前被构造函数拒绝（`:350`），P2.1b/P2.1c 落地后该守卫改为「spec + lanes>1 允许，但强制 D5（无 host checkpoint ring / 无 session catalog）且 graph 关闭（eager）」。`--spec mtp|dflash2 --max-concurrency > 1` 的「坍缩为 1 lane + 警告」届时取消（D3 已按此写入契约文档）。
- **已落地（2026-09-30）**：DFlash2 与 MTP 均已放开（见 §12.3/§12.2 状态行）；`tp2_generation_concurrency` 只对 `SpeculativeBackend::DFlash` 坍缩到 1 lane，MTP/DFlash2/None 得 `min(requested, 4)`。D5（无 host checkpoint ring / 无 session catalog）与 eager（`verify_graph_enabled_ = ... && lanes_ == 1U`）在多 lane 上一律生效。
### 12.5 P2.2 实现记录（`(bucket,B)` CUDA Graph + AR 预算 + `round_base` 常量）
**状态：已落地并验收（2026-10-01）。** 本节原先只写了与 P2.1 的交界；下面是完整实现与测量。

#### 图键与图家族
- `WindowGraph`（`tp2_generation_core.h:447-474`）增两个字段：`std::int32_t batch = 1;`（本批活跃 lane 数）与 `std::int32_t lane = -1;`（单列宽度捕获时烘进 kernel 的 state slot；设备绑定的批量条目为 −1）。`select_window_graph(std::vector<WindowGraph>&, visible_end, batch, lane)`（`.cpp:1171`）先过滤 `graph.batch != batch`，再在 `lane != -1` 时要求 `graph.lane == lane`，最后按 visible 区间选；`reusable_window_graph(..., batch, lane)`（`.cpp:1184-1193`）额外要求 `captured && round_base[0] >= shard_a_.round_base && round_base[1] >= shard_b_.round_base`（水位回退则重新捕获，重捕获复用同一 AR channel）。
- width==1 的条目**逐 lane 各捕获一份**（每 lane 的 state slot 是 kernel 参数，见 §12.1 的 `batch==1` 标量分支）；width≥2 的条目 `lane = -1`，slot 走设备向量。每条批量家族每桶条目数 = `2*lanes_-1`。
- 家族与桶数（`--max-context 131072`，`L = lanes_`）：`verify_graphs_` = 8 桶（MTP 用 `mtp_graph_profiles` = 12 桶；DFlash2 用 `dflash_graph_profiles(DFlash2, cap, drafts, 1)` = 6 桶，见 `planning/graph_profiles.cpp`）；`verify_batch_graphs_` = 桶数 × (2L−1)（`.cpp:480-548`）；`decode_batch_graphs_` = 8 × (2L−1)（`.cpp:564-598`）；`decode_graphs_` 仅 `lanes_ == 1` 时 8 条（lanes_>1 一律走 `decode_batch_graphs_`）；`mtp_chain_graphs_` = 12（`mtp_chain_body`，仅单 lane 走图，批量链是 eager 循环）。启动日志 `[tp2-graph] plain decode step: %s | batched: %s | verify step: %s | verify batch: %s | mtp chain: %s`（`.cpp:636-651`）。

#### AR channel 预算（D2）
- ctor（`tp2_generation_core.cpp:654-660`）在任何 capture 之前 `tp::DevicePair::reserve_ar_channels(verify_graphs_.size() + verify_batch_graphs_.size() + mtp_chain_graphs_.size() + decode_graphs_.size() + decode_batch_graphs_.size())`。`reserve_ar_channels`（`src/core/tp/device_pair.cu:482-528`）在 `ar_channels_` 非空时抛 `"tp allreduce: the rendezvous id space cannot grow once a channel exists"`，但此处尚无 channel，安全。
- 5 个 capture 点各自只取一个 channel 且被 `if (graph.ar_channel == tp::DevicePair::kNoArChannel)` 守护：`capture_verify_graph`（`:1273-1274`）、`capture_verify_batch_graph`（`:1320-1321`）、`capture_decode_graph`（`:1531-1532`）、`capture_decode_batch_graph`（`:1625-1626`）、`capture_mtp_chain_graph`（`:1764-1765`）。⇒ `create_ar_channel`（`device_pair.cu:530-547`）的触顶分支（`"tp allreduce: the captured graphs exceed the reserved rendezvous id channels"`）不可达。
- live 实测 `[mem] TP-2 rendezvous id channels reserved: N`（`_temp/p22d_*_run.log`）：MTP C=1 32、C=2 48（exact 臂）/ 84（graph 臂）、C=3 124、C=4 164；DFlash2 C=1 ≤16（低于打印阈值）、C=2 30 / 48、C=3 76、C=4 104 —— 全部等于上式在对应 `lanes_` 下的值，无 throw。

#### R2：`round_base` 启动常量与水位冻结
- `round_base = used()` 在批序言之后取（plain `:2277-2278`、spec `:2851-2852`），且序言里的分配全部改成对 `lanes_` 的**常量**大小（`counts`、`configs_dev`、五段 pinned staging），使 `round_base` 与逐轮活跃 lane 数无关。每轮用 `position_arena(ws_a, shard_a_.round_base, shard_a_.round_base)` 复位。
- 关键修复：MTP 批量链的 `Tensor positions = ws_a.alloc(DType::I32, {1, columns});` 原在 `for (std::int32_t step = 0; step < max_extent; ++step)` **之内**，`max_extent` 是每轮活跃 lane 的最大 extent ⇒ 分配次数逐轮变化 ⇒ verify 调用点 `used()` 漂移 ⇒ replay 报 `TP-2 batched verify CUDA Graph replay found a different workspace layout`。现提到循环外（`.cpp:3026-3030`），循环内只留 `cudaMemcpyAsync(positions.data, mtp_step_host.data(), mtp_stride*sizeof(std::int32_t), H2D, shard_a_.device.stream)`。
- 运行期护栏（调试用）：`run_verify_window_batch` / `run_plain_decode_step_batch` 在图 replay 前比较当前 `used()` 与捕获时的 `arena_begin`/`arena_bytes`，不一致即抛上面那条 layout 错误；`position_arena` 在 `target < floor` 时抛 `"TP-2 verify CUDA Graph was captured below this request's workspace"`。

#### 三条崩溃/别名修复（P2.2c 实测暴露）
1. **捕获栈地址（DFlash2 并发 `cudaErrorIllegalAddress`）**：`gdn_mix`（`text.cpp:1596-1604`）在 `recording && per_lane_conv` 时走 `:1751-1762`，原用**函数局部 host vector** 做 H2D staging（`std::vector<std::int32_t> base(lanes); ... copy_i32(base.data(), base_slots, s);`）。该 memcpy 被捕获成图节点后每次 replay 都读已失效的栈地址 ⇒ garbage slot ⇒ snapshot kernel 越界。改设备端生成：`ops::fill_i32_positions(base_slots, slots, window, s)`（`ops::fill_i32_positions` 增加 `stride` 参数，契约 `positions[i] = start + i*stride`，`T>0`/`stride>0`/`start + (T-1)*stride <= INT32_MAX`；`include/ninfer/ops/position.h`、`src/ops/kernel/position.cuh`、`src/ops/launcher/position.{h,cu}`、`src/ops/wrapper/position.cpp`（`stride < 1` 抛 `"fill_i32_positions: stride must be positive"`）、`tests/ops/test_position.cpp`）。单 lane 路线 `active_sequence_batch_ == 1` 使 `per_lane_conv` 为假、走 `state_.conv_slot(...)`（稳定设备指针），所以只有 batch≥2 的 verify window 会踩到；顺带删掉一次会同步的 pageable H2D。
2. **spec 路线 pinned verify staging 五段重叠**：`batch_window_section(section, stride) = base + section*stride` 写成 `ids=0/positions=1/valid=2/kv_rows=3/slots=4`，段 2/3/4 实际落在 `base + 2*lanes_`、`+3*lanes_`、`+4*lanes_`，与 `spec_ids` 区间重叠（width=6、`lanes_=2`：ids [0,12) vs valid [4,6)）⇒ host 写的 `valid/kv_rows/slots` 覆盖 `spec_ids`。表现为 MTP 2-lane `same=False/False`、`pn=4`、acceptance 0~10%（原本 72%）。改为 `[[nodiscard]] std::int32_t* batch_window_base() noexcept` + 显式 `spec_ids = base; spec_positions = spec_ids + id_span; spec_valid = spec_positions + id_span; spec_kv_rows = spec_valid + lanes_; spec_slots = spec_kv_rows + lanes_;`（`id_span = width*lanes_`）。plain 路线的 `batch_lane_host_section(section) = base + section*lanes_`（五段各 `lanes_`）无此问题。
3. 见上一小节的 `positions` 提升。

#### 验收测量（2026-10-01）
- 三臂 smoke `_temp/20261001-0500_p22c2b_smoke.ps1`（oracle = C=1；graph = C=2 缺省；exact = C=2 + `NINFER_TP2_VERIFY_BATCH_GRAPH=exact`；`--max-context 131072 --kv-dtype int8`、temperature 0、48 token）：MTP solo 与三次并发全 True/True，wall graph `884/856/851` vs exact `955/954/951`；DFlash2 全 True/True，wall graph `504/478/481` vs exact `550/545/545`，`draft=39/39`，无 `cudaErrorIllegalAddress`。
- 2-lane MTP solo 711 ms vs 1-lane 810 ms（无单请求回归）；plain 路线 C=4 单跑 `predicted_ms 1233` vs C=1 中位 1259（§9 P1.4d）⇒ P2.2b 消除了原先约 24% 的单请求代价。
- 存量测试：`test.ps1 -Filter 'position|tp2|tp_device|engine_options|serve_options'` 12/12（`tools/win_port/test.ps1`）；artifact 门禁 `ninfer_qwen3_5_tp2_forward_test.exe --artifact D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer` exit 0（自述「batched verify window lane-consistent under bounded route drift」）、`ninfer_qwen3_5_tp2_load_test.exe --artifact ...` exit 0。
- **口径收窄（重要）**：B≥3 且批内 prompt 位置差 ≥4 时，批量 round 共享的单个 attention envelope 会让非最长 lane 产生 ulp 级漂移，可能翻转一个低 gap token（实测约第 35/48 token），该 lane 输出与单跑不同；plain / MTP graph / MTP 全 eager 三条路径均可复现（故**不是 P2.2 引入**），同 prompt 的 DFlash2 C=3/C=4 未复现。属判据 A 的有界近邻并列，但 §12.2/§12.3 与 P1.8 的「逐字相同」只是漂移未触发时的观测。详见 §9 的 P2.2d 后那条 `[!]` 记录。

### 12.6 P2.3 设计（跨会话 retention 的 per-lane 化）
**状态：设计已定（2026-10-01），待施工。** 本节把 §5 E2 / §7 的 P2.3 一行展开成可施工的分阶段方案；实测规模**远大于**原先的一行估计（原估计隐含「把已有调用加一个 lane 参数」，实际是在批量路径里从零建一套 retention）。

#### 现状实测（为什么不是加一个 lane 参数）
- 15 个 retention 成员全部只在单 lane 的 `execute_walk`（`src/runtime/engine/tp2_generation_core.cpp:4360-5913`）里被引用：`host_checkpoint_*` 31 处、`reuse_source_` 7、`cached_prompt_tokens_` 7、`anchor_session_` 6、`sessions_` 6、`active_session_` 5、`cached_boundaries_` 5、`cached_state_valid_` 3、`block_anchor_*` 2、`live_state_valid_` 1。
- 两个批量执行器 `execute_plain_batch`（`tp2_generation_core.cpp:2166`）与 `execute_spec_batch`（`:2558`）对这些成员的引用数为 **0**（唯一例外是 `execute_spec_batch` 尾部提 catalog 的注释与 `dflash_context_frontier_` 1 处）⇒ 多 lane 路线**没有** retention 代码，P2.3 是在批量路径里新建。
- 12 个 `session_*` 函数全部以「整池 = 一个 device lineage」为前提：`session_drop`（`:3462`）只维护单个 `active_session_`；`session_evict_one`（`:3506`）靠 `device_resident` 跳过驻留 entry（隐含「最多一个驻留」）；`session_recall`（`:3954`）以 `active_session_`/`cached_state_valid_`/`shard_a_.host_checkpoints` 为全局；另有 `session_store_active`（`:3718`）、`session_restore`（`:3828`）、`session_capture_shared_state`（`:3894`）、`session_publish`（`:4163`）。
- 环形与镜像搬运按整池裸拷贝：`build_shard`（`:790`）`checkpoint.buffer = PinnedHostBuffer(shard.state_backing.bytes)`（含全部 lane）；`session_store_active:3742`、`session_restore:3851`、`session_capture_shared_state:3923` 都是 `state_backing.bytes` 的 `cudaMemcpyAsync`。`src/models/qwen3_5/program/dflash_round.cpp:282-309` 的 `copy_context_*` 硬编码 `ring_->slot_view(0)`。

#### 内存口径（决定镜像布局）
- 现状 lanes=1：`slots = host_state_slots(32) + divergence + block = 34`，每槽整池 73.4 MiB ⇒ 2495.6 MiB/shard（两卡约 5 GiB pinned）。
- 直接放开 D5 而不改布局：lanes=4 时 `34 × 293.6 = 9.98 GiB/shard`（约 20 GiB pinned）⇒ 不可接受。
- **决定（施工版，修正本小节初稿）：镜像按 lane 存，且环按 lane 分区。** 初稿写的是「每槽存 `lanes_` 张像」⇒ `slots × state_bytes` = 9.98 GiB/shard，与「验收」要求的「lanes=4 与 lanes=1 同量级」自相矛盾。施工采用：总槽数 `total = host_state_slots + divergence + block`（= 34）按 lane 均分，每 lane 得 `host_checkpoint_slots_per_lane_ = max(1, total / lanes_)` 个槽（lanes=1 → 34，lanes=2 → 17，lanes=4 → 8），**每槽只存一张紧凑单 lane 像**（`image_bytes = layers*(conv_layer_bytes + recurrent_layer_bytes)` ≈ 73.4 MiB）；lane 的槽区间 = `[lane*slots_per_lane, (lane+1)*slots_per_lane)`。总 pinned = `lanes_ * slots_per_lane * image_bytes`：lanes=1/2 → 2495.6 MiB/shard，lanes=4 → 2348.8 MiB/shard（同量级，满足验收）。
- 每 lane 的子布局与 lanes=1 整环同形：`fixed = divergence(1) + block(1)`；`tail = max(1, min(kReuseTailCheckpointCount, (per_lane - fixed)/2))`；`grid = per_lane - fixed - tail`；`stride = max(kReuseCheckpointStride, round_up(ceil(max_context/grid), 128))`。lanes=1 时逐值退化为原实现（total=34、per_lane=34、tail=2、grid=30、per_slot=`ceil(max_context/30)`）⇒ 单 lane 行为不变。
- 单 lane 像在整池里**不连续**：`LinearAttentionStatePool` 的 conv 形状 `{conv_channels, conv_width, slot_count}`、recurrent `{key_head_dim, value_head_dim, value_heads, slot_count}` 以 slot 为最外维、逐层分块（`src/core/linear_attention_state.cpp:96-99`）。搬运用现成的 `LinearAttentionStateSlotView`（`src/core/linear_attention_state.h:44-52`：`conv_layer0`/`recurrent_layer0`/`conv_layer_bytes`/`recurrent_layer_bytes`/`conv_layer_pitch_bytes`/`recurrent_layer_pitch_bytes`/`layers`）+ `cudaMemcpy2DAsync`：host 侧紧凑为 `[layers][conv_layer_bytes]` + `[layers][recurrent_layer_bytes]`，device 侧 stride 用 pitch。
- 落成三个私有静态 helper：`make_lane_state_geometry(const Shard&)`（用 `state->slot_view(0)` 算出 `conv_base`/`recurrent_base`/`conv_bytes`/`recurrent_bytes`/两个 pitch/`layers`/`image_bytes`）、`copy_lane_state(const LaneStateGeometry&, const void* device_base, std::int32_t lane, std::byte* host, cudaMemcpyKind, cudaStream_t)`（两次 `cudaMemcpy2DAsync`，host pitch 为紧凑字节数、device pitch 为层间 pitch；D2H 与 H2D 共用）、`zero_lane_state(...)`（两次 `cudaMemset2DAsync`，供 `ReuseSource::None` 用）。`LaneStateGeometry` 结构体声明在 `struct Shard` **之前**（`Shard` 以它作数据成员）。DFlash2 的 draft 像（`host_dflash*`）仍整池，留 Stage 2。

#### 目录（catalog）模型
- `SessionEntry.device_resident`（`tp2_generation_core.h:266`）→ `std::int32_t device_lane = -1`（−1 = host-resident）。不变式从「最多一个驻留」变为「每 lane 最多一个驻留」；`session_evict_one` 改为跳过驻留在任何 lane 上的 entry。
- `active_session_` → `std::array<std::size_t, kTp2GenerationMaxConcurrency> active_session_`（每 lane 一个）；`session_drop` 必须同时调整所有 lane 的索引。
- lane 与 session 的绑定：`drive_lane_queue` 按队列顺序把请求填进 lane，同一会话相邻轮次可能落到不同 lane ⇒ 复用只能走召回路径（`session_restore` 把状态写回该 lane 的 slot 与 KV 行），**不能假设 lane 粘性**。`ReuseSource::LiveState` 仅当「本 lane 的驻留 entry 正是该请求的会话」时可用。

#### 环形（checkpoint ring）
- `Shard::HostCheckpoint` 的 `position`/`dflash_frontier`/`prefill_id`/`valid` 改为 `std::array<..., kTp2GenerationMaxConcurrency>`（`buffer`/`dflash_buffer` 仍是单个 `PinnedHostBuffer`，每槽一张像）；`host_checkpoint_next`/`host_checkpoint_tail_next` 改为 per-lane 数组（`host_checkpoint_grid_slots` 仍是每 shard 一个常量）。`invalidate_host_checkpoints` 用 `checkpoint.valid.fill(false)` 保守清全部 lane。
- 写点：`execute_walk` 的 31 处 `host_checkpoint_*` 改为带 lane；批量路线新增等价写点（逐 lane，在其 prefill 结束时）。

#### 批量执行器接线（分阶段）
- 现状对 P2.3 有利：`execute_spec_batch` 已**逐 lane** prefill（`:2758` `for (std::uint32_t t0 = 0; !cancelled && t0 < lane.prompt_tokens;)`，draft sink 用 `active_lane_`），只有 decode 是批量；`execute_plain_batch` 同理。⇒ per-lane 复用深度天然可插：每 lane 在 prefill 前算 `reuse_depth`，chunk 循环从 `t0 = reuse_depth` 起步，`lane.result.reused_prompt_tokens`（`:2737` 现硬编码 0）随之填真值。
- **Stage 1（plain 路线）**：`session_recall(lane, prompt_tokens)` → `session_restore(entry, boundary, lane)` → per-lane 后缀 prefill → `session_store_active(lane)` / `session_publish(lane, ...)`。先只启用 host checkpoint 环（每 lane 自己的）；`state_snapshots`（整池、`kReuseSnapshotCount = 2`）的 per-lane 化留到 Stage 2。
- **Stage 2（spec 路线）**：MTP 同上；DFlash2 另加 draft 上下文的 per-lane 像（`host_dflash*`、`dflash_context_frontier_`、`dflash_round.cpp:282-309`）与 `state_snapshots` 的 per-lane 化。
- **Stage 3（守卫与文档）**：`lanes_ > 1` 的 D5 守卫（`tp2_generation_core.cpp:790` 一带的 `[mem] host checkpoint ring off ...` 与 session catalog 关闭）按路线逐步放开直至移除；`docs/serving.md` 的 D5 段与 §5 E2/§7 同步改写。

#### 施工进度（2026-10-01）
- **Stage 1a（per-lane retention 状态）已完成**：15 个 retention 成员抽成类作用域 `struct RetentionState`（`cached_prompt_tokens`/`cached_boundaries`/`cached_state_valid`/`active_session`/`anchor_session`/`live_state_valid`/`block_anchor_position`/`block_anchor_prefill_id`/`reuse_source`/`host_checkpoint_live_id`/`dflash_context_frontier`/`mtp_previous_draft`/`mtp_have_previous`）+ `std::array<RetentionState, kTp2GenerationMaxConcurrency> lane_retention_` + `retention(lane)`（含 const 重载）；`enum class ReuseSource` 提到类作用域；`session_*` 系列与 `adopt_generated_turn`/`reuse_path`/`execute_walk` 全部带 lane 参数；新增 `session_invalidate_all()`（批量失败与 AR stall 时清全部 lane）。头文件需 `#include "ninfer/tp2_capacity.h"`。
- **Stage 1b（镜像 helper + 环分区）已完成**：见上两小节。
- **Stage 1c（目录）已完成**：`SessionEntry.device_resident` → `device_lane`；`mark_device_resident` → `mark_device_lane(sessions, index, lane)`（先清掉其它 entry 上 `device_lane == lane` 的）；`session_evict_one` 跳过 `device_lane >= 0`；`session_store_active` 入口判据改为 `entry.device_lane != lane`；`session_publish` 的 `extends_resident` 判据改为 `device_lane == lane`。
- 验证：`build.ps1 -Jobs 16` exit 0（27 s）；`test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` **11/11 通过**（67.51 s）；`tp2_forward`/`tp2_load` artifact exit 0；`tp2_sessions`（单 lane retention 回归基线）见「验证计划」。
- **Stage 1d（批量路线接线）已完成**：`execute_plain_batch` 与 `execute_spec_batch` 逐 lane 跑 `session_recall`/复用扫描/`begin_gdn_state`/`publish`，整池 `cudaMemsetAsync` 删除（改为逐 lane `zero_lane_state`），`finalize` 加 `[this]` + 发布前置块（`LaneState.prefilled`）。
- **Stage 1e（守卫放开）已完成**：ctor `:383` 与 `:422` 的 `lanes_ == 1` 守卫删除（环与目录在任意 lane 数下都活），改为注释说明。
- **Stage 2a（helper 抽取）已完成**：新增私有 `struct LaneReuse { tokens; slot; next_host_checkpoint; }` 与五个 helper `scan_lane_reuse`/`restore_lane_gdn`/`restore_lane_dflash`/`publish_lane_prefill`/`invalidate_lane_prefill`，从 `execute_plain_batch` 的内联块与 `execute_walk` 的 `begin_dflash_state` 逐行提取。
- **Stage 2b（spec 路线接线）已完成**：`execute_spec_batch` 逐 lane 接入 recall/扫描/`restore_lane_gdn`/`restore_lane_dflash(…, false)`/chunk 级 Grid+Tail checkpoint/取消失效/`computed_prefill_tokens_ -= reuse`/`publish_lane_prefill`；`draft_frontier` 向量删除，全部改用 `retention(slot).dflash_context_frontier`。
- 验证（Stage 2b）：`build.ps1 -Jobs 16` exit 0（26-27 s）；`test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` **11/11 通过**（68.46 s）；artifact `tp2_forward`/`tp2_load` exit 0，`tp2_sessions` 仅剩既有已知失败（shared-system-prompt 首个采样分歧，`PLAN.md:100`）。活体验证见下条。
- **Stage 2c（`execute_walk` 的 per-lane 收尾）已完成**：串行回退（整批含 media 或 grammar）在 lanes>1 时以 `lane>=1` 跑 `execute_walk`，三处收口。（1）prefix-reuse 快照原 `snapshot_state` 写**整张** plane：保留给 `kRoundScratchSlot`（fold 整池还原），新增 `snapshot_lane_state`（`copy_lane_state(…, lane, state_snapshots[slot].data, D2D)`）替换四处 prefix 快照（pre-loop rewind slot≥1、取消分支 slot 0、chunk 循环 rewind slot≥1、prompt-end slot 0）。否则一个 lane 会把另一个 lane 的 plane-0 slice 改成自己**当前**池状态，而对方的 `cached_boundaries` 仍指向更浅的边界，DeviceSnapshot 复用就会还原到过深的状态（活体已证 `src=device` 是真路径）。（2）DFlash2/MTP fold 的 `fold_slots[1] = {0}` 改为 `{static_cast<std::int32_t>(lane)}`——`fold_verify_window` 把该 slot 当作 `source_state_slot`/`destination_state_slot`，写死 0 会污染 lane 0 且完全不 fold lane 1 的已提交列。（3）`execute_walk` 末尾的 `if (lanes_ == 1)` 守卫删除：lane 自己的 `cached_prompt_tokens`/`cached_boundaries`/`cached_state_valid` 必须与自己的 plane slice 同步发布，否则同一 lane 的边界与镜像会失配（这也消掉了 `:2131` 与 `:5810` 两条 D5 残留注释）。lanes=1 下三处均字节等价。
- 验证（Stage 2c）：`build.ps1 -Jobs 16` exit 0（25 s）。
- 活体四探针（`_temp/20261001-1100_p23_stage2b_routes.ps1`，作业 pwsh-389，四者全部 `PROBE_EXIT=0` 且 `--- errors ---` 为空）：**plain-pair** `src=host|live|device` 三源全中（`reuse=512 slot=8 src=host`、`reuse=874 slot=0 src=live`、`reuse=916 slot=0 src=device`）；**mtp-pair** 响应 `cache_n=874/512/874/913`，轨迹 `src=host|live`；**dflash2-pair** `cache_n=874/873/916/512`，轨迹 `src=device|device|host|live`；**plain-pair-ring**（`--max-private-continuations 1` ⇒ `session_capacity_=0`，仅靠 checkpoint ring）`cache_n=2806/2560/2560/2805`，轨迹 `src=device|host|host|device`。⇒ 批量 plain / MTP / DFlash2 三路线与 ring-only 路线的 per-lane 复用均已实证。

#### 验证计划
- 回归：`tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'`；artifact 门禁 `ninfer_qwen3_5_tp2_forward_test.exe` / `tp2_load` / `tp2_sessions`。`tp2_sessions` 需扩为多 lane：同一系统提示的两个会话在 C=2 下第二轮必须命中复用（`reused_prompt_tokens > 0`），且逐 lane 与单跑 oracle 在判据 A 下一致。
- 活体：`_temp` 探针扩为「C=2 两会话交替两轮」，检查 `[tp2-session]` 轨迹出现 recall；再跑 P2.2 三臂 smoke 确认无回归。
- 内存：`[mem] host checkpoint ring` 在 lanes=4 时必须与 lanes=1 同量级（约 2495.6 MiB/shard），而不是 9.98 GiB。

#### 风险
- **R-a（索引失效）**：`session_recall` 的边界选择依赖本 lane 的 `cached_prompt_tokens_`；`session_drop` 会平移 `sessions_`，所有「先取索引、后 erase」的顺序必须逐处复核（现有注释 `:4057-4058` 已强调，per-lane 后更易踩）。
- **R-b（部分失败）**：批量路线的 device 状态是一个池的多个 slot，`session_restore` per-lane 后只能写本 lane 的 slot ⇒ 某 lane 召回失败而其他 lane 已 prefill 时，按 D4 整批失败，不做部分回滚。
- **R-c（规模）**：这是一次约 1000–2500 行的重构，集中在 `src/runtime/engine/tp2_generation_core.{h,cpp}`；单 lane 行为（`tp2_sessions` 现有用例）是回归基线，必须先保证零变化。

### 12.7 anchor 网格对齐修复与 `tp2_sessions` plain 门禁现状（2026-10-01）
**性质：既有缺陷（Round 19 漏修点），非 P2.3 引入。**

#### 症状与定位
- `check_unaligned_dialogue`（本轮新增用例）在 plain 路线确定性失败：`a conversation whose entry kept no host slabs diverged from the oracle on its first sample`。
- 定位（作业 pwsh-406，`NINFER_TP2_TIMING=1`，日志 `_temp/p23_s2c_sessions_plain_chunks.log`）：engine B turn3 `prompt=88` 为 `reuse=0 suffix=88 chunks=2 host_writes=4`，oracle 同一 prompt 为 `reuse=0 suffix=88 chunks=1 host_writes=0`；两侧 `[tp2-reuse]` 字段逐字段相同 ⇒ 分歧来自 **chunk 切分**，不是状态残留。
- 根因：`execute_walk` 的 divergence anchor 把 chunk 截断在 `anchor_position = anchor_prefix - kReuseDivergenceMargin(8)`（engine B turn3 的 `[tp2-session] anchor entry=1 position=56` ⇒ `64-8=56`），而 `anchor_prefix` 由引擎历史（catalog entry 的 `host_shared_state`）决定 ⇒ 同一 prompt 的 chunk 计划随历史漂移，末列 logits 随之变化。
- 这正是 `docs/tp2-dual-5060ti-worklog.md:3444-3467` Round 19 已定论的缺陷，当时**只对 `snapshot_at` 修过**（`src/runtime/engine/tp2_generation_core.cpp:5441-5463` 的注释即那段修复：rewind 快照只由 `reuse`/`prompt_tokens`/`prefill_chunk` 决定），divergence anchor 是唯一漏网的截断点；其余截断点（`snapshot_at`、`next_host_checkpoint`、`block_position`）本就落在固定网格上。

#### 修复
- `src/runtime/engine/tp2_generation_core.cpp:5507-5520`：`anchor_target = anchor_prefix > kReuseDivergenceMargin ? anchor_prefix - kReuseDivergenceMargin : anchor_prefix;` 后，`anchor_position` 向下对齐到 `reuse + ((anchor_position - reuse) / prefill_chunk) * prefill_chunk`（仅当 `anchor_position > reuse`）。`anchor_divergence` 仍用对齐后的值。副作用：短 prompt 的 anchor 落到 `reuse` 时不再产生 divergence anchor，但所有截断点都在固定网格上 ⇒ 同一 prompt 的 chunk 计划可复现。
- 验证（作业 pwsh-408）：`build.ps1 -Jobs 16` exit 0；engine B turn3 变为 `chunks=1 host_writes=0`（与 oracle 一致）；`TP-2 unaligned dialogue (unaligned dialogue (plain)) passed`、`TP-2 replayed answer (replayed answer (plain)) passed`。
- 对 `run_scenario` 前三次调用逐调用证明为 **no-op**：opening(64) 与 other_a(48) 的 `shared_prefix=0` ⇒ `anchor_position=0` ⇒ `anchor_divergence=false`；a_second(88, `reuse=71`) 的 `anchor_target=72-8=64`，`64 > 71` 为假 ⇒ 同样 false。

#### 剩余失败：`a recalled conversation`
- 失败点 `tests/models/qwen3_5/test_tp2_sessions.cpp:483-487` 的 `compare_recall(label, "a recalled conversation", recalled_frontier=71, a_second, continued_answer)`（`recalled_frontier = 64+8-1 = 71`，非网格 ⇒ 只比较首采样）：`got [220 220 17 15 17 18 95859 15] expected [365 10715 277 198 8442 13516 1544 36133]`。
- 两侧形状：engine `a_second` `reuse=71 suffix=17 chunks=1 src=live`；oracle `continued` `reuse=64 suffix=24 chunks=1 src=device`。reuse 深度天然不同（recall 路径 vs 引擎自身 lineage）⇒ 末 chunk 宽度 17 vs 24 ⇒ 同属 Round 19 类「末列 logits 随 chunk 宽度漂移」，**Round 19 的修复覆盖不到 recall-vs-oracle**（它只能让「同一 reuse 深度」的计划确定化）。
- 旧基线对比：`profiles/sessions-plain-trace.log.err`（mtime 2026-09-29 15:53:26，早于 commit 47b82bc0）中 `a_second` 的 `[tp2-session]`/`[tp2-reuse]` 三行与当前**逐字节相同**（`entry 0 tokens=72 frontier=71 kv_end=71 prompt_end=64 shared_end=0 shared=72 reach=71 via=frontier resident=0`、`recall frontier=71 resident_depth=0 tokens=72 entries=2`、`-> reuse=71 slot=0 src=live`），且该次运行推进到更后面的 `a conversation behind a shared system prompt` 才失败 ⇒ 那次构建里 `a recalled conversation` 是通过的。
- 判定实验：作业 pwsh-419（脚本 `_temp/20261001-1200_head_baseline.ps1`）`git stash push -m p23_head_baseline -- src include tests` 切到 HEAD（pre-TP-2，d1a9b57b）后全量重建，跑同一份 plain 门禁（日志 `_temp/head_baseline_sessions_plain.log`），最后 `git stash pop` 还原。结果待填：HEAD 也失败 ⇒ 既有 test 脆弱性（按判据 A 记录或放宽该断言）；HEAD 通过 ⇒ TP-2 引入的数值回归，需在 `tp2_generation_core.cpp` 内二分。

#### 引擎侧改动审计（lane=1 逐字节等价性，用于收窄嫌疑）
- `src/models/qwen3_5/execution/text.cpp`：`gdn_mix` 的新分支全部由 `shard_config_ != nullptr && (Phase::Verify || recording) && active_sequence_batch_ > 1`（per-lane conv）或 `Phase::Verify && active_sequence_batch_ > 1` 守卫；`const std::int32_t width = T / record_rows;` 而 `record_rows = active_sequence_batch_ > 0 ? active_sequence_batch_ : 1`（lane=1 ⇒ 1）⇒ 走原 `ops::causal_conv1d_silu_split` 与 `width = T` 路径。`forward_tp2_prefill`/`forward_tp2_decode_window` 只加 `std::int32_t lane` 形参，并把硬编码 0 换成 `lane`（`ops::set_i32_scalar(b.kv_table_rows/state_source/state_destination, lane, ...)`、`fill_i32_positions(b.positions, first_position, 1, stream)`）；lane=0 时逐字节不变。`forward_tp2_window_batch`/`forward_tp2_decode_window_batch` 是新函数，lane=1 不走。
- `src/ops/{kernel,launcher,wrapper}/position.*`：HEAD 为 `fill_i32_positions(Tensor&, std::int32_t start, cudaStream_t)`（`positions[i] = start + i`），现为 `fill_i32_positions(Tensor&, std::int32_t start, std::int32_t stride, cudaStream_t)`（`positions[i] = start + i*stride`）；所有调用点传 stride=1。wrapper 新增 `stride < 1` ⇒ `"fill_i32_positions: stride must be positive"` 与溢出 ⇒ `"fill_i32_positions: range is outside non-negative I32"`。
- `src/models/qwen3_5/program/dflash_round.{cpp,h}` 只服务 MTP/DFlash2；`src/runtime/engine/model_instance.{cpp,h}` 只改 `normalize_engine_options`（`tp2_generation_concurrency` 归一化 + `[tp2-lane] --max-concurrency %u collapses to 1 lane` 提示）。
- ⇒ lane=1 下唯一可能带数值变化的文件是 `src/runtime/engine/tp2_generation_core.cpp`（+3575 行）。



### 12.8 Stage 2c 快照方向 bug（`copy_lane_state` D2D 写反）与 `a recalled conversation` 的修复（2026-10-01）
**性质：P2.3 Stage 2c 引入的回归。**（本节推翻 §12.7「剩余失败」一节的诊断：那次失败不是 Round 19 类 chunk 宽度漂移，而是快照方向写反导致的 GDN 状态污染。）

#### 症状
- plain 门禁（单 lane）`a recalled conversation` 确定性失败：`FAIL (plain): a recalled conversation diverged from the oracle on its first sample: got [220 220 17 15 17 18 95859 15] expected [365 10715 277 198 8442 13516 1544 36133]`（`tests/models/qwen3_5/test_tp2_sessions.cpp:483-487`）。
- HEAD 基线（pwsh-419，`git stash push -- src include tests` 切到 d1a9b57b 后全量重建）**通过**该用例，只失败在更后面的 `a conversation behind a shared system prompt` ⇒ 是 TP-2 引入的数值回归。
- 临时 logits 探针（pwsh-442）显示这是近邻并列：engine `a_second` `first=220`，`220=5.78125 271=5.71875 365=5.71875 62=5.65625 … gap=0.06250`；同一几何的对照（engine 与 oracle 各自的 `opening`，`prompt=64 reuse=0`）**逐位相同** ⇒ 前向本身无结构差异，差异来自两条浮点谱系。

#### 根因（逐行读代码）
- `copy_lane_state(geometry, device_base, lane, other, kind, stream)`（`src/runtime/engine/tp2_generation_core.cpp:3857-3894`）三个分支对第一个 device 指针的语义不一致：D2H/H2D 把 `device_base` 当**整池**（D2H 当源、H2D 当目的地），而 D2D 分支 `cudaMemcpy2DAsync(device_conv /*=device_base 侧*/, conv_pitch, source /*=other*/, conv_pitch, …)` 把 `device_base` 当**目的地**、`other` 当源。
- 两个「快照」调用点按「pool 是源、plane 是目的地」的习惯传参：`:5391-5396 snapshot_lane_state` 与 `:2361-2366 publish_lane_prefill` 都传 `(shard.state_backing.data, lane, shard.state_snapshots[slot].data, D2D)`。在 D2D 分支的实际语义下它们执行的是**反向**操作——把 plane 读回 live pool。
- 于是 `state_snapshots[0]`/`[1]` **从未被正向写入**：`:774-779` 只 `cudaMemset(shard.state_backing.data, 0, state_bytes)`，plane 是 arena 的原始设备内存。每次 prefix 快照都用这段原始内存**覆盖 live pool 的 GDN 状态**，其后的 decode 循环、发布的 session 状态、host checkpoint 全部建立在被破坏的状态上。
- 为什么既有 got-vs-oracle 检查一直通过：engine 与 oracle 跑同一份代码，破坏**对称**；且首采样发生在第一次覆盖之前（探针插在首采样之后仍逐位一致）。
- 两个真正的 *restore* 调用点（`restore_lane_gdn :2276-2279`、`begin_gdn_state :5220-5222`，均为 `DeviceSnapshot`）在旧 D2D 语义下**恰好正确**，这也是 bug 被长期掩盖的原因。
- 附带纠正（**已于 §12.11 更正**）：此处原判「`plan_linear_attention_state_pool`（`src/core/linear_attention_state.cpp:75-112`）把 slot 放在**最内维**（`{conv_channels, conv_width, slot_count}` / `{key_head_dim, value_head_dim, value_heads, slot_count}`），故 `conv_layer_pitch_bytes`（层间真实步长）≠ `conv_bytes`（单层单 slot），lanes>1 时会跨 slot 步进」是**误报**——它按行主序（最后一维连续）理解 NInfer 布局，而 NInfer 的 `Tensor` 是 **dim 0 最快变化**（`src/core/tensor.cpp:50-57 set_contiguous_strides`：`nb[0]=elem`，`nb[i]=nb[i-1]*ne[i-1]`），所以 slot 恰恰是最慢变化/最外层，`conv_layer_pitch_bytes` 只反映 arena 的层间 padding。结论、证据与同一轮发现并修复的真实缺陷见 §12.11。

#### 修复（`src/runtime/engine/tp2_generation_core.cpp`，4 处 edit）
- `snapshot_lane_state` 与 `publish_lane_prefill` 的 plane 写入改为「plane 在前、pool 在后」：`copy_lane_state(geometry, plane /*device_base*/, lane, pool /*other*/, cudaMemcpyDeviceToDevice, stream)`。
- `copy_lane_state` 头注释与 D2D 分支注释改写，明确「D2D 写进第一个 device 指针；restore 传 pool 在前，snapshot 传 plane 在前」。

#### 验证
- `build.ps1 -Jobs 16` exit 0（26 s 增量）。
- plain 门禁（pwsh-445）：`unaligned dialogue passed`、`replayed answer passed: … reused 70 of 58 prompt tokens`（与 HEAD 同值），随后推进到与 HEAD **逐字相同**的既有失败 `FAIL (plain): a conversation behind a shared system prompt diverged from the oracle on its first sample: got [2752 13 198 197 197 92 198 197] expected [467 419 538 13 198 197 197 92]` ⇒ `a recalled conversation` 已修复。
- 探针：engine `a_second` `prompt=88 reuse=71 first=365`，与 oracle `continued` 的 `first=365` 一致。
- §12.7 的 anchor 网格对齐修复**保留**（它让 chunk 计划与引擎历史解耦）；快照修复后 `replayed answer` 的复用计数回到 HEAD 的 `70 of 58` ⇒ 单 lane 可观测行为无变化。
- 完整闸门（pwsh-446）：`tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` **11/11 通过**，69.84 s（`tp_device_pair` 33.59 s、`linear_tp2_split_nvfp4` 25.17 s、`linear_tp2_split_fp8_head` 7.08 s、`linear_tp2_split_grouped_head` 1.68 s、`engine_options` 0.23 s、`serve_options` 0.03 s；5 个需要 artifact 的用例 Skipped）。
- artifact 用例（pwsh-447）：`ninfer_qwen3_5_tp2_forward_test.exe --artifact D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer` **exit 0**（末行「TP-2 single-token forward passed: … batched verify window lane-consistent under bounded route drift」，**不再**出现 `FAIL: the batched verify window diverged from its per-lane solo window`；唯一不匹配是 `lane=1 col=2 gap=1.03992 solo=4400 batch=3049 margin_batch=0.125` 的近邻并列）；`ninfer_qwen3_5_tp2_load_test.exe --artifact …` **exit 0**（`devices=0,1 layers=64 shard_gate=[8704,5120] shard_down=[5120,8704] head=[124160,5120] vocabulary-parallel`）。
- `tp2_sessions` 三条路线（pwsh-448，`NINFER_TEST_ARTIFACT=D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer`）：plain/mtp 只剩已记录的既有失败（`a conversation behind a shared system prompt`，`tests/models/qwen3_5/test_tp2_sessions.cpp:549-554`）；dflash2 只剩 `docs/tp2-dual-5060ti-worklog.md:6549-6554` 记录的既有失败（`a recalled conversation`，该 artifact 上 HEAD 同样逐 token 失败）。三条路线 `unaligned dialogue` 与 `replayed answer` 均 passed。
- Stage 2b 四个活体探针重跑（pwsh-450，日志 `_temp/p23_fix_stage2b_probes.log`）：plain-pair / mtp-pair / dflash2-pair / plain-pair-ring 全部 `PROBE_EXIT=0`，四段 stderr 无错误。**此前被方向 bug 破坏的 host-restore 路径现在真的被走到并正常**：mtp-pair `lane=0 prompt=953 cached=932 shared=851 replay_split=851 adopted=0 prefill_end=916 host=1/34 stride=16384 -> reuse=512 slot=8 src=host`；dflash2-pair `lane=0 prompt=913 … -> reuse=512 slot=8 src=host`；plain-pair-ring 两条 lane 均 `-> reuse=2560 slot=12/29 src=host`。`src=device`（reuse=874/873/916/2806）与 `src=live`（reuse=916/874/913）也正常。⇒ 修复在 batch executor（pair 模式即 `execute_plain_batch`/`execute_spec_batch`）与 host checkpoint 路径上均已实机验证。
- 遗留（不属本节，已由 §12.11 处置）：此处原记的「lanes>1 时 `copy_lane_state` 的逐层 2D 拷贝跨 slot 步进」经核为**误报**（更正见 `:612`）；但同一轮调查在 host checkpoint 的 target-state 镜像上发现一处**真实越界**，已在 §12.11 修复。


### 12.9 P2.4 多模态 per-lane 与「不同图片互相复用 KV」缺陷的修复（2026-10-01）

#### 施工（`src/runtime/engine/tp2_generation_core.{h,cpp}`）
- E1 `drive_lane_queue`（`:2082-2102`）删掉 media 的串行拒绝（`refusal = "media"`），grammar 拒绝保留（`refusal = "grammar"`）。多模态请求现在与文本请求一样进批，`bool batchable = lanes_ > 1` 对其成立。
- E2 抽出 `std::unique_ptr<qwen::execution::VisionPrefillSession> TP2GenerationCore::open_vision_session(qwen::PreparedPromptData& data, std::uint32_t reuse, qwen::execution::VisionPrefillPlan& plan)`（`:5015-5073`，声明 `tp2_generation_core.h:785`）：把 `execute_walk` 里原来的内联 Vision 会话构造（`plan_vision_control` + `build_vision_control` + `VisionPrefillSession`）收成一处，`execute_walk` 改为调用它（`:5284-5293`）。
- E4/E5 `execute_plain_batch` / `execute_spec_batch` 的 per-lane 循环内各自持有 `media` / plan / session：非 const 的 `length` 先经 `prepare_chunk` 截断（`Tp2VisionChunk media_chunk`），prefill 的 `media_ptr` 指向该 lane 自己的 chunk，循环结束后释放 session；计时拆成 `prompt_wall_seconds` / `vision_seconds` / `prefill_seconds`。spec 批的截断放在 `mtp_input_a` 分配之前，使 MTP priming 看到的是截断后的长度。

#### 实机验证（媒体进批）
- 双图对跑（`_temp/20261001-1400_p24_vision_pair.ps1`）：`[tp2-lane] batch=2 capacity=2 path=batched`，两条答案正确；solo 与 soloB 与批内逐字节一致；纯文本对照 prompt 43 token vs 图片 prompt 1069 token。
- 混合批（`_temp/20261001-1600_p24_vision_mixretry.ps1`，日志 `_temp/p24_mixretry.log`）：在 [0,6,10,14,18,22] ms 延迟里，attempt 2（10 ms）与 attempt 3（14 ms）真的成批（`batch=2 path=batched`，一条 media + 一条 text），两条都 HTTP 200 且答案正确 ⇒ media 进批完成。

#### 缺陷：HEAD 既存的「不同图片互相复用 KV」
- 症状（`_temp/20261001-1710_p24_media_reuse2.ps1`，C=1，依次 load_03 → load_04 → load_05 → load_03）：r1(load_03)=Green（整段 prefill `prompt_ms=810.7`），r2(load_04)=**Green**（`cache_n=1024`、`prompt_ms=210.7`），r3(load_05)=**Green**（`cache_n=1024`、`prompt_ms=215.7`），r4(load_03)=Green。从零跑时 load_04 是 Purple、load_05 是 Brown ⇒ 第 2 条起返回第 1 张图的答案。
- 根因：每张图合并成**相同的 placeholder token id**（所有图片 prompt 都是 1049 token、1024 个 merged vision token），而 TP-2 的前缀复用判定只看 token id（`scan_lane_reuse:2191-2197`、`session_recall:4553-4560`/`:4602-4604`、`execute_walk` 内联扫描 `:5190-5267`），媒体身份完全缺席。`src/models/qwen3_5/execution/text.cpp:2582-2624` 的 vision scatter 由 `control.scatter_indices` 驱动，复用边界落在图内时只重算后半段列 ⇒ 前半段仍是旧图 KV。
- 该缺陷**不是 P2.3/P2.4 引入**：`git show HEAD:src/runtime/engine/tp2_generation_core.cpp` 的对应位置（导出件 `_temp/head_tp2core_now.cpp:2536-2542`）是同一段 token-only 比较。单卡路线由 `docs/serving.md:1041-1044` 的契约覆盖（「A multimodal hit additionally requires matching token types, three-axis MRoPE positions, encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix skips Vision execution」），TP-2 路线违反了它。

#### 修复（`MediaSpan` 精确 item 比较）
- `src/models/qwen3_5/program/prefix_identity.{h,cpp}`：把匿名命名空间的 `same_item` 提升为公开的 `bool same_vision_item(const VisionItem&, const VisionItem&)`（比较 modality / grid / patch_begin / patch_count / content_digest / timestamps / token_spans），单卡路径复用同一份比较。
- `src/runtime/engine/tp2_generation_core.h`：新增 `struct MediaSpan { std::uint32_t begin = 0; std::uint32_t end = 0; models::qwen3_5::VisionItem item; }`；`SessionEntry` 加 `std::vector<MediaSpan> media`；`RetentionState` 加 `std::vector<MediaSpan> cached_media`；`session_recall` / `session_publish` / `scan_lane_reuse` / `publish_lane_prefill` 四个签名各加 `std::span<const MediaSpan> media`；新增 `collect_media_spans(data, limit)`（把每个 Vision item 的 `token_spans` 求包络为 `[begin,end)`；`end > limit` 的项——被 prompt 截断——不参与）与 `media_prefix_cap(cached, incoming, shared_prefix)`（双指针合并两张按 `begin` 有序的表；`begin >= shared_prefix` 的项不约束边界；同 `begin` 但 `same_vision_item` 为假 ⇒ 返回该 `begin`；一侧有另一侧没有 ⇒ 返回较早的 `begin`）。
- 调用点：`scan_lane_reuse` 的 token 比较之后（`:2255`）；`session_recall` 的 `active_shared` 与 host 候选 `shared` 之后（`:4628`/`:4674`）；`execute_walk` 内联扫描之后（`:5278`）；两个 batch executor 各传 `prompt_media`（`:2675-2685` / `:3204-3211`）。维护点：`publish_lane_prefill:2398`、`execute_walk` 尾 `:6042`、`session_recall` 命中 `:4723`、`session_publish` 的 `:4855`/`:4874`、`session_invalidate_active:4071`。
- 理由：token id 看不到媒体，而 token_types 与三轴 MRoPE positions 都由 media item（grid/digest/spans）决定，所以 item 精确比较与单卡契约等价，无需 O(n²) 收缩 `ResidentPrefixIdentity`。

#### 修复验证
- `build.ps1 -Jobs 16` exit 0（26 s 增量）。
- 缺陷用例重跑（`_temp/20261001-1710_p24_media_reuse2.ps1`）：r1 load_03=Green、**r2 load_04=Purple**、**r3 load_05=Brown**、r4 load_03=Green，四条全部 `cache_n=0` 整段 prefill（`prompt_ms≈808-811`）；`[tp2-reuse]` 显示 `shared=15`（图片 token span 起点）⇒ 边界被拉到图片之前。
- 同图复用不回归（`_temp/20261001-1700_p24_media_reuse.ps1`）：r2（load_01 重复）仍 `shared=1049 → reuse=1024 src=device`、`cache_n=1024 (97.6%)`、`prompt_ms=204.8`；r3（load_00）与 r4（load_01）由原来的 `reuse=1024 src=host`（错误）变为 `shared=15 → reuse=0` 整段 prefill。
- 历史含同图的纯文本追问不回归（`_temp/20261001-1500_p24_vision_mix.ps1` 的 `history` 用例）：t2-A `prompt=1111 cached=1083 shared=1083 → reuse=1082`、`cache_n=1082 (97.4%)`、`prompt_ms=98.2`，与修复前逐字相同；`mix` 用例仍成批且答案正确。
- 门禁（pwsh-478）：`test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` **11/11 通过**，68.97 s（`tp_device_pair` 33.80 s、`linear_tp2_split_nvfp4` 24.80 s、`linear_tp2_split_fp8_head` 6.64 s、`linear_tp2_split_grouped_head` 1.49 s、`engine_options` 0.21 s、`serve_options` 0.03 s；5 个需要 artifact 的用例 Skipped）。
- artifact 用例：`ninfer_qwen3_5_tp2_forward_test.exe --artifact D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer` **exit 0**（末行同 §12.8，唯一不匹配仍是 `lane=1 col=2 gap=1.03992 solo=4400 batch=3049 margin_batch=0.125` 的近邻并列）；`ninfer_qwen3_5_tp2_load_test.exe --artifact …` **exit 0**。
- `tp2_sessions` 三路线（后台 job pwsh-477，`NINFER_TEST_ARTIFACT=D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer`）：plain `EXIT=1`、mtp `EXIT=1`、dflash2 `EXIT=1`，**三个路线都只剩 HEAD 既有的失败**——plain/mtp 停在 `a conversation behind a shared system prompt diverged from the oracle on its first sample`（plain `got [2752 13 198 197 197 92 198 197] expected [467 419 538 13 198 197 197 92]`，mtp `got [467 419 538 317 197 197 92 317] expected [2752 317 197 197 92 317 197 197]`），dflash2 停在 `a recalled conversation diverged from the oracle on its first sample: got [59399 475 327 363 62 16 15 15] expected [365 4577 287 62 16 15 15 15]`；三路线的 `unaligned dialogue` 与 `replayed answer` 用例全部 passed。⇒ 媒体身份修复没有把新的失败引入任何路线。

### 12.10 P2.5 运维文档同步、并发档位定稿与收尾清理（2026-10-01）

#### 文档同步（G2）
- `docs/serving.md:68-74`：补并发档位的内存前提——lane 共享一个 KV 池，所以是分摊上下文上限而不是各自预留；linear-attention state arena 每卡每 lane 增长约 294 MiB，出厂 262,144 保持 `--max-concurrency 1`，2..4 lane 需要 `--max-context 131072`。
- `docs/serving.md:75-82`：把「a batch containing a Vision or tool-grammar request runs lane by lane」改为**只有 tool-grammar** 走 lane-by-lane（P2.4 后 Vision 进批，每 lane 在 Vision shard 上各自编码自己的媒体），并补一句复用前缀必须匹配媒体身份（checkpoint 携带每个 image/video item 的 digest、grid 与 consumer spans）。
- `docs/tp2-dual-5060ti.md:486-493`：删除过时的「cross-session retention is off there」，改为 P2.3 的 per-lane retention（checkpoint ring 与 session catalog 按 lane 切片，每 lane 独立 recall 自己的上一轮会话），并补 Vision 进批与媒体身份复用。
- `model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md:237-245`：TP-2 小节新增并发段落（`--max-concurrency 2..4` 把 queued requests 填进一个 decode round、共享 KV 池、+294 MiB/卡/lane、262144 ⇒ C=1 / 131072 ⇒ 2..4、实测 1.76-1.91x@2 与 2.80x@4 plain、2.07x@4 dflash2、1.48x@2 与 2.26x@4 mtp、近邻并列判据）。
- `AGENTS.md:35-39` **无需改动**：已经写着 TP-2 路线把 plain、DFlash2 与 MTP round 填到四个 queued requests。

#### 并发档位定稿
- 并发档位：`--max-concurrency 1..4`（单卡路线仍是 1..8）；默认 `1` 与历史逐字节一致，`lanes_ > 1` 时机械强制 `EagerExact`（`verify_graph_enabled_ = false`）。
- 上下文档位：`--max-context 131072` + 共享 KV + per-lane retention 是 C=2..4 的推荐配置；262,144 上 C>2 因 state arena 增长不可接受（会挤到 <150 MiB 余量）。
- 数字权威来源：§3.3（内存台账与 P1.2b 实测修正）、§12.5（P2.2 测量）、`docs/serving.md:76-91`（聚合吞吐与近邻并列）。

#### 收尾清理
- 删除 `src/runtime/engine/tp2_generation_core.cpp` 中临时的 `NINFER_TP2_LOGITS_PROBE` 探针块（原 `:5911-5945`，35 行；删除后 6746 → 6711 行），保留全部默认关闭的 `NINFER_TP2_*` 诊断开关。
- 删除探针后复验（job pwsh-478，`_temp/p25_build.log`/`p25_gate.log`/`p25_forward.log`/`p25_load.log`）：`build.ps1 -Jobs 16` **exit 0**；`test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` **11/11 通过**（68.55 s）；`ninfer_qwen3_5_tp2_forward_test.exe --artifact D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer` **exit 0**（唯一不匹配仍是 `two-lane verify window ... max_logit_diff=2.03125 argmax_mismatches=1 near_tie_flips=1`）；`ninfer_qwen3_5_tp2_load_test.exe --artifact …` **exit 0**。
- `tests/models/qwen3_5/test_tp2_forward.cpp` 经核实**没有**临时诊断代码（`kTestLanes = 2`、`kTestPoolPages = kTestCachePages * kTestLanes`、`.kv_table_rows/.slot_count/.record_capacity = kTestLanes`、`for (lane < kTestLanes)` 都是双 lane verify window 测试的正常组成部分），无需改动。
- 收尾（2026-10-01 完成）：本文件已移入 `docs/`，源码中 16 处裸文件名引用（`include/ninfer/tp2_capacity.h` 1、`src/runtime/engine/model_instance.cpp` 1、`src/runtime/engine/tp2_generation_core.h` 4、`src/runtime/engine/tp2_generation_core.cpp` 9、`tests/test_engine_options.cpp` 1）同步指向 `docs/PLAN-tp2-concurrency.md`；`_temp/` 产物已清理。
- 移动后复验：当时用户线上服务器（`C:\ninfer\infer-serve.exe`，pid 21000，`--max-concurrency 4`）在跑，`tools/win_port/build.ps1:164-173` 的护栏只按**进程名**判断就 `exit 4`（误判：它持有的是 `C:\ninfer\` 下的另一份 exe，不锁 `build-win/apps/ninfer-serve.exe`），故 dot-source `tools/win_port/vcvars.ps1` 后直接 `ninja -C build-win` —— **exit 0**；`test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` **11/11 通过**（69.93 s，五个 artifact 驱动的 tp2 用例照旧 Skip）。
### 12.11 「`copy_lane_state` 跨 slot 步进」遗留项的核查与 host checkpoint target-state 镜像越界的修复（2026-10-01）

#### 结论一：原遗留项是误报（NInfer 布局是 dim 0 最快）
- NInfer 的 `Tensor` 是 **dim 0 最快变化**：`src/core/tensor.cpp:50-57 set_contiguous_strides` 为 `stride = dtype_size; nb[0] = stride; for (i=1..3) { stride *= ne[i-1]; nb[i] = stride; }`；`slice(dim,start,len)`（`src/core/tensor.cpp:112-125`）只改 `ne[dim]` 并把 `start*nb[dim]` 加到指针上。
- 因此 `conv_shape {conv_channels, conv_width, slot_count}` 的内存布局是 `[slot][conv_width][conv_channels]`：slot 是最外层（最慢变化），每 slot 是连续的 `conv_bytes = conv_channels*conv_width*elem` 块，位于层内偏移 `slot*conv_bytes`；`recurrent_shape {key_head_dim, value_head_dim, value_heads, slot_count}` 同理（`slice(3,slot,1)` ⇒ `slot*recurrent_bytes`）。
- 直接证据：`src/core/linear_attention_state.cpp:199-209 conv_slot/recurrent_slot` 做 `slice(2/3, slot, 1).view({...})`。`Tensor::view()`（`src/core/tensor.cpp:99-108`）在非连续时抛 `std::invalid_argument("view requires a contiguous tensor")`，而它从未抛过 ⇒ 该 slice 是连续的。另 `:44-71 layer_stride_bytes` 校验层间 pitch 恒定（`current - previous == stride`，否则抛 `"... layer stride is not constant"`），`:162-176 slot_view` 据此给出 `conv_layer_pitch_bytes`（层间真实步长，含 arena padding）与 `conv_layer_bytes`（单层单 slot）。
- 旁证：同一「slot 是层内连续块」契约已被 4-slot 池的功能测试覆盖——`src/models/qwen3_5/state/state_image.cpp:381-458 copy_to_host/copy_from_host` 逐层用 `cudaMemcpyAsync(dst, conv_slot(layer,source).data, conv.bytes(), ...)` 整块搬运；`tests/models/qwen3_5/test_state_image.cpp:132-170 test_host_roundtrip`（`slot_count == 4`，每个 (layer,slot) 填不同字节、往返后逐 slot 校验）与 `:201-211`（`copy_slot(0,2)`/`zero_slot(1)` 的源隔离校验）全部通过。
- ⇒ `src/runtime/engine/tp2_generation_core.cpp:4028-4067 copy_lane_state`（`device_conv = base + conv_base + lane*conv_bytes`；D2D 两侧同布局整池，D2H/H2D 用 `conv_pitch`/`conv_bytes` 做 2D 步进）与 `:4069-4081 zero_lane_state` 的偏移与跨步**都是对的**。§12.8 `:612` 与 `:627` 已就地更正。

#### 结论二：同一区域的真实缺陷——host checkpoint 的 target-state 镜像按整池寻址
- 布局事实：`shard.host_checkpoints[index].buffer` 是 `PinnedHostBuffer(shard.lane_state_geometry.image_bytes, true)`（`:795-796`），即**一个槽只有一张紧凑单 lane 像**；`PinnedHostBuffer` 是精确分配（`src/core/arena.cu:251-264 cudaHostAlloc`，不额外扩）。lane 由**槽下标**表达——`:789 slots = lanes_ * host_checkpoint_slots_per_lane_`、`:5089 begin += lane * host_checkpoint_slots_per_lane_`、`:5095 host_checkpoints[begin + index]`。与 §12.6 `:526`/`:537` 的施工版决定一致（每槽一张像，否决「每槽 `lanes_` 张像」）。
- 三处拷贝却在 buffer 上又加了 `lane * image_bytes`：`restore_lane_gdn` 的 `HostCheckpoint` 分支（原 `:2339-2345`）、`snapshot_host_checkpoint`（原 `:5074-5078`）、`execute_walk` 内联的 `begin_gdn_state`（原 `:5399-5404`）。lane=0 偏移为 0 故正确；lane≥1 的读写落在 pinned 分配尾后约 73.4 MiB 的任意内存 ⇒ UB / 堆破坏。
- 为何长期未暴露：写点与读点用**同一个**越界偏移，checkpoint 往返自洽；且 lane 1 的 host 恢复路径确实被走到过（`_temp/p23_fix_stage2b_probes.log:106` plain-pair-ring `lane=1 prompt=2890 cached=2805 shared=2783 replay_split=2783 adopted=0 prefill_end=2805 host=5/34 stride=16384 -> reuse=2560 slot=29 src=host`，slot=29 属 lane 1 的 `[17,34)`），答案连贯。对照 `store_dflash_image`（`:1908-1911`）有显式护栏 `if (image.size() < (size_t)(lane+1)*lane_bytes) throw std::logic_error("draft context host image is smaller than the lane's draft ring")`，target-state 路径缺等价护栏才漏过。
- 其余镜像寻址经审计**正确、未改动**：`entry.host_state/host_prompt_state/host_shared_state` = `PinnedHostBuffer(shard.state_backing.bytes)`（整池，`:4237`/`:4310`/`:4325`）故 `+ lane*image_bytes` 正确；`entry.host_dflash*` = `PinnedHostBuffer(dflash_bytes)`（`:4263`/`:4277`/`:4291`，`dflash_bytes = lanes_ * lane_context_image_bytes()`）；`checkpoint.dflash_buffer` = `PinnedHostBuffer(dflash_image)`（`:974`，`dflash_image = lanes_ * dflash_lane_image`）。`session_capture_shared_state` 的 `from_device=false` 分支（`:4611-4613`）对源**故意不加** lane 偏移，因为其 `frozen[shard_index]` 就是 `checkpoint.buffer.get()`（本 lane 的专属 buffer），紧邻的 draft 分支两侧都加偏移是因为 `dflash_buffer` 是整池——两者都正确。

#### 修复（`src/runtime/engine/tp2_generation_core.{h,cpp}`）
- 三处删掉 `lane * image_bytes`，直接用 `checkpoint.buffer->data()`（`:2340-2347`、`:5111-5114`、`:5437-5444`），注释改写为「槽属于本 lane，buffer 就是该 lane 的紧凑像；lane 由槽下标（`begin + lane*slots_per_lane`）表达，绝不由槽内字节偏移表达」。
- `snapshot_host_checkpoint`（`:5107-5110`）加唯一 writer 的护栏：`if (checkpoint.buffer->size() < shard.lane_state_geometry.image_bytes) throw std::logic_error("TP-2 host checkpoint slot is smaller than one compact lane state image");`
- `make_lane_state_geometry`（`:3975-4022`，guard `:3989-4021`）末尾加失败即停的几何不变量校验：对每个 `lane ∈ [0, shard.state->slot_count())`、`layer ∈ [0, geometry.layers)` 断言 `conv_slot(layer,lane).data == base + conv_base + lane*conv_bytes + layer*conv_pitch`（recurrent 同理）且 `.bytes()` 等于 `conv_bytes`/`recurrent_bytes`，否则抛 `std::logic_error("TP-2 lane state geometry disagrees with the linear-attention pool: lane N layer M is not one contiguous state image per layer at the pool's layer pitch")`。这把结论一的布局契约变成机器检查，由 `tp2_forward`（lanes=2）/`tp2_load`/门禁每次覆盖。
- `src/runtime/engine/tp2_generation_core.h:128-132` 的 `LaneStateGeometry` 注释改写，明确「dim 0 最快 ⇒ slot 最慢变化/最外层；每 lane 是每层一个连续块、层间一个 pool pitch；紧凑 pinned 像是这些块首尾相接、无 arena padding」。

#### 验证
- `build.ps1 -Jobs 16` **exit 0**（32 s 增量，`_temp/p26_build.log`）。
- 门禁（`_temp/p26_gate.log`）：`test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` **11/11 通过**，72.83 s。
- `ninfer_qwen3_5_tp2_forward_test.exe --artifact D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer` **exit 0**（`_temp/p26_forward.log`；lanes=2 会执行新增几何护栏，未抛）。末行同 §12.10；唯一不匹配仍是 `two-lane verify window (4 columns x 2 lanes): max_logit_diff=2.03125 argmax_mismatches=1 near_tie_flips=1`（`lane=1 col=2 gap=0.882812 solo=4400 batch=3049 margin_batch=0.125` 的近邻并列），`duplicated-lane control: lane_gap=0 ... argmax_mismatches=0`。
- `ninfer_qwen3_5_tp2_load_test.exe --artifact …` **exit 0**（`_temp/p26_load.log`：`devices=0,1 layers=64 shard_gate=[8704,5120] shard_down=[5120,8704] head=[124160,5120] vocabulary-parallel`）。
- `ninfer_qwen3_5_state_image_test.exe` **exit 0** 且无 `FAIL:` 输出 ⇒ 4-slot 池的「slot 是层内连续块」契约测试（`test_host_roundtrip` 与源隔离校验）仍通过。
- ring 探针复跑（`_temp/20261001-1100_p23_stage2b_routes.ps1`，job pwsh-485）：四段 plain-pair / mtp-pair / dflash2-pair / plain-pair-ring 全部 `PROBE_EXIT=0`（`_temp/p26_ring.log`，`S2B_DONE`）。
  `plain-pair-ring` 的 lane 1 host 恢复路径（本次修复点）仍正常：`[tp2-reuse] lane=1 prompt=2848 cached=2805 shared=2783 replay_split=2783 adopted=0 prefill_end=2805 host=5/34 stride=16384 -> reuse=2560 slot=29 src=host`（slot=29 属 lane 1 的 `[17,34)` 区间）；其答案 `B3: cache_n=2560 prompt_ms=483.3 pred_ms=340.1 n=13 | The invariant is preserved because every writer names the sa` 与 lane 0 的 `A3: cache_n=2560 prompt_ms=243.0 pred_ms=580.3 n=13 | The invariant is preserved because every writer names the sa` 一致，与修复前逐字相同；`plain-pair`/`mtp-pair`/`dflash2-pair` 的 `src=device` 复用与 `mtp-pair`/`dflash2-pair` 的 `reuse=512 slot=8 src=host` 也全部复现。
- `tp2_sessions` plain 路线（`NINFER_TEST_ARTIFACT=D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer`，`NINFER_TEST_ROUTE=plain`，`_temp/p26_sessions_plain.log`，job pwsh-487）：`unaligned dialogue` 与 `replayed answer`（`reused 70 of 58 prompt tokens`）passed，随后停在**与 HEAD 逐字相同**的既有失败 `FAIL (plain): a conversation behind a shared system prompt diverged from the oracle on its first sample: got [2752 13 198 197 197 92 198 197] expected [467 419 538 13 198 197 197 92]`（`EXIT=1`）⇒ 单 lane（`slot_count() == 1`）路径下新增几何护栏通过且行为不变。
- 说明：修复只去掉越界偏移、不改变读写的内容语义（writer 与 reader 原本用同一个偏移），故功能行为不变；新增护栏使「ring 分区与 state 几何不一致」在任何 lane 上立即失败而不是静默越界。

### 12.12 审查修复计划的落地（A/B/C 项）与 MTP 多卡 warmup 的 peer arena 泄漏修复（2026-10-01）

逐项施工记录与验证见 `docs/PLAN-tp2-concurrency-review-remediation.md` §11；本节只记结论与对本文档有影响的部分。

#### 落地范围
- **P0.1＝A2**：批组建/trace/调度全部移入 `drive_lane_queue` 的 `try`，异常走 D4 收窄路径而不是逃出驱动线程。
- **P0.2＝C2**：多 lane 批次改由核心自持的驱动线程运行（`lane_driver_`，构造时启动、`stop_lane_driver()` 析构停止），`wait_lanes` 只入队并等待自己的 `complete`；`publish_lane` 每 lane 完成即发布 ⇒ 长请求不再扣住同批的短请求（探针：短请求 `done` 从整批等待降到 316–1858 ms，无 499）。
- **P1.1＝A3**：`build_shard` 增加启动断言（`lane_context_window() + (mtp_enabled_ ? mtp_drafts_ : 0) < per_lane * kPagedKVPageSize`），把「未发布页 = 未初始化 block-table 项」的关系钉死在启动期；`build_shard` 调用点下移到 lane 预算计算之后。
- **P1.2＝A4+A5**：删 `model_instance.cpp` 的「collapses to 1 lane」死 fprintf（`--spec dflash` 在归一化末尾直接被拒，无路线静默丢 lane）；删 `tp2_generation_core.cpp` 不可达的 `refusal = "route"` 分支。
- **P1.3＝B2+B5**：`docs/serving.md` 与 `src/serve/serve_options.h:79-83` 的 TP-2 容量口径改为 `tp2_generation_concurrency(max_concurrency)`（1–4 活跃 lane）+ `max_pending_requests` 排队。
- **P2.1＝C3/A1**：带 tools 的请求不再退化为串行 —— 每 lane 一个 `ToolCallConstraint`，按 lane 的列块（`[width*l, width*(l+1))`）掩码该轮 logits，prefill 首 token 同样掩码；`drive_lane_queue` 的 grammar 拒绝分支删除。
- **P2.2＝C1**：轮边界动态接纳（`try_pop_lane_queue` + `admit_lane`），空出的 slot 立刻接住批运行期间到达的请求；staging 全部按 `lanes_` 预留。
- **P3.1＝B3**：MTP 草稿链的图/host profile/AR channel 只在 `lanes_ == 1` 构建，多 lane 启动日志显示 `mtp chain: n/a`。
- **P3.2＝B6 / P3.3＝B1 / P3.4＝B4**：`build.ps1` 条件化说明、HTTP 线程池容量不变式注释、整轮失败爆炸半径注释，`docs/serving.md` 同步。

#### 新发现并修复：`TextContext::proposal_argmax` 的 peer arena 泄漏（P0.3）
- 现象：`--spec mtp --max-concurrency 4` 启动 warmup 报 `TP-2 batched verify CUDA Graph replay found a different workspace layout`（`tp2_generation_core.cpp:1551-1553`）；`--max-concurrency 1` 正常，DFlash2 多 lane 正常。
- 根因：`src/models/qwen3_5/execution/text.cpp:822` 的 `proposal_argmax` 只对本地 arena 取 scope（`:823`），而拆分 proposal head 的分支在 **peer 的 arena** 上分配（`:867`/`:876`/`:878` 以及 `:877` 的 `project` scratch）且 peer 侧无 scope ⇒ 每次调用把 peer 的 bump pointer 抬高约 403,456 B 永不回收。多 lane 路线每轮跑 `max_extent` 个 MTP 链步，`max_extent` 随轮预算变化（warmup 捕获轮 2、重放轮 0）⇒ verify 图捕获的 `arena_begin[1]` 无法复现。
- 归属：既有缺陷（`text.cpp` 最后一次改动是已提交的 `6a17bb2c`）；同文件其他 peer 分配点（`:494`/`:932`/`:1050`/`:2137`/`:2814`）都有 `auto peer_scope = peer.work_.scope();`，只有 `:858` 漏了。
- 修复：`:858` 之后补 `auto peer_scope = peer.work_.scope();`（含注释 +5 行）。
- 与本文档 `:500` 记录的 P2.2c 属同类（链步内的分配让 verify 调用点的 `used()` 漂移），P2.2c 只处理了本地侧（把 `positions` 提到循环外），peer 侧漏了。

#### 验证（2026-10-01）
- `ninja -C build-win` exit 0（`_temp/fix_build2.log`，`NINFER_TP2_LAYOUT_TRACE` 临时插桩已全部删除）。
- 门禁（`_temp/fix_gate.log`）：`test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` **11/11 通过**，72.51 s（五个 artifact 用例照旧 Skip）。
- `tp2_forward --artifact …` exit 0（`_temp/fix_forward.log`）；`tp2_load --artifact …` exit 0（`_temp/fix_load.log`）。
- `tp2_sessions`（`_temp/fix_sessions.log`）与基线 `_temp/p22_sessions.log` 的 `passed/FAIL` 行 **Compare-Object 完全一致**（仍停在既有的 shared-system-prompt 失败）。
- MTP C=4 `--draft-tokens 2` 双卡冒烟：warmup 通过（`listening on http://127.0.0.1:8099`），两条并发 `/v1/responses`（带 tools 的短请求 + 1729 token 长请求）均 HTTP 200，带 tools 的返回合法 `function_call`；逐步 trace 显示 shard B 水位在链步间恒定（修复前 `81920 → 444416 → 847872`，修复后恒为 `81920`）。
- MTP C=1 双卡冒烟：启动日志 `mtp chain: graph`、`rendezvous id channels reserved: 24`、HTTP 200 ⇒ P3.1 未破坏单 lane 路线。
- 上述 P0/P1/P2.1/P2.2/P3 与本次 peer scope 修复均**未提交**，在 work tree（仓库约定：仅在用户要求时提交）。
