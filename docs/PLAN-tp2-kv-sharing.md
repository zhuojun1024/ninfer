# PLAN：TP-2 共享 KV 动态预留（kv sharing）实施计划

> **本文件是 TP-2 共享 KV 项目的唯一活动计划**，也是跨上下文压缩的持久记忆；上下文被压缩或会话恢复后，先重读本文件再继续。
>
> - **基线**：分支 `feat/tp2-concurrency`，`HEAD = 94e6cc64`（11 commits，未 push）；Windows 工作树 `D:/Documents/workbench/ninfer`，构建 `build-win`（VS2022 + CUDA 13.3）。
> - **来源**：本文件由源码级评估固化（用户指令 m00683「落成文档并开始实施」，评估请求 m00552「评估动态共享。另可参考 C:/llama.cpp的做法」）。
> - **相邻权威**：`docs/PLAN-tp2-concurrency.md`（多并发主体计划，本文件是其 KV 语义的续篇）、`docs/serving.md`（对外契约）、`docs/maintainer/paged-kv-cache.md`（KV 物理模型与 execution row）、`docs/maintainer/engine-architecture.md`（单卡执行/调度契约）、`docs/cli.md`（启动旋钮）。
> - **语义**：本项目的共享 KV ＝ **一份物理池 + 准入时按请求完整预留 + 请求终态归还 + 无抢占**。与单卡路线的 KV 语义一致；不含每轮现找空位、驱逐空闲 slot、跨 lane 抢占。

---

## 0. 结论（已交付评估，作为实施基线）

**值得做，且是 TP-2 的正确收敛方向。** 今天的 TP-2 把一份物理 KV 池在启动期静态均分给 lane，等于把「机器总上下文」除以并发数；而单卡路线的语义是「一份共享池，谁跑谁用满」。两者在同一份产品契约下互相矛盾：

- 静态均分下 `--max-context 131072 + --max-concurrency 4` ⇒ 每 lane 512 页 = **32768 token**。用户实测（m00498）的失败用例 `prompt 14811 + output 17957 = 32768` 恰好撞满，第二轮直接 `HTTP 400 context_length_exceeded`；
- 动态共享下同两个请求各需 `pages_for_tokens(14811 + 32768) = 744` 页，`2 × 744 = 1488 ≤ 2048` ⇒ **两个会话同时跑**；四个同规模请求 `2976 > 2048` ⇒ 两个并发、两个排队（排队而非 400）；
- 动态共享下单个 `prompt 90000` 的请求也能跑（静态 32768 直接 400）。

实施分两段：**S1 文本 KV 动态预留/释放（主体）**、**S2 上限旋钮 `--lane-context`**；MTP 镜像（S3）与文档/验收（S4）随后。S1 完成即覆盖用户生产路线（`--spec dflash2`）与 plain 路线。

---

## 1. 目标与非目标

### 1.1 目标
- lane 之间**共享一份物理 KV 池**：一个请求单独运行时可以用满整池；多个请求重叠时按各自实际需要划分，而不是启动期均分。
- 准入时按请求的 **prompt + effective output** 完整预留页数，成功才 admit；池子不足时**排队**（FIFO），不报 400（只有 prompt 本身超过上限才 400）。
- 请求终态归还物理页，页回到池的空闲表后可被同批新到达的请求立即使用。
- 单卡路线（`--devices` 单卡 / `lanes_ == 1`）行为**逐位不变**。
- plain 路线 temperature 0 仍逐位确定；gate 11/11、forward/load、`tp2_sessions` 与基线一致。

### 1.2 非目标（首期不做）
- 每轮现找空位（llama.cpp `find_slot`）、驱逐空闲 slot、抢占、continuous batching。
- 碎片整理（host 路径不要求连续页，碎片只影响可容纳的最大单请求，正确性无碍）。
- 多模态 lane 的 KV 语义变化（沿用既有 per-lane 绑定）。
- `--kv-capacity` 在 TP-2 路线上仍被 `normalize_engine_options` 归一为 `--max-context`（见 `src/runtime/engine/model_instance.cpp:119`）；本计划不改这一点。

---

## 2. 现状：静态均分（代码证据，file:line 为 2026-10-02 的 `94e6cc64` 工作树）

| 位置 | 事实 |
|---|---|
| `src/runtime/engine/tp2_generation_core.cpp:441-443` | `pages_per_lane = pages_for_tokens(options_.max_context) / lanes_; lane_token_capacity_ = pages_per_lane * kPagedKVPageSize;` |
| `src/runtime/engine/tp2_generation_core.h:454-458` | `lane_context_window()`：`lanes_ == 1` 返回 `options_.max_context`，否则返回 `lane_token_capacity_ - margin`（MTP 时 margin = `mtp_drafts_ + 2`，否则 1） |
| `tp2_generation_core.cpp:1054-1122` | 启动期一次性 `pool.reserve` + `materialize` + `tables.publish(row, 0, handles)`，lane `l` 拿 `[l*per_lane, (l+1)*per_lane)`，最后一 lane 拿余数 |
| `tp2_generation_core.cpp:1081-1089` | A3 启动期断言：`lane_context_window() + write_tail < per_lane * 64` |
| `tp2_generation_core.cpp:1124-1176` | MTP 层 KV 同一套静态切分（shard 0） |
| `tp2_generation_core.cpp:2015-2021` | `submit()` 用 `lane_context_window()` 判 400 并 clamp output |
| 生产日志 | `[mem] shard 0 KV pages 2048 over 4 lanes = 512 pages (32768 tokens) per lane` |

池本身**已经是动态的**（这是本计划工作量可控的根本原因）：
- `src/core/paged_kv_cache.h:197` `class DeviceKVPagePool` 自带空闲页表（`free_page_runs_`/`page_allocated_`，`:262-275`）、`reserve()`（`:217`）、`materialize()`（`:228`）、`capacity_pages()`/`available_pages()`（`:208-211`）。
- `src/core/paged_kv_cache.h:339` `class KVExecutionTablePool` 的 `publish(row, logical_begin, pages, stream)`（`:353-354`）是**可重复、可增量**的发布：实现为 host shadow + `cudaMemcpyAsync`（`src/core/paged_kv_cache.cpp:774-833`），每次 admit 覆盖自己的范围即可。
- 单卡路线正是这个语义：`src/models/qwen3_5/program/storage/kv_store.h:243` `LogicalKVPageStore` 在 `materialize/dematerialize` 之间反复搬页。

---

## 3. llama.cpp 对照（`C:/llama.cpp`）

| 机制 | llama.cpp | 本计划 |
|---|---|---|
| 开关 | `common/common.h:573 bool kv_unified = false;`（默认静态均分）；`include/llama.h:406` | TP-2 默认即目标语义（一份池） |
| 静态均分算术 | `src/llama-context.cpp:288-304`：`n_ctx_seq = n_ctx / n_seq_max`，不可整除向下取整并告警 | 与 NInfer 今天 `:441-443` 完全同构 |
| 每 slot 上限 | `common/common.h:630 int32_t kv_unified_per_slot = 0;`；`tools/server/server.cpp:153-169` 用 `n_parallel * kv_unified_per_slot` 定池大小 | S2 的 `--lane-context`（默认 0 = 整池） |
| 分配器 | `src/llama-kv-cache.cpp:751-815 prepare()` 整批原子放置 + 失败回滚；`:898-1095 find_slot()` 每 ubatch 现找 cell | **不采用**：`find_slot` 是抢占/continuous batching 的一半，且每轮重发布会让 CUDA Graph 水位断言（`tp2_generation_core.cpp:1551-1555`）变脆 |
| 释放 | `tools/server/server-context.cpp:1667-1690 try_clear_idle_slots()` 与 `seq_rm` 绑定抢占 | 只在请求终态释放，无抢占 |
| 静态均分的代价（它自己的注释） | `tools/server/server-context.cpp:1420-1434`（`--cache-idle-slots`）：non-unified 下清空 slot 不释放任何可复用空间 | 与用户 m00498 的 400 同一机制 |

结论：**取 unified 的「一份池 + 上限旋钮」，不取它的现找 cell 与驱逐**。

---

## 4. 设计（方案 A）

定义：`need = prompt_tokens + effective_output`（`effective` 已由 `submit()` 按上限 clamp）；`margin = mtp_enabled_ ? mtp_drafts_ + 2 : 1`（沿用今天 `lane_context_window()` 的余量语义：MTP 的 draft 链会写到最长 lane 位置之后）；`write_tail = mtp_enabled_ ? mtp_drafts_ : 0`。

| # | 规则 |
|---|---|
| A1 | 准入时 `reserve_pages = pages_for_tokens(need + margin)`，在两个 shard 的文本池（以及 MTP 路线的 shard 0 MTP 池）上各 `reserve()` 一次；任一失败 ⇒ 释放已拿到的部分并**不 admit** |
| A2 | 池子不足 ⇒ 该请求**回插 FIFO 队首**并结束本轮接纳（不让更年轻的请求插队），下一轮再试；只有 `prompt > lane_admission_limit()` 才 400 |
| A3 | admit 成功后 `materialize` + `tables.publish(row, 0, handles, stream)` **覆盖本请求的全部预留页**；该 lane 的窗口 `kv_window = reserve_pages * kPagedKVPageSize - margin`，不变式 `kv_window + write_tail <= reserve_pages * 64`（由 `reserve_pages = pages_for_tokens(need + margin)` 保证，保留运行期断言） |
| A4 | 释放 = `leases.clear()`（每个 lease 析构即把物理页还回 `free_page_runs_`）+ 预留对象（全量 materialize 后 `reservation.pages_ == 0`，析构为无操作）；**不必清空 execution row**——窗口永不越过已 publish 范围，但每次 admit 必须 publish 自己完整的范围（A3 教训：`src/core/arena.cu:141-155` 的 `cudaMalloc` 不置零，未 publish 条目是垃圾物理页号） |
| A5 | 上限是**策略值** `lane_context_limit_ = min(policy, pool_tokens - margin)`，`pool_tokens = pages_for_tokens(max_context) * 64`；`policy` 默认整池，S2 由 `--lane-context` 收窄。取 `min` 是为了**不可能死等**：`need <= limit` ⇒ `reserve_pages <= 池页数` |
| A6 | `lanes_ == 1` 走原路径（启动期一次性 publish 整池，窗口 = `options_.max_context`），逐位不变 |
| A7 | 分配只发生在 **admit / retire** 两个边界 ⇒ 零轮内开销；不动 CUDA Graph 的 `arena_begin` 水位 |

释放时机：lane 在 `finalize()` 里发布完 host session 之后立刻归还（让同批新到达的请求能立刻复用这些页）；批次执行器收尾再对所有 lane 幂等归还一次作为兜底。

---

## 5. 改动点清单（S1，目标形态）

| 文件 | 改动 |
|---|---|
| `src/runtime/engine/tp2_generation_core.h` | `Shard` 增 `mtp_lane_pages`（`vector<vector<DeviceKVPageLease>>`）；删 `lane_token_capacity_`，增 `lane_context_limit_`；`lane_context_window()` → `lane_admission_limit()` + `lane_kv_margin()`/`lane_kv_write_tail()`/`lane_kv_window(pages)`；声明 `reserve_lane_kv/release_lane_kv/retire_lane_session/requeue_lane_front/lane_need_tokens` |
| `tp2_generation_core.cpp:433-449` | 静态切分算术 → `lane_context_limit_` 计算（含 `pool_tokens - margin` 下溢保护） |
| `tp2_generation_core.cpp:1054-1122` | 文本池：`lanes_ > 1` 时只 `acquire` 每 lane 的 execution row，不 reserve/publish；`lanes_ == 1` 保持原样 |
| `tp2_generation_core.cpp:1124-1176` | MTP 池同上（仅 shard 0） |
| `tp2_generation_core.cpp:2015-2021` | `submit()` 改用 `lane_admission_limit()` |
| 新函数（`try_pop_lane_queue` 附近） | `reserve_lane_kv(lane, need_tokens)` / `release_lane_kv(lane)` / `retire_lane_session(lane)` / `requeue_lane_front(pending)` / `lane_need_tokens(pending)` |
| plain 批执行器 `:2599-2625, 2710-2747, 3010-3012, 3048-3065` | `LaneState` 增 `kv_window`；初批与轮边界接纳先 `reserve_lane_kv` 再 admit，失败回插队首（初批还要从 `batch` erase）并 `break`；`finalize` 里 `session_publish` 之后 `retire_lane_session` + `release_lane_kv`；driver `catch` 对全部 lane 兜底释放 |
| spec 批执行器 `:3239-3260, 3361-3413, 3660-3662, 3721-3756, 3763-3773` | 同上；`lane_window` 改为该 lane 自己的 `kv_window` |
| `tp2_generation_core.cpp:6584` | 单 lane walk 的窗口改用 `lane_admission_limit()`（`lanes_ == 1` 时与今天等价） |
| 启动日志 | `[mem] shard %d KV pages %u over %d lanes = … per lane` → 共享池口径 |
| `docs/serving.md:68-76` | 「lanes share one KV pool, so they partition the context ceiling」→ 动态预留口径 |

---

## 6. 风险表

| # | 风险 | 等级 | 处置 |
|---|---|---|---|
| R1 | 大 output 请求饿死别人（客户端可把 `max_output_tokens` 放大到整池） | 高 | S2 的 `--lane-context`；S1 在启动日志打印「整池 = N token，单请求最大预留 = M 页」 |
| R2 | 「池子不足」误走 D4 整批失败路径 | 高 | A2 明确：不足 ⇒ 回插队首 + `break`，不抛异常 |
| R3 | A3 不变式被绕开（未 publish 的条目是垃圾页号） | 高 | A3 运行期断言 + 每次 admit 覆盖完整范围 |
| R4 | host KV / session retention 的 handle 向量变成 per-request | 中 | 保留 `kv_lane_handles[lane]` 语义为「该 lane 当前请求的页」；store/restore 都发生在请求驻留期内，`pages_for_tokens(frontier) <= reserve_pages` 恒成立 |
| R5 | 释放与同批新请求复用同页的时序 | 中 | 释放只改 host 侧计数；复用页的写入与 host 拷贝在同一 `device.stream` 上顺序执行 |
| R6 | 碎片化（池够但无连续 run） | 中 | 正确性无碍（host 路径不要求连续）；文档记录「实际并发度可低于 lane 数」 |
| R7 | 实际并发度低于 lane 数（C=4 只容两个 744 页请求） | 中 | 写进 `docs/serving.md` 与启动日志 |
| R8 | MTP 池未同步动态化 ⇒ 静默越界写 | 高 | S1 必须同时改 MTP 池（同一 `reserve_pages` 语义）；或 S3 未完成前把 MTP 路线的上限压回静态均分（不采用，直接同改） |
| R9 | `tp2_sessions` 等既有失败被误认为回归 | 低 | 与基线 `Compare-Object` 逐行比对（既有 plain shared-system-prompt 失败保持不变） |

---

## 7. 分阶段步骤

| 阶段 | 内容 | 完成判据 |
|---|---|---|
| **S1** | 文本 + MTP 池动态预留/释放；`lane_admission_limit_`；接纳路径三处（plain 初批/plain 轮边界/spec 初批/spec 轮边界） | `ninja` 0；gate 11/11；forward/load 0；`tp2_sessions` 与基线一致；双卡冒烟 V1–V6 全绿（见 §9.2；单卡 `--devices 0` 半项未跑） |
| **S2** | `--lane-context`（`0` = 整池）+ 校验 + `docs/cli.md` | 选项可解析；越界值被拒；文档同步；见 §9.4（已完成） |
| **S3** | （S1 已含 MTP 池；此处只做剩余镜像）`mtp_page_handles` 死字段清理、host checkpoint 口径复核 | 无残留引用；见 §9.3（已完成） |
| **S4** | `docs/serving.md` / 本文件施工记录 / 验收记录 | 文档与代码一致；见 §9.1–§9.4（已完成） |

---

## 8. 验证方案

### 8.1 必过门禁
- `ninja -C build-win` exit 0；`tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 11/11；
- `ninfer_qwen3_5_tp2_forward_test.exe --artifact …` / `…_load_test.exe --artifact …` exit 0；
- `$env:NINFER_TEST_ARTIFACT=…; tp2_sessions_test.exe` 的 passed/FAIL 行与基线 `Compare-Object` 一致。

### 8.2 行为验收（双卡 `ninfer-serve --devices 0,1 --max-context 131072 --max-concurrency 4`）
| # | 场景 | 期望 |
|---|---|---|
| V1 | 两个请求各 `prompt 14811 + max_output_tokens 32768` 同时提交 | 两个都 200，且都跑完（今天第二个 400） |
| V2 | 四个同规模请求同时提交 | 两个运行、两个排队；第一个结束后排队者自动开始；无 400/499 |
| V3 | 单个 `prompt 90000` 请求（C=4） | 200（今天 400） |
| V4 | 长请求运行中到达短请求 | 短请求立即被接纳（P2.2 动态接纳不回归） |
| V5 | 单卡（`--devices 0`）与 C=1 | 输出与基线一致 |
| V6 | plain 路线 temperature 0 | 逐位确定 |

### 8.3 观测点
- 启动日志：共享池页数与「单请求最大预留」；
- `[tp2-lane] admit slot=… live=…`；
- 池不足时应出现回插队首的 trace（可选 env 开关）。

---

## 9. 施工记录

### 9.1 S1：共享池动态预留/释放（已提交 `d0931a2d`；基线 `HEAD=94e6cc64`）

**改动**
- 头文件：`Shard` 增 `mtp_lane_pages`；删 `lane_token_capacity_` → `lane_context_limit_`；`lane_context_window()` 拆成 `lane_admission_limit()`（`lanes_==1 ? options_.max_context : lane_context_limit_`）+ `lane_kv_margin()`（MTP 时 `mtp_drafts_+2`，否则 1）+ `lane_kv_write_tail()` + `lane_kv_window(pages)`；新增 5 个声明。
- ctor：静态切分算术 → `pool_tokens = pages_for_tokens(max_context) * kPagedKVPageSize`、`lane_context_limit_ = pool_tokens - lane_kv_margin()`（`pool_tokens <= margin` 抛错）。
- 文本/MTP 启动块：`lanes_ == 1` 保持「整池一次 reserve+materialize+publish」；`lanes_ > 1` 只 `acquire` 每 lane 的 execution row（不 reserve、不 publish），日志改 `[mem] shard %d KV pool %u pages (%u tokens) shared by %d lanes` / `[mem] shard %d MTP KV pool %u pages shared by %u lanes`。启动期 A3 断言删除（不变式移入 `reserve_lane_kv`）。
- `reserve_lane_kv`：先清该 lane 可能残留的 lease（失败批次自愈）→ `pages_for_tokens(need + margin)` → 运行期 A3 断言 → 两 shard（`mtp_enabled_` 时再加 shard 0 的 MTP 池）逐个 `reserve`，任一失败立即 `return false`（已拿的由 reservation 析构还池）→ 每个池 `materialize` + `publish(row, 0, handles, stream)`。
- `release_lane_kv`：清两 shard 的 lease/handles 与 shard 0 的 MTP lease/handles；清该 lane 的 retention（`cached_*`、`cached_state_valid`、`live_state_valid`、`reuse_source`、`anchor_session`、`block_anchor_*`、`invalidate_host_checkpoints`）；把所有 `sessions_[].device_lane == lane` 置 -1；`active_session = kNoSession`。
- `retire_lane_session`：`session_store_active(lane)` 成功即返回，否则 `session_evict_one()` 重试；**不调用 `session_drop`**（`sessions_` 是 vector，drop 会让 `active_session` 索引失效）。
- 执行器：`LaneState` 增 `kv_window`；初批与轮边界接纳先 `reserve_lane_kv`，失败 ⇒ `requeue_lane_front`（初批再从 `batch` erase，否则驱动收尾安全网会 publish 空结果）+ `break`/`continue`；`admit_lane` 返回 true 后才写 `lanes[index].kv_window`（`admit_lane` 开头 `lane = LaneState{}` 会清它）；`finalize` 在 `session_publish` 之后、`publish_lane` 之前调 `retire_lane_session` + `release_lane_kv`；spec 轮的 `capacity_left` 改用该 lane 自己的 `kv_window`；`execute_walk`（单 lane 不可达路径）改用 `lane_admission_limit()`；driver `catch` 里对 `0..lanes_-1` 兜底 `release_lane_kv`。
- `docs/serving.md:68-76` 改为共享池动态预留口径。

**写码时确定的非显然点**
- 释放必须同时失效 lane 的「设备驻留」声明：物理页立刻可能被别的 lane 拿走，而 `scan_lane_reuse`/`session_recall` 的三种来源（LiveState/DeviceSnapshot/HostCheckpoint）都以「本 lane 设备 KV 仍持有该前缀」为前提 ⇒ 不失效会静默跳过 prefill 读别人的 KV。
- 退役时先 `session_store_active` 再释放，跨请求复用改走 host slab（下一轮 `session_recall` → `session_restore` 拷回新页）；`active_session = kNoSession` 让该 host 条目重新成为候选（`:5120` 只在 `index == active_session` 时跳过）。
- 接纳上限 = 整池 - margin ⇒ 只要 `prompt + effective <= lane_context_limit_` 就必然能在一张空池里放下，「池子不足」只可能是等别的 lane 退役，不会死锁。

### 9.2 S1 验证记录（2026-10-01/02，基线 `HEAD=94e6cc64`；随 `d0931a2d` 提交）

**门禁（全部通过）**

| 项 | 结果 |
|---|---|
| `ninja -C build-win` | `BUILD_EXIT=0`（43 步，含 `tp2_generation_core.cpp.obj`） |
| `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` | `100% tests passed out of 11`，70.54 s，`GATE_EXIT=0`（5 个 artifact 驱动用例 Skipped 属预期） |
| `ninfer_qwen3_5_tp2_forward_test.exe --artifact …` | exit 0；`duplicated-lane control: lane_gap=0 solo_gap=2.03125 batch_lane0_gap=0 argmax_mismatches=0`；两 lane verify 窗口 `max_logit_diff=2.03125 argmax_mismatches=1 near_tie_flips=1 legacy_vs_lane0_diff=0` |
| `ninfer_qwen3_5_tp2_load_test.exe --artifact …` | exit 0；`TP-2 dual-shard load passed devices=0,1 layers=64 shard_gate=[8704,5120] shard_down=[5120,8704] head=[124160,5120] vocabulary-parallel` |

**双卡行为冒烟**（`ninfer-serve --devices 0,1 --max-context 131072 --max-concurrency 4 --spec dflash2 --draft-tokens 5 --host-kv-mib 20480`，port 8099，`POST /v1/responses`）

启动日志从旧口径变成新口径（两个 shard 都是）：

```
[mem] shard 0 KV pool 2048 pages (131072 tokens) shared by 4 lanes
```

（旧：`[mem] shard 0 KV pages 2048 over 4 lanes = 512 pages (32768 tokens) per lane`）

| 场景 | 结果 |
|---|---|
| 单请求 `prompt 39804`（> 静态上限 32768）+ `max_output 16` | **HTTP 200**，27.0 s（旧代码必 400） |
| 两个并发请求，各 `prompt 39804 + max_output 32768` | **两个都 200**；27.5 s / 54.7 s ⇒ 第二个等第一个退役后自动开始（排队而非 400） |
| 单请求 `prompt 89546` + `max_output 16` | **HTTP 200**，71.6 s（旧代码必 400） |

预留算术自洽：`need = 39804 + 32768 = 72572` ⇒ `pages = pages_for_tokens(72573) = 1134`，`2 × 1134 = 2268 > 2048` ⇒ 必然串行，与观测到的「第二个耗时约 2 倍」一致。`status=incomplete` 是 `max_output_tokens=16` 撞顶，不是错误。图口径 `[tp2-graph] plain decode step: eager(exact) | batched: graph | verify step: eager | verify batch: graph | mtp chain: n/a`；`engine ready 30.9 s` / `warmup complete 1.2 s`。

**行为验收（V1–V6，双卡 `--max-concurrency 4`，`POST /v1/responses`）**

| # | 场景 | 结果 |
|---|---|---|
| V1 | 两个并发，各 `prompt 39804 + max_output 32768` | 都 200；27.5 s / 54.7 s（第二个等第一个退役后自动开始，非 400） |
| V2 | 四个并发（各 `prompt 19968 + max_output 32768`） | 全部 200；`25474 / 52510 / 38471 / 51652 ms`（两波），trace `[tp2-lane] admit slot=0 live=2`，无 400/499 |
| V3 | 单个 `prompt 89546`（C=4） | 200，71.6 s（静态切分下必 400） |
| V4 | 长请求运行中到达短请求 | 短请求被立即接纳：long 27324 ms / short 23861 ms |
| V5 | C=1（同二进制 `--max-concurrency 1`，S1 在 `lanes_ <= 1U` 全部早退） | 与 C=4 行为一致：同一复用探针在 C=1/C=4 都得 `cached=5749`；`tp2_sessions` 行与基线逐字一致。**单卡 `--devices 0` 半项未跑**（TP-2 artifact 是双 shard 布局，单卡是另一套 artifact/布局） |
| V6 | plain 路线 `temperature 0` | 答案通道逐位确定：顺序 2 次、并发 2 次 × `reasoning.effort` `none`/`xhigh` 四种组合下 message 文本哈希全同（`Blue` / `Red`）；思考通道只在同一 batch 形状内可复现（顺序两次同哈希、并发一对同哈希，但顺序 vs 并发 178 vs 193 字符）⇒ 与「批内 attention envelope 不同 ⇒ ulp 级近邻并列可翻转」的既有口径一致，非 S1 引入 |

**`tp2_sessions` 基线比对**（`NINFER_TEST_ARTIFACT=D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer`）

- 默认 plain：`unaligned dialogue passed`、`replayed answer passed`、`FAIL (plain): a conversation behind a shared system prompt diverged from the oracle on its first sample: got [2752 13 198 197 197 92 198 197] expected [467 419 538 13 198 197 197 92]`，`EXIT=1`。
- `NINFER_TEST_ROUTE=dflash2`：`unaligned dialogue (dflash2) passed`、`replayed answer (dflash2) passed: the turn after a re-rendered answer reused 70 of 58 prompt tokens`、`FAIL (dflash2): a recalled conversation diverged from the oracle on its first sample: got [59399 475 327 363 62 16 15 15] expected [365 4577 287 62 16 15 15 15]`，`EXIT=1`。
- 两串 FAIL 与 `docs/PLAN-tp2-concurrency.md:659` 记录的 HEAD 基线**逐字相同**（且 S1 在 C=1 全部早退）⇒ 无 sessions 回归。同源既有问题见 `docs/tp2-dual-5060ti-worklog.md:6549-6555`（`dflash_solo` 的 `a_continued` 首采样不一致，已 `git stash` 验证与当时改动无关）。

**`--spec mtp` 多 lane 冒烟**（`--spec mtp --draft-tokens 2`，C=4）

- 启动日志 `[mem] shard 0 MTP KV pool 2048 pages shared by 4 lanes`（MTP 池只挂在 shard 0，同样共享）；`[tp2-graph] … mtp chain: n/a`（多 lane 不建链，P3.1 门生效）。
- M1 单个 `prompt 39804 + max_output 16` ⇒ 200（24.5 s）；M2 两个并发 `prompt 19968 + 16` ⇒ 都 200（21.7 / 22.1 s）；M3 两个并发 `prompt 39804 + max_output 32768` ⇒ 都 200（25.5 / 50.7 s，池不足 ⇒ 串行）；M4 两个并发 tools 请求 ⇒ 都 200 且各产出 `get_weather` 调用（工具文法在批内多 lane 正常）。

**跨请求复用往返（S1 的核心行为变化）**

- 探针 `_temp/s1_diag.mjs`：R1 = 长 prompt，R2 = 把 R1 的 `output` 数组原样回放 + 一个追问（只有这种形状能命中 entry 的 frontier 边界）。
- C=4：`[tp2-session] entry 0 tokens=5750 frontier=5749 kv_end=5749 prompt_end=5718 shared_end=0 shared=5750 reach=5749 via=frontier resident=0` + `recall frontier=5749 resident_depth=0 tokens=5750 entries=1`；usage `cached_tokens=5749`，耗时 3964 → **511 ms**。
- C=1 对照（同二进制）：同样 `cached=5749`（383 ms）⇒ `release_lane_kv` → `session_store_active` → `session_recall` → `session_restore` 的往返在多 lane 路线上确实工作，且与单 lane 行为一致（S1 的核心风险「释放后复用静默失效」被证否）。
- 反例（既有语义，非 S1 引入）：客户端自己重新渲染助手回合（`asst('ok')`）时 `shared=5717` 比 R1 的 prompt（5718）少 1（chat template 只在最后一个 user turn 后追加 generation prompt），`offered[kind] > shared` 把 frontier / prompt_end / shared_end 三个边界全拒 ⇒ `reach=0`、`cached=0`。该判据属 `session_recall`（`tp2_generation_core.cpp:5257-5381`）与 `scan_lane_reuse` 共有，S1 前后一致。

---

### 9.3 S3：镜像清理与口径复核（已提交 `d0931a2d`）

- 删除死字段 `Shard::mtp_page_handles`：它只在单 lane 启动块被 `clear`/`reserve`/`push_back` 填充，全仓无任何读取点——host checkpoint 导出读的是 `mtp_lane_handles[lane]`（`src/runtime/engine/tp2_generation_core.cpp:5059` 与 `:5163`），而 `mtp_lane_handles[0]` 已经持有同一批 handle。头文件里「`mtp_pages` 与 `mtp_page_handles` 是单 lane 路线 pin 住、host checkpoint 导出用的整池列表」的注释随之改成只提 `mtp_pages`（保活整池 lease）与按 lane 索引的四件套（`mtp_lane_pages`/`mtp_lane_handles`/`mtp_rows`/`mtp_views`）。
- host checkpoint 口径复核（双卡 `--spec mtp --max-concurrency 4` 启动日志）：`[mem] host checkpoint ring 32 slots x 73.4 MiB/lane/shard`、`[mem] host-checkpoints shard 0 slots 32 (grid 3 + tail 3 + divergence 1 + block 1) x 73.4 MiB | stride 43776 tok | pinned 2349.0 MiB`（32 × 73.4 = 2349 ✓），与 S1 之前一致——S1 只改变「什么时候拍镜像」，不改变每 lane 的槽位几何。
- 回归：`ninja -C build-win` exit 0（43 步）；`tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 11/11（70.67 s）；`--spec mtp --max-concurrency 1` 单 lane 启动 + 一次 `POST /v1/responses` ⇒ 200（正是被改的启动分支）。

---

### 9.4 S2：`--lane-context` 上限旋钮（已提交 `d0931a2d`）

**语义**：`0`（默认）= 整池，S1 行为逐字节不变；`>0` = 一个 lane 最多能接纳多少 token（对齐 llama.cpp `--kv-unified-per-slot`），用来按策略收窄「先到的大请求独占整池」；`--max-concurrency 1` 路线忽略它（单 lane 本来就独占上下文）并在启动日志明说。

**改动点**
- `include/ninfer/types.h:181` 新增 `std::uint32_t lane_context = 0;`（`EngineOptions`，公共 API）。
- `src/serve/serve_options.h:37` 同名字段；`src/serve/serve_options.cpp:181` 解析 `--lane-context N`；`:403` 校验 `lane_context <= max_context`，否则 `--lane-context must be 0 or at most --max-context`；usage 文本加 `[--lane-context N]` 与一行说明。
- `src/serve/generation_service.cpp:243` 映射进 `EngineOptions`。
- `src/runtime/engine/model_instance.cpp:65`（`validate_options`）与 `:112`（`normalize_engine_options` 的 TP-2 分支）各加一条同样的校验：越界值**报错而不是静默夹紧**，保证操作员的数字与核心对外宣称的上限一致。
- `src/runtime/engine/tp2_generation_core.cpp:451-463`：在 `lane_context_limit_ = pool_tokens - margin` 之后按策略取小（`lane_context != 0 && lane_context < limit` ⇒ `limit = lane_context`），并打印 `[mem] TP-2 lane admission ceiling %u tokens (--lane-context %u)`；`:471-478` 在 `lanes_ == 1U && lane_context != 0` 时打印 `[mem] --lane-context %u is ignored at --max-concurrency 1: the single lane owns the whole context`。
- 测试：`tests/test_serve_options.cpp:164-176`（解析进 ServeOptions + 超过 `--max-context` 被拒）、`tests/test_engine_options.cpp:107-125`（归一化保留 + 越界被拒）。
- 文档：`docs/serving.md:851`（选项表新行）、`docs/serving.md:1057-1060`（准入段落补策略说明）、`docs/cli.md:279-281`（KV 段落指向 serving）。

**验收**（`ninja -C build-win` exit 0，586 步；`test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 11/11，含两条新断言；脚本 `_temp/run_s2_verify.ps1`，日志 `_temp/s2_c4.err` / `_temp/s2_c1.err`）

| 场景 | 命令 | 结果 |
|---|---|---|
| 启动日志（C=4） | `--max-concurrency 4 --lane-context 32768` | `[mem] TP-2 lane admission ceiling 32768 tokens (--lane-context 32768)`，同场 `[mem] shard 0 KV pool 2048 pages (131072 tokens) shared by 4 lanes` |
| 大 prompt（39340 tok + max_output 16） | 同上 | **HTTP 400**（上限生效；同一量级 prompt 在默认 `0` 下是 200——S1 的 V3 用 89546 tok 已证） |
| 中 prompt（19010 tok + max_output 4096） | 同上 | HTTP 200，`status=completed`，14.6 s |
| 启动日志（C=1） | `--max-concurrency 1 --lane-context 32768` | `[mem] --lane-context 32768 is ignored at --max-concurrency 1: the single lane owns the whole context` |
| 大 prompt（39340 tok + max_output 16） | 同上 | **HTTP 200**（C=1 忽略上限，整池可用），26.2 s |

**未覆盖**：`--lane-context` 的公平性/碎片化收益未量化（R1 的量化留到有真实多请求负载时再做）。

---

### 9.5 操作面补充：400 报文算术、容量告警、架构文档例外说明（已落盘）

**背景**：S1 把「每个 lane 固定 1/lanes 上下文」换成共享池后，操作员拿到 400 时无法从报文判断是「策略上限」还是「池子本身」，长 prompt 撞顶导致输出被截断也没有任何日志。本节补上这三处（用户 m01348 批准）。

- **400 报文带算术**（`src/runtime/engine/tp2_generation_core.cpp:2017-2043`）：`submit` 的 `context_length_exceeded` 分两支——
  - 策略绑定（`--lane-context` 就是上限）：`prompt exceeds this lane's context capacity: prompt 39340 tokens > lane ceiling 32768 tokens (--lane-context 32768); raise --lane-context to widen it`
  - 池子绑定（默认整池，**示例报文**；该分支只在 `(pool_tokens − margin, max_context]` 窄带内可达，见发现 2，故未做端到端实测）：`… prompt 17200 tokens > lane ceiling 16383 tokens (KV pool 256 pages x 64 tokens = 16384 tokens minus a 1-token write margin, --lane-context 0 = whole pool); raise --max-context to widen it`
- **前端准备阶段的 400 也带 token 数**（`src/models/qwen3_5/frontend/frontend.cpp:256-263` 的助手新增 `prompt_tokens`/`prompt_count_is_capped` 参数，调用点 `:849`（`encoded.input_ids.size()`，cap 命中 ⇒ `at least`）与 `:922`（`token_ids.size()`，精确值）；`src/models/qwen3_5/frontend/processor.cpp:935-940`、`:1066-1071` 同样补 `prompt at least N tokens`）：`prepared prompt exceeds Engine max_context 16384: prompt at least 16385 tokens`。这是 prompt 超 `--max-context` 时操作员唯一能看到的报文（见下文发现 2），tokenizer 上限使其只能给下界，故写 `at least`。
- **容量告警**（同文件 `:2044-2055`）：`limit_reason == FinishReason::ContextCapacity`（上限而非客户端决定了输出预算）时打印一行 `[tp2-capacity] prompt %u + requested output %u exceeds the %u-token lane context ceiling; the output budget is clamped to %u`；未截断的请求不打印。
- **架构文档例外说明**（`docs/maintainer/engine-architecture.md:45-51` 新增一段）：写明 TP-2 多 lane 是「同一份 paged KV 池 + 接纳时预留 prompt+预算+写入余量 + 池子不足留队首 + 唯一 400 是 prompt 超上限 + `--lane-context` 收窄 + 不抢占」的具体形态，并指出公平性由操作员的策略保证而不是调度器。
- **文档同步**：`docs/serving.md:415-417` 的 400 段落补一句 TP-2 口径（上限是 lane 的接纳上限：默认整池，`--lane-context` 可收窄，报文说明是哪一支在约束）；`docs/PLAN-tp2-concurrency.md:287` 的护栏条目注明报文后来改为带算术的两支。

**验收**：`ninja -C build-win` exit 0；门禁 11/11；双卡冒烟（`_temp/run_s2c_verify.ps1`、`_temp/run_s2d_verify.ps1`）——

| 场景 | 期望 | 实测（2026-10-02） |
|---|---|---|
| C=4 `--max-context 131072 --lane-context 32768`，prompt 39340 + 16 | 400，策略绑定报文 | HTTP 400（72 ms）：`prompt exceeds this lane's context capacity: prompt 39340 tokens > lane ceiling 32768 tokens (--lane-context 32768); raise --lane-context to widen it` |
| C=4 `--max-context 131072 --lane-context 32768`，prompt 19010 + 32768 | 200，且 `[tp2-capacity]` 一行 | HTTP 200 completed（14.5 s），`[tp2-capacity] prompt 19010 + requested output 32768 exceeds the 32768-token lane context ceiling; the output budget is clamped to 13759`（= 32768 − 19010 + 1） |
| C=4 `--max-context 16384`（默认整池），prompt 长于 16384 | 400，报文给出 prompt 下界 | HTTP 400（38 ms）：`prepared prompt exceeds Engine max_context 16384: prompt at least 16385 tokens` |
| C=1 `--max-context 16384`，prompt 9510 + 16384 | 200，告警按单 lane 上限打印 | HTTP 200，`[tp2-capacity] prompt 9510 + requested output 16384 exceeds the 16384-token lane context ceiling; the output budget is clamped to 6875`（= 16384 − 9510 + 1） |

**验证中发现并回填的两件事**：

1. `--max-context` 的**服务端默认值是 8192**（`src/serve/serve_options.h:32`）。第一轮脚本把 `--max-context` 从公共参数里拿掉后，C=4 `--lane-context 32768` 的服务器启动即退（exit 1），stderr 首行 `ninfer-serve: --lane-context must be 0 or at most --max-context`。所以「策略绑定」分支的验收必须显式给 `--max-context 131072`。
2. **prompt 超过 `--max-context` 时，前端准备阶段先拒绝**（`src/models/qwen3_5/frontend/frontend.cpp:846`、`src/models/qwen3_5/frontend/processor.cpp:935-940` 与 `:1066-1071`），因此核心的**池子绑定分支只在 `(pool_tokens − margin, max_context]` 这个窄带内可达**（带宽 = `margin − (pool_tokens − max_context)`：dflash2 `margin=1`、页对齐余量 0 时正好 1 token；MTP `--draft-tokens 15` 时最多 17 token），实际是防御性分支。对操作员真正有用的是前端那条，本次也给它补了 token 数；由于前端 tokenizer 以 `max_context + 1` 为上限（`encode_rendered_chat(..., impl_->max_context + 1U)`、`EncodeOptions{.max_tokens = encode_limit}`），计数只能给下界，报文因此写 `prompt at least 16385 tokens`。

### 9.6 `kReuseSnapshotCount` 2→1：设备端只留 prefill 末态边界（未提交，m01730）

**动机**：`state_arena = std::make_unique<DeviceArena>((2 + kReuseSnapshotCount) * state_bytes)`（`src/runtime/engine/tp2_generation_core.cpp:836-837`）与 `dflash_snapshot_arena = std::make_unique<DeviceArena>(kReuseSnapshotCount * dflash_image)`（`:1026`）说明每 shard 常驻 `kReuseSnapshotCount + 2` 份整平面 state（live + 各复用边界 + round scratch）与 `kReuseSnapshotCount` 份草稿环。C=4 相对 C=1 的 2168 MiB 里 1334 MiB 是这批快照的账（2 份 state 边界 1174.5 + 1 份草稿环 160），只有约 834 MiB 是多并发本身。

**为什么 1 是下限**：`kRoundScratchSlot = kReuseSnapshotCount`（`src/runtime/engine/tp2_generation_core.h:68`），而 `publish_lane_prefill` 用 `state_snapshots[0]` 存 prefill 末态（`tp2_generation_core.cpp:2514-2525`），slot 0 同时是会话 slab 里 prompt 末态镜像的唯一来源（`session_store_active` → `entry.host_prompt_state`/`entry.host_dflash_prompt`，`:5079-5100`）。再降 0 会让 round scratch 与 prefill 末态抢同一个平面，所以本次只砍 rewind 槽（slot 1..count-1）。

**改动点**（一行常量 + 5 处边界口径）

| 位置 | 改动 |
|---|---|
| `src/runtime/engine/tp2_generation_core.h:55-64` | `kReuseSnapshotCount = 2` → `1`，注释改成「rewind 不值一个设备平面：每槽每 shard 又是一份整平面 state，主机 checkpoint 环已持有同一批状态，slot 0 因会话 slab 读 prompt 末态镜像而保留，故 1 是下限」 |
| `src/runtime/engine/tp2_generation_core.h:274-277` | `dflash_snapshots` 注释去掉「slot 1 是它后面的 rewind」 |
| `src/runtime/engine/tp2_generation_core.cpp:1016` | 草稿环 arena 注释 "the two reuse boundaries" → "the reuse boundaries" |
| `src/runtime/engine/tp2_generation_core.cpp:6045-6050`、`:6063` | `[tp2-reuse]` 跟踪里 `cached_boundaries[1]` 的越界读改成循环取最深的 rewind 槽（`rewind_boundary`） |
| `src/runtime/engine/tp2_generation_core.cpp:6465-6470` | 取消路径 `cached_boundaries[1] = 0` 的越界写改成循环清零 |

**语义后果**
- rewind 槽的所有循环本来就是 `for (slot = 1; slot < kReuseSnapshotCount; ++slot)`（`:6048`、`:6340`、`:6468`、`:6500`、`:6559`），count=1 时全部空转：既没有设备端 rewind 平面，也没有 `snapshot_at[1]` 对 prefill chunk 的夹紧（`:6500-6503`）。
- **C=4 上这份平面是死分配**：`release_lane_kv`（`:2664`）在 `lanes_ > 1` 时每个请求终态都被调用，会清 `cached_boundaries` 与 `cached_state_valid`，因此 `scan_lane_reuse`（`:2331-2400`）的整个边界扫描从第二轮起就不进入（trace `src=none`）。C=4 的跨轮复用走 `session_recall`/`session_restore` 主机 slab，与本次改动无关。
- **C=1 才是唯一语义变化点**（`release_lane_kv` 对 `lanes_ <= 1U` 早退）：落在「上一轮 prompt 末尾一个 chunk 之内、但不等于末态」的复用请求，改由主机 checkpoint 环的 tail 子环给边界（`[mem] host-checkpoints shard N slots 34 (grid 24 + tail 8 + divergence 1 + block 1) x 73.4 MiB | stride 8192 tok`，tail 覆盖末尾 8192 token）。**复用深度不变，只是恢复源从设备快照（D2D）变成 H2D**（73.4 MiB state + 40 MiB 草稿/lane）。

**验收**（`tools/win_port/build.ps1` BUILD_EXIT=0，增量 29–30 s；门禁 `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 11/11，71.48 s）

显存（同一 binary 与参数，只差 `--max-concurrency` 与常量；count=1 日志 `_temp/cnt1_c1.err`/`_temp/cnt1_c4.err`，基线 `_temp/s2_c1.err`/`_temp/s2_c4.err`）：

| 项目 MiB | count=2 基线 | count=1 | 差 |
|---|---|---|---|
| C=4 shard 0 `state` | 1174.5 | 880.9 | −293.6 |
| C=4 shard 0 `draft-snap` | 320.0 | 160.0 | −160.0 |
| C=4 shard 0 `free` | 726.0 | 1180.0 | +454.0 |
| C=4 shard 1 `state` | 1174.5 | 880.9 | −293.6 |
| C=4 shard 1 `free` | 1424.0 | 1718.0 | +294.0 |
| **C=4 两卡合计** | — | — | **+748** |
| C=1 shard 0 `state`/`draft-snap`/`free` | 293.6/80.0/1996.0 | 220.2/40.0/2108.0 | +112 |
| C=1 shard 1 `state`/`free` | 293.6/2322.0 | 220.2/2394.0 | +72 |
| **C=1 两卡合计** | — | — | **+184** |

⇒ **C=1→C=4 的增量从 2168 MiB 降到 1420 MiB**（DFlash2 round 行同步缩小：C=1 `ring 40.0 | context 90.3 | frame 3.4 | proposal 3.3` → C=4 `ring 160.0 | context 211.2 | frame 12.1 | proposal 4.0`）。

行为（复用深度不变，A/B 同一探针 `_temp/cnt1_rewind.mjs`）：

| 场景 | count=2 | count=1 |
|---|---|---|
| C=1 rewind 带（R2 = FILLER×285 + 另一条尾句，5431 tok / shared 5418） | `cached=5120`（707 ms），trace `reuse=5120 slot=1 src=device` | `cached=5120`（648 ms），trace `reuse=5120 slot=28 src=host`（`host=5/34 stride=8192`） |
| C=4 双轮 recall（`_temp/s1_diag.mjs` 探针） | `cached=5749`（511 ms） | `cached=5749`（470 ms） |

双卡 artifact 链（`_temp/run_cnt1_artifact.ps1`）：`tp2_forward_test` exit 0（`duplicated-lane control: lane_gap=0 solo_gap=2.03125 batch_lane0_gap=0 argmax_mismatches=0`；两 lane verify `max_logit_diff=2.03125 argmax_mismatches=1 near_tie_flips=1`，与基线逐字相同）；`tp2_load_test` exit 0；`tp2_sessions_test` 完整日志里只有一条 FAIL——plain divergence 用例，输出与 HEAD 基线逐字节相同（`docs/tp2-dual-5060ti-worklog.md:6549-6554`）⇒ 无回归（本次 sessions 运行是 plain 路线，dflash2 用例未跑）。

**未覆盖**
- rewind 复用的恢复从 D2D 变 H2D，本次只证明复用深度与端到端延迟同档，未做 PCIe 带宽级分项计时。
- vision prompt 的 chunk 夹紧语义与纯文本不同（`:6490-6503` 的 vision cap 与 snapshot clamp 顺序），未构造 vision 复用用例。
- 回退方式：常量改回 2 并重建（数组尺寸与槽循环自动跟随）。

---

## 10. 未决问题与已知限制
- ~~`--lane-context` 的默认值~~（已定，m01348）：默认 `0` = 整池，即 S1 的「先到的大请求可以独占整池」；需要按策略收窄时由操作员显式给值。不再改默认。
- ~~400 报文带 lane 容量算术 + `prompt + max_output` 超预算告警~~（**已做**，m01348 第 3 项）：见 §9.5；核心报文分「策略绑定 / 池子绑定」两支，前端准备阶段报文补 `prompt (at least) N tokens`，输出预算被上限夹紧时打印 `[tp2-capacity]`。
- ~~`docs/maintainer/engine-architecture.md:41` 的 TP-2 例外说明~~（**已做**，m01348 第 3 项）：见 `docs/maintainer/engine-architecture.md:45-51`。
- 碎片化的量化（不同请求尺寸序列下可容纳的最大请求）尚未测量。
- **已知代价（S1 引入）**：每个请求终态都会走 `retire_lane_session` → `session_store_active`，把该 lane 的 frontier KV + state 镜像 D2H 到 host slab（满上下文时约 73.4 MiB/shard），下一轮复用再经 `session_recall` → `session_restore` 拷回新页。这是「页可立即易主」的直接后果，多轮会话的吞吐代价待量化（R4）。
- **已知代价（count=1，未提交，m01730）**：设备端不再保留 rewind 边界（`kReuseSnapshotCount = 1`），C=1 落在「上一轮 prompt 末 chunk 内」的复用改由主机 checkpoint 环的 tail 子环提供边界，恢复走 H2D（73.4 + 40 MiB/lane）；复用深度实测不变。C=4 不受影响（该平面本就是死分配）。回退：`src/runtime/engine/tp2_generation_core.h:64` 常量改回 2。详见 §9.6。
- **仍未验证**：单卡 `--devices 0` 半项（TP-2 artifact 是双 shard 布局，不适用）；碎片化的量化（不同请求尺寸序列下可容纳的最大请求）；R4 的多轮会话吞吐代价；vision prompt 的 rewind 复用用例。
