# TP-2 headroom 优化执行计划

依据：docs/tp2-concurrency-headroom.md 第 6 节优先级表。逐项实现、逐项验证、逐项记录。

## 执行顺序与状态

| 序 | 项 | 报告优先级 | 状态 |
|----|-----|-----------|------|
| 1 | plain batch round 计时埋点 | P1 | 代码完成，构建+测试通过 |
| 2 | plain 路线不雕 round-scratch 状态平面 | P1 | 代码完成，构建+测试通过 |
| 3 | 放宽 batch 形成窗口 | P0 | 代码完成，构建+测试通过 |
| 4 | 中途加入的 prefill 与存活 lane 的 decode 交错 | P0 | 代码完成，构建+测试通过，已实测（见下） |
| 5 | 初始 prefill 期间消费队列 | P0 | 与 4 合并后**回退**：证明为 no-op（理由见下） |
| 6 | --lane-context 多 lane 默认值/告警 | P1 | 代码完成（告警，不做硬默认），构建+测试通过 |
| 7 | TTFT/total 口径计入排队时间 | P1 | 代码完成，构建+测试通过 |
| 8 | 修正 docs/serving.md 的 262,144/C=1 指导 | P2 | 完成（1/2/3/4/6/7 已在 docs/serving.md 说明） |
| 9 | 输出预算按剩余量动态收缩/页回收 | P2 | 分析后**不改**（破坏 all-or-nothing 不变式），结论进报告 |
| 10 | 投机批链改用每 lane envelope | P2 | **已实施**：删除 batch 对 host split 数的截断 + 新增 batch/solo 逐位 parity 回归；A/B 显示 plain 中性、接受率收益未复现（噪声内）。见「项 C 实现与验证」 |

### 用户第二轮增补（m01111）

| 序 | 项 | 状态 |
|----|-----|------|
| A | spec/MTP 路线 prefill 交错（`pump_spec_round`） | 代码完成，构建通过，A/B 实测成立（见下） |
| B | spec 路线 `round_ms`/`rest` 计时口径（A 的测量前提） | 代码完成，构建通过 |
| C | 项 10 Ops 核按列 split（删除 batch 对 host split 数的截断） | 代码完成 + 新增 batch/solo 逐位 parity 回归测试；ops 测试通过（旧 ops 二进制该测试 FAIL）；A/B 实测完成——plain decode 中性、接受率收益未复现（见下） |
| D | P3 批量 prefill（顺带消除 admission 位置差） | 设计完成（_temp/20261002-1800_item3_batched_prefill.md），**未实现**：收益受 width≤16 与 GDN record 约束（见下） |
| E | 可分页 host checkpoint ring（报告 P3 行） | 代码完成（PinnedHostBuffer → HostBuffer(Pageable)），构建通过；模型测试（lanes 全过 / sessions 与旧基线逐字相同）；空指针回归已修；延迟实测完成——两卡省 5.29 GiB 页锁，recall 请求 +10 ms（噪声内） |

### 执行顺序说明

6 与 7 提到 4/5 之前：7 是 4/5 效果的测量口径（没有排队时间的 TTFT 无法证明交错有效），
6 独立且极小。这与「先做埋点」的既有做法一致。

## 验证方式（仓库验证表）

- C++ 改动：构建受影响目标（tools/win_port/build.ps1）+ tools/win_port/test.ps1 相关过滤测试。
- 性能/延迟：在声称范围实测（本机 2×RTX 5060 Ti，--devices 0,1 起服务）。
- 文档改动：链接 + `git diff --check`。

## 关键实现备忘（源码定位）

- 驱动线程与形成窗口：src/runtime/engine/tp2_generation_core.cpp:2277-2391，
  kBatchFormationWindow :2250，kBatchFormationMaximum 新增于 :2278。
- plain 路线：execute_plain_batch :3174，pump_decode_round lambda :3405，
  初始 admission（S1 循环）:3870-3887，主循环 :3890-4030，中途 admission :3903-3930。
- admit_lane（plain）:3554-3870；spec 对应 :4195-4514。
- 状态平面雕刻 :851-863；state_snapshots[kRoundScratchSlot] 只在 spec/walk 使用。
- lane-context :443-478（新增告警分支）。
- TTFT 口径：submit() :2108；两处 admit_lane 置 queue_wait；两处 finalize 求和；
  execute_walk :7151/:7156/:8021。

## 已确认的物理约束（决定了哪些项值得做）

- prefill 由 PCIe（卡 2 Gen4 x4，~7 GB/s）限制：0.368-0.375 ms/token，2.67k tok/s 天花板。
- decode 由 DRAM 权重流限制：C=4 时约 317 GB/s。
- 因此 prefill 与 decode 争用的是**不同**资源，并行理论上可行；但共享 workspace arena 与
  跨卡 in-kernel allreduce 的 rendezvous 是软件障碍。
- 4 条 10.8k prompt 的 prefill 串行总工作 = 22 s，任何调度都不可能更快；交错只改变
  延迟分布（每个 lane 的 TTFT 上界从「整段 prefill」降到「一个 chunk」）。

## 项 4/5 设计（已实施，plain 路线）

**调度策略**：任何时刻**只有一个** lane 在 prefill（保持既有串行 FIFO 顺序与每个 lane 当前的
TTFT），已持有 token 的 lane 在这个 lane 的两段 chunk 之间各跑**一个 decode round**。
轮转式（每个未完成 lane 每轮一个 chunk）被否决：那会让所有 lane 在聚合墙钟时刻同时完成
prefill，lane 0 的 TTFT 从 ~5.5 s 退化到 ~22 s。

报告里「短 lane decode 37.7 → 2.5 tok/s = 15x」是**延迟**假象（tokens/s 按整段阻塞 prefill
摊薄），不是单轮成本；chunk 级交错正好修掉这一点。GPU 总工作量不变。

**实现：pump 回调，不是拆分状态机**。把 decode round 主体提成
`execute_plain_batch` 内的 `auto pump_decode_round = [&]() { ... };`（定义在
`active.reserve(lane_capacity);` 之后、`admit_lane` 之前，因此 admit_lane 可直接调用，
不需要 std::function 也不需要前置声明）。主循环里的 round 主体改成调用它；
`admit_lane` 的 chunk 循环顶部、上一个 chunk 的 `scope_a/scope_b` 已析构之后，
加 `if (!active.empty()) { pump_decode_round(); <恢复 slot 绑定> }`。
`admit_lane` 仍是同步的、串行顺序不变。

**关键不变式**：
- decode round 必须精确从 `shard_a_.round_base`/`shard_b_.round_base` 开始（捕获的 CUDA Graph
  把 `arena_begin[0/1]` 烘进去了，`run_plain_decode_step_batch` 在
  `workspace->used() != graph->arena_begin[0]` 时抛
  `"TP-2 batched decode CUDA Graph replay found a different workspace layout"`）。
  所以 round_base 的捕获点从「reserve 循环之后」**前移**到「pump lambda 定义之后」。
- `admit_lane` 的 ws_a 净分配为 0（今天中途 admission 已经跑在
  `position_arena(ws_a, round_base, round_base)` 之前，任何持久的分配本来就会被回卷掉）。
- 一列 round 会把标量 GDN slot 对重新广播（src/models/qwen3_5/execution/text.cpp:2370-2378
  的 `batch == 1` 分支），所以 pump 返回后要恢复
  `set_linear_state_slots(active_lane_, active_lane_)`。
- 单实例 vision session 前提**不被破坏**：只有无媒体/带媒体 lane 各自串行，任何时刻仍只有
  一个 prefill 在跑。

**范围**：只做 plain 路线。spec 路线（execute_spec_batch）保留每 lane 串行 prefill，
在报告中记为剩余空间。

## 项 7 改动点（已实施）

- `TP2GenerationCore::Request`（src/runtime/engine/tp2_generation_core.h:102）加
  `std::chrono::steady_clock::time_point submitted{};`。
- `submit()`（.cpp:2108）置 `request->submitted = Clock::now();`。
- 两处 admit_lane 置 `lane.result.engine_timing.queue_wait_seconds =
  duration<double>(lane.begin - pending.request->submitted).count();`
- 两处 finalize + execute_walk：`first_token_seconds`/`total_seconds` 改为
  `prepare_seconds + … + queue_wait`（对齐单卡口径）；`prompt_wall_seconds` 与
  `prefill_seconds` **保持**从 admit 起算（它们定义 prefill tok/s 速率）。
  这条是必须的：src/serve/generation_service.cpp:447-457 会
  `ttft = prepare + max(0, first_token - prepare)`、`total = prepare + max(0, total - prepare)`，
  即 timings.total_seconds **必须**含 prepare。
- 下游无需改动：`GenerationEngineTiming::queue_wait_seconds`
  （src/runtime/engine/request_record.h:70-73）与 `| queue <dur>`
  （src/serve/operational_log.cpp:262-265）已存在。

## 项 5 结论（已实现后回退，勿再加回）

在 S1 初始 admission 循环末尾加 `try_pop_lane_queue()` 后编译通过，但分析证明它是 no-op：
S1 循环按 `batch.size()` 顺序 admission，循环结束后主循环**立刻**执行
`while (active.size() < lane_capacity) { try_pop_lane_queue(); ...; admit_lane(...) }`，
两者在同一墙钟时刻、同样严格串行地 prefill 新成员，TTFT 无差别，差别只是成员待在哪个
`batch` 向量里。报告中项 5 的前提（初始 prefill 期间不查队列）实际已被既有中途 admission
覆盖；2+2 分裂由项 3 修好，残余的「新来者等当前 prefill 的 lane」是单 lane 串行 prefill 的
固有性质，需 P3 批量 prefill。已回退，文件回到 8082 行。

## 项 4 实测结果（_temp/20261002-1442_pump.ps1，plain，int8，245760，C=4）

prompt：长 = longSys(160) = 15,310 token，短 = 54 token，max_output=24，reasoning=low。

| 场景 | 短 lane total | 短 lane decode | 机制 |
|------|---------------|----------------|------|
| 报告基线（无交错，B3） | 6.0 s | 2.5 tok/s | 整段 prefill 阻塞 |
| 交错，chunk 1024 | 7.5 s | 3.1 tok/s | 每 chunk（~0.55 s）1 个 decode round |
| 交错，chunk 256 | **2.8 s** | **8.5 tok/s** | 每 chunk（~0.19 s）1 个 round |

- 反向顺序（长先、短后）仍要等：短 lane queue 9.8-9.9 s，TTFT 9.9 s vs 短先时的 114 ms。
  这是「任何时刻只有一个 lane prefill」的固有代价，项 4 不改变 admission 顺序。
- 新埋点直接证明机制（chunk 256，NINFER_TP2_TIMING=1）：
  短先的长 lane 那次 batch 打 `prefill_chunks=61 pump_rounds=14`——短 lane 拿到 24 token
  后退出 active，pump 随之停止；长先短后打 `prefill_chunks=61 pump_rounds=1`；四长并发打
  `prefill_chunks=180 pump_rounds=46`。
- 项 3 生效：四长并发从报告的 `batch=2` 变成 `batch=4`（chunk 1024）/ `batch=3`（chunk 256，
  50 ms 上限仍把第 4 条切成第二波）；800 ms 间隔的 2+2 正确地不合并。
- chunk 256 的每 lane prefill 速率降到 1.34-1.52k tok/s（1024 时 1.69-1.94k），与报告成本模型一致：
  小 chunk 用 prefill 速率换 decode 平滑度。
- 客户端 SSE 在 reasoning + 24 token 下多数为 0 个 text delta，不能当 token 时钟；
  权威数据是 `done |` 与 `[tp2-time]` 行。

## 项 9 结论（不改，写入报告）

`reserve_lane_kv`（:3027）在 admission 前对两卡 + MTP cache 做**全量 all-or-nothing** 预留，
设计注释 :3004-3010 明说「一个池现在放不下就什么都不留，调用方把请求放回队首……
等待永远是等另一条 lane 退役，不会死锁」。按剩余量收缩或回收页面会让 lane 在 decode 中途
需要**增长**，而该路线没有 mid-request 背压/抢占（AGENTS.md：无 active-request preemption），
安全实现需要新增「按需驱逐并增长连续 KV window」路径，不是 P2 项的量级。实测 66 MiB/卡空闲
是「必然完成」保证的代价，且上界由客户端 `max_output_tokens` 决定；操作杠杆是 `--lane-context`
（项 6）或更小的客户端预算。报告按「已分析、不改、附理由」记录。

## 项 10 归因实测（配置：131072 + int8 + --spec mtp --draft-tokens 5）

| 组 | 接受率 |
|----|--------|
| 单条长（15,310）单跑 C=1 | 46/70 = **65.7%**（4.0 tokens/lane-round） |
| 四条长同批 C=4 | 45.7 / 54.2 / 34.7 / 38.0%（均 43.2%，3.0 tokens/lane-round） |
| 长 + 短 C=2 | 长 41.1%，短（54 token）**36.2%** |
| 单条短单跑 | 45.3% |
| 两条短同批 / 四条短同批 | 54.3 / 55.7% 与 53.3 / 55.3 / 43.2 / 42.4% |

结论：回归真实（宽窗口入批 65.7% → 34.7-54.2%，tokens/lane-round −25%），但判别实验证明
「批处理本身」不背锅（短窗口同批与单跑无差别）⇒ 机制是**宽窗口的批内 split policy**：
`mtp_forward_decode_batch` 的 envelope 只有一个宽度（tp2_generation_core.cpp:4815-4817 注释
「The envelope steers split policy only … the widest lane bounds every lane's window」）。
修复面在 Ops 核（每列 envelope / 按列 split policy），不在 engine；verify 是捕获的图，
改签名要重捕获并用 FP32/FP64 oracle 验证逐 token 一致性。**本次不做**，已列入报告剩余空间。
开放问题：触发者是「窗口绝对值宽」还是「lane 间位置差」——短窗口实验只能排除前者以外的批处理本身。

## 项 A 实测结果（spec/MTP 路线交错，_temp/20261002-1745_specpump.ps1）

配置：`--devices 0,1 --max-context 131072 --kv-dtype int8 --max-concurrency 4
--spec mtp --draft-tokens 5 --host-state-slots 32 --host-kv-mib 20480`；长 = longSys(160) =
15,310 token，短 = 54 token，max_output=64，reasoning=low。ARM C = 短先发、长 +400 ms；
ARM B = 长先发、短 +400 ms。

| 配置 | HEAD ARM C 短 lane | 交错 ARM C 短 lane | HEAD ARM C 长 lane | 交错 ARM C 长 lane |
|------|--------------------|--------------------|--------------------|--------------------|
| chunk 1024 | 总 8.8 s / decode 3.1 tok/s | **总 3.3 s / 8.9 tok/s** | 9.0 s | 9.5 s |
| chunk 256 | 总 11.1 s / decode 2.5 tok/s | **总 2.0 s / 15.7 tok/s** | 11.4 s | 12.1 s |

- ARM B（后到者）两边都不变：短 lane queue 7.6 s（chunk 1024）/ 10.0 s（chunk 256）。
  这是「任何时刻只有一个 lane prefill」的固有代价，交错不改变 admission 顺序。
- 长 lane 反而 +0.5-0.7 s（pump 开销 ~5%），总 GPU 工作量不变。
- 机制证据：`[tp2-time] spec-batch … prefill_chunks=16 pump_rounds=6`（交汇的短 lane 在长 lane
  的 16 个 chunk 之间被 pump 了 6 个 decode round）。
- 实现与 plain 的差异：`pump_spec_round` 定义前必须**上移** `shard_[ab]_.round_base` 捕获、
  `round`（dflash_round）、`dflash_execution`、`lane_extent`、`append_slots/starts/ends`、
  `mtp_pack`、`mtp_step_host`、`mtp_selectors_h`（admit_lane 对这些零引用）；chunk 循环顶部
  调用后要恢复 `set_linear_state_slots(active_lane_, active_lane_)`（spec 的 chunk 前向不自设 slot，
  靠 admit 入口的绑定；一列 round 会经 text.cpp 的 `batch == 1` 路径重发标量 slot）。

## 项 B 结论（spec 计时口径）

spec 路线从不累加 `round_ms`，`[tp2-time] spec-batch` 的 `avg_round` 恒为 0.00，无法作为
项 A 的轮成本证据。修正：

- 删除死成员 `Tp2RoundTiming::fold_ms` 与 3 处 `timing.fold_ms += timing.elapsed(5, 6);`
  （fold 在同步之后运行，round 内 `elapsed()` 读未完成的 event 对恒返回 0.00）；
  `report()` 的该字段由 `fold=` 改为 `rest=`，值 = `(round_ms - mtp - verify - accept - copy) / n`。
- `pump_spec_round` 加回合级 `round_start` 与 `timing.round_ms +=`（与 plain pump 同口径）。

## 项 E 设计（可分页 host checkpoint ring）

动机见 docs/tp2-concurrency-headroom.md:134-151 §2.5：C=4/262,144 时 ring = 36 槽 × 73.4 MiB =
2,642.6 MiB/卡（两卡 5.1-8.2 GiB 页锁），而它是 host KV 之外最大的一块 pinned 内存。

改动（沿用 `HostBuffer(HostPinning::Pageable)`，不留双路径）：

- src/runtime/engine/tp2_generation_core.h:195/:200：`HostCheckpoint::buffer`/`dflash_buffer`
  由 `unique_ptr<PinnedHostBuffer>` 改 `unique_ptr<HostBuffer>`。
- src/runtime/engine/tp2_generation_core.cpp:929/:1117：改 `std::make_unique<HostBuffer>(bytes,
  HostPinning::Pageable)`；`[mem]` 行加 `(pageable)`。
- 镜像传递路径改回**裸映射 + 长度**（与既有 `copy_lane_state` 的约定一致）：
  `store_dflash_image(Shard&, void*, std::size_t, lane)`、`load_dflash_image(Shard&, const void*,
  std::size_t, lane)`、`session_capture_shared_state(..., const void* const* frozen,
  const void* dflash_frozen, lane)`，相应的 `frozen[2]`/`frozen_draft`/`block_draft` 局部变量改
  `const void*`，调用点传 `->data()`/`->size()`。理由：ring 槽（可分页）与 session slab
  （pinned）共用同一条拷贝路径，PinnedHostBuffer 与 HostBuffer 无共同基类。
- 正确性：ring 不在任何 CUDA Graph 捕获区；所有访问只经 `->data()/->size()`；
  `copy_lane_state` 用 cudaMemcpy2DAsync（pageable 合法，代价是驱动内部 staging + 宿主线程阻塞，
  即 store/recall 变慢，而非错误）。无测试断言 pin/槽数。
- 验证口径：构建 + `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'`
  + 实机（`--host-state-slots 32` 时 ring 35/36 槽）对比 prefill tok/s 与 store/recall 耗时。
- 文档同步：docs/serving.md、docs/tp2-dual-5060ti.md、docs/tp2-concurrency-headroom.md:134-151/:383/:558；
  docs/PLAN-tp2-concurrency.md:342 的「lanes=4 ⇒ 9.98 GiB/shard」是过时初稿。

## 项 E 实现与验证

已按上述设计落盘（不留双路径）：h:195/:200 改 `unique_ptr<HostBuffer>`；cpp:929/:1117 改
`make_unique<HostBuffer>(bytes, HostPinning::Pageable)`；`store_dflash_image`/`load_dflash_image`/
`session_capture_shared_state` 的镜像参数改「裸映射 + 长度」，`frozen[2]`/`frozen_draft`/`block_draft`
改 `const void*`；`[mem]` 行加 `(pageable)`。构建 BUILD_EXIT=0（26-28 s，多次）。

- 模型测试（`NINFER_TEST_ARTIFACT=D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer`）：
  `ninfer_qwen3_5_tp2_lanes_test.exe` 全过（LANES_EXIT=0；mid-batch arrival 0.729842 s / 3.89515 s，
  对照既有基线 0.59 s / 3.67 s，同量级）；`..._sessions_test.exe` 仍失败，但失败行与旧 pinned 基线
  （_temp/cnt1_sessions_full.log）**逐字相同**，且两日志的 `^TP-2 ` 行 Compare-Object 无差异
  ⇒ 无新增失败（该测试在首个 FAIL 处即停，旧日志亦然）。
- **完整模型测试口径 + HEAD 基线对照（最终二进制）**：
  `$env:NINFER_TEST_ARTIFACT='D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer'; & tools/win_port/test.ps1 -Filter 'qwen3_5_tp2'`
  → 67% passed out of 6（load/forward Skipped、sessions Failed、dflash_solo Failed、
  dflash_append Passed 47.51 s、lanes Passed 69.56 s）。把全部改动 stash 掉重建 HEAD（464 s）后同 filter 复测，
  **两个失败逐字复现**：sessions 的 `got [2752 13 198 197 197 92 198 197] expected [467 419 538 13 198 197 197 92]`、
  dflash_solo 的 `the recalled walk diverged from the from-scratch walk on its first sample:
  [59399 475 327 363 62 16 15 15] vs [365 4577 62 16 15 15 15 15]`
  ⇒ 均为**既有失败**，与本轮改动无关。证据 _temp/20261002-1745_modelverify_new.md。
- 运行期证据：`[mem] host checkpoint ring 36 slots x 73.4 MiB/lane/shard (pageable)`、
  `[mem] host-checkpoints shard 0/1 slots 36 (...) x 73.4 MiB | stride 43776 tok | pageable 2642.6 MiB`
  （对照旧行 `... | pinned 734.1 MiB`，11 槽）。
- **空指针回归（已修）**：把 `unique_ptr` 句柄改成「裸指针 + 长度」时，`checkpoint.dflash_buffer->data()`
  在 `dflash_buffer == nullptr` 的 MTP 路线（`shard.dflash_round == nullptr`，dflash_buffer 只在
  `if (dflash_round != nullptr)` 的循环里分配）变成无条件解引用 ⇒ 长 prompt prefill 收尾处
  0xC0000005 访问违例（确定性，两次复现 wall 都是 9.9 s）。已改为判空后取值（cpp:2773/:5923），
  并把 `checkpoint.buffer` 的判空显式化（:2768）。教训：`unique_ptr<T>` → 裸指针的每一处
  `.get()` 都必须变成判空后的 `->data()`。
- **延迟实测（pinned vs pageable，同 harness 同代，只差 ring 的 pinning）**：
  `_temp/20261002-1700_ringio.ps1 -Tag pinned2|pageable2` + `_ringio.mjs`（13,559 token 前缀 +
  4 条 98.2% 命中 recall 请求，`--max-context 131072 --host-state-slots 32`）。`[mem]` 行按实际
  pinning 打印（`pinned 2642.6 MiB` / `pageable 2642.6 MiB`）。recall 请求（req#2..#5）total 均值：
  pinned 495.5 ms、pageable 505.3 ms（+10 ms，2%）；TTFT 均值 pinned 296.5 ms、pageable 292.8 ms。
  ⇒ 代价落在 run-to-run 噪声内。两臂 `[tp2-kv]` store/recall 计时相同（store 125.7-151.4 ms /
  recall 67.9-80.1 ms）——它测的是 paged KV（HostKVArena，本来就可分页）而不是 ring 镜像，可作旁证。
- `--host-kv-pinned` 控制的 HostKVArena 仍可 pin（该三元表达式未被本项改动）。

## 项 C 实现与验证（删除 batch 对 host split 数的截断）

机制（设计调研 subagent c0f16d5e，报告 _temp/20261002-1730_item10_ops_envelope.md）：设备侧
`active_split_count = min(逐列自然 split 数, host 传入值)`（src/ops/softmax_attention/dense/causal_cache/small_t.cuh:119），
而 host 传入值 `causal_attention_split_capacity` 在 `batch_size > 1` 时被 `grid_limit`/`page_limit` 截断
（src/ops/softmax_attention/dense/causal_cache/small_t.cu:229-246）。于是批内每一列的归约树比该 lane 单跑时浅，
数值逐位不同；窗口越宽、批越大，截断越狠（Int8/Scale=1/K=5：B=4 cap=10、B=2 cap=20）。

改动：`causal_attention_split_capacity` 删除 `batch_size` 形参（launch.h:26-33），q_heads==24 分支直接
`return capacity;`（capacity 本身已是 envelope 上界；device 侧仍按逐列自然数夹紧，短列多出的 split 立即 return），
11 个调用点同步改 4 参。H12/H16 分支本来就直接 return capacity，不受影响。

回归测试（新增）：tests/ops/softmax_attention/causal_cache.cpp 的 `run_batch_single_parity_case`——
batch=4、width=4、visible=4104（>4096 使 BF16 旧代码必截断）、5 种 storage，把「批内第 b 列」与
「该 lane 单独跑」的输出逐位比较。判据：修复后逐位相等。

| 二进制 | `ninfer_softmax_attention_test` |
|--------|----------------------------------|
| 新 ops（修复后） | `PASS causal_softmax_attention public-contract correctness`，OPS_EXIT=0；容差 oracle 无回归（nvfp4 mae=0.00537 / k8v4 mae=0.00537） |
| 旧 ops（stash 回退） | OPS_EXIT=1，`FAIL causal_softmax_attention public-contract correctness`，5 storage × 4 lane 全部出现 1-3 ULP 差异（例 bf16 lane=0 element 4 batched=0x3a20 solo=0x3a1f） |

**A/B 实测（同 harness 同代，唯一差异 = src/ops 的 6 个文件）**：

| 配置 | 旧 ops | 新 ops |
|------|--------|--------|
| plain decode C=1/C=2/C=3/C=4（每 lane tok/s） | 36.8 / 35.0·35.9 / 31.9·32.7·33.6 / 29.9·30.6·31.3·32.2 | 37.2 / 35.0·35.8 / 32.0·32.8·33.7 / 29.9·30.6·31.4·32.3 |
| plain `[tp2-time] avg_round` C=1..4 | 27.15 / 13.89 / 9.97 / 7.84 ms | 26.85 / 13.92 / 9.92 / 7.85 ms |
| spec 短臂 P4 每 lane 轮成本 mtp+verify | 21.05 ms | 23.57 ms |
| spec 长臂四长 每 lane 轮成本 mtp+verify | 51.65 ms | 50.11 ms |
| spec 接受率 短臂 P4 均值 | 50.1% | 51.4% |
| spec 接受率 长臂四长均值 | 49.6% | 40.3% |
| spec 接受率 **单长（控制项，代码路径无变化）** | **53.3%** | **41.0%** |

⇒ plain decode 完全中性；接受率差异全在 run-to-run 噪声内（单跑控制项自己漂 12 个点），
**不能证明**接受率收益。结论：本项保留，价值是**数值一致性**（parity 测试守住），不是吞吐优化。
报告 §8.7 与 §6 的措辞已据此改写。

## 项 D 结论（P3 批量 prefill，未实现）

设计报告 _temp/20261002-1800_item3_batched_prefill.md（161 行）。可复用入口只有
src/models/qwen3_5/execution/text.cpp:2939-3100 `forward_tp2_window_batch`（per-lane kv 行/状态槽/valid_columns
都已是 [batch] 绑定，width=max(chunkLen)）。两条硬约束：

- attention batch>1 时每 lane width≤16（causal_softmax_attention.cpp:241-243），且 batch>1 永不选 Prompt 路由
  （:382）⇒ 大 T 批量 prompt 前向不支持，只能 width≤16 的小 chunk。
- TP-2 GDN 非 record 路径 batch>1 强制 width==1（text.cpp:1748-1749、:1917-1918），只有
  `GdnStateAction::RecordForReplay` 支持 width>1 ⇒ 批量 prefill 必须走 record+fold。

收益判断：width≤16 下「减少 kernel 启动」相对单 lane 大 T prefill 不自动成立；确定收益是
**admission 位置差被限制在 chunk 宽度内**（收敛投机批的单值 envelope）。且 `window_batch` 走非 overlap
层循环（text.cpp:3076/3078），会丢掉 L2 AR‖MMA overlap 的 +11%（docs/tp2-decisions.md:281-285）。
⇒ 本次不做；报告 §4 已记录该结论（与 §4.2「批量 prefill 提升不了长 prompt 聚合吞吐」一致）。

## 记录

- 项 1/2/3/6/7：tools/win_port/build.ps1 全量重建 BUILD_EXIT=0（712 s，含 msvc_deps_prefix 修复）；
  6/7 增量 BUILD_EXIT=0（29 s）；项 1（新 plain 埋点）增量重建 BUILD_EXIT=0（28 s，job pwsh-59）；
  `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 12/12 通过
  （6 个模型相关 TP-2 测试因无权重被 skip）。
- 项 4：回退后重建 BUILD_EXIT=0（27 s，job pwsh-53）；实测见上。
- 项 8：docs/serving.md 已补 1/2/3/4/6/7 的说明（含实测数字）。
- 项 10：MTP 接受率归因实测（`_temp/20261002-1550_mtp.mjs`）：S=单条长跑，A=四长同 batch，
  B=长+短（位置不同）。注意 `--max-context 245760 --kv-dtype int8 --spec mtp` 启动即 OOM
  （`cudaMalloc failed`，39 s），必须回到文档口径 `--max-context 131072 --kv-dtype int8`。
- **第二轮最终验证（本轮全部改动，工作树未提交）**：`tools/win_port/build.ps1` BUILD_EXIT=0（470 s，
  全量重建——ops 头文件改动会触发全链）；`tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options|softmax_attention'`
  **15/15 通过**（307.98 s，含 ninfer_softmax_attention_test 205.62 s 的 batch/solo parity 回归、
  nvfp4 14.26 s、k8v4 15.23 s；6 个模型 TP-2 测试无 artifact 被 skip）；`git diff --check` 退出 0。
  模型测试用 artifact 单跑的口径与 HEAD 基线对照见「项 E 实现与验证」。
