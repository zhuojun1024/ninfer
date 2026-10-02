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
| 10 | 投机批链改用每 lane envelope | P2 | 归因已确认（宽窗口入批接受率下降），修复落在 Ops 核，未做 |

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
