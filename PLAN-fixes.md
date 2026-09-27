# NInfer 评审缺陷修复计划（2026-09-27）

> 本文件是当前唯一的活动修复计划，也是跨上下文压缩的持久记忆。来源：一次独立全架构评审
> （8 个子代理 + 逐条人工复核），结论见上一轮对话。本文件只保留要修什么、怎么修、怎么验、
> 进度到哪。已知且不改的项见 PLAN.md §3.3/§3.4/§3.7 与 docs/tp2-decisions.md。

## 0. 复核后修订（相对评审报告的更正）

- **A3（reserve_materialization 提前 move prompt）可达性下调为"近乎不可达"**：
  resource_manager.h:429 已用 program.resource_revision() 校验过 plan revision，而 program.cpp:318
  唯一的"非取消 Aborted"用同一个 revision 比较再做一次；单 worker 期间 revision 不变 ⇒ 该 Stale
  分支实际不可达。仍按防御性修复处理（顺序矛盾真实存在）。
- **A5（P2P 传输）当前目标不可达**：本机 p2p_=false（消费级驱动屏蔽）。修法是把它改成正确实现或
  显式拒绝，避免"P2P 被授权的机器上静默算错"。

## 1. 修复项总表

| # | 项 | 严重度 | 状态 |
|---|---|---|---|
| P1 | A1 TP-2 会合 id 通道上限（MTP K>=3 必然耗尽） | 高 | 已实现并验证 |
| P2 | A2 DevicePair 移动构造漏搬 watch_ + 移动赋值泄漏 | 高 | 已实现并验证 |
| P3 | A4 begin_capture/end_capture 非异常安全 | 中高 | 已实现并验证（编译+双卡件） |
| P4 | A3 reserve_materialization 提前 move prompt（防御） | 中（可达性低） | 已实现并验证（既有 Stale 断言仍过） |
| P5 | A6+E1 采样 NaN 到 INT_MAX 越界写 / token_counts 无长度域校验 | 中 | 安全部分已实现并验证；排序项延后 |
| P6 | B2 服务层并发容量与引擎归一化分裂 | 中 | 已实现并验证 |
| P7 | F1 graft_components 漏掉 Uses/资源引用，共享对象可被静默覆盖 | 中高 | 已实现，py_compile 通过 |
| P8 | A5 P2P allreduce/sendrecv 正确性 | 潜伏（实为确定性逻辑错误） | 已实现，本机无法实测 |
| P9 | B1 worker catch(...) 一律升级为 Engine 永久失败 | 架构 | 复核后不改（符合 §7.4） |
| P10 | F2 GDN input_columns 逐层重复 48 份（文档/实现不一致） | 低 | 已实现并验证 |
| - | B5 4 份释放例程收敛；E3/E4；F3/F4；A7 守卫 | 低 | 记录不阻塞 |

## 2. 各项设计

### P1 - 会合 id 通道由图桶数推导（A1）
- 根因：kArMaxChannels=16 固定；每个首次捕获的 WindowGraph 占一个通道；MTP 下 verify_graphs_ 与
  mtp_chain_graphs_ 各按 mtp_graph_profiles(max_context,K) 建桶。逐字复算桶数：K=1/2 到 8+8=16
  （恰好用满）；K=3 到 9+9=18；K=4 到 11+11=22；K=5 到 12+12=24。第 17 个通道即抛 logic_error，
  该桶此后永久无法捕获。
- 方案：通道容量变为构造后可预留的成员。
  - device_pair.h：kArDefaultChannels=16 + 成员 ar_channel_capacity_；新增
    void reserve_ar_channels(std::size_t count)（必须在任何 create_ar_channel 之前调用；
    in-kernel 传输不可用时 no-op）。
  - device_pair.cu：容量替换 3 处分配 / 1 处检查；实现 reserve_ar_channels（重新 cudaHostAlloc
    count 个 id cell + 两卡 cudaMalloc count 个标量，再替换旧块）。
  - tp2_generation_core.cpp：图桶向量建完后调用
    pair_.reserve_ar_channels(verify_graphs_.size() + mtp_chain_graphs_.size() + decode_graphs_.size())。
- 验证：ninfer_tp_device_pair_test 新增"预留 N 通道后连续创建 N 个成功、第 N+1 个抛错"用例；
  编译 ninfer-serve。
- 预期：K>=3 不再触顶；K=2 有余量。

### P2 - DevicePair 移动语义（A2）
- 根因：移动构造初始化列表漏 watch_，也未清空 other.watch_（目标 watch_ 空 -> 诊断失效；源仍持有
  线程且其映射指针已归目标 -> 目标先析构时 UAF 读）；移动赋值覆盖指针前未释放目标既有 pinned/
  device 缓冲（泄漏）。
- 方案：移动构造搬移 watch_ 并把 other.watch_ 置空；移动赋值在覆盖前 stop_ar_watchdog() 并释放
  host_a_/host_b_/arrival_host_*/stall_host_/base_*（按当前设备绑定释放），再接管 other 并清空 other。
- 验证：ninfer_tp_device_pair_test（含 move 用例）通过。

### P3 - 捕获 RAII（A4）
- 根因：三处裸调用 pair_.begin_capture(); capture_group(...); pair_.end_capture();，捕获体抛出时
  capturing_ 永久为真 -> 后续捕获永久失败，且后续 eager 集合走"已捕获"分支。
- 方案：tp2_generation_core.cpp 加局部 RAII（ArCaptureGuard，析构调 end_capture），三处改为 guard
  作用域内调用 capture_group。end_capture 对已关闭状态是 no-op，保留幂等。
- 验证：真实双卡件 append/solo 用例；捕获期抛异常的定向用例。

### P4 - reserve_materialization 不消费 prompt（A3，防御）
- 根因：engine_core.h 以 std::move(request->prompt) 传入；Program 一进 start_resource_transaction
  就把 prompt 移走，随后 revision 门失败返回 Aborted/stale，请求留在 pending 且 prompt 为空。
- 方案（已实现）：把"移动"推迟到 Program 的 revision 门之后。
  - Program::start_resource_transaction(ResourcePlan&&, PreparedPrompt&, CancellationFlagView)：
    门失败直接返回，不碰 prompt；通过后才 PreparedPromptAccess::take(std::move(prompt))。
  - resource_manager 保持 PreparedPrompt&& 形参（调用方授权消费），但以左值转发给 Program，
    于是 Stale 返回时调用方的 prompt 仍完整。engine 调用点保持 std::move(request->prompt)。
  - 这样既修了缺陷，又不用改 20 多处测试调用点（临时量仍可绑定 &&）。
- 验证：编译 ninfer_resource_manager_test / ninfer-serve；既有 Stale 断言（2418/2434）必须仍通过。

### P5 - 采样 NaN 安全 + token_counts 契约（A6/E1）
- 根因：单块贪心 bi=INT_MAX 起步，全 NaN 时输出 INT_MAX 并 atomicAdd(token_counts[INT_MAX]) 越界；
  speculative 贪心接受同样以 INT_MAX 作 terminal/id 并写 token_counts。
- 已实现：1) sampling.cuh 与 speculative_round.cuh 的贪心初值 bi/best_index = 0（0 是合法 id），
  全 NaN 行不再产出 INT_MAX。2) sampling.cuh 的 atomicAdd 前置 [0,token_domain) 校验。
- 延后（需独立 oracle 验收）：ordered_score_bits / sampling_bf16_tile_sort_key 的 NaN 排序与
  sampling_better 的"忽略 NaN"对齐。tile key 的 0 是空槽哨兵，NaN→最低会与之冲突，必须连同
  linear_topk/候选选择一起做数值验收，不能顺手改。记录在案，本轮不改。
- 验证：ninfer_sampling_test、ninfer_speculative_round_test 编译通过；全 NaN 定向用例见"待补"。

### P6 - 服务层容量取归一化后的引擎容量（B2）
- 根因：GenerationService 与 HttpServer 都用 ServeOptions 的 max_concurrency+max_pending_requests
  建 RequestCapacity/线程池；TP-2 路线引擎归一化为 1+1，于是服务层放行的请求被引擎当 overloaded
  拒绝（而不是在 FIFO 里等待），线程池也按错误的容量限额。
- 方案（已实现）：serve_options.h/.cpp 新增 effective_request_capacity(const ServeOptions&)，
  与 normalize_engine_options 对齐（device_b>=0 → 1+1）；generation_service.cpp 与 http_server.cpp
  都改用它，两处数字同源。
- 验证：编译 ninfer-serve；ninfer_serve_options_test。

### P7 - graft_components 漏掉 Uses/资源引用（F1）
- 根因：Entry.binding_objects 只扫顶层 bindings；uses[].auxiliaries 与 components[].resources 同样
  引用 object，却不在 users 里。plan_graft 的"共享对象安全"检查因此漏判：一个被非嫁接组件通过
  auxiliary/resource 引用的对象仍可能被覆盖，违背模块文档的承诺。
- 方案（已实现）：Entry 新增 object_components（object id -> 引用它的组件名集合），覆盖三处引用
  （binding 名、Use.parameter、component 名）；plan_graft 的 foreign 检查改用它。
- 验证：py_compile（本机无 3.11，用 3.13 语法检查，已注明）+ 现有转换测试。

### P8 - P2P 正确性（A5）
- 复核结论（比报告更严重）：P2P allreduce 不只是竞态，而是**逻辑错误**——两个 peer copy 就地互写：
  copy1 把 data_a 覆盖成 data_b，copy2 却还要读原始 data_a，无临时 buffer 时无论怎么排序都不可能
  正确。sendrecv 的 P2P 分支对"send_a 与 recv_a 互为别名"（头文件明确允许）也会让另一侧的 copy
  读到被覆盖的源。
- 方案（已实现）：删除两条 P2P 快速路径，P2P 主机改走映射 pinned 的 in-kernel transport（把建
  transport 的条件从 !p2p_ && !force_host_staging_ 改为 !force_host_staging_，避免 P2P 主机被
  整体降级到每次集合都同步的 host staging），超 staging 的载荷走 host-staging。
  p2p_ 保留为能力上报（p2p_available()），transport 不再发 peer copy。
- 验证：本机无 P2P，删掉的代码路径无法本地实测；保留能力上报与既有 move/析构断言。

### P9 - Engine 失败升级收窄（B1）
- 现状：worker_loop 的 catch(...) 一律 fail_all_locked，failed_ 永不复位。
- 复核结论（已实施）：判定为设计如此，本轮不改。engine-architecture §7.3 的请求级拒绝只发生在
  Program mutation 之前；mutation 之后的异常按 §7.4 必须整机失败。worker catch(...) 与文档一致，
  要改属产品级设计变更，不在本修复范围。

### P10 - GDN input_columns 逐层重复（F2）
- 根因：recipe.py::prepare 为每个 Use 的每个 auxiliary 各建一个对象。GGUF 配方里 48 个 GDN 层各
  一个 gdn/output Use（每层 1 个 input），于是同一个 6144×int32（24,576 B）的 input_columns
  排列被存了 48 份（约 1.125 MiB），与
  docs/maintainer/artifact-container.md §8「同一向量可被任意多个用途引用，loader 只绑定一次」矛盾。
  C++ loader（prepare.cpp::Bindings::input_columns）本来就按 (width, parts) 去重，所以只需修 recipe。
- 方案（已实现）：Recipe.prepare 按 (format, layout, shape, data) intern auxiliary 对象，相同字节
  只分配一个 TensorSpec，多个 Use 引用同一 object id。首个出现处的编号/顺序不变。
- 验证：
  1) tests/convert/test_recipe.py 新增 test_identical_auxiliaries_are_stored_once
     （3 个 Use 共用一个 input_columns → prepared.auxiliaries 恰 1 个，且三个 Use 都引用它）。
  2) 真实工件计数（读目录 JSON）：
     - Qwen3.8-27B-GSQ-RCO-IQ3_S-dflash2q4.ninfer：48 个 Use → 48 个对象，内容同 hash
       1c3c6784c6d0，各 24576 B，合计 1.125 MiB（修复前的形态）。
     - Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer：48 个 Use → 1 个对象（0.023 MiB），
       证明"一个对象被 48 个 Use 引用"的表示已被真实工件与 loader 接受。
     两份报告的 objects 差 1296-1249=47=48-1，与该去重一致。

## 3. 验证与提交策略

- C++ 改动：受影响 target 编译 + 相关 ctest 套件（Windows 用 tools/win_port/test.ps1）。
- 每项一个 commit（Conventional Commits，小写 type）；P1/P3/P4 可合并为一次 TP-2 修复提交。
- Python 改动：python -m py_compile + 现有测试。

## 4. 进度

- [x] P1 会合通道预留（已实现；device_pair.cu/h、tp2_generation_core.cpp、新增测试）
- [x] P2 DevicePair 移动语义（已实现；move ctor/assign + watch_ 重指与重启）
- [x] P3 捕获 RAII（已实现；ArCaptureGuard ×3）
- [x] P4 reserve_materialization prompt（已实现；移动推迟到 Program revision 门之后）
- [x] P5 采样 NaN / token_counts（安全部分已实现；排序归一部分延后，见 P5 节）
- [x] P6 服务层容量（已实现；effective_request_capacity）
- [x] P7 graft Uses/资源引用（已实现；object_components）
- [x] P8 P2P（已实现：删除错误的 peer-copy 快速路径，P2P 主机改用 in-kernel/host staging）
- [x] P9 Engine 失败升级（复核后判定为设计如此，本轮不改：7.3 请求级拒绝只发生在 Program mutation 之前，
  mutation 后的异常按 7.4 必须整机失败。worker catch(...) 与文档一致，本轮不改，需产品级设计变更）
- [x] P10 GDN input_columns 去重（已实现；Recipe.prepare 按字节 intern auxiliary）

### 验证记录（2026-09-27，Windows build-win，ctest）

构建：`cmake --build build-win --target ... -j 12` 全绿（含 ninfer-serve）。
测试：`tools/win_port/test.ps1 -Filter 'sampling|speculative_round|serve_options|engine_options|resource_manager|tp_device_pair'`。

通过：
- ninfer_tp_device_pair_test（34.3 s，PASS）。新用例生效：
  `reserve channels: overflow refused as expected: ... reserved rendezvous id channels`，
  且 `[mem] TP-2 rendezvous id channels reserved: 24`。既有 move / 各 rung / host-staging 用例均通过。
- ninfer_sampling_test（PASS）。新增 `nan_greedy_contract`（全 NaN 行 → token 0，counts[0]++，
  越界写会被 GuardedDeviceBuffer 守卫页抓住）通过。
- ninfer_speculative_round_test（PASS）。
- ninfer_sampling_defaults_test（PASS）。

与本轮改动无关的既有失败（均不在本改动文件内，逐条确认）：
- ninfer_resource_manager_test："candidate-stratified reuse closure"（MaterializationPlanner 排序）。
  已用 baseline 证明：stash 掉本轮 program.h/.cpp + resource_manager.h + 该测试文件后重建，仍以同一断言失败。
- ninfer_engine_options_test："TP-2 route kept a host KV arena"（model_instance.cpp 的
  normalize_engine_options 保留 host_kv_capacity；本轮未改该文件）。
- ninfer_serve_options_test："server reasoning-effort default was not resolved"
  （serve/translate.h 的 resolve_semantics；本轮未改）。新增的 effective_request_capacity 断言通过
  （失败输出里没有本轮的提示串）。

Python：`tests/convert` + `tests/artifact` 共 54 passed（含 P7 py_compile、P10 新增用例）；
`py_compile tools/convert/graft_components.py`、`py_compile tools/convert/recipe.py` 通过。
本机没有 Python 3.11（PATH 上只有 3.13.13，conda 无 envs），改用 3.13 完成。

未完成/无法本地验证：
- P5 的 NaN 排序归一部分（延后，需独立 oracle）。
- P8 的 P2P 路径：本机 p2p_available=0（消费级驱动屏蔽），删掉的 peer-copy 路径无法本地实测。
