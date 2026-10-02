# PLAN：TP-2 多并发审查修复计划

> **本文档把两路独立结论合并为一份修复计划**：① 对 TP-2 多并发改动的**只读源码审查**（会话 `session-a6dd89fc-759c-435e-8759-2024e33f22c7`，审查对象＝工作树相对基线 `d1a9b57b` 的改动）；② 本会话对**用户实机日志**的复现与诊断（用户报「第二个会话一直排队、过一会就中断」）。
>
> - **复核基准**：HEAD `bc3c8a33`（分支 `feat/tp2-concurrency`）。审查报告的每条结论都已在该 HEAD 上重新定位复核，结果记在每条的「HEAD 复核」栏；行号以该 HEAD 为准。
> - **编号约定**：`A*`＝审查确认缺陷；`B*`＝审查风险/脆弱点；`C*`＝本会话实测的并发行为缺陷；`P0–P3`＝修复阶段。审查报告中的 `D1–D5`/`R1–R6` 与本文档 `A/B` 一一对应（见每条的「审查编号」）。**注意：本文档的 A/B/C 与相邻计划 `docs/PLAN-tp2-concurrency.md` 第 0 节的产品决策 `D1–D6` 不是同一套编号。**
> - **相邻权威**：`docs/PLAN-tp2-concurrency.md`（Phase 0 决策、Phase 1/2 施工图与第 12 节逐条修复记录）、`docs/serving.md`（对外合同）、`docs/tp2-decisions.md`（已定案）。
> - **范围**：本文档只写「要修什么、为什么、怎么验」，不重复既有计划已固化的架构决策与施工细节。

---

## 0. 结论速览

两路结论覆盖的是**两条不同的线**：审查线是**正确性与健壮性**（批组建段的异常逃逸、per-lane KV 块表的未发布区、死代码与矛盾文档），实测线是**并发行为**（批次运行期间不接纳新请求、响应被整条队列扣住、带 tools 的请求完全不并发）。两线**只有一处直接重叠**：审查 `A1`（grammar 成员把整批拖回串行）＝ 实测 `C3`（带 tools 的请求在 TP-2 上恒为串行），二者是同一根因的两种表述。另有 `B5`（lane＝批内位置而非对话绑定）与 `B4`（整批失败/全目录作废）**约束**动态入批（`C1`）的落地，属间接耦合。

**建议修复顺序**：`A2`（永久死锁）→ `C2`（响应扣留→499）→ `A3`（KV 后备断言）→ `A4/A5`＋文档（`B2/B5`）→ `C3/A1`（grammar 并发，用户场景的真正修复）→ `C1`（动态入批）→ `B3/B6/B1/B4` 清理。

| 编号 | 严重度 | 一句话 | 审查编号 | 与并发修复方案的关系 |
|---|---|---|---|---|
| A1 | 中 | 批内任一 grammar 成员把**整批**拖回串行，文档低估影响面 | D1 | **直接重叠**于 C3；修法有两种，见第 5 节 |
| A2 | 中 | `drive_lane_queue` 中位于 `try` 之外的代码抛异常 ⇒ TP-2 路线**永久死锁** | D2 | 新增（本方案未覆盖） |
| A3 | 中/低 | per-lane KV 块表只发布 `per_lane` 项，其余是未初始化设备内存，越界即静默破坏 | D3 | 新增 |
| A4 | 低 | `--spec dflash` 的「collapse 为 1 lane」警告先打印、随后立刻抛错退出；注释与文档矛盾 | D4 | 新增 |
| A5 | 低 | `refusal = "route"` 分支不可达（死代码） | D5 | 新增 |
| B1 | 风险 | 服务容量不变式（余量恰为 1）未文档化；池满时是关连接而非 HTTP 429 | R1 | 新增 |
| B2 | 风险 | 4 lane 下主机 checkpoint 网格变粗 5.3×（非正确性缺陷） | R2 | 新增 |
| B3 | 风险 | 多 lane 路线仍构建 `mtp_chain_graphs_` 与 12 个 AR channel，纯浪费且日志误导 | R3 | 新增 |
| B4 | 风险 | 整批失败＋`session_invalidate_all` 的爆炸半径是**全目录** | R4 | 与 C1 的失败语义重定义耦合 |
| B5 | 风险 | `docs/serving.md` 称「每个 lane 独立回忆自己的对话」，实际是「lane＝批内位置」 | R5 | **约束** C1（动态入批后 lane 归属更不稳定） |
| B6 | 风险 | `build.ps1` 的 `Repair-MsvcDepsPrefix` 超范围且会 touch 全部 C++ 源 | R6 | 新增 |
| C1 | 高 | 批次只在批开始组批、且跑到全部 lane 结束才返回 ⇒ 运行期间新请求只能等下一批 | — | 本方案 P2.2（新增） |
| C2 | 高 | 响应按**整批**发布，且驱动者线程被整条队列扣住 ⇒ 先完成的请求被拖到 499 | — | 本方案 P0.2（新增） |
| C3 | 高 | 带 tools 的请求在 TP-2 上**完全不并发**（grammar 无 per-lane mask） | ＝A1 | 本方案 P2.1（重叠） |

---

## 1. 来源与复核方法

| 来源 | 内容 | 权限/性质 |
|---|---|---|
| 审查会话 `session-a6dd89fc-759c-435e-8759-2024e33f22c7` | 只读源码审查报告：`D1–D5` 确认缺陷、`R1–R6` 风险、一张「已验证不是缺陷」清单、验证局限说明 | 只读，未改任何文件 |
| 本会话（用户实机） | 用户日志 `C:/ninfer/serve-win.log` 的解读；`build-win/apps/ninfer-serve.exe` 同参数复现（端口 8099，`NINFER_TP2_LANE_TRACE=1`）；源码定位 | 实测 |

复核方法：对审查报告的每条结论，在 HEAD `bc3c8a33` 上用 read/grep 重新定位到行号并确认代码形态；对 `A2`/`A3` 另核了触发机制本身（`PreparedPromptAccess::view` 确实抛异常；`DeviceArena` 确实用 `cudaMalloc` 且不清零）。凡复核后与审查报告有出入的，在条目内明确写出。

---

## 2. 审查线：确认缺陷（A1–A5）

### A1（中）批内任一 grammar 成员把**整批**拖回串行 —— 文档低估了影响面

**审查编号**：D1　**HEAD 复核**：存在（行号一致）

证据：`src/runtime/engine/tp2_generation_core.cpp:2089-2100`

```cpp
bool batchable         = lanes_ > 1;
const char* refusal    = batchable ? nullptr : "route";
if (batchable) {
    for (const auto& member : batch) {
        const auto& data = qwen::PreparedPromptAccess::view(member->request->prompt);
        if (data.tool_call_output != nullptr) {
            batchable = false;
            refusal   = "grammar";
            break;
        }
    }
}
```

`batchable` 是**整批**开关，不是逐成员开关。命中后 `src/runtime/engine/tp2_generation_core.cpp:2119-2126` 把批内**每一个**成员交给 `execute_lane`：

```cpp
} else {
    // P1.4a runs one lane of the batch at a time; ...
    for (std::size_t lane = 0; lane < batch.size(); ++lane) {
        batch[lane]->result =
            execute_lane(*batch[lane], static_cast<std::uint32_t>(lane));
    }
}
```

而 `execute_lane`（`src/runtime/engine/tp2_generation_core.cpp:2164-2172`）走完整单请求 walk（`execute` → `execute_walk`）。因此 `--max-concurrency 4` 下只要有 1 个 tool-grammar 请求入批，另外 3 个普通请求也一起从 2.80× 掉到 1×。

`docs/serving.md:79-80` 的表述是「only a tool-grammar request runs lane by lane instead of as one round」，把影响面说成了一个请求。

**为什么是真问题**：grammar 请求在 agent 工具循环里是常态而非边缘情形；`src/runtime/engine/tp2_generation_core.cpp:2087-2088` 的注释（批量 decode 步没有 per-lane mask，所以 grammar 不能进批量轮）本身是对的，缺陷只在「要么全批、要么全串行」的粒度。

**建议**：
1. 短期（止血）：把 grammar 成员从批中**摘出来**单独串行走，其余成员照常走 `execute_plain_batch`/`execute_spec_batch`。注意 `batch` 的顺序即 lane 位置（见 B5），摘出后需重排剩余成员的 slot。
2. 目标态：在 batched round 里实现 per-lane grammar mask（见 C3 / P2.1），摘出法随之退场。
3. 文档：`docs/serving.md:79-80` 改为「a batch containing a tool-grammar request runs every member lane by lane」。

### A2（中）`drive_lane_queue` 中位于 `try` 之外的代码抛异常 ⇒ TP-2 路线**永久死锁**

**审查编号**：D2　**HEAD 复核**：存在（行号一致）

证据：`src/runtime/engine/tp2_generation_core.cpp:2063-2110`

```cpp
void TP2GenerationCore::drive_lane_queue() {
    for (;;) {
        std::vector<std::shared_ptr<PendingRequest>> batch;
        batch.reserve(lanes_);                                    // :2066  可抛 std::length_error
        {
            std::unique_lock<std::mutex> queue(lane_queue_mutex_);
            ...
            while (batch.size() < lanes_ && !lane_queue_.empty()) {
                batch.push_back(std::move(lane_queue_.front()));  // :2074  可抛 std::bad_alloc
                lane_queue_.pop_front();
            }
            if (batch.empty()) {
                lane_driver_active_ = false;                      // :2078  唯一的正常复位点
                lane_queue_cv_.notify_all();
                return;
            }
        }
        bool batchable      = lanes_ > 1;
        const char* refusal = batchable ? nullptr : "route";
        if (batchable) {
            for (const auto& member : batch) {
                const auto& data = qwen::PreparedPromptAccess::view(member->request->prompt);  // :2093  可抛
                ...
            }
        }
        ...
        try {                                                     // :2110  ← 唯一的保护从这里才开始
```

`PreparedPromptAccess::view` 确实会抛：`src/models/qwen3_5/frontend/frontend.cpp:732-735`

```cpp
const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    return *prompt.data_;
}
```

异常在 `:2110` 之前逃出 `drive_lane_queue` 时的状态：

- `lane_driver_active_` 已在 `src/runtime/engine/tp2_generation_core.cpp:2039` 置 `true`，无人复位；
- 该批成员已被 `:2074-2075` 从 `lane_queue_` 摘除，且 `member->complete` 全为 `false`；
- `wait_lanes` 的等待**没有超时**（`src/runtime/engine/tp2_generation_core.cpp:2035`）；
- 后续请求只会走 `lane_driver_active_ == true` 的等待分支（`:2034-2041`），**再也不会有人成为驱动者**。

结果：驱动者自己拿到异常，批内其余成员与之后所有排队请求**永久阻塞**，整条 TP-2 路线只能靠重启进程恢复。相比之下 `catch (...)`（`src/runtime/engine/tp2_generation_core.cpp:2127-2146`）已把「批失败」正确落成「所有成员失败」，问题只在于它的覆盖范围少了一段。

**建议**（推荐第 1 种）：
1. 把 `:2065-2109` 整段移入 `try`（即把 `try` 的起点提到 `batch.reserve` 之前），让批组建失败也走 D4 的整批失败路径；
2. 或加 scope guard，保证任何退出路径都复位 `lane_driver_active_ = false`、`notify_all()`，并把未完成成员统一置 `failure`；
3. 另建议给 `wait_lanes` 的等待加一个与 `--pending-timeout-ms` 同源的上限，作为这类不变式破坏的最后兜底。

### A3（中/低）per-lane KV 块表只发布了 `per_lane` 项，未发布项是**未初始化设备内存**

**审查编号**：D3　**HEAD 复核**：存在；另核了 publish 语义与 arena 未清零

证据：`src/runtime/engine/tp2_generation_core.cpp:1034-1064`（文本 KV）与 `:1085-1125`（MTP KV）

```cpp
const std::uint32_t pages    = pages_for_tokens(capacity);
const std::uint32_t per_lane = pages / static_cast<std::uint32_t>(lanes);
...
const std::uint32_t count = (lane + 1 == lanes) ? (pages - assigned) : per_lane;
...
shard.kv_rows.push_back(tables.acquire(lane));
tables.publish(shard.kv_rows.back().handle(), 0, handles, shard.device.stream);
```

`handles` 只有 `per_lane`（末 lane 为余数）个，但该执行行的逻辑容量仍是**整条** `max_context`：`src/models/qwen3_5/state/decoder_state.cpp:25-28,50-52`（`logical_pages = page_count(capacity)`，池级检查只要求 `physical_page_groups >= logical_pages`）。

`publish` 只校验上界、不要求填满（`src/core/paged_kv_cache.cpp:774-780`），`publish_indices` 只 `cudaMemcpyAsync` 发布的那一段（`src/core/paged_kv_cache.cpp:824-833`），而 `DeviceArena` 用 `cudaMalloc` 分配且**不清零**（`src/core/arena.cu:141-155`）。因此设备端块表里那些条目是**垃圾物理页号**。

今天不会触发，是因为三处主机侧算术把它挡住了：`src/runtime/engine/tp2_generation_core.cpp:1964-1967` 拒绝超窗 prompt、`src/runtime/engine/tp2_generation_core.h:441-445` 的 `lane_context_window()` 把窗口收在 `per_lane*64 - margin`、spec 路径每轮 `capacity_left` 再夹一次（`src/runtime/engine/tp2_generation_core.cpp:3444-3449`）。三处**都没有后备校验**，也不在同一条不变式的命名下：一旦有人放宽窗口（动态配额、复用别 lane 空闲页），后果不是断言失败而是**静默写到别人的物理页或池外**。基线（lanes=1）时 `per_lane == pages`，行是被填满的 ⇒ 这是本次改动**新引入**的脆弱性。

**建议**：
1. `build_shard` 加启动断言，把不变式写死，例如 `if (per_lane * kPagedKVPageSize < lane_context_window() + 1 + (mtp_enabled_ ? mtp_drafts_ : 0)) throw std::logic_error("TP-2 lane context window exceeds the pages published to its execution row");`
2. 或让 `KVExecutionTablePool` 记录「已发布范围」，使设备端查询有据可查；
3. 至少在 `src/runtime/engine/tp2_generation_core.cpp:1051` 与 `src/runtime/engine/tp2_generation_core.h:441-445` 两处互引注释，写明「窗口＝已发布页数×64−margin」是硬绑定。

### A4（低）`--spec dflash` 的「折叠为 1 lane」警告与实现/文档矛盾

**审查编号**：D4　**HEAD 复核**：存在

证据：`src/runtime/engine/model_instance.cpp:99-113` 会打印

```cpp
"[tp2-lane] --max-concurrency %u collapses to 1 lane on the %s route (see docs/PLAN-tp2-concurrency.md)"
```

但同一函数 `src/runtime/engine/model_instance.cpp:135-138` 立即终止启动：

```cpp
if (options.speculative.backend == SpeculativeBackend::DFlash) {
    throw std::invalid_argument(
        "TP-2 generation supports --spec mtp and --spec dflash2 (not --spec dflash)");
}
```

`include/ninfer/tp2_capacity.h` 的 `tp2_generation_concurrency` 只在 `DFlash` 上返回 1，所以该 fprintf 是**死路径**；`src/runtime/engine/model_instance.cpp:99-103` 的注释（"rather than failing startup"）与 `docs/serving.md:75-76`（"still collapses ... with a warning"）都与实现矛盾。

**建议**：删掉 `src/runtime/engine/model_instance.cpp:107-113` 的 fprintf，注释改为「DFlash v1 在下方被拒绝，这个常量返回 1 只是历史原因」；`docs/serving.md:75-76` 改为「`--spec dflash` is rejected at construction on this route」。

### A5（低）`refusal = "route"` 分支不可达

**审查编号**：D5　**HEAD 复核**：存在

证据：`src/runtime/engine/tp2_generation_core.cpp:2089-2090` 的 `bool batchable = lanes_ > 1;` 恒为 `true`：`drive_lane_queue` 的唯一调用点是 `src/runtime/engine/tp2_generation_core.cpp:2047`（在 `wait_lanes` 内），而 `wait_lanes` 的唯一调用点是 `src/runtime/engine/tp2_generation_core.cpp:2000-2001`，守卫为 `if (owner_->lanes_ > 1)`。

**建议**：`batchable` 直接初始化为 `true`（或删掉 route 分支），只保留 grammar 判据；`src/runtime/engine/tp2_generation_core.cpp:2083-2084` 的注释同步。

---

## 3. 审查线：风险与脆弱点（B1–B6）

### B1 服务容量不变式未文档化，且「丢弃连接」而非 HTTP 429

**审查编号**：R1　**HEAD 复核**：存在

`src/serve/http_server.cpp:223-229` 把线程池配成 `worker_count = C+P+1`、队列上限 `C+P`。本仓库定制的 httplib 在队列满时 `enqueue` **返回 false**（`third_party/cpp-httplib/httplib.h:11704-11723`），而 `Server::listen_internal` 是在**调用 listen 的线程**里 accept 后 enqueue，失败只记 `Error::ResourceExhaustion` 并关闭 socket（`third_party/cpp-httplib/httplib.h:13652-13737`）⇒ **客户端看到连接被关闭，拿不到 HTTP 429**。

今天走不到这条路径，只因为 `src/serve/generation_service.cpp:276-298` 的 `acquire_request_lifetime` 在 `active >= maximum` 时立刻抛 `RequestError(RequestErrorKind::Overloaded, "inference request queue is full")`，使在飞请求 ≤ C+P，而池容量是 (C+P+1) workers + (C+P) 队列位，余量恰为 1。

**建议**：在 `src/serve/http_server.cpp:223` 上方加注释固定两个前提（accept 线程不占 worker；准入失败不阻塞线程），并写明「余量 = 1 是刻意的」。

### B2 checkpoint 网格在 4 lane 下变粗 5.3×

**审查编号**：R2　**HEAD 复核**：存在（非缺陷）

`src/runtime/engine/tp2_generation_core.cpp:389-397` 按 lane 均分主机 checkpoint 环后，`grid` 槽数减少：默认 `host_state_slots` + lanes=4 ⇒ `per_lane=8, fixed=2, flexible=6, tail=3, used=5, grid=3`；`--max-context 131072` 时 `per_slot = ceil(131072/3) = 43691` ⇒ `host_checkpoint_stride_ = max(8192, round128(43691)) = 43776`（lanes=1 时为 8192）。

**不是**「超出预算」缺陷：`lanes_ * floor(total/lanes_) <= total` 恒成立，`:380-383` 的注释是保守的，最多比预算少 `lanes-1` 个槽。实际影响只是网格型前缀复用在 4 lane 下更粗（短 prompt 不受影响，尾环仍覆盖最后 8192 token）。

**建议**：`docs/serving.md` 的 retention 段补一句「多 lane 下网格 checkpoint 间距会按 lane 数变粗」。

### B3 `mtp_chain_graphs_` / `mtp_chain_host_` / 12 个 AR channel 在多 lane 路线是纯浪费

**审查编号**：R3　**HEAD 复核**：存在

`src/runtime/engine/tp2_generation_core.cpp:560-569` 只要 `mtp_enabled_` 就构建 `mtp_chain_graphs_`（12 条 profile），`:664-666` 为其预留 12 个 AR channel。唯一使用者 `mtp_propose_window`（`src/runtime/engine/tp2_generation_core.cpp:1799`，图选择在 `:1812`）只在 `src/runtime/engine/tp2_generation_core.cpp:6393` 被调用，而那里在 `execute_walk`（单 lane 串行走查）内部；多 lane 路线走 `execute_spec_batch`，MTP 链步直接调 `ctx_a.mtp_forward_decode_batch` + `ctx_a.mtp_propose_batch`，**从不进入** `mtp_propose_window`。同时启动日志（`src/runtime/engine/tp2_generation_core.cpp:650-656`）在多 lane 下仍打印 "mtp chain: eager(bucket)"，误导运维。

**建议**：这两个资源（及 AR channel 预留）改为只在 `lanes_ == 1U && mtp_enabled_` 时构建，多 lane 时日志字段显示 `n/a`。

### B4 整批失败 + `session_invalidate_all` 的爆炸半径是**全目录**

**审查编号**：R4　**HEAD 复核**：存在

`src/runtime/engine/tp2_generation_core.cpp:2127-2146` 的 `catch (...)` 按 D4 把整批置为同一失败并调用 `session_invalidate_all()`；`abort_if_ar_stalled`（`src/runtime/engine/tp2_generation_core.cpp:3954-3973`）内部也先 `invalidate_host_checkpoints(); session_invalidate_all();` 再抛 `RequestError(Unavailable, ...)`。即一次 allreduce 会合失败会作废**所有 lane**的复用状态（`session_invalidate_all` 定义在 `src/runtime/engine/tp2_generation_core.cpp:4107-4110`）。

**建议**：确认是否有意为之；若是，在 `docs/serving.md` 明确「一次 AR stall 会丢弃全部复用状态」。与 P2.2（动态入批）的失败语义重定义一并决定。

### B5 `docs/serving.md` 的「每个 lane 独立回忆自己的对话」与实际绑定不符

**审查编号**：R5　**HEAD 复核**：存在（非正确性缺陷）

`docs/serving.md:77-79` 称 checkpoint 环与会话目录按 lane 切片，「so each lane recalls its own previous conversation independently」。实际绑定是「lane ＝ 批内位置」：`src/runtime/engine/tp2_generation_core.cpp:2494-2495` 的注释写明 slot 同时是批内位置、KV 执行行与 GDN 状态槽，`:2515` `lanes[index].slot = static_cast<std::uint32_t>(index);`。

**已确认不是正确性缺陷**：`session_recall` 能把主机侧会话恢复到任意 lane，且 `src/runtime/engine/tp2_generation_core.cpp:4699` 的跳过条件（`entry.device_lane >= 0`）保证同批两条 lane 不互相偷取活会话；checkpoint 的 `valid[lane]` 只在写入它的 lane 上置位（`publish_lane_prefill` `src/runtime/engine/tp2_generation_core.cpp:2406-2412`）。问题只在文档暗示的稳定「对话↔lane」绑定不存在，设备态复用命中率随到达顺序变化。

**建议**：改为「the checkpoint ring and the session catalog are sliced by lane, so device-resident reuse depends on which lane a request lands on; a recall onto a different lane falls back to the host-resident session」。**这是 P2.2 的前置**：动态入批会让 lane 归属进一步不稳定。

### B6 `tools/win_port/build.ps1` 的改动超出本任务范围，且会强制全量重编

**审查编号**：R6　**HEAD 复核**：已随 `6a17bb2c` 一起提交

该改动新增 `Repair-MsvcDepsPrefix($BuildDir)`，修复本身必要且正确（原 `rules.ninja` 的 `msvc_deps_prefix` 是 CMake 按 CP936 误解码 cl 中文提示产生的乱码，导致 ninja 依赖记录全为 `#deps 0`、改头文件不触发重编）。但它与 TP-2 多并发无关，且「touch 全部 C++ 源」会让每次构建触发全量重编。

**建议**：保留该修复（依赖追踪已恢复），但把 touch 限定在「探测到前缀不一致」时才发生；若已是该行为，在注释里写明。

---

## 4. 实测线：并发行为缺陷（C1–C3）

复现环境：`build-win/apps/ninfer-serve.exe`，端口 8099，与用户脚本同参（`--devices 0,1 --max-concurrency 4 --max-context 131072 --spec dflash2`），`NINFER_TP2_LANE_TRACE=1`。用户侧日志：`C:/ninfer/serve-win.log`。

### C1（高）批次运行期间不接纳新请求：组批只在批开始，且批跑到全部 lane 结束才返回

**证据（代码）**：`src/runtime/engine/tp2_generation_core.cpp:2063-2158` 的 `for(;;)` 每次迭代只组一个 batch 并把它跑完；`execute_plain_batch` 内部 `while (!active.empty())`（`src/runtime/engine/tp2_generation_core.cpp:2843`）、`execute_spec_batch` 内部同理（`src/runtime/engine/tp2_generation_core.cpp:3408`）；serial 路线更是一次跑完每个成员整段 walk（`:2119-2126`）。组批窗口 `src/runtime/engine/tp2_generation_core.cpp:2058` `constexpr std::chrono::microseconds kBatchFormationWindow{3000};` 只有 3 ms ⇒ 只有「同一瞬间」到达的请求才可能同批。`:2105-2109` 每次组批打印一次 `[tp2-lane] batch=%zu capacity=%u path=%s%s`，日志里一次组批只有一行，正是「整批跑到结束」的直接证据。

**证据（实测）**：
- 无 tools：req#8（1024 token，引擎 7.5 s）11:23:57.861 到 → `batch=1 capacity=4 path=batched`；req#9（8 token，晚 1.5 s）11:23:59.373 到 → **另一条** `batch=1 capacity=4 path=batched`，直到 11:24:05.34 才开跑：**引擎时间 127 ms，墙钟 6.1 s**。
- 用户日志：req#2（tools，15,163 prompt）独占 11:11:29.4→11:15:52.05（4m23.4s）；req#3/req#4 在 11:11:33 到达，直到 11:15:52.05 才开跑（req#4 total 7.9 s）。

**影响**：第二个会话的 TTFT 由前一个会话的整段生成时间决定 —— 即用户报的「第二个会话一直排队」。

**建议**：P2.2 动态入批。

### C2（高）响应按**整批**发布，且驱动者线程被整条队列扣住

**证据（代码）**：
- 发布粒度：`src/runtime/engine/tp2_generation_core.cpp:2147-2156` 在整批跑完后才 `for (auto& member : batch) member->complete = true; lane_queue_cv_.notify_all();`。等待者只在每批结束时被唤醒 ⇒ 先完成的 lane 被同批慢 lane 拖住。
- 驱动者扣留：`wait_lanes`（`src/runtime/engine/tp2_generation_core.cpp:2020-2051`）里找不到驱动者的线程**自己当驱动者**并持 `execution_mutex_` 调用 `drive_lane_queue`（`:2046-2047`），而 `drive_lane_queue` 只在**队列清空**时才 return（`:2151-2155`）⇒ 驱动者自己的请求即使早已跑完，也要等整条队列排空才拿到结果。

**证据（实测）**：
- 3×tools 同时到达 → `batch=3 capacity=4 path=serialgrammar`：req#5 total 1.4 s、req#6 434 ms、req#7 435 ms，**三条都在 11:22:53.242 才 done**。
- 用户日志：req#1（无 tools，200 prompt/64 out）整段 walk 在 11:11:27.86→29.36 跑完（throughput prefill 200 tok + decode 63 tok；`total 773 ms`），但 `req#1 done` 记在 **11:15:59.950**，并伴随 `req#1 response failed during transport | HTTP 499 | client disconnected`。

**影响**：先完成的请求被扣住 4m31s ⇒ 客户端超时/断开（HTTP 499）；即使同批，快 lane 也要等慢 lane。这与 A2 是**不同**的缺陷：A2 是异常路径把路线卡死，C2 是正常路径把响应压住。

**建议**：P0.2（驱动者与请求线程解耦 + 完成即发布）。

### C3（高）带 tools 的请求在 TP-2 上完全不并发（＝ A1 的实测面）

**证据（代码）**：grammar mask 只存在于 serial walk：`src/runtime/engine/tp2_generation_core.cpp:5224-5235`（`tool_constraint` 构造）、`:5546-5564`（`tool_mask_dev`/`tool_mask_columns`）、`:5931-5936`（prefill 首 token 掩码）、`:6300-6329`（serial DFlash2 窗口掩码）、`:6478-6501`（serial MTP 窗口掩码）；batched 路线**完全没有**。任一成员带 tools ⇒ `:2089-2100` 整批 `refusal="grammar"` ⇒ `:2119-2126` 逐成员整段 walk。

**证据（实测）**：
- tools：req#10（1024 token）11:24:07.498 → `batch=1 path=serialgrammar`；req#11（8 token，晚 1.5 s）11:24:08.995 → 又一条 `batch=1 path=serialgrammar`，11:24:15.989 才跑：**引擎 337 ms，墙钟 7.0 s**。
- 3×tools 同批 → 整批串行且整批发布（同 C2 的实测）。

**影响**：dsh 每个请求都带 tools ⇒ TP-2 的并发度**恒为 1**，`--max-concurrency 4` 对该负载完全无效。

**建议**：P2.1（batched round 的 per-lane grammar mask）。

---

## 5. 重叠分析（本文档要回答的问题）

| 本文档修复项 | 审查结论 | 关系 |
|---|---|---|
| **P2.1（C3）** | **A1** | **直接重叠**：同一根因（批量轮没有 per-lane grammar mask）。审查用「缺陷」视角描述其**影响面**（一个 grammar 成员拖垮整批），实测用「行为」视角描述其**结果**（带 tools 的请求恒串行）。 |
| P2.2（C1） | 无 | 互补。审查报告未覆盖「批次运行期间不接纳新请求」；`src/runtime/engine/tp2_generation_core.cpp:2957-2974` 的头注释还明确写着 continuous batching out of scope。 |
| P0.2（C2） | 无 | 互补。审查报告未覆盖「正常路径下响应按整批发布 / 驱动者被队列扣住」；`A2` 是**异常**路径的相邻问题，二者修法可以合并到同一段代码，但判据不同。 |
| — | A2/A3/A4/A5/B1/B2/B3/B4/B6 | 本修复方案未覆盖的新增项。 |
| P2.2 的前置 | B5 | **间接耦合**：`C1` 落地后 lane 归属进一步不稳定 ⇒ B5 的文档口径必须先改。 |
| P2.2 的语义决定 | B4 | **间接耦合**：动态入批改变批内成员构成，D4「整批失败」与 `session_invalidate_all` 的爆炸半径必须一并重新定义。 |

### 5.1 A1 与 C3 的两种修法对比

| 方案 | 做法 | 改动面 | 对 dsh 场景（每个请求都带 tools） | 对混合批（1 grammar + N 普通） |
|---|---|---|---|---|
| 审查建议（摘出法） | 把 grammar 成员从批里摘出单独串行，其余成员保持批量 | 小（改 `drive_lane_queue` 的分派与 slot 重排） | **无效**：所有成员都是 grammar ⇒ 仍然全串行 | 有效：普通成员恢复 2.80× |
| 本方案 P2.1（per-lane mask） | 在 batched round 里拼 `[vocab, width*lanes]` 掩码，lane l 占 `width*l` 列块；每 lane 的约束状态按 accepted token 推进；prefill 首 token 也要掩码 | 中大（batched round + prefill 首 token + 约束对象 per-lane 化） | **有效** | 有效 |

结论：两者不冲突 —— 摘出法可作为**短期止血**（P1.4），per-lane mask 是**目标态**（P2.1）。只做摘出法无法解决用户当前场景。

---

## 6. 统一修复计划

### P0 止血（小改动，可独立提交）

**P0.1 ＝ A2（永久死锁）**
- 改动：`src/runtime/engine/tp2_generation_core.cpp:2065-2109` 移入 `try`（或 scope guard）；`wait_lanes` 等待加超时兜底。
- 验收：构造一个 `view` 抛异常的入批路径（单测直接注入 `PreparedPromptData == nullptr` 的 prepared prompt，或对批组建段做单元级注入），确认路线不会挂起、批内成员与后续请求都拿到失败结果。

**P0.2 ＝ C2（响应扣留 → 499）**
- 改动：
  1. **驱动者解耦**：把 `lane_driver_active_` + `drive_lane_queue` 从「请求线程自任驱动者」改为核心拥有的驱动线程（构造时启动、析构时停止），或保留 leader 但每批结束后若驱动者自己的请求已完成就交棒（`src/runtime/engine/tp2_generation_core.cpp:2020-2051`、`:2147-2156`）。
  2. **完成即发布**：把「整批跑完统一 `complete = true`」改为「每个 member 完成即发布」。批量路线在**轮边界**判定：一轮结束后把已 finished 的 member 移出 active 并立即发布（plain 的 `active.swap(live)` 与 spec 的 `still` 收集已有等价压缩逻辑，可直接复用）；serial 路线每跑完一个 lane 即发布。
  3. `NINFER_TP2_LANE_TRACE` 增加 per-lane 发布字段，便于回归。
- 验收：重跑本会话的「长请求 + 1.5 s 后短请求」探针（无 tools 与 tools 各一组）⇒ 短请求的 `done` 时刻贴近它自己的引擎时间；用户日志形态的 `HTTP 499` 不再出现。

### P1 正确性与文档（审查项）

- **P1.1 ＝ A3**：`build_shard` 加启动断言（见 A3 建议 1），并互引注释。
- **P1.2 ＝ A4 + A5**：删 `src/runtime/engine/model_instance.cpp:107-113` 死路径、改 `:99-103` 注释与 `docs/serving.md:75-76`；`src/runtime/engine/tp2_generation_core.cpp:2089-2090` 去掉不可达的 route 分支。
- **P1.3 ＝ B2 + B5 + 文档**：`docs/serving.md:75-76`、`:77-79`、`:79-80` 三处口径；retention 段补网格变粗说明；`src/serve/serve_options.h:79-82` 仍是旧口径（「TP-2 归一化为 1 活跃 + 1 排队」），一并改。
- **P1.4 ＝ A1 的缓解（摘出法）**：仅当 P2.1 暂缓时执行。

### P2 吞吐（用户场景的真正修复）

- **P2.1 ＝ C3/A1（batched per-lane grammar mask）**
  - 布局：`tool_mask_dev = [vocab, constraint_columns * lanes]`，lane `l` 占列块 `[width*l, width*(l+1))`（与既有窗口列约定 `t = column + width*lane` 一致）。
  - 约束状态：每 lane 一个 `ToolCallConstraint` 实例（或等价的可复制状态），按该 lane 本轮 accepted 的 token 推进；prefill 首 token 也必须掩码。
  - 需要新增：`ops::apply_token_mask` 的按列块变体（或等价内核）、batched round 里 mask 的构建与 H2D、prefill 首 token 的 per-lane 掩码路径。
  - 验收：带 tools 的并发请求出现 `path=batched`；per-lane oracle 与同 lane 单跑对比（近并列判据）；grammar 合法性用工具调用回归用例检查。
- **P2.2 ＝ C1（运行中批次动态接纳）**
  - 改动：把 per-lane 的 setup/prefill 抽成可调用单元（例如 `admit_lane(PendingRequest&, slot)`），在每轮边界从队列取新请求并分配空出的 slot；`lanes`/staging 按 `lanes_` 预分配（`batch_lane_host_`、`mtp_pack`、`accept_host` 等已是 lane 容量）。
  - 语义：D4「整批失败」需要重新定义（部分成员已在运行）；`session_invalidate_all` 的爆炸半径（B4）一并决定。
  - 验收：用户日志形态的双会话场景（第一个会话长、第二个会话在它运行中到达）⇒ 第二个会话在同一批的后续轮里开始产出，TTFT 不再等于前一个会话的整段时长；gate 11/11 与三个 artifact 测试全绿。

### P3 清理与观测

- **P3.1 ＝ B3**：`mtp_chain_graphs_`/`mtp_chain_host_`/AR channel 仅 `lanes_ == 1 && mtp_enabled_` 构建；日志字段多 lane 显示 `n/a`。
- **P3.2 ＝ B6**：`build.ps1` 的 touch 条件化。
- **P3.3 ＝ B1**：容量不变式注释。
- **P3.4**：本轮已提交的两项（`34cb16ca` serial DFlash2 帧按 lane 0 切片、`bc3c8a33` view 报错带形状）不属本计划范围，但它们是 A1/C3 场景下的直接 500 故障修复，须在 P2.1 的回归里保持。

---

## 7. 验收矩阵

| 修复项 | 验证方式 | 判据 |
|---|---|---|
| P0.1（A2） | 注入 `view` 抛异常的批组建路径（单测或临时注入） | 无挂起；批内与后续请求均收到失败结果；`lane_driver_active_` 复位 |
| P0.2（C2） | 「长请求 + 1.5 s 后短请求」探针（无 tools / tools 各一组）+ 用户日志形态回归 | 短请求 `done` ≈ 其引擎时间；无 `HTTP 499`；per-lane 发布 trace 出现 |
| P1.1（A3） | 双卡启动 `--max-concurrency 4 --max-context 131072` | 断言不触发；若人为放宽窗口则抛错 |
| P1.2–P1.3 | 代码审阅 + 文档 diff | 无死路径；文档与实现一致 |
| P2.1（C3） | 双卡 serve 冒烟：2×tools 并发；grammar 合法性用例 | `path=batched`；工具调用输出合法；per-lane oracle 近并列判据通过 |
| P2.2（C1） | 双会话到达时序探针（长 + 运行中到达的短） | 第二个会话在第一批的后续轮开始；TTFT 与前者整段时长解耦 |
| 全量回归 | `tools/win_port/test.ps1` 过滤 `tp2`/`tp_device`/`engine_options`/`serve_options`；`ninfer_qwen3_5_tp2_forward_test` / `tp2_load_test` / `tp2_sessions_test`（`NINFER_TEST_ARTIFACT`） | gate 11/11；forward/load exit 0；sessions 保持既有结果（含已知的 shared-system-prompt 失败） |

---

## 8. 审查已确认「不是缺陷」的关键点（避免重复审查）

| 项 | 证据 |
|---|---|
| P1.8 标量槽回填在两处批量入口都在 | `src/models/qwen3_5/execution/text.cpp:2361-2365`、`:3047-3051` |
| 单列批的图键设计正确 | `src/runtime/engine/tp2_generation_core.cpp:542-549`、`:629-637`；`run_plain_decode_step_batch:1679`、`run_verify_window_batch:1477`；`select_window_graph` 要求 `graph.batch == batch` 精确相等（`:1233-1239`） |
| AR channel 不会触顶 | `src/runtime/engine/tp2_generation_core.cpp:664-666` 在任何 capture 之前预留全部图族 |
| 图复放的 workspace watermark 一致 | `:2838-2841` 一次设定 `round_base`；`:2882-2885` 每轮回卷；复放前断言 `workspace->used() == graph->arena_begin[i]`（`:1502-1505`、`:1703-1706`） |
| 退役 lane 用原始 slot 而非压缩列号 | `:2877-2878`、`:2893`、`:3466` |
| 采样 token-count 缓冲不别名 | `:2533-2550` 按 `base + index * public_tokens` 切片；`:2557` `configs_dev` 为 `sizeof(SamplingConfig) * lanes_`；契约见 `include/ninfer/ops/sampling.h:45-78` |
| `records_for`/`fold_for` 有边界检查 | `src/runtime/engine/tp2_generation_core.h:248-259` |
| `submit` 的减法回绕有分层覆盖 | `:1964`（多 lane）与 `src/runtime/engine/engine.cpp:335-339`（单 lane） |
| `restore_lane_gdn` 只动自己那条 lane | `:2316-2348`（`ReuseSource::None` 走 `zero_lane_state`） |
| 批内不会读脏的 `active_lane_` | 18 处引用已核对；批量路径只在逐 lane prefill 循环里写读 |
| `state_backing` 语义正确 | `:776-781` `state_backing.bytes == state_bytes`（池本身），spec 每轮整池快照/回滚不误伤快照槽 |
| MTP 层不绑 GDN 状态槽是无害的 | MTP 层是纯 attention+MLP，无 GDN mixer |

---

## 9. 验证局限与未决问题

1. **审查是源码级的**：审查报告引用的实测数字（1.76–1.91×@2、2.80×@4、1.41× 长度不均配对、state +293.6 MiB/shard/lane）来自既有计划与 `docs/serving.md` 已记录的门禁结果，审查期间未重新构建运行。
2. **ctest 覆盖不到本改动**：`ninfer_qwen3_5_tp2_load/forward/sessions/dflash_solo/dflash_append_test` 会被 ctest **Skipped**（`tests/models/qwen3_5/test_tp2_forward.cpp:158` 需要显式 `--artifact`，`:163` 需要两块 GPU），且该测试自带 `build_shard_state`（`:120-129`）**不经过** `TP2GenerationCore::build_shard` ⇒ 覆盖不到 A3 的块表发布路径。**真正的回归门是双卡 `ninfer-serve --devices 0,1` 冒烟。**
3. **逐字节 oracle 不适用于批量路线**：批内共享一个由最长 lane 决定的 attention envelope ⇒ ulp 漂移可翻转近似并列 token（`docs/serving.md:93-98`）。批量正确性必须用近并列判据。
4. **A2 与 A1/C3 都无法被现有测试暴露**：前者需要批组建期抛异常的输入，后者需要双卡 + grammar 与普通请求同时到达。两者都需要新测试（见验收矩阵）。
5. **未决**：D4「整批失败」在动态入批（P2.2）下如何重定义；`session_invalidate_all` 是否应缩小到失败批（B4）；per-lane grammar mask 的约束状态是否需要按 lane 复制整棵约束树（成本待估）。

---

## 10. 与既有文档的关系

| 文档 | 关系 |
|---|---|
| `docs/PLAN-tp2-concurrency.md` | Phase 0 决策（`D1–D6`）与 Phase 1/2 施工图；本文档的 A/B/C 项落地后应在该文件第 12 节追加一条施工记录，并在此处回填状态 |
| `docs/serving.md` | 对外合同；P1.3 修正其 TP-2 段落 |
| `docs/tp2-decisions.md` | 已定案清单；A3/A4/A5 属于「实现与文档不符」，不改变已定案 |
| `docs/maintainer/paged-kv-cache.md` | A3 涉及的 execution row 发布语义的权威 |

---

## 11. 施工记录

| 项 | 状态 | 落盘位置 | 验证 |
|---|---|---|---|
| P0.1（A2） | 已落盘 | `src/runtime/engine/tp2_generation_core.cpp:2172-2211`：批组建、trace、调度全在 `try` 内，异常走 D4 路径而不是逃出驱动线程 | 编译绿；gate 11/11（未单独注入抛异常的输入） |
| P0.2（C2） | 已落盘 | 驱动线程 `drive_lane_queue`（`:2135-2218`）、完成即发布 `publish_lane`（`:2097-2130`）、构造函数起线程 + `stop_lane_driver()` | 「长请求 + 1.5 s 后短请求」探针：短请求 `done` 从整批等待降到 1858 ms（P0.2 后）、316 ms（P2.2 后）；无 `HTTP 499` |
| P2.2（C1） | 已落盘 | `try_pop_lane_queue()`（`:2522-2531`）；plain `admit_lane` lambda `:2747-2992` + 初批 `:2994-2996` + 轮边界接纳 `:3030-3050`；spec `admit_lane` lambda `:3397-3659` + 初批 `:3660-3662` + 轮边界接纳 `:3721-3741`；两路线 `lane_capacity = lanes_`（`:2602`/`:3243`），staging（`tool_mask_host`/`accept_host`/`mtp_pack`/`append_*`）全部按 `lanes_` 预留 | 见下 |

P2.2 实测（双卡 `--devices 0,1 --max-context 131072 --max-concurrency 4 --spec dflash2`，`NINFER_TP2_LANE_TRACE=1`，日志 `_temp/p22_serve.log`）：

- **plain 路线**：长请求运行中 +1.5 s 到达的短请求 `done=316 ms`（P0.2 后为 1858 ms），长请求 4135 ms 照常跑完。
- **tools 路线**（P2.1 起不再退化为串行）：长 tools 请求运行中 +1.5 s 到达的短 tools 请求 `done=898 ms`，长请求 3835 ms；短请求产出合法 `call:get_weather({"city":"Paris"})` ⇒ 动态入批的 lane 同样走 per-lane mask。
- **四条 tools 同时到达**：2 条组批 + 2 条经 `admit slot=2 live=3` / `admit slot=3 live=4` 动态入批 ⇒ 该轮出现 `[tp2-lane] grammar masked … window columns of 4 lanes`，P2.1 遗留的 lane 2/3 掩码列块端到端跑到。
- **回归**：`tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` ⇒ `100% tests passed out of 11 (70.73 s)`；`ninfer_qwen3_5_tp2_forward_test.exe --artifact …` 与 `…_load_test.exe --artifact …` exit 0；`tp2_sessions` 属 §9 既有失败。

**P2.2 未决**：D4「整批失败」在动态入批下仍是整批 + `session_invalidate_all()` 全目录，B4 的爆炸半径未缩小，留给 P3.4/B4 一并决定。

### P1 施工记录（P1.1–P1.4）

| 项 | 状态 | 落盘位置 | 验证 |
|---|---|---|---|
| P1.1（A3） | 已落盘 | `tp2_generation_core.cpp` 的 `build_shard` 增加启动断言：`lanes_ > 1` 时要求 `lane_context_window() + (mtp_enabled_ ? mtp_drafts_ : 0) < per_lane * kPagedKVPageSize`，否则抛 `TP-2 lane context window exceeds the pages published to its execution row`；注释点明 `DeviceArena` 不置零 ⇒ 越界请求会读到未初始化 block-table 项并写到别的 lane 的物理页。`build_shard` 调用点同时下移到 lane 预算计算之后 | 双卡 `--max-concurrency 4 --max-context 131072` 启动不触发；gate 11/11 |
| P1.2（A4+A5） | 已落盘 | `src/runtime/engine/model_instance.cpp:97-107`：删掉「collapses to 1 lane」死 fprintf，注释改写为「`--spec dflash` 在归一化末尾直接被拒，没有路线会静默丢 lane，常数里的 DFlash 规则在此不可达」；`tp2_generation_core.cpp` 删掉不可达的 `refusal = "route"` 分支 | 编译绿；gate 11/11 |
| P1.3（B2+B5+文档） | 已落盘 | `docs/serving.md` 三处口径（lane = 批内位置的绑定、4 lane 下 checkpoint 网格约粗 4 倍、动态接纳）；`src/serve/serve_options.h:79-83` 改为 `tp2_generation_concurrency(max_concurrency)`（1–4 个活跃 lane）+ `max_pending_requests` 排队的正确描述 | 文档 diff 审阅 |
| P1.4（A1 摘出法） | 未执行 | 被 P2.1 的真修复取代：dsh 的每个请求都带 tools，摘出串行化没有收益 | — |

### P2.1 施工记录（C3/A1）

- 落盘位置：`tp2_generation_core.cpp` plain 轮 `:3097-3102`（掩码应用）、spec 轮 `:3963-3968`/`:4150-4155`（构建 + `ops::apply_token_mask`）；`drive_lane_queue` 的 grammar 拒绝分支删除。每 lane 一个 `ToolCallConstraint`，mask 张量 `[vocab, window_live]`，lane `l` 占列块 `[width*l, width*(l+1))`（与窗口列约定 `t = column + width*lane` 一致），prefill 首 token 同样掩码。
- 验证：`_temp/p21_serve.log` 出现 `path=batched` 与多次 `[tp2-lane] grammar masked N window columns of M lanes`；P2.2 探针里 4 条带 tools 请求同轮出现 `grammar masked … of 4 lanes`，短请求产出合法 `call:get_weather({"city":"Paris"})`；MTP C=4 冒烟（见下）同样以带 tools 的 `/v1/responses` 返回 200 + `function_call`。

### P3 施工记录（B3/B6/B1/B4）

- **P3.1＝B3**：`tp2_generation_core.cpp:558-562` 门槛改为 `mtp_enabled_ && lanes_ == 1U`；启动日志 `mtp chain` 字段在 `!mtp_enabled_ || lanes_ > 1U` 时显示 `n/a`；`mtp_propose_window` 增加 `mtp_chain_host_ == nullptr` 守卫（`:1846-1849`）。可达性：`mtp_propose_window` 唯一调用点在 `execute_walk` 内，而 `execute_walk` 只在 `lanes_ == 1` 分支被调用。
- **P3.2＝B6**：`tools/win_port/build.ps1:134` 加注释说明该修复本来就是条件化的（前缀一致时直接返回，不做全量 touch + 删 `.ninja_deps`）。
- **P3.3＝B1**：`src/serve/http_server.cpp:224-234` 注释固定容量不变式（accept 线程不占 worker；`acquire_request_lifetime` 在 `active >= max_concurrency + max_pending_requests` 时抛 `RequestError(Overloaded)`），并说明 `worker_count = queued_requests + 1` 的余量 1 是刻意的（httplib `ThreadPool::enqueue` 满时返回 false，客户端只会看到断连而不是 429）。
- **P3.4＝B4**：`drive_lane_queue` catch 内注释说明爆炸半径＝全部 lane 是刻意选择（一轮是一次 collective）；`docs/serving.md` 增补「A round is also the unit of failure…」。
- 回归：`_temp/fix_gate.log` ⇒ `100% tests passed out of 11 (72.51 s)`（5 个 artifact 用例按设计 Skipped）；`ninfer_qwen3_5_tp2_forward_test.exe --artifact …` exit 0；`…_load_test.exe --artifact …` exit 0；`tp2_sessions` 见下。

### P0.3（P3 验证期新发现）：MTP 多卡 warmup 失败 —— `proposal_argmax` 的 peer arena 泄漏

- 现象：`--spec mtp --max-concurrency 4` 启动 warmup 报 `TP-2 batched verify CUDA Graph replay found a different workspace layout`（抛出点 `tp2_generation_core.cpp:1551-1553`）；`--max-concurrency 1` 正常；DFlash2 多 lane 不受影响。
- 根因：`src/models/qwen3_5/execution/text.cpp:822 TextContext::proposal_argmax` 只在本地 arena 取 scope（`:823 auto proposal_scope = work_.scope();`），但拆分 proposal head 的分支在 **peer 的 arena** 上分配 `hidden_peer`（`:867`）、`part_peer`（`:876`）、`merged_peer`（`:878`）以及 `project(..., peer.work_, ...)`（`:877`）的 scratch，peer 侧没有对应 scope ⇒ 每次调用把 peer 的 bump pointer 抬高约 403,456 B 且永不回收。
- 触发链：多 lane 路线每轮跑 `max_extent` 个 MTP 链步（`ctx_a.mtp_propose_batch`），`max_extent` 随该轮预算变化（warmup 捕获轮 2、重放轮 0）⇒ verify 图捕获时 `arena_begin[1] = round_base + max_extent * 403456`，重放轮对不上，断言失败。逐步 trace 见 `_temp/p3_step_trace.log`：修复前 B 侧 `81920 → 444416 → 847872` 而 A 侧不动，修复后 B 侧恒定 `81920`。
- 归属：**既有缺陷**，不是本轮改动引入——`text.cpp` 在本次工作树里只有这一处新增，最后一次改它的是已提交的 `6a17bb2c`。同文件其他 peer 分配点（`:494`/`:932`/`:1050`/`:2137`/`:2814`）都有 `auto peer_scope = peer.work_.scope();`，只有 `:858` 漏了。
- 修复：`:858` 之后补一行 `auto peer_scope = peer.work_.scope();`（含 4 行注释，共 +5 行）。
- 验证：C=4 `--draft-tokens 2` 冒烟 warmup 通过（`listening on http://127.0.0.1:8099`），两条并发 `/v1/responses`（带 tools 的短请求 + 1729 token 长请求）均 HTTP 200，带 tools 的返回合法 `function_call`；C=1 仍 `mtp chain: graph`、channels 24、HTTP 200；gate 11/11、forward/load exit 0。

### 会话缓存单/多并发一致性（m02934）

背景：`m02832` 的 A/B 探针发现 C=1 能复用系统前缀（`cache 2,048 (74.8%, long anchor)`、TTFT 422 ms），而 C≥2 恒 `cache 0 (0.0%)`、TTFT 1.6 s。本轮逐条比对三条 prefill 路线——`execute_walk`（C=1）、`execute_plain_batch`、`execute_spec_batch`（C≥2）——的接线与共享辅助函数，结论：两处真实缺口（GAP A、GAP D）、一处架构限制（GAP B，只记录）、一处误报（GAP C）。

| 项 | 状态 | 落盘位置 | 验证 |
|---|---|---|---|
| GAP A：两条批量路线没有 walk 的锚点机制，`entry.host_shared_end` 恒为 0，新会话的共享前缀在 `session_recall` 被 `offered > shared` 拒掉 | 已落盘 | 把 walk 的锚点规划与捕获抽成三个成员：`plan_lane_anchors`（`src/runtime/engine/tp2_generation_core.cpp:2435-2531`）、`capture_lane_anchor_from_device`（`:2533-2546`）、`capture_lane_anchor_frozen`（`:2548-2583`）；`LaneReuse` 增 `shared_prefix`；walk 改为调用（`:6761`/`:6771`/`:6991`）；plain 路线四处接线（plan `:3246`、from-device `:3259`、`reuse_path(reuse, anchors.block_frontier, …)` `:3262`、chunk clamp + Divergence/Block 写 `:3350-3351`、frozen `:3446`），spec 路线同构（`:3986`/`:4000`/`:4002`/`:4096-4097`/`:4200`） | 见下 |
| GAP D：批量路线取消 prefill 时无条件 `invalidate_lane_prefill`，丢掉已完成 chunk 的部分进度（walk 会发布它） | 已落盘 | 新增 `publish_partial_prefill`（`:2701-2738`，镜像 walk 取消路径 `:6777-6812`）；两处取消分支改为 `prefilled > 0` 时发布、否则失效（`:3428-3437`、`:4182-4191`）；`prefilled` 提升到 chunk 循环之外（`:3282`、`:4019`） | 见下 |
| GAP B：walk 在 decode 轮写 Grid 检查点（`:7616+`），批量路线不写 | 不实现（记录为架构限制） | C≥2 每次 `finalize` 都走 `release_lane_kv` → `invalidate_host_checkpoints(lane)`，ring 被清空；能跨请求存活的只有 catalog 的三类镜像（frontier / prompt-end / block-shared）。要让 decode 尾部的 grid 边界可用，需给 `SessionEntry` 再加一类镜像，而 host 预算 10240 MiB/shard 只装得下 2 个 3960 MiB 会话（`[tp2-session] host budget …`），收益（客户端重渲染答案时尾部重 prefill 的减少）不抵成本 | 分析结论 |
| GAP C：两条路线发布的 frontier 语义不一致 | 误报 | plain `:3448` / spec `:4202` 的 `publish_lane_prefill(lane.prompt_tokens, …)` 与 walk `:7647-7655` 的 `frontier = prompt_tokens + sampled - 1` 在各自坐标系里是同一点（最后一个被采样的 token 不再前向） | 代码比对 |

GAP A 实测（编译后；探针 `_temp/m03350_probe.mjs` + `_temp/run_m03350_probe.ps1` / `_temp/run_m03420_probe.ps1`，5 请求共享前 2,722 token；修复前 C=2/C=4 全 `cache 0`、TTFT 1.6 s）：

| 臂 | A1 | A2/A3/B1/B2 |
|---|---|---|
| `floor0_c2`（spec dflash2，conc=2，`--session-retention-floor 0`，prompt 2,738） | `cache 0`，2043 ms | `cache 2,048`，869/963/937/883 ms |
| `long_c2`（spec，floor 4096，prompt 10,839） | `cache 0`，7054 ms | `cache 10,240`，1043/976/964/926 ms |
| `plain_c2`（无 `--spec`，floor 0，2,738） | `cache 0`，2249 ms | `cache 2,048`，1203/1229/1387/1276 ms |
| `spec_c4`（conc=4，floor 0，2,738） | `cache 0`，2136 ms | `cache 2,048`，1063/1158/1251/1358 ms |

长 prompt 臂正是用户真实场景（> retention floor）：命中 10,240/10,839 = 94.5%。`_temp/m03350_long_c2.err` 印证链路：A1 存下 `anchor entry=0 position=10240`；A2 候选 `… host_prompt_end=10839 shared_end=10240 shared=10823 reach=10240 via=shared` → recall → `reuse=10240 src=live`。短 prompt 臂必须先 `--session-retention-floor 0` 才建 entry：`session_publish` 在 `history.size() < session_retention_floor_tokens_` 时早退（`:5895`），4096 是用户配置而非代码缺陷。

GAP D 实测（`_temp/m03470_cancel.mjs` + `_temp/run_m03470_cancel.ps1`，prompt 10,839，C1 在 2 s 时断开连接 ⇒ 服务端 `cancellation.requested()` 在 chunk 循环命中）：

- spec：C1 取消后 `anchor entry=0 position=4096`；C2 重试同一 prompt `recall frontier=4096 … reuse=4096 src=live` ⇒ `cache 4,096`、4414 ms（修复前该重试为全量 prefill、`cache 0`）；C3 同族新会话仍 `cache 10,240`、966 ms。
- plain：部分进度 5120 ⇒ C2 `cache 5,120`、3872 ms；C3 `cache 10,240`、1269 ms。

C=1 回归（`_temp/run_m03480_c1.ps1`，conc=1 / floor 4096 / prompt 2,738）：A1 `cache 0` 1933 ms，A2–B2 均 `cache 2,048`（686/719/694/810 ms）——walk 改用共享助手后行为不变。

### 准入预算的 off-by-one：C≥2 下被 clamp 的请求永不准入（m03586）

背景：用户问「开 2 个会话、总上下文超过 `--max-context` 时会发生什么：输出被截断？卸载到内存排队？还是直接报错？」。追源码时发现 `lane_need_tokens` 比真实 KV 需求多算一个 token，与 `submit` 的 `capacity_output = ceiling - prompt + 1`（有意为「最后一个被采样的 token 不再前向」留位）叠加后，预留页数比池页数多 1 ⇒ 请求永不准入。

机制：池页数 `P = pages_for_tokens(max_context)`、池 token 数 `T = 64P`、`margin = lane_kv_margin()`；C≥2 且 `--lane-context 0` 时准入上限 `W = T - margin`。`submit`（`src/runtime/engine/tp2_generation_core.cpp:2035-2081`）在 `prompt > W` 时回 400，否则 `effective = min(requested, W - prompt + 1)`。预留 `pages_for_tokens(prompt + effective + margin)`（`:2800-2802`）：

- `prompt + requested ≤ W`：不 clamp，`pages ≤ P` ✓；
- `prompt + requested = W + 1`：不 clamp（`requested == capacity_output`），`prompt + effective = W + 1` ⇒ `pages = P + 1` ✗；
- `prompt + requested ≥ W + 2`：clamp 到 `W - prompt + 1` ⇒ 同样 `W + 1` ⇒ `pages = P + 1` ✗。

`reserve_lane_kv` 全有或全无（`:2804-2831`），失败即 `requeue_lane_front` 放回队首（`:2922-2926`）；驱动线程在队列非空时不等待（`:2198-2212`）⇒ 忙转、客户端无限等待。TP-2 路线的 `pending_deadline` 形参在定义处未命名、被忽略（`:2022-2026`），`--pending-timeout-ms` 对提交后的等待阶段不生效。`:2798-2799` 的注释（"a request that fits the policy always fits an empty pool … never a deadlock"）因此不成立。

修复：`lane_need_tokens` 改为 `prompt + (budget == 0 ? 0 : budget - 1)`（`:2929-2936`），与单卡计划的 `reserved_context_tokens`（`src/models/qwen3_5/program/planning/request_plan.cpp:258-262`）一致——最后一个被采样的 token 不再前向，不需要槽位。C=1 不经过批量路线（`reserve_lane_kv` 在 `lanes_ <= 1U` 时直接返回 true，`:2805`；驱动线程仅在 `lanes_ > 1U` 时启动，`:721`），故单并发路径逐字节不变。

实测（`_temp/m03620_cap.mjs` + `_temp/run_m03620_cap.ps1`，双卡 `--max-context 4096`、`--spec dflash2`、prompt 3,740、`max_output_tokens 4096`）：

| 臂 | 修复前 | 修复后 |
|---|---|---|
| C=2 对照（`max_output_tokens 20`，不 clamp） | 200 / 2429 ms | 200 / 2517 ms |
| C=2 clamp（`max_output_tokens 4096`） | **无响应**：客户端 30 s 超时（`http=0`），服务端 30 s 内 36,238 行 `[tp2-lane] batch=1 capacity=2 path=batched`、日志 1.52 MiB（忙转） | 200 / 2371 ms、`status=completed`、7 行 lane trace、日志 0 MiB；`[tp2-capacity] … clamped to 356` |
| C=1 clamp（同一请求） | 200 / 2589 ms、clamped to 357 | 200 / 2451 ms、clamped to 357（不变） |

对外行为（回答原问题）：单个请求的 prompt 超过上限 ⇒ HTTP 400 `context_length_exceeded`；输出预算超过上限 ⇒ 不报错，clamp 输出预算（C=1 起即如此）；两个会话总量超过池 ⇒ 后到的请求留在 FIFO 队首等前一个 lane 退役（不截断、不报错；「卸载到内存」是另一套机制，即 S1 的会话级 host 留存）。修复前「输出预算被 clamp」在 C≥2 恰好落进上面的挂死分支。

顺带发现的两个相邻缺口（已在下一节修复）：(1) TP-2 路线的 `pending_deadline` 被忽略 ⇒ 排队阶段没有 `--pending-timeout-ms` 强制（`docs/serving.md:1039-1045` 的对外说法对 TP-2 不成立）；(2) 排队中的请求不检查 `cancellation`（只在 executor 内检查，`:3280`/\`:4020\`/\`:6781\`/\`:7105\`）⇒ 客户端断开后该请求仍占队并会被执行。

### 排队阶段的超时与取消：提交者自己强制（m03710）

背景：上一节末尾的两个相邻缺口。TP-2 路线的 `submit` 形参带 `pending_deadline` 但定义处未命名、被忽略，`wait_lanes` 只等自己的成员 `complete` ⇒ `--pending-timeout-ms` 对提交后的排队阶段完全不生效（`docs/serving.md:1039-1045` 的对外说法对 TP-2 不成立）；排队中的请求也不检查 `cancellation`（只有 executor 内的检查，`:3280`/`:4020`/`:6781`/`:7105`），客户端断开后仍占队、仍会被执行。

机制：单卡路线的 admission（`src/runtime/engine/engine_core.h:1579-1597`）先查绝对 deadline 再查 `cancelled`：超时 ⇒ `RequestError(QueueTimeout, "inference request expired while waiting for admission")`（HTTP 503 `request_queue_timeout`，`src/serve/generation_service.cpp:69-74`），取消 ⇒ `complete_detached_cancelled`（499）。TP-2 把同一对检查搬到提交者：`admitted` 是唯一的交接位，只在 `lane_queue_mutex_` 下改——谁把成员从 FIFO 取出（驱动的批量形成，`src/runtime/engine/tp2_generation_core.cpp:2294-2313`；或批次中途的 `try_pop_lane_queue`，`:2889-2913`）就置位，`requeue_lane_front`（`:3044-3051`）放回队首时清零。`admitted == false` 期间由提交者强制 deadline 与取消；置位后交回 executor 既有检查。

deadline 与取消都无法 signal 进 `lane_queue_cv_`（deadline 不是受锁保护的状态变化，取消标志是 HTTP 线程写的 atomic），故 `wait_lanes` 的等待上限 250 ms（`kQueuedCancelPollInterval`，`:2149`）并每轮复查；deadline 优先于取消，与单卡一致。退休方式与驱动自己的 pop 相同：`drop_expired_lane`（`:2172`）存下提交者要 rethrow 的 `QueueTimeout`（不发 preview），`drop_cancelled_lane`（`:2164`）发出与 executor 首个取消检查一致的 terminal Cancelled preview。驱动与 `try_pop_lane_queue` 也拒绝把 lane 交给已过期/已取消的成员，所以 requeue 的成员不会在提交者睡着时被抢先执行。

实测（`_temp/m03720_queue.mjs` + `_temp/run_m03720_queue.ps1`；双卡 `--max-context 4096`（池 64 页）、`--spec dflash2`、prompt 2,748、`max_output_tokens 1024`（req1 长生成 18 s 占满池，req2 与它同尺寸 ⇒ 必然排队）；req2 在 req1 开始后 600 ms 发出）：

| 臂 | 客户端 | 服务端 |
|---|---|---|
| timeout（C=2，`--pending-timeout-ms 2000`） | req2 **HTTP 503**，2016 ms，`inference request expired while waiting for admission`；req1 200 / 18.3 s / 1024 tok | `publish tokens=0` ⇒ `WARN req#2 failed during generation … HTTP 503 request queue timeout`（req1 仍在解码，证明 req2 从未拿到 lane） |
| cancel（C=2，30000 ms，req2 在 2505 ms abort） | req2 客户端 AbortError；req1 200 / 18.5 s | `publish tokens=0` ⇒ `req#2 done … cancelled … prompt 0 / output 0` |
| control（C=2，短请求两发，均能准入） | 两个 200（463/291 ms） | 无 `publish tokens=0` |
| C=1 回归（短请求两发） | 两个 200（379/325 ms） | 无 `publish tokens=0` |

C=1 不经过该路径（`wait_lanes` 只在 `lanes_ > 1` 时由 `Submission::wait` 调用；C=1 直接在 `execution_mutex_` 上排队，无 deadline 强制，与改造前逐字节一致）。

## 四项收尾：413 报文、KV 碎片化、S1 往返代价、C=1 vs C=4（m03963/m03964）

### 1. HTTP 413 只应来自 `--max-request-mib`（已修）

根因：`docs/serving.md:420-422` 的契约是「413 `request_too_large` 保留给 JSON 解析之前就超过 `--max-request-mib` 的裸请求体」，但 vendored cpp-httplib 对 `application/x-www-form-urlencoded` 另有一条 8192 字节上限：`CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH`（`third_party/cpp-httplib/httplib.h:133-134`）在普通 Handler 路由的 `Server::read_content`（`httplib.h:13326-13336`）里直接置 413 并 return false，与 `payload_max_length_` 无关，且发生在任何 handler 之前 ⇒ 应用层无法拦截，报文还会错误地宣称「超过了配置的上限」。

修法：8 条带 body 的路由（7 个 POST + 1 个 DELETE，`src/serve/http_server.cpp:492-529`）改走 httplib 的 content-reader 路由（`HttpServer::register_post`/`register_delete`，`src/serve/http_server.h:77-81`；`buffer_request_body`，`src/serve/http_server.cpp:41-61`）。该路径经 `read_content_core`（`httplib.h:13350-13425`）只在 `:13409` 按 `payload_max_length_` 判超限，无 form 上限、不做 `parse_query_text`（无 DoS 放大）；`pre_routing_handler_`（鉴权，`httplib.h:13747`）仍在读 body 之前运行。`handle_unrendered_http_error` 与 `tests/test_http_error_handler.cpp` 未改。

实测（`_temp/m04130_413.mjs`；双卡 `--max-request-mib 1`、`--spec dflash2`）：

| 请求 | 修复前 `61F09142…` | 修复后 |
|---|---|---|
| 18,234 B、`application/json`（对照） | 200 / 2,538 ms | 200 / 2,427 ms |
| 18,234 B、`application/x-www-form-urlencoded` | **413**「exceeds the configured payload limit of 1048576 bytes」（body 只有 18 KB） | **200** / 2,357 ms |
| 2,097,202 B、`application/json` | 413 | 413「…1048576 bytes」 |
| 2,097,181 B、`application/x-www-form-urlencoded` | 413 | 413「…1048576 bytes」 |

流式回归：content-reader 路由下 SSE 照常（`STREAM http=200 dataLines=20/22 eventLines=20/22 completed=true`，C=1 与 C=4 各一次）。

### 2. KV 池碎片化：外部碎片为 0，内部碎片 = 页取整 + 整个输出预算持有到退役

结构上准入不可能因碎片失败：`DeviceKVPagePool::reserve` 只判断 `pages == 0 || pages > available_pages()`（`src/core/paged_kv_cache.cpp:266-270`），预留是纯计数器，不要求物理连续；物理连续性只影响 `contiguous_run_count`（`:253-264`）与 `materialize` 的相邻偏好（`:288-320`），即只影响 `copy_to_host`/`copy_from_host`（`:576-660`）的 `cudaMemcpy2D` 批次数。

新增 env 门控 trace `NINFER_TP2_KV_TRACE=1`（`src/runtime/engine/tp2_generation_core.cpp:2922-2932`）在 `reserve_lane_kv`（`:3003-3015`）、`release_lane_kv`（`:3033-3042`）、`session_store_active`（`:5628-5637`）、`session_restore`（`:5765-5777`）打印池计数、该 lane 的 run 数与 store/recall 墙钟。

实测（`_temp/m04130_kv.mjs` + `_temp/run_m04130_mix.ps1`；双卡用户配方、`--max-context 245760` ⇒ 池 3,840 页/卡、1,081,344 B/页 ≈ 3.96 GiB/卡）：34 次 reserve/release、18 次 store/recall，`runs` **全部为 1/1**（两卡都是单一连续 run），尺寸覆盖 1/44/60/86/107/171/234 页并含并发交错 ⇒ 未观测到物理碎片。

内部碎片（预留页 vs 实际存储页）：

| 请求 | 预留页 | store 页 | 闲置 |
|---|---|---|---|
| prompt 10,839 + budget 64 | 171 | 170（frontier 10,857/10,863） | 1 页 = 1.03 MiB/卡 |
| prompt 5,418 + budget 64 | 86 | 85–86 | 0–1 页 |
| prompt 2,738 + budget 64 | 44 | 44 | 0 |
| prompt 10,836 + budget 4096（20 tok 即停） | 234 | 170 | **64 页 = 66.0 MiB/卡（132.0 MiB 两卡）** |
| prompt 2,735 + budget 4096（25 tok 即停） | 107 | 44 | **63 页 = 65.0 MiB/卡** |

并发峰值：两个 234 页请求同时持有 ⇒ `allocated 468/3840`（12.2%），`available` 最低 3,372/3,840（87.8%）。结论：浪费是内部的、且是既定策略（整份输出预算一次性预留、持有到退役、不缩容、不抢占，m01589/m01553），不是分配器碎片；它的可见代价是「本可容纳的请求因为整份预留而排队」，而不是失败或截断。

### 3. S1 会话 host 往返的吞吐代价

| 操作 | 页数 | KV 字节/卡 | ms | KV 单项下界（两卡合计） |
|---|---|---|---|---|
| store | 170 | 175.3 MiB | 8 次：79.0 / 中位 ≈108 / 133.8 | 3.13 GiB/s |
| store | 86 / 85 | 88.7 / 87.7 MiB | 67.3 / 71.9 | 2.4–2.6 GiB/s |
| recall（frontier） | 160 | 165.0 MiB | 6 次：60.9 / 中位 64.5 / 66.6 | 5.00 GiB/s |
| recall（shared） | 160 / 80 | 165.0 / 82.5 MiB | 59.5 / 37.7 | 5.42 / 4.27 GiB/s |

（KV 单项下界 = 只算 KV 页字节、不含同区间内的 lane state D2H/H2D 与 dflash 草稿镜像 ⇒ 实际带宽更高。store 只搬实际 frontier，不搬整份预留。）

对照收益：10,845 token 冷 prefill 6.4 s（1.69k tok/s）→ 命中 10,240 token 后 TTFT 0.45–0.47 s（605 token @1.33–1.53k tok/s）。一次完整往返 store+recall ≈ 0.11 + 0.065 = **0.17 s**，即约 3% 的收益；代价固定且有界（每次退役 ~0.11 s，与之后是否复用无关）。

### 4. C=1 vs C=4 单请求配对测量

`_temp/m04140_pair.mjs` + `_temp/run_m04140_pair.ps1`；同一 prompt（10,845）、同一 `max_output_tokens 256`、同一用户配方，只改 `--max-concurrency`；每臂 3 次连发 + 1 次流式。

| 指标 | C=1 | C=4 |
|---|---|---|
| R1 冷 TTFT | 6.4 s | 6.4 s |
| R1 冷 prefill | 1.69k tok/s（10,845 tok） | 1.70k tok/s（10,845 tok） |
| R1 冷 decode | 115.8 tok/s（dflash2 201/269） | 104.9 tok/s（198/283） |
| R1 客户端总时长 | 8,676 ms | 9,215 ms |
| R2 复用 TTFT | 406 ms（cache 10,240，long anchor） | 466 ms（cache 10,240，private endpoint） |
| R2 prefill / 客户端 | 1.53k tok/s / 2,599 ms | 1.33k tok/s / 3,001 ms |
| R3 复用 TTFT / 客户端 | 405 ms / 2,481 ms | 453 ms / 4,082 ms |
| 设备占用（卡0/卡1） | 14,924 / 14,678 MiB | 15,512 / 15,134 MiB |
| state arena（卡1） | 146.8 MiB | 587.2 MiB |
| host checkpoint ring（卡1） | 35 槽 × 73.4 MiB，stride 10,240，pinned 2,569.2 MiB | 36 槽 × 73.4 MiB，stride 81,920，pinned 2,642.6 MiB |
| 卡1 free | 620.0 MiB | 164.0 MiB |

结论：多 lane 机制对「单独一个请求」的 prefill 吞吐没有影响（1.69k vs 1.70k tok/s，TTFT 都是 6.4 s）；冷启动端到端 +6%（decode 差异由 dflash2 接受率方差主导：74.7% vs 70.0%，n=3 不可分辨）。稳定的差异是复用路径（C≥2 走 host slab：TTFT +15%，406→466 ms）与显存（+588/+456 MiB 设备 + 73.4 MiB pinned，卡1 free 620→164 MiB）。C=4 R3 的 decode 80.1 tok/s（接受率 48.5%）说明单机 n=3 的 decode 对比不可控，只有 prefill 与显存两项可判读。
