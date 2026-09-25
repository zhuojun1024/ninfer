# Round 17 ①：plain decode 上 CUDA Graph — A/B 报告

## 三臂配置（唯一差异是 `NINFER_TP2_DECODE_GRAPH`）

`D:/LLM/qwen3_8_27b_w4a4_w8a8.ninfer`、`--devices 0,1`、`--max-context 204800`、`--kv-dtype fp8`、
`--no-prefix-reuse`、`--host 127.0.0.1 --port 3457`、无 `--spec`（plain 路线）、
`NINFER_TP2_TIMING=1`；曲线 `-Sizes 1024,32768 -Repetitions 3 -DecodeTokens 256`。

| 臂 | 含义 | decode tok/s @1018 | @32768 | prefill tok/s @1018 | @32768 | avg_round @32768 |
|---|---|---:|---:|---:|---:|---:|
| `exact` | eager + 逐请求精确 envelope（= 改动前代码路径） | 31.11 | 29.79 | 1634.6 | 1506.7 | — |
| `0` | eager + 桶 envelope（`ordinary_graph_profiles`） | 31.10 | 29.80 | 1635.9 | 1504.9 | — |
| 缺省（graph） | 捕获 graph + 桶 envelope | **35.03** | **33.39** | 1628.4 | 1505.0 | 29.93 ms |

- decode：1K **+12.6%**、32K **+12.1%**（3 reps，sd 约 0.01 tok/s）。
- 换算成轮时间：eager 32.1/33.6 ms → graph 28.5/29.9 ms，即每轮省 **约 3.6 ms**（与 PLAN 预估的 2.91 ms 发射间隙同量级）。
- prefill 三臂一致（±0.5%，噪声内）⇒ 改动只作用于 decode，符合设计。

## 数值比对（7 条固定 prompt，temperature 0，160 tokens）

| 比较 | 结果 |
|---|---|
| 捕获 graph vs eager 桶 envelope（换发射方式） | **7/7 完全一致** |
| eager 精确 envelope（复跑） vs 桶 / vs graph | 7/7 完全一致 |
| eager 精确 envelope（首跑） vs 其余 | prompt 0 差 13 字符 |
| eager 精确 envelope 首跑 vs 同配置复跑 | prompt 0 差 13 字符 |

结论：
- 捕获-回放与 eager 发射**逐位一致**（核心正确性判据通过）。
- 桶 envelope 与精确 envelope 一致（与 PLAN Round 17 前置结论 1 吻合）。
- 唯一一次差异出现在同一模式 `exact` 的两次独立运行之间，**可归因于运行间不可复现性，而非本改动**：无 `--spec` 路线跨进程不保证逐位复现；同一进程内重复同一 prompt 集也会分叉（疑似前缀复用使第 8 次请求走 cached-prefill 而非整段 prefill）。
- 因此 token 级比对在本轮只能作为「无回归」证据，不能作为逐位判据；与模式相关的差异为 **0**。

## 实现要点

- 新增 `TextContext::forward_tp2_decode_window`（`text.cpp`）：`forward_tp2` 的逐行克隆，唯一差异是 token/position 由 `copy_i32`（H2D memcpy，可捕获）而非 `ops::set_i32_scalar`（host 值，不可捕获）写入，envelope 由参数给出。
- `tp2_generation_core` 新增 `DecodeStepMode{Graph,EagerBucket,EagerExact}` 与 `std::vector<WindowGraph> decode_graphs_`；`NINFER_TP2_DECODE_GRAPH` = 缺省 `graph` / `0` → `EagerBucket` / `exact` → `EagerExact`；非 in-kernel allreduce（P2P/host staging 会同步 host）时强制 `EagerExact`。
- 桶来自 `qwen::detail::ordinary_graph_profiles(options_.max_context)`，`visible_begin = profile.min + 1`、`visible_end = min(max_context, profile.max + 1)`；桶 envelope 逐位精确（前置结论 1）。
- 启动打印 `[tp2-graph] plain decode step: <graph|eager(bucket)|eager(exact)> | verify step: <...>`。

## 落盘

`profiles/bench/tp2_decode_graph/`：`curve-{exact,bucket,graph}.json`、`greedy-{exact,bucket,graph,exact2,exact2b}.jsonl`、
`serve-*.log`（含 `[tp2-graph]` 与 `[tp2-time]`）。

## 构建脚本（用户指出的问题）

`tools/win_port/build.ps1` 重写：`Start-Process` 文件重定向代替管道 + 硬超时 + `ninja` 子进程管道预检 +
运行中的 `ninfer-serve` 占用 exe 预检；沙箱下 8 秒内以可读原因退出，正常下 37 秒完成增量构建。
根因：`ninja` 为每个子进程建管道，沙箱拒绝建管道 ⇒ ninja 在第一个 job 前阻塞，无输出、无子进程、0 CPU。

## Round 17 ②：AR 按张量大小切传输策略 — A/B 报告

### 策略

`NINFER_TP2_AR_STRATEGY=size`（缺省）/ `kernel`（A/B 参照，全部走原切片路径）。

| payload | 传输 | 依据 |
|---|---|---|
| ≤ 64 KiB（decode 单 token delta = 10 KiB；MTP verify 4 token = 40 KiB） | 单 block；`bump_ar_token` 融进 AR kernel 的 thread 0（`fuse_bump`），发射次数 2→1；线程数仍为 `kArThreads`；独享 2×64 KiB mapped staging，槽位不与 prefill 槽别名 | 延迟问题：每轮 128 次调用，固定开销占大头 |
| > 64 KiB（prefill） | 不变：切片 + 独立 bump + 24 MiB staging | 带宽问题，已贴 PCIe 链路地板 |

跨卡定序仍靠 mapped-pinned arrival token 自旋；融合把计数器的读-改-写放进同一个 kernel，由 thread 0 用
`__threadfence_system()` 发布，peer 从不读对端的计数器。

### 实测（同一二进制，缺省 decode graph，fp8 KV，204800）

| 臂 | decode tok/s @1018 | @32768 | avg_round @32768 | prefill tok/s @32768 |
|---|---:|---:|---:|---:|
| `kernel`（参照） | 35.12 | 33.48 / 复跑 33.48±0.02 | 29.86 ms | 1506.4 |
| `size` 首版（256 线程） | 35.07 | 33.35 | 29.95 ms | 1504.4 |
| `size` 终版（1024 线程，保留） | **35.34** | **33.65** | **29.70 ms** | 1505.8 |

- 终版：**+0.63% @1K、+0.51% @32K**（avg_round −0.16 ms）。
- 首版 256 线程是回归（−0.2%/−0.4%）：收益来自省掉一次发射，亏损来自缩小 block。payload 的真正成本是
  mapped-host 对端往返，group 循环需要并行度，所以线程数保持 `kArThreads` 才是正确配置。
- prefill 是天然对照：三臂全部在 0.1% 内不变 ⇒ 大 payload 路径确实没被动到。
- 与实现前预估的 +1–2% 相比：落在**下沿**（+0.5–0.6%）。

### 数值

`size`（两种线程数）vs `kernel`：7 条固定 prompt **7/7 完全一致**。融合路径与切片路径逐位相同，
因为逐元素的加法顺序没有变。

### 未实现的一臂（动工前已与你确认）

大 payload 的 copy-engine 传输：prefill 已贴链路地板（10–20 MiB 是带宽问题），且其跨卡定序必须依赖
跨设备 `cudaStreamWaitEvent`，仓库内也已有「copy engine 更慢」的实测；预期 ≈0，故未实现。
