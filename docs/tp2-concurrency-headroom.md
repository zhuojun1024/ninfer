# TP-2 多并发实现的剩余优化空间

> 分析对象：`ninfer-serve --devices a,b`（TP-2 专用路线）的多 lane 批处理实现。
> 范围：显存/内存占用、prefill/decode 速度、TTFT 延迟。
> 结论基于源码审读 + 本机实测（2026-10-02，2×RTX 5060 Ti，HEAD `e9c8ec17`）。
> 已有结论（docs/tp2-decisions.md、docs/tp2-dual-5060ti.md、docs/PLAN-tp2-concurrency*.md）不再重复验证，只做增量。

## 0. 结论摘要

TP-2 多并发的 **decode 聚合已经接近该实现的理想值**：4 lane 等长 batch 实测 3.31×（理想 4×），
损失 17% 全部来自随列数线性增长的那部分（单轮 26.5 ms → 32.0 ms）。权重流已被摊薄，
这块没有数量级的空间。

真正剩余的、用户可见的空间有三类：

| # | 方向 | 实测证据 | 量级 |
|---|------|----------|------|
| 1 | **TTFT / 混载延迟**：prefill 逐 lane 串行；中途加入的 lane 在 round 顶部整段阻塞 decode | 4×10.8k 并发：最后一个 lane 客户端 TTFT ≈23 s（引擎日志只报 5.5 s）；短请求跟在长请求后被阻塞 5.1 s；短请求在前时其 decode 37.7 → 2.5 tok/s（15×） | 最大且最确定 |
| 2 | **显存浪费**：plain 路线的 round-scratch 状态平面完全未使用；KV 按整个输出预算预占并持有到请求结束 | C=4 时 293.6 MiB/卡（两卡 587 MiB）纯浪费；预算 4096 只输出 20 token 后仍占 64 页 = 66 MiB/卡 | 中 |
| 3 | **文档与度量**：`docs/serving.md` 的「262,144 必须 C=1」已过时；plain batch 路线无 round 计时埋点 | 现在 C=4 在 262,144 fp8/MTP K=2 可正常启动并跑满 4 请求（剩 811/1,769 MiB）；`[tp2-time] plain-batch` 恒为 0.00 | 低（但阻塞后续优化） |

已接近硬件上限、不建议继续投入的方向：prefill 聚合吞吐（受卡 2 的 Gen4 x4 链路限制，
2.67k tok/s 天花板）、decode 权重流（已占 75-90% 带宽）、allreduce 调用次数
（减少调用必然牺牲权重切分，见 docs/tp2-decisions.md）。

**执行状态**（2026-10-02，逐项执行见 §6 状态列、§8 与 §9 记录）：§6 清单里的 P0/P1 项已全部实现并实测；
项 5 分析后判定无需改动；项 9 分析后判定不改（会破坏预留不变式）；第二轮增补的四项（投机路线
prefill 交错、项 10 的 Ops 修复、P3 批量 prefill、可分页 host checkpoint ring）中三项已实施并实测
——项 10 的价值是**数值一致性**（接受率收益未复现）。P3 批量 prefill 先量后判：批量窗口
`forward_tp2_window_batch` 在 W=16/B=4 下 1.24 ms/lane-token，而单 lane 1024-chunk 是
0.587 ms/lane-token，**批量慢 2.1×** ⇒ 确定不做（打平需 W×B≈1216 列/轮，是 ops 上限 width ≤ 16 的
19 倍）。剩余空间只剩「先放宽 ops 批量域再谈批量 prefill」与硬件（卡 2 的 PCIe 槽位）。

## 1. 实现现状（本报告用到的结构）

- 每轮一个 compact decode batch，lane 数 = `--max-concurrency`（1..4），lane 身份 = batch 中的位置。
- 驱动线程 `drive_lane_queue`（src/runtime/engine/tp2_generation_core.cpp:2280-2391）持有
  `execution_mutex_` 跨整个 batch 执行；新请求靠 **mid-batch admission** 加入，不另起 batch。
- 路由：plain → `execute_plain_batch`（:3142-3867）；MTP/DFlash2 → `execute_spec_batch`（:3868+）；
  单 lane 走 `execute_walk`（lanes>1 时 `execute_walk` 是死代码）。
- `admit_lane`（plain :3345-3641）在一个 lambda 里完成该 lane 的**整段同步 prefill**：
  分块循环 `for (t0 = reuse; t0 < prompt_tokens;)` 调 `forward_tp2_prefill(...)`（:3468-3605），
  末尾采样首 token。注释 :3453-3455 原文：「Prefill is serial per lane, so the single startup
  arena is reused sequentially and only one session is ever alive」。
- 每 lane 的 KV 从共享 paged pool 动态预留（`reserve_lane_kv` :2938-3000）；
  线性注意力状态 arena 按 lane 分槽（`slot_count = lanes_`，:826）。

## 2. 显存 / 内存

### 2.1 实测：262,144 上限下各并发的设备预算

配置：`--max-context 262144 --kv-dtype fp8 --host-state-slots 32 --host-kv-mib 20480`，
加载后无请求，读 nvidia-smi 与启动 ledger（`_temp/20261002-1350_mem.ps1`）。

| 配置 | GPU0 used | GPU1 used | state/卡 | record/卡 | ledger free 0/1 |
|------|-----------|-----------|----------|-----------|-----------------|
| MTP K=2 + `--lm-head-draft` C=1（出厂配置） | 15,028 | 14,074 | 146.8 | 2.6 | 270.0 / 1,218.0 |
| 同上 C=2 | 15,178 | 14,222 | 293.6 | 5.1 | 122.0 / 1,070.0 |
| 同上 C=4 | 15,482 | 14,522 | 587.2 | 10.2 | 0.0 / 770.0 |
| plain C=4 | 14,338 | 14,338 | 587.2 | 0.0 | 952.0 / 952.0 |

（总显存 16,310.6 MiB/卡；ledger 的 free 列是 cudaMemGetInfo，在该驱动下偏悲观，
以 nvidia-smi 为准：C=4 MTP 实际剩 829 / 1,789 MiB。）

**每 lane 增量成本 = (15,482 − 15,028)/3 = 151.3 MiB/卡 ≈ state 146.8（2 平面 × 73.4）+ record ~5。**
KV pool 由 `--max-context` 固定（4,096 页），**不随 lane 数增长**。

端到端验证（`_temp/20261002-1440_shipped_c4.ps1`，出厂配置 + C=4 + 4 条并发 243-token 请求）：
4 条全部 200、各输出 256 token，wall 5,289 ms，每 lane decode 59.5/54.4/51.8/53.8 tok/s
（聚合 219.5 tok/s），MTP 接受率 80.0-87.1%，运行后剩 811 / 1,769 MiB。

⇒ `docs/serving.md` 的「262,144 配置保持 `--max-concurrency 1`；2-4 lane 需要
`--max-context 131072`」是**过时指导**：它写于状态 arena 为 293.6 MiB/卡/lane
（`kReuseSnapshotCount = 2` 时代）的时候。当前 146.8 MiB/卡/lane，C=4 在 262,144 可正常启动并运行。

### 2.2 浪费 1：plain 路线完全未使用的 round-scratch 状态平面

> **已修（项 2）**：plain 路线不再雕刻该平面。实测启动台账 `state 293.6 | round 0.0`（改前 round 也是 293.6），每卡省 293.6 MiB。见 §8.1。

`state_arena` 的平面数（:839-845）：

```cpp
const std::size_t first_state_plane = host_checkpoint_stride_ != 0 ? 1U : 0U;
const std::size_t state_planes      = 1U + shard.state_snapshots.size() - first_state_plane;
```

在 host ring 打开（默认）时 `first_state_plane = 1`，于是只雕出 slot 1（= `kRoundScratchSlot`）；
slot 0（复用边界）由 host ring 承担（:2785-2788 注释明确说明）。

而 `state_snapshots[kRoundScratchSlot]` 的**全部引用点**只有：
src/runtime/engine/tp2_generation_core.cpp:4701、:4886、:5079（`execute_spec_batch`）
和 :7761、:7804（`execute_walk`，且都在 `dflash2_enabled_` / MTP 分支内）。
`execute_plain_batch`（:3142-3867）**零引用**（全文件 grep 确认）。

⇒ plain 路线上该平面被分配、被 memset，却从不读写。代价 = 73.4 MiB × lanes /卡：

| 配置 | 浪费/卡 | 两卡合计 |
|------|---------|----------|
| plain C=1 | 73.4 MiB | 146.8 MiB |
| plain C=2 | 146.8 MiB | 293.6 MiB |
| plain C=4 | 293.6 MiB | 587.2 MiB |

修法（数行）：把 :839 的跳过条件从「ring 是否存在」扩成
`host_checkpoint_stride_ != 0 || !(mtp_enabled_ || dflash2_enabled_)`，即 spec 路线之外不雕 scratch 平面。

### 2.3 浪费 2：整个输出预算被预占并持有到请求结束

> **已分析，不改（项 9）**：按剩余量收缩会破坏 all-or-nothing 预留不变式，使 lane 需要中途增长 → 无背压即死锁。操作杠杆是 `--lane-context`（项 6）。见 §8.6。

`lane_need_tokens`（:3113-3116）= `prompt_tokens + (budget == 0 ? 0 : budget - 1)`，
`reserve_lane_kv` 在 admission 时一次性预留全部页，**不收缩、不抢占**，直到 lane 退役。

实测（前序工作，`NINFER_TP2_KV_TRACE=1`）：

- 10,836 token prompt + budget 4096、实际只输出 20 token → 预留 234 页 / 实际写入 170 页，
  64 页 = **66.0 MiB/卡（两卡 132 MiB）空闲占用**。
- 2,735 + budget 4096、输出 25 token → 107 / 44 = 63 页 = 65.0 MiB/卡。

对外表现是「本可容纳的请求排队」，不是失败或截断。在高并发 + 大 `max_output_tokens` 的
客户端（工具类请求常带大预算）下，这会直接压低实际并发度。

可选修法：budget 按剩余量动态收缩（释放尾部页）、或对「已完成但未退役」的 lane 做页回收、
或把默认 `--lane-context` 设成 pool_tokens / max_concurrency（见 2.4）。

### 2.4 风险：`--lane-context` 默认 0 允许单 lane 独占整个 pool

> **已实施（项 6）**：多 lane 且 `--lane-context 0` 时启动打印告警（含上限 token 与 lane 数），不做硬默认（硬默认会拒绝长于 pool/lanes 的 prompt）。见 §8.8。

`src/serve/serve_options.h:37` `std::uint32_t lane_context = 0;`；
tp2_generation_core.cpp:443-450 在 lanes>1 时把 ceiling 设为 `pool_tokens - margin`，
只有操作员显式给 `--lane-context` 才会压低。

⇒ 默认配置下，先到的单条大请求可以合法占用几乎整个 KV pool，后到的 lane 一直排在 FIFO 头部
等它退役——**默认配置下并发度可能退化为 1**。这与 2.3 的「全额预占」叠加。
建议：多 lane 路线把 `--lane-context` 默认取 `pool_tokens / lanes_`（或至少打印一条警告）。

### 2.5 Host 侧 pinned checkpoint ring

`[mem] host-checkpoints shard N slots … x 73.4 MiB | stride … | pinned …`：

| 配置 | slots | stride | pinned/卡 |
|------|-------|--------|-----------|
| 262,144 C=1 | 35（grid 24 + tail 8 + div 1 + block 1 + prompt-end 1） | 11,008 tok | 2,569.2 MiB |
| 262,144 C=2 | 36（grid 8 + tail 7 + …） | 32,768 tok | 2,642.6 MiB |
| 262,144 C=4 | 36（grid 3 + tail 3 + …） | 87,424 tok | 2,642.6 MiB |
| 245,760 C=4（dflash2） | 36 | 81,920 tok | 4,082.6 MiB |

这是**不可分页的 host 内存**，两卡合计 5.1-8.2 GiB。收益是避免重 prefill（一次
10,240 token 命中把 TTFT 从 6.4 s 降到 0.45 s）；代价是 store/recall 约 0.17 s/次
（store 3.13 GiB/s、recall 5.00 GiB/s，见前序实测）。

可调项（不是缺陷）：ring 的 pin 是为了 DMA 带宽；若改为可分页，代价是那 0.17 s 变慢，
换来 GiB 级 host 内存。C≥2 时 grid 从 24 槽掉到 3 槽（stride 11,008 → 87,424 tok），
即文档所述的「4 lane 下 grid 粗约 4×」。

### 2.6 其他

- `--kv-dtype`：int8 在同 ceiling 下比 fp8 **大** 48 MiB（64 元素 scale group）；
  nvfp4/k8v4 更小但有值精度代价。想要更大 context 或更多 headroom，这是唯一的 KV 侧旋钮。
- 权重 + CUDA context 10,448.6-11,528.6 MiB/卡是绝对主导项，不可压缩（除换切分方案）。
- 没有观察到 KV pool 外部碎片（前序 34 reserve/release、18 store/recall，runs 全为 1）。

## 3. decode 速度

### 3.1 实测扩展曲线

plain，bf16 KV，`--max-context 8192`，243 token prompt，4 lane 等长且都打到 256 token 上限
（`_temp/20261002-1310_scale.ps1`，每 lane 速率取自 `done |` 行的 `decode X tok/s`）：

| C | 每 lane tok/s | round ms | 聚合 tok/s | 理想 | 效率 |
|---|---------------|----------|------------|------|------|
| 1 | 37.7 | 26.5 | 37.7（1.00×） | 1× | 100% |
| 2 | 35.0 / 35.9 | 28.2 | 70.9（1.88×） | 2× | 94% |
| 3 | 33.5 / 32.1 / 32.8 | 30.5 | 98.4（2.61×） | 3× | 87% |
| 4 | 30.3 / 32.2 / 31.5 / 30.9 | 32.0 | 124.9（3.31×） | 4× | 83% |

**轮成本模型：round(B) ≈ 26.5 + 1.85 × (B − 1) ms。**
即每多一列只多 1.85 ms（≈ 单 lane 轮的 7%），理想（纯权重流受限）是 0。
C=4 损失 17% 的聚合增益；若把边际列成本减半，C=4 可达 ≈3.62×。

> 与文档的差异：docs/serving.md 与 docs/PLAN-tp2-concurrency.md 记的 plain 路线聚合是
> C=2 1.76-1.91×、C=4 2.80×。本次复现了 C=2（1.88×），但 C=4 测得 **3.31×**，因为本次四
> 条 lane 完全等长（都打到 256 token 上限）。文档自己也写了「轮的成本等于最长成员的成本：
> 输出长度差两倍的一对只测到 1.41×」——**2.80× 混淆了轮成本与 lane 完成时间不均衡**。
> 等长测量才是纯粹的轮成本上界；真实负载的聚合低于此，取决于长度离散度。

### 3.2 边际 1.85 ms/列 的去向

可分离的部分（估算，按 448 GB/s 与实测 7 GB/s 链路）：

| 项 | 每列/轮 | 依据 |
|----|---------|------|
| GDN 线性注意力状态读写 | ~0.33 ms | 73.4 MiB/卡/lane 的 state image，读+写 |
| allreduce payload | ~0.19 ms | hidden 5120 × 2 B × 2 delta × 64 层 = 1.31 MB/列，链路 ~7 GB/s |
| 全注意力 KV 读 | ~0.07 ms（499 tok 上下文）；4,543 tok 时 ~0.62 ms | 见下 |
| 未解释 | ~1.2 ms | 见下 |

KV 读不是主因：单 lane 上下文从 499 → 4,543 token，round 只从 26.5 → 27.12 ms
（Δ0.62 ms / +4,044 token，即 0.153 µs/token）。KV 字节数涨了 9×，轮成本只涨 2.3%，
所以 KV 读在 4.5k 上下文时也只占一轮的 ~2%。

剩下 ~1.2 ms/列 **无法在现有埋点下分离**：plain batch 路线没有 round 计时（见 3.4），
候选是 host 侧每 lane 工作（sampling config 暂存、D2H、commit、publish、session 记账）
与批核函数的每列低效（split 策略、grid 形状）。

### 3.3 投机路线：批处理的边际收益更小

docs/PLAN-tp2-concurrency.md:441-462 的实测（本报告未重测）：

| 路线 | C=2 | C=4 |
|------|-----|-----|
| plain（等长） | 1.88× | 3.31× |
| MTP K=5 | 1.48× | 2.26× |
| DFlash2 draft=5 | — | 2.07× |

原因：单请求的投机解码已经把权重流按 K+1 列摊薄，批处理只能再摊 lane 维。
**这是设计取舍，不是缺陷**：要单用户延迟用 `--spec`，要多用户吞吐用并发。

同文档记录的副作用：MTP 批链的 envelope 是**整个 batch 一个值**（max_position + step），
非最长 lane 的 MTP 头会漂移、接受率下降（lane1 34/62 vs 单跑 36/52）。
作者把它归因于 envelope；本次未独立隔离（也可能是 prompt 内容差异），列为待确认项。
若确认，修法是给批链传每 lane 的 positions/envelope（批链本来就是 eager 循环）。

### 3.4 度量缺陷：plain batch 路线没有 round 埋点

> **已修（项 1）**：plain 路线有专用埋点 `[tp2-time] plain-batch … decode=/grammar=/sample=/readback=/prefill_chunks=/pump_rounds=`。见 §8.2。

`execute_plain_batch`（:3142-3867）只有 :3837-3838 的
`timing.committed += …; ++timing.rounds;` 和 :3847 的 `timing.report("plain-batch")`，
**没有 `timing.record(…)` / `close_round(…)`**。于是每次 C≥2 的 plain 运行都打印：

```
[tp2-time] plain-batch rounds=375 committed=375 avg_round=0.00ms mtp=0.00 verify=0.00
accept=0.00 copy=0.00 sync_wait=0.00 fold=0.00 prefill=0.0ms
```

spec 路线（:4541/4697/4716/4779/4787/4978/5075/5090-5100）与单 lane walk（:7340-7902）
都有埋点。**文档里所有 plain 路线的聚合数字都来自这条没有埋点的路径**，
这既妨碍继续优化，也让线上问题难以定位。修法：照抄 spec 路线的 7 个 event。

### 3.5 建议（decode）

1. 补 plain batch 的 round 埋点（低成本、解锁后续一切分解）。
2. 分解出那 ~1.2 ms/列 后，针对大头下手：
   - 若 host 侧：把每 lane 的 commit/publish/sampling 暂存移出关键路径，或与下一轮 GPU 工作重叠。
   - 若核函数 split 策略：注意 `causal_attention_chunk_tokens` 对 q_heads≠24（TP-2 shard 几何）
     恒返回 6（src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:25-35），
     即 TP-2 上没有为 batch 形状做 split 平衡。
3. 不要指望消除权重流；那是被摊薄的部分，已经是收益来源。

## 4. prefill 速度

### 4.1 成本模型

- 单 lane 4,543 token：prefill 2,264.6 ms = **0.498 ms/token（2.01k tok/s）**。
- 单 lane 10,826 token：1.99k tok/s；10,845 token（dflash2 路线）：1.69-1.70k tok/s。
- 传输边际斜率 0.368-0.375 ms/token（与 chunk 宽度无关）⇒ **卡 2 的 Gen4 x4 链路（~7 GB/s）
  给出 2.67k tok/s 的聚合天花板**。
- 已 shipped 的 `--prefill-overlap 256`（N=4，64 对齐子块，AR 走
  `DeviceContext::collective_stream`）把 0.587 → 0.516 ms/token（−12%）；
  该优化在 `forward_tp2_prefill` 内部（text.cpp:2661），因此批路线的 inline prefill 同样受益。
- 剩余可挖：0.498 − 0.368 ≈ **0.13 ms/token（26%）**，即把 AR 完全藏进计算。
  对应单 lane 上限 ~2.7k tok/s。

### 4.2 为什么「批量 prefill」提升不了长 prompt 的聚合吞吐

`docs/PLAN-tp2-concurrency.md:446` 的作者估算：「若 prefill 也批量，B=2→1.68×、B=4→2.72×」。
**这个估算只对短于一个 chunk 的 prompt 成立**：

1. 每 token 的 allreduce 字节数不因批量而减少（T 维线性），所以传输时间总量不变。
   既然 prefill 已经贴着 2.67k tok/s 的链路天花板，批量不能提高聚合吞吐。
2. 权重读的摊薄已经发生：拟合 chunk 512（0.625 ms/tok）与 chunk 1024（0.587 ms/tok）得
   每 token ≈0.549 ms + 每 chunk ≈39 ms 固定，即 1024-token chunk 的权重读只占 ~7%。
   批量能省的是这 7% 的一部分，不是主要矛盾。
3. 内核层面也做不到：批量形式要求每 lane 宽度 ≤ `kMaximumVerifyTokens`(16)
   （causal_softmax_attention.cpp:241-242 与 :412-413），`kMaximumBatchSize = 8`；
   而 batch=1 的 prompt 核用 `positions[0]` 作为单一基准（prompt_nvfp4.cuh:126）。
   即现有 batched 核只能做「B lane × ≤16 token」的 verify 窗口，
   要支持「B lane × 1024 token」需要**新写一个 ragged 大 T 多序列 prompt 核**。
4. 即便写出来，workspace arena 也要跟着长：192 MiB 只覆盖单 lane 1024-token chunk 的
   ~140 MiB 峰值，B lane 需要 ~4×，会吃掉 2.1 的显存 headroom。

**批量 prefill 的真正价值是 lane 完成时间均衡（TTFT），不是吞吐。**
对长 prompt，聚合上限是链路给的 2.67k tok/s（当前 1.99k 的 1.34×）。

### 4.3 硬件是最大单点杠杆

卡 2 插在芯片组 Gen4 x4 槽（~7 GB/s）。挪到 CPU 直连 Gen5 x8 槽，prefill 上限随之抬升，
且对 decode 的 AR 也有利。这已在 docs/tp2-decisions.md 记为「最大单点杠杆」。

## 5. TTFT

### 5.1 实测（plain，`--max-context 245760 --kv-dtype int8`，C=4）

`_temp/20261002-1310_ttft.ps1`：

**B1：4 条 ~10,826 token 不同 prompt 同时发**

| 指标 | 值 |
|------|-----|
| 客户端 wall | 22,616 / 22,980 / 23,454 / 23,974 ms |
| 引擎每 lane TTFT | 5.5 s（全部） |
| 引擎每 lane total | 22.3 / 17.1 / 12.0 / 7.1 s |
| lane admission 时刻 | 13:41:23.7 / 29.2 / 34.8 / 40.2（≈5.5 s 间隔） |
| 聚合 prefill | 1.99k tok/s（= 单 lane，无批量） |

⇒ prefill 严格串行：第 k 条 lane 的真实客户端 TTFT ≈ k × 5.5 s。
lane trace 显示 4 条几乎同时到达的请求只组成了 `batch=2 capacity=4`，
另外两条在 `admit slot=2/3` 时才中途加入。

**B2：长请求先，400 ms 后短请求（54 token）**

- 客户端：长 6,503 ms / 短 6,098 ms。
- 短请求引擎行：`TTFT 69.6 ms | total 1.0s | prefill 796.6 tok/s (54 tok)`，
  即它在队列里等了 ~5.1 s，**只有长请求 prefill 完成后才开始**。

**B3：短请求先，400 ms 后长请求**

- 客户端：短 6,023 ms / 长 6,411 ms。
- 短请求引擎行：`TTFT 84.5 ms | total 6.0s | decode 2.5 tok/s`。
  首 token 84.5 ms 就出了，但剩余 15 个 token 被长 lane 的整段 inline prefill 阻塞，
  decode 从单跑的 37.7 掉到 **2.5 tok/s（15×）**，total 从 <1 s 变 6.0 s。

### 5.2 根因

> **已实施（项 3/4/7）**：形成窗口放宽到 3 ms+50 ms 上限、plain 路线 chunk 间 pump 一轮 decode、TTFT/total 口径计入排队时间。见 §8.3/§8.4/§8.8。

1. **初始 admission 串行 prefill**：plain :3647-3659 / spec :4401-4413 的循环对每个成员
   顺序调用 `admit_lane`，期间不消费队列、不做 decode。第 k 条 lane 的 TTFT = k × prefill。
2. **中途 admission 在 round 顶部整段阻塞 decode**：plain :3695-3725 / spec :4484-4504
   在每轮开头 `while (active.size() < lane_capacity)` 里调 `admit_lane`，
   新人的**整个 prefill**（10.8k token 时 ~5.5 s）在计算流上同步跑完，
   期间所有存活 lane 的 decode 全部停摆（B3 的 15× 塌陷）。
   注意 admission 只在主 decode 循环的 round 边界检查，初始 prefill 期间不检查。
3. **batch 形成窗口只有 3 ms**：`kBatchFormationWindow{3000}`（:2250），
   `lane_queue_cv_.wait_for(queue, kBatchFormationWindow)`（:2291）。
   4 条同时发的请求因此被切成 2+2，后两条要等前两条的首轮 decode 才能开始。
4. **TTFT 度量是 admission 相对的**：引擎的 TTFT/total 时钟从该 lane 被 admission 起算，
   不含排队与串行化。**TP-2 多 lane 下客户端感知的 TTFT 最高可达日志值的 ~4×**
   （B1：日志 5.5 s，客户端 ~23 s）。这会让任何基于日志 TTFT 的容量规划失真。

### 5.3 建议（TTFT，按性价比排序）

1. **让中途加入的 prefill 与存活 lane 的 decode 交错**（最高价值）。
   当前是「整段 prefill 阻塞」。可行的中间形态：
   - 在 `admit_lane` 的分块循环里，每跑一个 chunk 就回到主循环跑一轮 decode
     （即把 `admit_lane` 拆成「准备 + 每轮推一个 chunk」的状态机）。
     代价：chunk 粒度决定阻塞时长，1024 token ≈ 0.5 s 仍然偏粗。
   - 混载时用更小的 chunk（如 256 → ~0.15 s 阻塞），无存活 lane 时用 1024。
     按 4.1 的拟合，chunk 256 的 per-token 成本 0.549 + 39/256 = 0.701 ms/token（−28%），
     但这只发生在混载窗口内。
   - 更彻底：把新人的 prefill 放到第二条流上与 decode 真正并行。decode 的 AR 只占链路
     ~6%（B=4 轮 5.24 MB / 32 ms ≈ 164 MB/s vs 链路 7 GB/s），链路有余量；
     真正的争用是权重流（两者都读全部权重，448 GB/s 共享）。
     结果是总吞吐守恒、但新人的 TTFT 从「等一个 round」变成「自己的 prefill 时间」。
2. **初始 prefill 期间也检查队列**：在 :3647-3659 的循环里每完成一个成员的 prefill
   就 `try_pop_lane_queue`，让 B1 的 2+2 变成一次性组满。
3. **放宽 batch 形成窗口**：把「固定 3 ms 等一次」改成「只要 `batch.size() < lanes_`
   且队列非空就继续等到一个小的上限（如 20-50 ms）」。队列空时不增加任何延迟
   （:2284-2287 已经是空闲即睡）。
4. **修正 TTFT 口径**：把 admission 前的排队时间计入对外 TTFT（或额外输出一个
   `queue_ms` 字段），否则多 lane 的延迟数据不可用。
5. **prefix reuse 仍是最大的单点 TTFT 杠杆**（已 shipped）：10,240 token 命中把
   6.4 s 冷 prefill 降到 0.45 s。已知的剩余缺口：C≥2 时 `finalize` 会
   `release_lane_kv → invalidate_host_checkpoints(lane)`，decode 尾部的 grid ring 被清空，
   只有 catalog 的 frontier / prompt-end / block-shared 三类镜像存活
   （多轮会话与共享 system 前缀仍可命中；只共享中段的场景不行）。
   补齐需要再加一类 `SessionEntry` 镜像，且 host 预算 10,240 MiB/分片只放得下
   2 × 3,960 MiB 的整段会话。

## 6. 建议清单

| 优先级 | 项 | 类型 | 预期收益 | 成本 | 状态（§8 为本次执行记录） |
|--------|-----|------|----------|------|--------------------------|
| P0 | 中途 admission 的 prefill 与 decode 交错（状态机 / 小 chunk / 第二条流） | 延迟 | 混载 TTFT 数量级改善；消除 15× decode 塌陷 | 中-高（调度重构） | **已实施**（plain 路线 pump 回调，每 chunk 一个 decode round）：chunk 256 时短 lane 2.8 s / 8.5 tok/s。§8.4；**第二轮已覆盖投机路线**（spec/MTP pump）：chunk 256 时短 lane 2.0 s / 15.7 tok/s。§9.1 |
| P0 | 初始 prefill 期间也消费队列 | 延迟 | 突发请求不再 2+2 分裂 | 低 | **无需改**：既有中途 admission 已覆盖；2+2 分裂由形成窗口修掉。§8.5 |
| P0 | 放宽 batch 形成窗口（非空则继续等到上限） | 延迟 | 突发组满 batch | 低 | **已实施**（3 ms 窗口 + 50 ms 上限）：四并发从 `batch=2` 变 `batch=4`。§8.3 |
| P1 | plain 路线不雕 round-scratch 平面 | 显存 | C=4 省 293.6 MiB/卡（两卡 587 MiB） | 极低（数行） | **已实施**：启动台账 `state 293.6`（4×73.4）、`round 0.0`。§8.1 |
| P1 | `--lane-context` 多 lane 默认取 pool/lanes 或至少告警 | 并发度 | 避免单请求独占 pool 导致并发退化 | 低 | **已实施（仅告警）**：硬默认会拒绝长于 1/lanes 的 prompt。§8.8 |
| P1 | 补 plain batch 的 round 埋点 | 度量 | 解锁 decode 边际成本分解 | 低 | **已实施**：`[tp2-time] plain-batch … decode=/grammar=/sample=/readback=/prefill_chunks=/pump_rounds=`。§8.2 |
| P1 | TTFT 口径加入排队时间 | 度量 | 多 lane 延迟数据可用 | 低 | **已实施**：`ttft`/`total` 从 core 受理起算，与 `| queue` 同口径。§8.8 |
| P2 | 输出预算按剩余量动态收缩 / 页回收 | 显存/并发 | 实测 64 页 = 66 MiB/卡 空闲占用 | 中 | **分析后不改**：破坏 all-or-nothing 预留不变式，会引入中途增长死锁。§8.6 |
| P2 | 修正 docs/serving.md 的 262,144/C=1 指导 | 文档 | 避免误导部署 | 低 | **已实施**：并补齐 1/2/3/4/6/7 的说明。§8.9 |
| P2 | 投机批链改用每 lane envelope（需先确认接受率归因） | 吞吐 | MTP C≥2 每 lane 接受率 | 中 | **已实施（数值一致性）**：批内每列与单独解码逐位一致（新增 parity 回归测试）；接受率收益未复现，plain decode 中性。§8.7 |
| P3 | 批量 prefill（需新 ragged 大 T 多序列核 + 更大 workspace） | TTFT 均衡 | 只均衡 lane 完成时间，聚合吞吐上限 1.34× | 很高 | **实测确认不做**：attention batch>1 每 lane width ≤ 16 且 GDN 须 record+fold；实测批量窗口 W=16/B=4 = 1.24 ms/lane-token vs 单 lane 1024-chunk 0.587 ⇒ **慢 2.1×**（打平需 W≈300）。§9.4 |
| P3 | host checkpoint ring 改可分页 / 减槽 | host 内存 | 省 GiB 级 pinned host | 低-中 | **已实施**：ring 槽改 pageable，两卡省 5.29 GiB 页锁；同 harness recall 请求 +10 ms（噪声内）。§9.5 |
| — | 卡 2 挪到 CPU 直连 Gen5 x8 槽 | 硬件 | prefill 上限 + decode AR | 换硬件 | 未做 |

## 7. 复现方式

脚本（workspace 根 `_temp/`，命名 `<YYYYMMDD-HHmm>_<desc>`）：

- `_temp/20261002-1310_scale.ps1` + `_scale.mjs`：C=1..4 等长 lane decode 扩展。
- `_temp/20261002-1310_ttft.ps1` + `_ttft.mjs`：B1/B2/B3 三组 TTFT 串行化实验。
- `_temp/20261002-1350_mem.ps1`：262,144 下 C=1/2/4 与 plain C=4 的显存预算。
- `_temp/20261002-1420_ctx.ps1` + `_ctx.mjs`：4,543 token 上下文的 C=1/C=4 prefill 与 decode。
- `_temp/20261002-1440_shipped_c4.ps1`：出厂配置 262,144 + C=4 端到端验证。
- `_temp/20261002-1442_pump.ps1` + `_pump.mjs`：项 4 交错实测（A=4 长并发 / B=长先短后 / C=短先长后 / D=2+2），
  `--prefill-chunk 1024` 与 `256` 各一遍。
- `_temp/20261002-1550_mtp.mjs`（配 `_mtp5.ps1`）：项 10 归因——S=单条长单跑、A=4 条长同批、B=长+短。
- `_temp/20261002-1620_mtpshort.mjs`（配 `_mtpshort.ps1`）：项 10 判别——短 prompt 同刻入批（C=1/2/4）。
- `_temp/20261002-1630_item10.ps1`：项 10 同代 A/B（`-Bin oldops|newops`，long 与 short 两臂）。
- `_temp/20261002-1745_specpump.ps1` + `_specpump.mjs`：投机路线 prefill 交错 A/B（`-Bin head|pumped`，
  短先/长先两臂，chunk 1024 与 256）。
- `_temp/20261002-1700_ringio.ps1` + `_ringio.mjs`：ring pinned/pageable 对照（`[tp2-kv]` store/recall
  与 98.2% 命中 recall 请求的 TTFT/total）。

环境要点（踩坑记录）：

- **没有 `--spec none`**：传它会打印 `invalid speculative backend: none` 并退出；
  省略 `--spec` 即 plain 路线。
- 必须传 `--chat-template D:/LLM/chat_template.jinja`：artifact 自带模板不接受
  `instructions` 映射到的 Developer 角色（src/serve/openai_responses_state.cpp:132-141），
  否则每个 /v1/responses 都被 400 拒绝。
- 模型是**位置参数**，且 PATH 需含 FFmpeg 与 libcurl 的 bin 目录。
- **MTP 的显存口径**：`--max-context 245760 --kv-dtype int8 --spec mtp` 启动 39 s 后
  `cudaMalloc failed: cudaErrorMemoryAllocation` 退出；MTP 实测必须用文档口径
  `--max-context 131072 --kv-dtype int8`（本报告 §8.7 用的就是它）。
- `_temp/20261002-1310_scale.ps1` 按 `-Stamp` 拼客户端脚本名，必须先复制
  `20261002-1310_scale.mjs` 成 `<Stamp>_scale.mjs`，否则 node 找不到脚本、done 行为空。
- 两卡均需空闲；服务进程全局唯一，重链接前先停服务。

## 8. 本次执行记录（2026-10-02）

逐项执行 §6 清单。源码改动集中在 `src/runtime/engine/tp2_generation_core.cpp/.h` 与 `docs/serving.md`；
构建入口 `tools/win_port/build.ps1`（增量 27-29 s，`BUILD_EXIT=0`），测试入口
`tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` → `100% tests passed out of 12`
（6 个模型相关 TP-2 测试因本机无权重 skip）。实测环境：2×RTX 5060 Ti、`--devices 0,1`、int8、
prompt 15,310 / 54 token、`max_output_tokens 24`、`NINFER_TP2_TIMING=1 NINFER_TP2_LANE_TRACE=1`。

### 8.1 项 2：plain 路线不再雕 round-scratch 状态平面

- 依据：`state_snapshots[kRoundScratchSlot]` 只在投机路线被引用
  （`execute_spec_batch` :4701/:4886/:5079、`execute_walk` 的 DFlash2/MTP 分支 :7761/:7804），
  `execute_plain_batch` 零引用；`snapshot_state`（整池快照 :6875-6880）**没有空指针保护**，
  所以门控必须精确为 `mtp_enabled_ || dflash2_enabled_`（`snapshot_lane_state` 自己有空指针保护 :6889）。
- 实测：启动台账 `shard N … | state 293.6 | … | round 0.0 | … | free 1414.0 of 16310.6 MiB`，
  `round` 由改前的 293.6 降到 0.0 ⇒ C=4 每卡省 293.6 MiB。`state 293.6` 是每 lane GDN 状态
  （4×73.4），未变。

### 8.2 项 1：plain batch 的 round 埋点

新增 `Tp2RoundTiming::close_round_plain` / `report_plain`，plain 路线不再借用投机路线的
`mtp/verify/accept/copy` 相位名。实测（chunk 256）：

```
[tp2-time] plain-batch rounds=69 committed=69 avg_round=112.17ms decode=27.20 grammar=0.00 sample=0.04 readback=0.20 sync_wait=0.03 prefill_chunks=180 pump_rounds=46
[tp2-time] plain-batch rounds=46 committed=46 avg_round=26.76ms decode=15.38 grammar=0.00 sample=0.03 readback=0.10 sync_wait=0.01 prefill_chunks=61 pump_rounds=1
[tp2-time] plain-batch rounds=46 committed=46 avg_round=64.70ms decode=27.89 grammar=0.00 sample=0.04 readback=0.18 sync_wait=0.02 prefill_chunks=61 pump_rounds=14
```

`rounds` 按**已提交列**计数，所以 `avg_round` 是每 lane 分摊的单轮成本（与投机路线的
`verify` 口径可比）；`pump_rounds` 是「在别的 lane 的 prefill chunk 之间跑的 decode round 数」。

### 8.3 项 3：放宽 batch 形成窗口（3 ms + 50 ms 上限）

形成逻辑改为 `take_from_queue()` 循环 + `formation_deadline = now + 50 ms`：队列只要非空且未满
就继续等一个 3 ms 窗口。实测四长并发从报告基线的 `[tp2-lane] batch=2 capacity=4` 变为
`batch=4 capacity=4`（chunk 1024）；chunk 256 时仍是 `batch=3` + 第二波 1 条（50 ms 上限
不跨 800 ms 间隔的 2+2，实测正确不合并）。单条请求只多付一个 3 ms 窗口。

### 8.4 项 4：prefill chunk 间 pump 一个 decode round（plain 路线）

**策略**：任何时刻只有一个 lane 在 prefill（保持 FIFO 顺序与每个 lane 的 TTFT），已持有 token
的 lane 在这个 lane 的两段 chunk 之间跑**一个 decode round**。轮转式 prefill 已否决：它会让所有
lane 在同一墙钟时刻完成 prefill，lane 0 的 TTFT 从 ~5.5 s 退化到 ~22 s。

**实现**：round 主体提成 `execute_plain_batch` 内的 `auto pump_decode_round = [&]()`（定义在
`active.reserve` 之后、`admit_lane` 之前，所以 admit_lane 可直接调用）；`admit_lane` 的 chunk
循环顶部、上一个 chunk 的 `DeviceArena::Scope` 析构之后调用它并恢复
`set_linear_state_slots(active_lane_, active_lane_)`（一列 round 的 `batch == 1` 分支会重播
标量 slot 对，src/models/qwen3_5/execution/text.cpp:2370-2378）。`round_base` 捕获点前移到
pump 定义之后（缓存的 CUDA Graph 烘死了 `arena_begin`，布局不一致会抛
`"TP-2 batched decode CUDA Graph replay found a different workspace layout"`）。

**实测**（长 15,310 token，短 54 token 提前 400 ms 提交）：

| 场景 | 短 lane total | 短 lane decode | 机制 |
|------|---------------|----------------|------|
| 报告基线（无交错，B3 10.8k） | 6.0 s | 2.5 tok/s | 整段 prefill 阻塞 |
| 交错，chunk 1024 | 7.5 s | 3.1 tok/s | 每 chunk（~0.55 s）1 个 round |
| 交错，chunk 256 | **2.8 s** | **8.5 tok/s** | 每 chunk（~0.19 s）1 个 round |

- 反向顺序（长先短后）仍要等：短 lane `queue 9.8-9.9 s`、TTFT 9.9 s，而短先时 TTFT 114 ms。
  这是「单 lane 串行 prefill」的固有代价，项 4 不改变 admission 顺序。
- 埋点直接证明机制：短先（长 lane 61 个 chunk，`pump_rounds=14`——短 lane 拿满 24 token 退出
  `active` 后 pump 停止）；长先（`pump_rounds=1`）；四长并发（`prefill_chunks=180 pump_rounds=46`）。
- 代价：chunk 256 的每 lane prefill 速率 1.34-1.52k tok/s vs chunk 1024 的 1.69-1.94k，
  与 §4.1 成本模型一致（小 chunk 用 prefill 速率换 decode 平滑）。**只改了 plain 路线**，
  投机路线仍逐 lane 串行 prefill（剩余空间）。
- 客户端 SSE 在 reasoning + 小 max_output 下多数为 0 个 text delta，不能当 token 时钟；
  权威数据是 `done |` 与 `[tp2-time]` 行。

### 8.5 项 5：初始 prefill 期间消费队列——**无需改动**

曾实现（S1 循环末尾 `try_pop_lane_queue()`）后回退：S1 按 `batch.size()` 顺序 admission，循环
结束后主循环**立刻**执行 `while (active.size() < lane_capacity) { try_pop_lane_queue(); …; admit_lane(…); }`，
两者在同一墙钟时刻、同样严格串行地 prefill 新成员，TTFT 无差别，差别只是成员待在哪个
`batch` 向量里。报告该项的前提（初始 prefill 期间不查队列）实际已被既有中途 admission 覆盖；
2+2 分裂由项 3 修掉。

### 8.6 项 9：输出预算预占——**分析后不改**

`lane_kv_pages`（:3023）=`pages_for_tokens(need + lane_kv_margin())`，
`reserve_lane_kv`（:3027）在 admission 前对两卡 + MTP cache 做**全量 all-or-nothing** 预留，
`lane_need_tokens`（:3202）=`prompt + budget - 1`。设计注释 :3004-3010 原文：「The reservation is
all-or-nothing and happens before the request is admitted, so a pool that cannot cover it right now
leaves nothing behind … the wait is always for another lane to retire, never a deadlock.」
按剩余量收缩/回收会让 lane 在 decode 中途需要**增长**，而该路线没有 mid-request 背压或抢占
（AGENTS.md：无 active-request preemption），安全实现要新增「按需驱逐 + 增长连续 KV window」路径，
不是 P2 项的量级。66 MiB/卡空闲上界由客户端 `max_output_tokens` 决定，操作杠杆是
`--lane-context` 或更小的客户端预算。

### 8.7 项 10：投机批链的每 lane 接受率——**数值机制确认，接受率收益未复现；修复已实施**

配置：`--max-context 131072 --kv-dtype int8 --spec mtp --draft-tokens 5`（245760+int8 启动即 OOM）。
`done |` 行新增的 `mtp accepted X/Y (P%)` 直接给每 lane 接受率。

**旧 ops（含 batch 对 host split 数的截断）基线**：

| 组 | lane | 接受率 |
|----|------|--------|
| 单条长单跑（C=1） | — | 40/75 = **53.3%** |
| 四条长同批（C=4） | 各 lane | 48.2 / 64.9 / 39.4 / 45.9%（均 49.6%） |
| 长 + 短（C=2，位置差 15k） | 长 / 短 | 47.5% / 46.7% |
| 单条短单跑 | — | 54/145 = 37.2% |
| 两条短同批（C=2） | 各 lane | 53.3 / 54.1% |
| 四条短同批（C=4） | 各 lane | 47.1 / 52.8 / 53.6 / 46.7%（均 50.1%） |

**数值层机制已确认**：旧 ops 上「同一 lane 批内跑」与「单独跑」**逐位不同**——
`causal batch/solo parity d256-h24-kv4 bf16 lane=0: element 4 batched=0x3a20 solo=0x3a1f`
（5 种 storage × 4 lane 全部 1-3 ULP 差异，OPS_EXIT=1）。原因是
`causal_attention_split_capacity`（src/ops/softmax_attention/dense/causal_cache/small_t.cu:217-247）
在 batch>1 时把 host 侧 split 数截断到 `grid_limit/page_limit`，而 device 侧
`active_split_count = min(逐列自然 split 数, host 值)`（small_t.cuh:119）——每列的归约树因此被改短，
与单独解码同列不同。

**修复（A1）**：`small_t.cu` 直接 `return capacity`（capacity 已是 envelope 上确界），
`causal_attention_split_capacity` 删除 `batch_size` 形参（launch.h:26-29），11 个调用点同步。
新增回归测试 `run_batch_single_parity_case`（tests/ops/softmax_attention/causal_cache.cpp，
batch=4 / width=4 / context=4100 / EXACT envelope）钉死「批内 lane == 单独跑」逐位相等：
新 ops PASS（`nvfp4 mae=0.00536884`、`k8v4 mae=0.0053744`，容差 oracle 无回归），旧 ops FAIL。

**但接受率没有复现收益**（同 harness 同代 A/B，唯一差异 = src/ops 的 6 个文件）：

| 配置 | 旧 ops | 新 ops |
|------|--------|--------|
| plain decode C=1/C=2/C=3/C=4（每 lane tok/s） | 36.8 / 35.0·35.9 / 31.9·32.7·33.6 / 29.9·30.6·31.3·32.2 | 37.2 / 35.0·35.8 / 32.0·32.8·33.7 / 29.9·30.6·31.4·32.3 |
| plain `[tp2-time] avg_round` C=1..4 | 27.15 / 13.89 / 9.97 / 7.84 ms | 26.85 / 13.92 / 9.92 / 7.85 ms |
| spec 短臂 P4 每 lane 轮成本 mtp+verify | 21.05 ms | 23.57 ms |
| spec 长臂四长 每 lane 轮成本 mtp+verify | 51.65 ms | 50.11 ms |
| spec 接受率 短臂 P4 均值 | 50.1% | 51.4% |
| spec 接受率 长臂四长均值 | 49.6% | 40.3% |
| spec 接受率 **单长（控制项，代码路径无变化）** | **53.3%** | **41.0%** |

⇒ 单跑控制项自己就漂了 12 个点，本 A/B 的接受率差异全部落在 run-to-run 噪声内，
**不能证明**接受率收益；plain decode 完全中性，spec 轮成本 −3%/+12%。

**结论**：项 10 保留，但它的价值是**数值一致性**——批内每列与单独解码逐位一致，由 parity 测试守住；
它不是接受率或吞吐优化。§0/§6 的措辞据此改写。

- **开放问题**（未隔离）：触发 split 变化的是「窗口绝对值宽」还是「lane 间位置差」——短窗口实验
  只能排除批处理本身；构造零位置差的宽窗口需要 admission 不再串行（即 P3 批量 prefill，但按 §9.4 的
  实测它慢 2.1×）或核内探针。
- envelope 注释位置（引用更正）：当前在 src/runtime/engine/tp2_generation_core.cpp:4436
  （原文「The envelope steers split policy only; the kernels take their visible set from the
  per-lane positions, and the widest lane bounds every lane's window.」）。

### 8.8 项 6 / 项 7

- 项 6：多 lane 且 `--lane-context 0` 时启动打印
  `[mem] TP-2 lane admission ceiling 245759 tokens (the whole KV pool): one lane may reserve all of it, so --max-concurrency 4 requests cannot be admitted together until it retires; set --lane-context to cap one lane`。
  不做硬默认——默认 pool/lanes 会让长于 1/lanes 的 prompt 直接 `context_length_exceeded`。
- 项 7：`submit()` 记 `request->submitted`，两处 admit_lane 置
  `lane.result.engine_timing.queue_wait_seconds`，两处 finalize 与 `execute_walk` 把
  `first_token_seconds`/`total_seconds` 改为 `prepare_seconds + … + queue_wait`（对齐单卡口径，
  src/runtime/engine/engine_core.h:873-882）。这条是必须的：src/serve/generation_service.cpp:447-457
  会 `ttft = prepare + max(0, first_token - prepare)`、`total = prepare + max(0, total - prepare)`。
  `prefill_seconds`/`prompt_wall_seconds` **保持**从 admission 起算（它们定义 prefill 速率）。
  实测 `done |` 行新增 `| queue 7.9s` 等字段，与服务器端 `total` 自洽。

### 8.9 项 8：docs/serving.md

补上：形成窗口行为（项 3）、pump 交错与实测数字（项 4）、plain `[tp2-time]` 行（项 1）、
`state/round` 台账变化（项 2，原文已有 round-scratch 只在投机路线的说明）、启动告警（项 6）、
TTFT/total 口径（项 7）。校验：`git diff --check` 无空白错误。

### 8.10 改动清单

- `src/runtime/engine/tp2_generation_core.cpp`：24 个 hunk，全部为上述项（含在 `.h` 里给
  `TP2GenerationCore::Request` 加 `submitted` 时间戳）。
- `src/runtime/engine/tp2_generation_core.h`、`docs/serving.md`、本报告、
  `docs/PLAN-tp2-headroom-execution.md`（逐项执行计划与记录）。
- 未做（记录为剩余空间）：P3 批量 prefill（§9.4，实测慢 2.1×）、卡 2 换 PCIe 槽。

## 9. 第二轮执行记录（2026-10-02，用户增补四项）

用户第二轮增补（m01111）：投机路线（spec/MTP）的 prefill 交错；项 10 的 Ops 核每列 envelope/split
policy 修复；P3 批量 prefill（顺带消除 admission 位置差）；可分页的 host checkpoint ring。
逐项计划与设计见 docs/PLAN-tp2-headroom-execution.md 的「用户第二轮增补」。

### 9.1 项 A：投机路线的 prefill 交错——**已实施并实测成立**

plain 路线的 pump 回调（§8.4）原样搬到 spec 路线：`execute_spec_batch` 的回合体抽成
`pump_spec_round` lambda（src/runtime/engine/tp2_generation_core.cpp），`admit_lane` 的 chunk
循环顶部在「上一轮 chunk 的 scope 已归还 arena」之后调用它，主循环只保留 retire 前言 +
中途 admission + `pump_spec_round()`。策略不变：**同时只有一个 lane 做 prefill**，已经在解码的
lane 在它的 chunk 之间各跑一个 decode 回合。因为只有一条 lane 在 prefill，视觉会话的单实例约束
不受影响；pump 不 retire/admit，不会递归回 admit_lane。

A/B（同 harness `_temp/20261002-1745_specpump.ps1` + `_specpump.mjs`，二进制只差本项）：

| 配置 | HEAD（无 pump） | 有 pump |
|------|-----------------|---------|
| chunk 1024，ARM C 短 lane（54 token） | 总 8.8 s / decode 3.1 tok/s | **总 3.3 s / 8.9 tok/s** |
| chunk 256，ARM C 短 lane | 总 11.1 s / decode 2.5 tok/s | **总 2.0 s / 15.7 tok/s** |
| chunk 1024，ARM C 长 lane（15,310 token） | 9.0 s | 9.5 s |
| chunk 256，ARM C 长 lane | 11.4 s | 12.1 s |
| ARM B 后到短 lane（queue） | 7.6 s / 10.0 s | 7.6 s / 10.0 s（不变） |

- 直接证据是新增计数器 `prefill_chunks`/`pump_rounds`：chunk 1024 时长 lane 的 prefill 拆成
  16 个 chunk、其间给已解码 lane 跑了 6 个回合（`[tp2-time] spec-batch ... prefill_chunks=16 pump_rounds=6`）。
- 长 lane 慢 0.5-0.7 s（~5%），是 pump 回合的插入成本。
- **run-to-run 抖动很大**（同一配置复测短 lane 3.3 s → 5.6 s），所以结论以 `pump_rounds` 计数器与
  chunk 256 的数字为主证据。
- ARM B 里后到的短请求仍要排队（7.6/10.0 s 不变）——这是「一次只一个 lane 做 prefill」策略的固有
  代价，不是缺陷。

### 9.2 项 B：spec 路线计时口径——**已实施**

`Tp2RoundTiming` 删除恒为 0 的 `fold_ms` 与 3 处 `timing.fold_ms += timing.elapsed(5, 6)`
（round 内查询未完成 event 恒为 0），`report()` 第 5 个字段改为 `rest=%.2f` =
`(round_ms − mtp − verify − accept − copy) / rounds`；`pump_spec_round` 补上回合
`round_start`/`timing.round_ms +=`，spec 行从此与 plain 行同口径（此前恒 `avg_round=0.00`）。

### 9.3 项 C：项 10 Ops 修复——见 §8.7

修复已实施（数值一致性：批内每列与单独解码逐位一致），接受率收益未复现；parity 测试钉住不变量。

### 9.4 项 D：P3 批量 prefill——**调研 + 宽度代价实测后确认不做**

设计调研（_temp/20261002-1800_item3_batched_prefill.md，161 行）结论：

- 唯一可复用的多 lane×多 token 入口是 `TextContext::forward_tp2_window_batch`
  （src/models/qwen3_5/execution/text.cpp:2939-3100）：已有 per-lane `kv_table_rows`/`state_slots`
  [batch] 与 per-lane `valid_columns` [batch]（`width=max(chunkLen)`、`valid_columns[i]=chunkLen[i]`
  即可表达不等长 chunk）。`forward_tp2_prefill` 与 `forward_tp2_decode_window_batch` 都不可用。
- 两条硬约束：attention batch>1 时每 lane **width ≤ 16**（src/ops/softmax_attention/dense/causal_cache/
  causal_softmax_attention.cpp:241-243，kMaximumVerifyTokens=16），且 batch>1 **永不选 Prompt 路由**；
  TP-2 GDN 非 record 路径 batch>1 **强制 width==1**（text.cpp:1748-1749、:1917-1918），
  批量 prefill 必须走 `GdnStateAction::RecordForReplay` + fold。
- 掩码不是问题（lane 隔离靠 per-lane KV 行，lane 内因果窗靠设备侧 pos，clamp 尾列不 append）；
  envelope 只是 launch/workspace 承诺，不是掩码。
- 收益判断：width ≤ 16 时「减少 kernel 启动」不自动成立；**确定收益只是把 admission 位置差限制在
  chunk 宽度内**（收敛 MTP 的单值 envelope）。且 `forward_tp2_window_batch` 走非 overlap 层循环
  （text.cpp:3076/3078），会丢掉 L2 allreduce‖MMA overlap 的 +11%（docs/tp2-decisions.md:281-285）。

**宽度代价实测（决定最小切片是否值得写）**。写切片之前先量出切片**唯一新增的那个 op**——一次
B lane × W 列的 `forward_tp2_window_batch`（含模型内的 GDN record 路径）——的真实成本。
`[tp2-time] spec-batch` 的 `verify` = `close_round` 的 elapsed(1,2)，恰好包住该调用。配置：
artifact D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer，TP-2 双卡，int8，131072，C=4，
prefill-chunk 1024，`--spec dflash2 --draft-tokens 5|15`，ARM P4 四短同批（harness
_temp/20261002-1925_w16probe2.ps1）。注意 MTP 无法到 width 16
（src/product/speculative_options.h:41-43 限制 draft-tokens ≤ 5），只有 dflash2（≤15）能。
口径：`++timing.rounds` 每提交一列 +1（src/runtime/engine/tp2_generation_core.cpp:3590 与 :4815），
一轮有 B 个 lane 各提交一列 ⇒ 每轮成本 = 报告值 × B（B=4）。

| 配置 | 每轮 avg_round | 每轮 mtp | 每轮 verify | verify 每列 |
|---|---|---|---|---|
| W=6（drafts 5） | 80.9 ms | 17.4 ms | 58.8 ms | 2.45 ms |
| W=16（drafts 15） | 108.6 ms | 24.0 ms | 79.6 ms | 1.24 ms |

拟合 **verify(W, B=4) = 46.3 ms 固定 + 0.52 ms/列**；对照 §4.2 实测的单 lane prefill 拟合
**39 ms/chunk 固定 + 0.549 ms/token**。⇒ 批量 prefill W=16,B=4 = **1.24 ms/lane-token**，
现有单 lane 1024-chunk = **0.587 ms/lane-token**，**批量慢 2.1×**。机制：固定成本（约 40 ms 的
权重流）在批量轮只摊到 W×B=64 列，单 lane chunk 摊到 1024 列（16× 差距）。批量要打平需
W×B ≈ 1216 列/轮，即 B=4 时 W≈300，是 ops 上限 `kMaximumVerifyTokens=16` 的 19 倍。
⇒ 方案 A（engine 侧最小切片）的价值预期为负，唯一能赢的区间是 prompt < ~56 token（本就一个
chunk）；要真正拿到批量 prefill 的收益必须先做方案 B（放宽 ops 批量域，400-800 行、风险最高），
而 §4.2 已论证即便如此也提高不了长 prompt 的聚合吞吐（链路天花板）。**本项收口为不做**，
保留为剩余空间。

### 9.5 项 E：可分页的 host checkpoint ring——**已实施并实测**

ring 槽从 `PinnedHostBuffer`（cudaHostAlloc Portable|Mapped）改为 `HostBuffer`(`HostPinning::Pageable`)，
并把镜像传递路径统一为裸映射 + 长度（`store_dflash_image`/`load_dflash_image`/
`session_capture_shared_state` 的 `frozen` 参数），因为两种 host buffer 没有共同基类。
ring 不在任何 CUDA Graph 捕获区内，所以可分页不影响捕获；`--host-kv-pinned` 控制的
HostKVArena 不受影响（仍可 pin）。

同 harness A/B（`_temp/20261002-1700_ringio.ps1` + `_ringio.mjs`：13,559 token 前缀 +
4 条 98.2% 命中 recall 请求，`--max-context 131072 --host-state-slots 32`，二进制只差 ring 的 pinning）：

| 配置 | `[mem]` 行 | recall 请求 total（4 条均值） | TTFT 均值 |
|------|-------------|------------------------------|-----------|
| pinned | `host-checkpoints shard 0/1 slots 36 ... | pinned 2642.6 MiB` | 495.5 ms | 296.5 ms |
| pageable | `... | pageable 2642.6 MiB` | 505.3 ms | 292.8 ms |

⇒ 两卡共 **5.29 GiB（2 × 2642.6 MiB）页锁内存变为普通 host commit**，recall 路径代价 ≈ +10 ms
（2%，落在 run-to-run 噪声内；TTFT 反而略降）。`[tp2-kv]` 的 store/recall 计时测的是 paged KV
（HostKVArena，本来就是可分页）而非 ring 镜像，两臂相同（store 125.7-151.4 ms / recall
67.9-80.1 ms），可作旁证。

模型测试（`NINFER_TEST_ARTIFACT=D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer`，
`tools/win_port/test.ps1 -Filter 'qwen3_5_tp2'`）：最终二进制 6 项中 4 项有结论——lanes **Passed**
（69.56 s）、dflash_append **Passed**（47.51 s）、sessions Failed、dflash_solo Failed，load/forward
Skipped。把本轮全部改动 stash 掉重建 HEAD（464 s）后同 filter 复测，**两个失败逐字复现**：
sessions 的 `got [2752 13 198 197 197 92 198 197] expected [467 419 538 13 198 197 197 92]`、
dflash_solo 的 `the recalled walk diverged from the from-scratch walk on its first sample:
[59399 475 327 363 62 16 15 15] vs [365 4577 62 16 15 15 15 15]` ⇒ 均为**既有失败**，不是本轮回归。

### 9.6 本轮改动清单

- `src/runtime/engine/tp2_generation_core.cpp`：spec 路线 pump 交错（`pump_spec_round` 抽取 +
  admit_lane chunk 循环调用 + 主循环只留 `pump_spec_round()`）、计时口径（删 `fold_ms`、`rest`、
  spec 回合累加）、ring 改可分页（分配点、镜像传递签名、`[mem]` 行按实际 pinning 打印）。
- `src/runtime/engine/tp2_generation_core.h`：ring 槽类型与镜像传递签名。
- `src/ops/softmax_attention/dense/causal_cache/`：`launch.h`、`small_t.cu`、`small_t_fp8.cu`、
  `small_t_k8v4.cu`、`small_t_nvfp4.cu`、`causal_softmax_attention.cpp`（删 batch 对 host split
  数的截断，`causal_attention_split_capacity` 去掉 batch 形参）。
- `tests/ops/softmax_attention/causal_cache.cpp`：新增 batch/solo 逐位 parity 回归。
- `docs/serving.md`：spec 路线交错、计时行口径、`--host-state-slots` 可分页。
- 未做（剩余空间）：P3 批量 prefill（§9.4，实测慢 2.1×，需先放宽 ops 批量域）、卡 2 换 PCIe 槽。

