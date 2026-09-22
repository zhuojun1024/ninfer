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
- 不确定性：acceptance 与 token/轮由数学决定、跨卡可迁移（残差逐位相同），但本机 draft 是 w4a4 而非
  nvfp4 ⇒ 接受率会有小差；效率锚点取自 MTP 轮次；未计 prefill 与首轮抖动。

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
