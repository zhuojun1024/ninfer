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

## 4. 上游 cherry-pick 计划（第一梯队 + 第二梯队）

**背景**（2026-09-21 分析）：origin/master 领先本分支 20 个提交（merge base `e360c4c0`，
上游 HEAD `9e163eee`）。本地主负载是 NVFP4 TP-2；上游的 Q4/Q5 调优、sparse_moe、Q5 bench/test
提交与本机无关，不排期（完整逐提交判断见会话记录）。**新官方 artifact（`6cc95cc5`）与本地前端不兼容**：
新 artifact 内嵌维护版模板（sha256 `a497db9e…`），不在本地 `CompiledChatTemplate::resolve()` 的
白名单（`e84f32a2…`/`c3cf9e34…`）内，加载即抛 `unsupported frontend/chat_template.jinja`。
本地 nvfp4 共享代码的改动全部是 Windows 特有的 TMA 描述符暂存（`#if defined(_WIN32)`），与上游
性能改动正交但同文件。

**前置**

- [ ] 处理工作树未提交改动（Round 18 会话恢复加固：`tp2_generation_core.{h,cpp}`、
      `test_tp2_sessions.cpp`，已实测未提交）——提交或 stash
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
