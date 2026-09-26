# NInfer TP-2 计划（2× RTX 5060 Ti · Qwen3.8-27B NVFP4）

> **唯一的活动计划**，也是跨上下文压缩的持久记忆。整理：2026-09-25（整理前全文入 worklog，本文件只保留
> 关键信息与未完成事项）。
> - 完整历史：`docs/tp2-dual-5060ti-worklog.md` —— 含四份 PLAN.md 全文逐字归档（2026-09-21 上游
>   cherry-pick 重规划版；2026-09-22 交付收尾版；2026-09-24 TP-2 会合协议重构后版；2026-09-25 本次整理前
>   全文，含 r67–r70 草稿量化战役、TP-2 传输低频失败排查与 TP-2 编排/会合工作）。
> - 交付说明、推荐配置与实测数据：`docs/tp2-dual-5060ti.md`；Windows 原生移植：`docs/windows.md`。
> - 定案决策速查（否决方案、根因、方法论规则、worklog 指针）：`docs/tp2-decisions.md`——重复调研已定案话题前必读。

---

## 1. 关键信息

- **分支**：`feat/windows-native-port`（Windows 原生 + WSL2 双树）。origin/master 领先的 20 个提交已全部
  cherry-pick 完成（前端 jinja 链 + NVFP4 性能三项），端到端 A/B 无退化。
- **硬件**：2× RTX 5060 Ti 16G（SYS 拓扑、无 P2P、448 GB/s/卡）。WSL2 构建树 `/home/zhuojun/ninfer`；
  Windows 工作树 `D:\Documents\workbench\ninfer`（构建树 `build-win`）。
- **主负载**：Qwen3.8-27B NVFP4，TP-2（权重按 shard 切半，本地自建半几何 shape）。草稿后端：`--spec dflash2`
  （K=5/7，K=7 为二进制默认）或 `--spec mtp`（K=2、`--lm-head-draft`）；`--spec dflash`（v1）在构造期拒绝。
- **现状**：交付目标 ①–⑤ 全部完成；vision、Windows 移植、多会话 KV 池、decode graph + AR 策略均已落地实测；
  DFlash2 已在 TP-2 放行（未对齐复用不再弃稿）；TP-2 会合协议重构完成（host 权威 id + 有界自旋 +
  超时失败语义）；DFlash2 草稿 Q4 全集合已升入官方配方（r69，用户确认，最终件可发布）；verify graph
  B7 验收通过（聚合 +5.2%）。
- **关键数字**（262,144 配置，WSL）：prefill 1,588 tok/s；decode 59.0 tok/s（MTP K=2）；Windows 131,072 配置
  decode 56.9 tok/s（与 Linux 持平）。DFlash2 K=7 在 4096 上下文贪心档约 71 tok/s；245,760/k8v4（draftall 件）
  扫描 K=5 96.4 / K=7 96.1 tok/s（交错复核 93.3 / 96.9）；roofline 修正上限 ≈108 tok/s（原归档 90–180 的
  180 不可达）。r69 最终件相对 r66 省 257.77 MiB（−1.08%），接受率无可测代价（≤1σ），轮时代价 ~+0.4%
  （proposal 步 +0.19 ms）。

**推荐运行配置**（WSL 8088 / Windows 8099 同配方，Windows 侧 `--max-context 131072`）：

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh` 是进程级默认思考强度（请求体优先）；`--chat-template` 随第一梯队摘取
可用（模板由内嵌 Jinja 执行）。2026-09-23 实测：DFlash2（K=5/7）decode 领先 MTP ~20%、prefill 代价 1% 内
⇒ 追求吞吐用 `--spec dflash2 --draft-tokens 7`（扫描配方为 245,760/k8v4，依据见归档）。

**最终件**（r69/r70，均官方配方无 `--override` 转换，均含 text/vision/mtp/dflash2，草稿剖面四件一致：
q4 88 / q8 7 / bf16 567）：`D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer`、
`…_w4a4_dflash2_final.ninfer`（18.42 GB）、`…_swift15_dflash2_final.ninfer`（21.58 GB）、
`…_thinkingcap_dflash2_final.ninfer`（22.52 GB）。

### 环境与运维要点

| 项 | 值 |
|---|---|
| 构建（WSL） | `bash tools/tp_bootstrap/r55_build.sh`（rsync + `cmake --build build_dyn -j 8`，成功标记 `BUILD_EXIT=0`，约 2–3 分钟；WSL 构建树现为 `build_dyn`（Ninja），`build_r35.sh` 仍指向已删除的 `build/`，不可用） |
| 构建（Windows） | `tools/win_port/configure.bat` + `build.bat`（VS2022 + CUDA 13.3，`-DCMAKE_CUDA_ARCHITECTURES=120a`） |
| 服务 | WSL 8088（`serve_supervise.sh`）；Windows 8099（`tools/win_port/serve.ps1`，默认前台；自测服务必须用 harness 后台 job，`Start-Process` 起的进程会随工具调用结束被杀） |
| 运行 PATH（Windows） | FFmpeg（`D:\ffmpeg-dev\…\bin`）与 libcurl（`D:\curl-dev\…\bin`）必须在 PATH，否则 `STATUS_DLL_NOT_FOUND`（三件套 solo/sessions 无它们会在加载期 `0xC0000135`） |
| artifact | `D:/LLM/qwen3_8_27b_nvfp4.ninfer`（23.7 GB，旧 artifact，模板 `c3cf9e34…`）；新官方 artifact（模板 `a497db9e…`）随第一梯队可用；r69/r70 最终件见上 |
| 进程纪律 | 全机同一时刻只有一个模型进程（WSL 8088 与 Windows 8099 互斥）；Windows 重新链接 exe 前先停服务（LNK1104）；`build-win/apps/ninfer-serve.exe` 不会自动同步 `C:\ninfer\`，需手动复制并以 `Get-FileHash` 比对 |
| 工具链 | 嵌套 `pwsh` → `wsl -e bash -lc` 吞 `$var` 与重定向 ⇒ 命令写成脚本文件放 `tools/tp_bootstrap/` 再执行；单次阻塞调用上限 600 s ⇒ 长任务用后台作业 |
| GPU/WSL 状态（2026-09-23 起） | 机器可见 3 卡：nvidia-smi 序 0/2＝5060 Ti、1＝Tesla T10；`--devices 0,1` ⇒ CUDA 序数取到两张 5060 Ti（与 nvidia-smi 序错位），TP-2 配对不受影响。WSL 侧 CUDA 全线段错误 ⇒ 测试改走 Windows `tools/win_port/test.ps1`，WSL build_dyn 只作编译验证 |

---

## 2. 已完成摘要（每条一行，细节见归档）

- ① MTP 一致性：根因＝分片未写 replay 记录（fold 消费空日志），已修复；判据为「无退化 + 质量同档 + 有加速」。
- ② 基准：prefill 615→1,588 tok/s；decode plain 31.4 / MTP K=2 50.5–54.0 → 59.0 tok/s。
- ③④ KV / 上下文：五种 KV dtype 可用；65,536 → 131,072 → 262,144（词表/隐层并行切分，−1.2 GiB/卡）。
- ⑤ 文档：`docs/tp2-dual-5060ti.md` + `docs/windows.md`。
- Vision：静态分片分工（MTP→shard 0、vision→shard 1），单图上限 8,192 token，无新增配置。
- Windows 原生移植：VS2022 + CUDA 13.3 构建；CLI/serve/TP-2 全部跑通，性能与 Linux 持平；TMA 描述符
  mapped-pinned 修复。
- 优化：exact-batch verify graph（−4.4 ms/轮）、plain decode graph（+12.6%）、AR 按 payload 切传输（+0.63%）、
  MTP draft 链图化（−0.2~0.3 ms/轮）；AR∥MMA 子块流水实测慢 5%，已否决回退。
- Round 19/20：chunk 计划与 ring rewind 解耦（1a/1b/1c），同一 prompt 的结果不再依赖引擎历史；host KV arena
  默认可分页（`--host-kv-pinned` 才锁页，锁页被拒自动回退）。
- 多会话 KV 池：host KV 换入换出 + 5 会话 LRU + 共享前缀镜像；召回 KV 与 device 逐字节一致。
- 上游 cherry-pick：第一梯队 5 项（jinja 前端链 + `--chat-template`）、第二梯队 3 项（NVFP4 TMA 路由、
  SwiGLU epilogue、非 256 整除 token）全部落地；e2e A/B 无退化（decode +1.3%~+5.8%）。
- 官方 NVFP4 工件接受率调查：合成工件与自转件一致、官方件最低 ⇒ 差异来自 text 权重，与组件无关。
- TP-2 显存不平衡：方案 A（reduced proposal head 按词表切分，−170 MiB/卡）与方案 B（selector 移至 shard 1，
  −246/+246 MiB）均已落地，输出逐字节一致；新增 `DevicePair::sendrecv` 做逐字节跨卡交换，修掉 BF16-add
  allreduce 搬 I32/FP32 的 sNaN 位型风险。
- TP-2 会合协议重构（2026-09-23，事故驱动）：Phase 1 有界自旋（env `NINFER_TP2_AR_TIMEOUT_MS`）+ 超时失败
  语义（HTTP 503 + 会话毒化重新 prefill）；Phase 2 host 权威 64 位 id（经图内 memcpy node 送达）；Phase 3 大
  payload 事件路径实测否决；Phase 4 partial 合并审计定案形态不存在。回归矩阵零回归。
- K 扫描与 prefill 成本归因（2026-09-23）：DFlash2 K=5≈K=7（**不要用 K=6**——可复现每轮成本凹陷）；纯 MTP
  K=3/4 平局且同 K 落后 DFlash2 ~20% ⇒ 生产继续 `--spec dflash2`；开投机几乎不付 prefill 代价；prefill
  1,905→569 tok/s（0→245k）＝深度无关固定项（AR ≈65% 已贴 Gen4 x4 链路地板、软件无空间）+ 深度项（245k 处
  attention 占 70%）；GQA 6× 请求冗余被测量门**否证** ⇒ L1 取消。
- DFlash2 draft 量化 r54–r66（2026-09-23/24，opt-in → 官方）：draft gate/up/down/output/kernel/feature 投影
  q4（r62–r66，−173.05 MiB）；draftall 件 −715.6 MiB；接受率裁决定案 30-rep 采样主导（15-rep 臂间噪声
  ~±2 pp）。
- DSH/agent-loop 前缀复用（2026-09-24）：TP-2 路线 `--chat-template` 被静默忽略已修（frontend
  `chat_template_path` 透传）；模板 `|trim` 假设**实测否定**；新增常驻两轮用例与文档。
- 首个生成 token 不发布（2026-09-24）：TP-2 prefill 首 token 现 `preview_model → commit → publish`；回放
  回答直达 `src=live`；新测试用例 `check_first_token_published`。
- 复用门控与弃稿（2026-09-24）：「未对齐复用必须弃 masked draft」的前提**被实测推翻**（同轮：弃稿 22.5 vs
  保留 87.1 tok/s）⇒ 删除门控、`draft_context_declined`、`reuse_grid` 与会话均值定价；契约：网格对齐与
  oracle 逐 token 一致、chunk 内只保证 boundary crossing。
- ② 融合 QKV→q4（r67，2026-09-24/25）：不需要新融合 kernel（decode 走 3× `ops::linear` 行视图；append 走
  既有逐层通用分支）；qkv4 件 −79.69 MiB，三件套全过；30-rep 接受率无可测代价；真实代价仅 proposal 步
  +2.8%（~+0.4% 端到端）。
- ① selector codebook→q4（r68，2026-09-25）：格式无关 plan + 按存储 scale 独立解码的 q4 op oracle（FP64）；
  loader 不再钉格式；唯一派发点 `run_candidate_selector()`；peer selector 245.0→66.9 MiB；cb4 30-rep
  接受率无代价。
- r69 官方配方提升 + 最终件（2026-09-25，用户确认）：完整 DFlash2 draft Q4 集合（15 QKV + 2 codebook → q4）
  进入 `tools/convert/official_recipes.py::_optional`；最终件 vs r66：接受率 ≤1σ、三件套全过、peer
  selector 66.9 MiB ⇒ 可发布（省 257.77 MiB、轮时代价 ~0.4%）。
- r70 四家族最终件（2026-09-25）：w4a4 / swift15 / thinkingcap 变体各产一件，append 冒烟全过；ThinkingCap
  tokenizer 坑修复（前端三处校验：Split 预分词器、`added_tokens_decoder`、`add_bos_token`；规范
  tokenizer.json 替换 + config 合并；serve 启动 + HTTP 问答通过）。
- TP-2 编排与会合（2026-09-25）：`forward_tp2_window` 消灭隐式默认参数（1a）；shard mask 命名常量 + 握手
  协议总表 + packed 断言（③，断言抓到一个真实累加器遗漏）；1b give-up abandon-output（窗口内 give-up 从
  illegal address 逐字节复现变为设计好的 503）；B7 验收：verify graph 聚合 +5.2%（6 reps/臂，定档需 ≥24）。
  全批数值零影响，三件套 digest 逐位不变。
- TP-2 传输低频失败排查（2026-09-25）：定性——give-up 落在 captured 窗口内时内核把不一致数据当地址用而崩，
  host 侧检查来不及；长跑不复现（≤1/1500），注入 700 逐字节复现。处置已实施：看门狗默认开启（大 dump 每次
  stall 限流一条）、超时默认 2000→10000 ms、abandon-output；触发源未决。

---

## 3. 未完成事项

### 3.1 交付前收尾

- [ ] 用推荐配置复测并发 C=1/2/4 与流式/stop 冒烟（Round 31/34 的结论基于 bf16 配置）。
- [ ] 交付时按 AGENTS.md 移除本文件（PLAN.md）；决定 `tools/tp_bootstrap/` 下约 300 个一次性诊断脚本的
  保留/清理范围（`build_r35.sh`、`serve_*`、`r37+r38+r39+r4x` 系列与 `docs/tp2-dual-5060ti.md` 引用为
  可复现流程）。

### 3.2 Windows 单卡 CLI tiny artifact 子任务（用户决定；前置已就绪）

目标：产出小到装进单张 16 GiB 卡的 `.ninfer`，用 Windows `ninfer.exe` 单卡跑通「加载 → 生成 → 采样」。
前置（Round 54.9）：CPU 版 torch 在 `build-win\torch-venv`（torch 2.14.0+cpu）；转换器接口已确认。
下一步：① 官方 recipe 直接跑 tiny checkpoint（`--recipe qwen3_8_27b_nvfp4 --components text`）；
② 不成功再写 tiny recipe/override；③ 生成脚本放 `tools/win_port/tiny_model.py`（合成 tiny config +
随机 bf16 权重，HF 命名）；④ 成功则写入 `docs/windows.md` 并提交。

### 3.3 未修缺陷与已知局限

- TP-2 路线 `bf16` KV + `--spec mtp` 在第一次 prefill 就崩（`prompt.cu:63` `cudaErrorInvalidValue`）；
  fp8 + MTP 正常（推荐配置不受影响）。排查入口：bf16 prompt-attention 的 launch 参数。
- MTP 路线在 `execution/text.cpp:486` 附近同样用 BF16 相加的 allreduce 搬 I32 proposal ids ⇒ 落在
  signaling-NaN 位型时会被静默改写（DFlash2 的等价交换已改 `DevicePair::sendrecv` 逐字节；MTP 未改，
  验证成本高）。
- MTP3 ≥ 70 tok/s 门槛未达（纯 decode 上限 K=2 50.5–54.0）。
- TP-2 路线未跑 perplexity 评测（质量证据为同提示多采样 A/B）。
- per-shard arena 的 `memory_summary()` 仍报 `pages 0/0`（仅显示口径问题）。
- 温度 0 不保证逐位 argmax：DFlash2 稀疏接受 + 每请求随机 seed（`translate.cpp:51-57`）使解码在温度 0 下
  也不确定 ⇒ 需要可复现 A/B 时加 `--greedy`（既有行为，未改动）。
- `ninfer_qwen3_5_tp2_sessions_test` plain 路线「a conversation behind a shared system prompt」首采样分叉：
  既有失败（r57 基线件与 r66 q4all 件逐 token 复现；dflash2/mtp 路线通过）⇒ 阻挡「三件套全绿」（sessions
  门禁现按 dflash2 路线跑）。
- TP-2 prefill 的首个 sample 未应用工具语法掩码（prefill 与 decode 路径不对称，独立遗留项）。
- 前缀复用：文本协议回放无法保证逐 token 复现（客户端重分词与采样边界不同、或边界落在 chunk 内——契约只
  保证 boundary crossing）；保证逐 token 的唯一杠杆是携带 token id 的续写协议（未排期）。
- 两次低频失败（与改动无关、重跑全过，排查见 §3.6）：① r67 轮 q4all 臂 K=7 `cudaErrorIllegalAddress`
  （req#14）；② r69 轮 final 臂 K=5 首次运行 req#14 HTTP 503 且无 FATAL 日志。

### 3.4 暂缓 / 可选（未排期）

- Round 15 剩余：工具调用约束真实模型端到端复测（75 条探针）；WARN 诊断；路线 B（完整 GBNF）；
  TP-2 `validate_licensed_tokens` 守卫。
- Round 14 候选（用户暂缓）：① system+tools 前缀末尾加锚点（修新会话/压缩后第一轮 46.9 s，可省约 11 s）；
  ② prefill 吞吐（suffix 0.95–1.13k tok/s、冷启 1.35–1.38k）；③ 每请求 ~0.3 s 固定开销。
- 归档 §11.6：MTP priming 对图片列用占位 embedding（无收益证据）；单图上限 8,192 硬编码（让出 shard 1 可
  重拿更大 envelope）；视觉编码期 shard 0 空闲（可接受）。
- 归档 §6：KV dtype 扫描补全（nvfp4/k8v4 只验证了可启动与吞吐，无质量数据）。
- 若要从源权重真正复刻官方工件的 DFlash2 draft，需先获取其 BF16 源检查点（本机只有 FP8/EXL3/GGUF）。
- L2：AR 与 MMA 重叠（环形 staging + 事件同步 + 子块流水）：浅层上限 ~1.5×；大重构，跨卡 rendezvous 死锁
  风险（worklog Round 35c 已评估）。
- 若仍攻 attention：只剩计算路径效率（FP8 PV + 压缩非张量 FP32 遍数），且需先解锁 profiler 权限（ncu
  2025.4.1 已装，`ERR_NVGPUCTRPERM` 被拒；管理员 PowerShell 或 NVIDIA Developer Settings 允许 GPU
  performance counters）。预期 attention 项 ~1.2–1.4×（245k 端到端 ~1.1–1.3×）。
- 硬件杠杆：卡 1 从芯片组 Gen4 x4 槽（~7 GB/s）移到 CPU 直连 Gen5 x8 槽（~20 GB/s）：零代码，浅/中上下文
  1.78×、245k 1.15×；需主板有空槽。
- DFlash2 K=6 每轮成本凹陷（三轮恒 25.7 rounds/s，低于 K=5（27.9）与 K=7（26.6））：疑似按窗口宽度
  分桶/对齐的核函数边界效应；`NINFER_TP2_TIMING=1` 取逐相位每轮耗时排查。
- DFlash2 prefill 稳定慢 3–4% 的确切来源（未做逐相位归因；`NINFER_TP2_TIMING=1` 拆 prefill 每相位）。
- B7 定档：verify graph A/B 为 6 reps/臂（聚合 +5.2%、逐请求 t=0.99 不显著）；需 ≥24 reps/臂 才能定档。
- 1a 残留：两侧实参收进同一 struct（两个调用点已显式且一致，收益递减）。
- DFlash2 TP-2 时期未实测项：Windows 原生构建的真实 free（台账 `free` 取自 `cudaMemGetInfo`，WSL2 少报
  约 1 GiB）；草稿按 head/row 切分的数值等价性；K=15 的 workspace 峰值；目标 5 层残差跨 shard 汇聚的每步
  开销；草稿每层 allreduce 的延迟；稀疏拒绝采样在 TP-2 分片 logits 下的等价性。
- 门禁修订决策（与提升分开）：字面门禁「贪心 |Δacc| ≤ 1.0pp」K=7 未过（−1.60pp，已证明为跨臂文本漂移
  混淆）；提升已按用户确认完成（r69），30-rep 采样主导裁决为事实方法——若将来正式修订门禁，在此一处做。

**设计约束（勿重复调研）**：整份复制 draft 不可行——满上下文（262144 / fp8 / MTP K=2）下 free 仅
697/1217 MiB，而草稿 2.07 GiB/卡；切分后每卡约 1.04 GiB。提议条件是目标 5 个 block 的 residual 拼接投影，
TP-2 把 64 层切两卡 ⇒ 需跨卡 handoff（两卡 residual 逐位相同，只捕 shard 0 即可）。候选 selector 直读
全词表 codebook（约 254 MiB），必须整份。DFlash2 与 MTP 互斥：选 DFlash2 可释放 shard 0 的 MTP 权重
430 MiB + KV 516 MiB。每轮成本构成、D1–D4 决策与各轮否定结论见归档。

### 3.5 环境受阻的未做项（已用替代路径覆盖）

- [ ] WSL 侧 op 测试 + 字节一致 + 前端夹具测试：**WSL2 CUDA 驱动崩溃**（`cudaGetDeviceCount()` 内 PTX JIT
  segfault，GPU 被 Windows 服务占用）；jinja 测试通过证明二进制无误，op 测试改在 Windows 侧跑。Windows 侧
  已过（NVFP4 A4/A16、frontend、jinja；`NINFER_OP_REPORT_STATS=1` 错误指标在容差内）⇒ WSL build_dyn
  只作编译验证。

### 3.6 TP-2 传输低频失败（剩余）

- **定性（已证）**：有界自旋 give-up 后，本轮各 collective 的偏和**只在本地**（对端贡献缺失），而 round
  会继续跑；give-up 落在 **captured 窗口内部**时，窗口自己的 kernel 把不一致的数据当**地址**用 ⇒
  illegal address，host 侧任何检查都来不及（注入 700 逐字节复现）；只有 eager prefill 的 give-up 才走成
  干净 503 + 下一请求恢复。
- **触发源未决**：自然长跑（累计 1500 请求、看门狗开启）0 事件，自然发生率 ≲1/1500；需要 >2 s 的
  host/device 延迟，或一次真实 desync（id 复用/错配）。看门狗默认开启；失败时另存 `*-FAILED.log` /
  `*.arwatch.txt`（判读按 id 差值量级：整组 0＝该侧从未到达 arrival 写入；1＝OFF-BY-ONE；整块＝
  BLOCK-LEVEL；其余 SKEW）。
- **(b) 让 give-up 安全（待决策）**：host 侧检查已被证伪；give-up 后毒化/中止语义，或窗口内做索引的算子
  对输入做范围约束——都是架构级改动，需要先确定确切算子。
- **工具边界**：`CUDA_LAUNCH_BLOCKING=1` / compute-sanitizer 与本传输不兼容（A 侧 launch 变阻塞，B 侧
  内核要等它返回后才 launch ⇒ 第 1 次 collective 必然 give-up，自死锁）⇒ 无法用 launch 串行化工具做
  kernel 级归因，只能靠源码级探针二分。


---

## 4. 架构精读与缺陷分析（2026-07-18，追加）

> 本节为一次独立的全架构精读（不使用子代理），在旧 TP-2 计划之上追加，不改动上文。

### 4.1 精读范围

通读 `docs/maintainer/` 三份架构权威文档（engine-architecture、paged-kv-cache、resource-scheduling），
逐层精读 C++ 核心：`src/core`（paged_kv_cache、host_kv_arena、tp/device_pair、arena、
linear_attention_state、gdn_replay_records）、`src/runtime`（engine_core、scheduler、
tp2_generation_core、contract、kv_capacity）、`src/serve`（generation_service、http_server/transport、
openai/anthropic 路由、translate）、`src/ops`（sampling、causal_attention、kv_cache_append）、
`src/models/qwen3_5`（execution/text、program/decode、frontend/output_session、prefix_identity、
dflash_round），并结合 git 历史核对上文 §3.3 已知缺陷。

### 4.2 确定性 / 潜在 bug

**复核结果（2026-09-26）：本节唯一的「确认 latent bug」是误报，已撤销——`zero_pages` 的两种 plane
order 都正确。** 推导与可执行判据如下。

- **〔撤销〕`src/core/paged_kv_cache.cpp` — `zero_pages` 的 HeadMajor 分支"越界写"（原文 2026-07-18）**：
  原文断言对 HeadMajor 布局（平面形状 `[X,64,N_phys,H]`）的
  `cudaMemset2DAsync(dst=base+first*nb2, dpitch=nb3, width=count*nb2, height=H)` 第 i 行前移
  `i*elem_size`，总写范围超出目标块 (H−1)×elem_size 字节，并建议改为 1D
  `cudaMemsetAsync(base+first*nb2, 0, count*nb2)`。
  **复核：前提错了——`ne` 的次序不是 numpy 那样的"最后一维最内层"。**
  `Tensor::set_contiguous_strides`（`tensor.cpp:50-57`）从 `nb[0]=dtype_size` 起逐维相乘 ⇒
  **`ne[0]` 最内层**，而 `plan_device_kv_page_pool`（`paged_kv_cache.cpp:90-93`）给 HeadMajor 建的是
  `{leading_extent, kPagedKVPageSize, physical_pages, head_extent}` ⇒
  `nb[2] = 64*leading*es`（**单页字节数**）、`nb[3] = physical_pages*nb[2]`（**head 步长**）、
  `ne[3] = head_extent`（head 数；既有用例已钉住 `ne[2]=pages`/`ne[3]=heads`，见
  `tests/test_kv_cache.cpp:247-251`）。于是 `dpitch=nb[3]` 正是"每行一个 head"，第 i 行覆盖
  `[base+first*nb2+i*nb3, +count*nb2)` ＝ head i 的页区间（行 0 不再"恰覆盖"，各行等距），
  末字节 `first*nb2+(H−1)*nb3+count*nb2 ≤ H*nb3` ＝ 平面末尾（run 长度 ≤ 总页数）⇒ **无越界**；
  且 `dpitch = physical_pages*nb2 ≥ count*nb2 = width`，满足 `cudaMemset2DAsync` 的 pitch 约束。
  原文把 shape 初始列表当成了内存顺序，才把 `nb[3]` 读成元素大小；它建议的 1D memset **只清零 head 0**，
  其余 head 的页保持原值，不是正确实现。
  **判据**（新增 `tests/test_kv_cache.cpp` 的 `exercise_zero_pages`）：两种 plane order 各取中间 run
  `[2,5)` 与末页 `[7,8)`，先把整池填确定型位型、跑 `zero_pages`、再逐平面**全字节**比对期望
  （run 内全 0、run 外逐字节不变）。当前实现 **PASS**；把实现临时换成原文建议的 1D memset 后 **FAIL**
  （新增用例 4 条 + 既有 `selective zero left page payload bytes`，共 5 条）⇒ 该用例确实能判定这件事。
  同时 `paged_kv_cache.h` 的 `PagedKVPlaneOrder` 补上"枚举名指最慢维、两种顺序的 shape 与
  page/head 步长"注释，避免复现同一误读。

### 4.3 上文 §3.3 已知缺陷核实

①②③④⑧ 在当前代码**均已修复**（证据见 4.2 与 §3.3 各条括注）；⑤（give-up 安全）已缓解但仍是开放架构问题；
⑥⑦⑨⑩ 为已知限制/未决，本次未展开。

### 4.4 设计缺陷与隐患（架构层面）

1. **双执行核心漂移风险**：TP-2 走独立 `TP2GenerationCore`，绕过 EngineCore 调度器/CUDA Graph/
   ResourceManager（单请求 lockstep）。请求生命周期、容量、取消、前缀复用、checkpoint ring 等逻辑在两条
   路径重复实现，须长期保持语义一致——已多次产出 prefill/decode 不对称类缺陷（工具掩码、首 token 时序、
   MTP ids 交换皆实例）。最大维护性隐患。
2. **TP-2 in-kernel allreduce give-up 安全性开放**：超时落在 captured 窗口内时 kernel 清零本侧输出片避免
   不一致数据被当地址，但本轮仍以混合 real/zero 数据完成，靠 host `ar_stalled()` + 收敛检查转失败。
   已缓解（有界自旋+看门狗+`abort_if_ar_stalled`），「让 give-up 本身安全」待决策。
3. **host-staging CPU allreduce 回退是性能悬崖**：in-kernel 传输不可用时回退到 host staging，逐元素加在
   CPU host loop 完成且同步双流，大 prefill 载荷极慢。文档化回退，但构成显著性能断崖。
4. **TP-2 前缀复用容量硬上限**：device reuse snapshot 固定 `kReuseSnapshotCount=2`（每槽 73 MiB/shard），
   更深分歧靠 host checkpoint ring（一次 PCIe 往返）。显式容量权衡，深分歧场景复用成本被结构性抬高。

### 4.5 排查后排除的疑点

- **`engine_core.h` worker 循环尾部 `queue_cv_.wait_for(1ms)`**：初判给每个 decode round 加 1 ms 延迟，
  复核**不成立**——各执行分支（control/prefill/decode）都 `continue` 直接回环顶，1 ms 等待只在「无执行
  单元（action==Wait）」空转路径到达，忙时为紧循环。
- `host_kv_arena`（first-fit + coalesce + plan/apply 偏移一致）、`kv_capacity`（溢出检查完备）、
  `sampling`（greedy/stochastic 双路、RNG 纯函数可 graph replay）、`causal_attention`（online-softmax
  split merge、envelope 校验）、`output_session`（UTF-8 增量解码、stop 匹配、reasoning 通道分离、
  preview/commit）、`prefix_identity`（FNV 风格 digest）均审毕，实现严谨，未见确定性缺陷。

### 4.6 结论

整体工程质量高：边界校验、溢出检查、生命周期所有权、数值 oracle 约定到位，§3.3 已知缺陷当前代码全部已修复。
本次唯一「确认 latent bug」（`zero_pages` HeadMajor 越界写）已按 §4.2 复核**撤销**——原文用错了 `ne` 次序，
代码两种 plane order 都正确，并已补上字节级用例；**4 项架构级隐患**中「TP-2 双核心漂移」与「TP-2 give-up
安全性」最值得后续投入（前者见 §4.7 实施记录，后者仍未决策）。

### 4.7 §4.4.3 解决方案调研（host-staging CPU allreduce 性能悬崖）（2026-09-26）

> 对 §4.4 第 3 点（in-kernel 传输不可用时回退 host staging，CPU host loop 逐元素加 + 同步双流，
> 大 prefill 载荷极慢）的可用方案调研。仅调研与评估，未改代码。

**现状清点（代码证据）**

- 三档传输（`src/core/tp/device_pair.cu`）：in-kernel（mapped pinned staging + arrival id，
  `815-871`；当前生产路径，已测到距 Gen4 x4 链路双向带宽下界 2%：`809-814` 注释）→ P2P
  （`872-895`，需 `cudaDeviceCanAccessPeer`，WSL2 下 `542-552` 探测失败）→ host staging 回退
  （`896-928`）。
- 回退触发条件仅两个：(a) 构造时 `cudaHostAllocMapped` 失败（`in_kernel_available_==false`，
  `553-623`）；(b) 载荷 > `kInKernelArBytes`（24 MiB，`249`）。
- 触发 (b) 在当前产品参数下**不可达**：prefill chunk 钳在 `kPrefillChunkMaximum=1024`
  （`tp2_generation_core.cpp:164,2545-2547`），最大载荷为 mixer/MLP delta
  5120×1024×2 = 10 MiB、gather activation 6144×1024×2 = 12 MiB，均 < 24 MiB。但「载荷 ≤ staging」
  只是数值巧合——chunk 钳位（`2545-2547`）只引用 `kPrefillChunkMaximum`，并未从
  `in_kernel_bytes_` 推导，是隐含不变式而非显式约束。现实触发是 (a)：非 WSL2 / mapped-pinned
  不可用的环境。
- 悬崖成本分解（`896-928`）：① **标量单线程 CPU `__hadd` 循环**（`919-921`）——单线程约
  0.5–2 GB/s，10 MiB 载荷约 5–20 ms/次（in-kernel 同载荷 2.7 ms），是主导项；② 两条计算流
  `cudaStreamSynchronize`（`913-914`）——每层全流水线中断；③ **不可 graph 捕获**——
  `tp2_generation_core.cpp:385,451` 在 `in_kernel_allreduce()==false` 时把 MTP 链、decode、
  verify 图全部降级 EagerExact。PCIe 流量与 in-kernel 相同（每卡 2× 载荷），故悬崖不是带宽问题，
  是「CPU 计算 + 全同步 + 无图」问题。
- 放大倍数：每层 2–3 个 collective（`text.h:564,576,589`）× 64 层，每 chunk 128–192 次；
  回退时单 chunk allreduce 成本从 ~0.35 s 升到 2–4 s 量级。
- 测试覆盖缺口：`tests/test_tp_device_pair.cpp` 的最大载荷用例是 2433024 B（`666`），
  >24 MiB 的回退路径与 mapped-pinned 不可用路径**均无现有用例**（`51` 注释声称三路径位精确一致，
  但回退路径在本机上从未被执行）。

**外部佐证（web）**

- 消费级双卡 PHB 拓扑的标准驱动**不开 P2P**；裸机也需打补丁的 open kernel module（BAR1）才可用
  （[vLLM forum：双 5090 SHM vs P2P](https://discuss.vllm.ai/t/dual-rtx-5090-tp-2-shm-vs-patched-bar1-p2p-cumem-single-pass-2-7-mean-point-estimate/2870)：
  P2P 开启平均仅 +2.67%、workload 相关，另一组多轮测量为 −5–7%）。WSL2 为 Hyper-V PCI passthrough +
  Microsoft 托管内核，补丁模块不可行。→ P2P 路线对本产品目标（WSL2 双 5060 Ti）**不适用**，
  且裸机收益也只是百分之几。
- NCCL 在无 P2P 时走 **SHM transport**（GPU 经 PCIe 写共享 host 内存、对端 GPU 读回，
  [Demystifying NCCL](https://arxiv.org/html/2507.04786v1)；
  [NCCL issue #1838](https://github.com/NVIDIA/nccl/issues/1838)）——与 NInfer 的 in-kernel
  mapped-pinned 路线同构，说明 in-kernel 路径已是该环境的标准解，host 侧不存在更优传输路线。

**方案评估**

| # | 方案 | 效果 | 成本/风险 | 结论 |
|---|---|---|---|---|
| S1 | **修复回退本身**：CPU 加改 SIMD + 多线程（AVX2/AVX-512，float 域加回 BF16 RNE 以保位精确），载荷分块 + 双缓冲流水（D2H(i+1) 与加(i) 重叠） | 回退从「~5–10× 悬崖」降为「链路受限」，与 in-kernel 同量级；不引入图降级以外的额外损失 | 改动限于 `device_pair.cu` host 侧循环；位精确可由现有三路径一致断言直接验证 | **推荐（主）** |
| S2 | **扩大 in-kernel staging**：`kInKernelArBytes` 24→48/96 MiB（成本仅 host pinned RAM）+ 让 prefill chunk 钳位从 `in_kernel_bytes_` 推导，把「载荷 ≤ staging」变成显式不变式 | 封死触发条件 (b)，对更宽模型/更大 chunk 免疫 | 一个常量 + 一行钳位；风险极低 | **推荐（辅）** |
| S3 | P2P（裸机 BAR1 补丁） | 裸机百分之几、workload 相关 | WSL2 不可行；引入驱动依赖 | **排除** |
| S4 | copy-engine 事件式大载荷路径 | — | 已实测排除：`device_pair.cu:813-814`「copy engines were slower」；`test_tp_device_pair.cpp:422-427` 的 ceiling 探针（`NINFER_TP2_AR_COPY_BENCH`）确认 sliced in-kernel 已距链路下界 2% | **排除（已有内部证据）** |
| S5 | 载荷量化（delta 降 FP8，字节减半） | 带宽减半 | 改变 residual 数值语义，需全套 oracle 合格；in-kernel 已在链路下界，当前目标无收益 | **超出当前 scope，不做** |

**建议**

- 一批做 **S1 + S2**（diff 限于 `device_pair.cu` + `tp2_generation_core.cpp` 钳位 +
  `tests/test_tp_device_pair.cpp`）：回退变链路受限、(b) 触发被封死、chunk 钳位显式化。
  图降级（EagerExact）保留——host 同步是回退的固有属性，S1 后非 WSL2 环境的退化只剩「无图」，
  不再是悬崖。
- 验证：`test_tp_device_pair.cpp` 增加 >24 MiB 载荷用例（走回退路径，三路径位精确断言复用）；
  加一个可强制回退的测试钩子（如跳过 mapped-pinned 初始化的 env），对回退路径做前后 A/B 计时。

**裸机 Windows 适用性**：结论原样成立。典型桌面 PHB 拓扑下 P2P 同样不被授权（闭源驱动对
GeForce 拒绝 P2P；授权依据 NVIDIA 芯片组允许表，消费级芯片组不在内；[club-3090 六门模型](https://github.com/noonghunna/club-3090/blob/master/docs/PCIE_P2P.md)），
传输分层与 WSL2 一致，S1+S2 收益/代价逐条成立。唯一分叉：若 P2P 恰好被授权（允许表内工作站
芯片组 / PIX 交换拓扑），`DevicePair` 自动走 P2P 路径，host staging 悬崖不存在、S1+S2 不需要；
该场景的正确改进是 kernel-based P2P 交换（可 graph 捕获），超出当前目标范围。

**实施结果（2026-09-26）：S1 + S2 已落地**

- **S1**（`src/core/tp/device_pair.cu`）：回退归约改为 SSE2 向量加（8 BF16/迭代；BF16→FP32 左移、
  FP32 加、按 `__float2bfloat16` 做 RNE 回舍并把 NaN 规范化到 `0x7fff`）+ 32 MiB 以上按 host 线程
  分片（最多 8）。结构为 `2×D2H → 双流同步 → 归约 → 2×H2D`——先实现的 1 MiB 分块流水实测比单次
  大拷贝**慢约 1 ms**（20 次小 H2D 的 CE 开销 > 重叠收益），故回退为单次拷贝。
  实测（`NINFER_TP2_AR_STAGING_BENCH=1`）：10 MiB 载荷旧 ≈ 35.5 ms（标量加 32.4 + 传输 3.1）→
  新 **5.2–8.0 ms**（5–7×）；50 MiB 旧 ≈ 175.7 ms → 新 **20.5 ms**（8×）。
  线程阈值取 32 MiB 的依据：10 MiB 时 8 线程 add 4.1–4.4 ms 反慢于单线程 2.1–2.5 ms（Windows
  线程创建约数百微秒），50 MiB 时 8 线程 8.4 ms 胜单线程 12.0 ms。
- **S2**（`device_pair.{h,cu}` / `tp2_generation_core.{h,cpp}`）：`kInKernelArBytes` 24 → 48 MiB；
  新增 `DevicePair::in_kernel_allreduce_bytes()` 与 `TP2GenerationCore::prefill_chunk_width()`，
  三处 chunk 计算点统一由 staging 容量反推，「载荷 ≤ staging」由数值巧合变为显式不变式。
- **验证**：`ninfer_tp_device_pair_test` PASS。新增 25 MiB（留在 in-kernel；旧 24 MiB 下会掉回退）、
  50 MiB（超 staging ⇒ 回退 + 分片）、强制回退对上的 3 MiB+16 / 50 MiB / 深队列 / sendrecv，以及
  736 例定向位型（ties 12 / inf 84 / nan 247）——两条传输路径都位精确。
- **触发条件复核（2026-09-26 续）**：把回退的 D2H/H2D 换成「内核直写 mapped pinned」这条路线被
  实测和可达性复核否决——同载荷（每卡 20 MiB 双向）copy engine 只比内核路径慢约 14%
  （3.14 vs 2.70 ms），换掉它对回退整体只值 ≤8%（回退总时里 2.1 ms 是 CPU 加法，S1 已到底）；
  而**生产配置下按尺寸触发的回退不可达**：prefill 最大载荷 10–12 MiB（hidden 5120 / gather 6144
  × chunk 1024），词表并行头合并是 `[V,1]`（生产 prefill 传 `logits_columns = nullptr`）与
  verify 窗口的 `[V, W≤16]` ≤ 8 MiB，全在 48 MiB 内。能触发回退的主机恰恰是「mapped pinned
  不可用」那一类——那里内核直写按定义不可能。
- **改为 S6 · staging 阶梯**：预留逐级退化 48→24→12 MiB（两级 parity 槽/卡；12 MiB 是"本产品任何载荷
  都不收窄"的档位，可捕获的最宽载荷是词表分片头合并 7.6 MiB），保住 in-kernel
  传输并让 S2 的 chunk 钳位自动适配较小档位，取代原来的「全有或全无 → 每个 collective 都 host
  同步」。`NINFER_TP2_AR_STAGING_MIB` 给测试复现受限主机用。
- **顺带修复（阶梯验证中发现）**：回退的 pinned staging 没有 parity，两条独立 compute stream 共用一块 ⇒
  下一次调用的 D2H 覆盖上一次调用的 H2D 正在上传的半区，设备收到局部假和（表现为 logits 变化、下游
  甚至 `illegalAddress`）。改为两 parity 槽 + 增长前同步两条流；新增 4 MiB 深队列回归用例（20 KB 那档
  因 H2D 太快而抓不到）。
