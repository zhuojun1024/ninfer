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
- L2：AR 与 MMA 重叠——**已实现并实测：+11% prefill（N=4）**。实现：prefill chunk 切成 N 个子块（64 对齐），
  每个子块的 collective 走第二条流（`DeviceContext::collective_stream`），与下一子块的 mixer/FFN 重叠
  （`run_layers_tp2_overlap`，选项 `--prefill-overlap W`=子块宽度，默认 256，0 关）。实测（chunk 1024、~6.3k prompt、7 reps 取中位）：N=0 0.587 ms/token、N=2 0.554、**N=3 0.612（比基线还差）**、
  **N=4 0.516（−12%，两批次复现）**、N=5 0.547、N=8 0.655 ⇒ 最优点确在 **N=4（子块 256）**，且 **N 不可插值**
  （320/384 落进更慢的调度档），定档应按"子块宽度 = 256"而非按 N 外推（chunk 512 复验：串行 0.625 →
  256 宽 overlap 0.572，−8.4%，且快过 chunk 1024 串行基线 0.587）；数值与单块逐位一致（tp2_forward_test
  的 chunk-split 不变量在 N=2/N=4 仍 exact）。三条前置证据：§5.4 的 divisor 臂（AR 是
  严格串行的 48% 块时）、`tools/tp_bootstrap/bench_ar_overlap.cu`（AR 不被占满 SM 的 compute 饿死，引擎粒度
  下只膨胀 7–18%）、以及 Round 12 的 −5% 只是其 host 侧调度形态的问题。**剩余**：默认值/N 的选择、与 decode
  路径无关、以及跨卡 rendezvous 的失效语义（见 §3.6）。
- 若仍攻 attention：只剩计算路径效率（FP8 PV + 压缩非张量 FP32 遍数），且需先解锁 profiler 权限（ncu
  2025.4.1 已装，`ERR_NVGPUCTRPERM` 被拒；管理员 PowerShell 或 NVIDIA Developer Settings 允许 GPU
  performance counters）。预期 attention 项 ~1.2–1.4×（245k 端到端 ~1.1–1.3×）。
  **2026-09-27 启动为 §6**（含质量风险、变体与门禁）。
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

---

## 5. 上游 ninfer-all 性能项移植计划（2026-09-26 追加）

> 来源：`D:\Documents\workbench\ninfer-all` —— 多 fork 整合线（基座 `ashalliants/ninfer-3090` v0.11.0，
> 多数由 Warlax 编写），其提交在基线移动后逐条重放，故 commit hash 与本地不共享；用 `git patch-id --stable`
> 对齐后，ninfer-all master 的 1770 条非合并提交中 1113 条与本地同补丁，657 条本地缺失（其中 perf/bench 119 条）。
> 本节只收对 **2×5060 Ti TP-2** 有收益的项，按收益/成本排序。
>
> 判据（§1 与 `docs/tp2-dual-5060ti.md`）：decode 是每 shard 权重流带宽 bound（10.15 GB/forward ÷ 448 GB/s
> ≈ 22.7 ms 下限，实测 round 38.2 ms）；另有 128 次/token allreduce（9–18 µs）、MTP 草稿链 host 串行
> 7–10 ms、launch 间隙 ~2.9 ms（1009 kernels/forward）、16 GiB 显存硬约束。排序：减少每卡权重字节 >
> 减少 launch/内核数 ≈ 减少通信次数 > 提高 prefill 算力 > 纯算力优化。

### 5.1 执行顺序与状态

| # | 项 | 上游提交 | 状态 |
|---|---|---|---|
| 1 | per-device `cudaFuncSetAttribute` 缓存（第二张卡 smem 上限未抬升） | `9adeb90c`（修 19 处） | **完成** 2026-09-26 |
| 2 | 170 SM 硬编码 → 运行时设备 SM 数 | `d7bbcf57`、`7afc8e17` | **完成**（3 处代码；1 处待测量） |
| 3 | 减内核/栈帧：RDC-off + attention gate 折叠 + INT8 scale 走 smem | `ea74dca4`、`77c9cc39`、`a941c9f4` | 3a **实测不支持已回退**；3b/3c 暂缓（见下） |
| 4 | 减少每卡权重字节：embedding/head 与 MTP 专家 Q4/Q6 | `1da31697`、`f1982279`、`f8c75edd` | 待决策：本 artifact 是 FP8 词表/头，等价动作＝把要流的 FP8 张量改 NVFP4（~17%，需质量 A/B） |
| 5 | verify 段：small-T tensor-core 内核 + GDN record 窗口 staging | `ce2df46b`、`21df3069` | 5b 已移植（Op 级 −22~27%，端到端测不出）；5a 复核后不做 |
| 6 | 大件（需产品决策）：KV codec / 设备路线 profile / fast prompt kernel | `eb9f7a23`/`ad26b362`/`2ba10da2`、`0b2f8384`/`81861a07`、`4303e604` | 待办 |

### 5.2 各项范围与验证

**1. per-device `cudaFuncSetAttribute`**
- 事实：该属性是「当前设备上该函数」的属性；函数局部 `static` 让第一个设备记录结果、第二个设备跳过调用。
  本地已有正确实现 `src/ops/common/cuda_smem.h::ensure_max_dynamic_shared_memory`（按 (kernel, device)
  记忆，槽满后退化为无条件调用，仍正确），但只有 4 个 `small_t` 站点接入。
- 待修（10 处函数局部 `static`，schedule 允许到 99 KiB > 48 KiB）：
  `src/ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu:59`、
  `src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu:34`、
  `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu:295`、
  `src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu:33`、
  `src/ops/linear/bf16/bf16_launch.cuh:42`、`src/ops/linear/bf16/shapes/n256_k5120.cu:13`、
  `src/ops/linear/fp8/fp8_launch.cuh:59`、`src/ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu:51`、
  `src/ops/linear_add/fp8/fp8_linear_add_a8.cu:32`、`src/ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu:36`。
- 同一机制的三处重复实现一并收敛到该 helper（逻辑等价）：`prompt_fp8.cu:27`、`prompt_k8v4.cu:22`、
  `prompt_nvfp4_non_rdc.cu:19` 的 `static bool attr_done[64]`，以及 `prompt.cu:28-35` 的每次发射都调用。
- 做法：全部改走 `ensure_max_dynamic_shared_memory(reinterpret_cast<const void*>(kernel), bytes)`，
  保留既有 `if constexpr (>48 KiB)` 条件。
- 验证：`cmake --build build-win -j` 通过；`git grep 'static const cudaError_t'` 归零；相关 op 测试
  （`ninfer_tp_device_pair_test`、attention/linear 套件）通过；第二张卡上跑含 >48 KiB 内核的路径不再
  返回 `cudaErrorInvalidValue`。

**实施结果（2026-09-26）**

- 10 处函数局部 `static` 全部改走 `ensure_max_dynamic_shared_memory`：
  `ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu`、
  `ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu`、
  `ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu`、
  `ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu`、`ops/linear/bf16/bf16_launch.cuh`、
  `ops/linear/bf16/shapes/n256_k5120.cu`、`ops/linear/fp8/fp8_launch.cuh`、
  `ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu`、`ops/linear_add/fp8/fp8_linear_add_a8.cu`、
  `ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu`。
- 4 处重复实现收敛到同一 helper：`prompt.cu`（原来每次发射都调）、`prompt_fp8.cu`、
  `prompt_k8v4.cu`、`prompt_nvfp4_non_rdc.cu`（原来各自的 `static bool attr_done[64]`）。
- `git grep 'static const cudaError_t'` 现为 0；所有站点统一带 `#include "ops/common/cuda_smem.h"`。
- 证据：`pwsh -File tools/win_port/build.ps1 -SkipProbe` 全量构建 exit 0（192 步，89 s）；17 个受影响
  套件全绿（device、tp_device_pair、rmsnorm、gated_rmsnorm、gdn_gating、softmax_attention、
  rmsnorm_rope、attn_input_proj、gdn_input_proj、dynamic_grouped_conv×2、linear_fp8_a16/a8、
  linear_bf16_a16、linear_add_bf16/fp8、linear_swiglu_fp8，317 s）。
- **真实产物验证（2026-09-26）**：跑双卡件 `D:\LLM\qwen3_8_27b_swift15_dflash2_final.ninfer`（20.10 GB）：
  `ninfer_qwen3_5_tp2_dflash_append_test` **PASS**（47 s，两卡同时执行前向 + DFlash2 proposal）；
  `ninfer_qwen3_5_tp2_dflash_solo_test` 在 `D:\LLM\qwen3_8_27b_w4a4_dflash2_q4all.ninfer` 上 **PASS**（41 s）。
  两个失败项都是**既有且已记录**的，与本轮改动无关：`tp2_sessions_test` 默认三路线下的 plain 分叉
  （worklog:6117 规定该门禁走 `NINFER_TEST_ROUTE=dflash2`）；`tp2_dflash_solo_test` 在 swift15 件上的稳定
  分叉（worklog:6548 记「`git stash` 后在 HEAD 重建、逐 token 相同地失败」，候选原因是该 artifact 与测试
  基线不匹配）。⇒ 第二张卡上 >48 KiB 内核的路径已由真实两卡前向覆盖。
- 后续候选（未做，属项目 3 范围）：仍有无条件 `cudaFuncSetAttribute` 的每次发射调用
  （`context_kv_materialize/materialize.cu`、`linear/gguf/ggml_bridge_mmq.cuh`、
  `linear/nvfp4/nvfp4_w4a4_tma.cu`、`linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu`、
  `chunked/{output,prepare_wy_wu,state_passing}.cu`、`linear_topk/{fp8,q4,q8}_m64.cu`）。它们正确，但每次
  发射都过一次 driver；TP-2 每次 forward ~1009 次发射会累到 host 侧，值得在有计时证据时一并收敛。

**2. 170 SM 硬编码**
- 落点与改法：`src/ops/launcher/rmsnorm.cu:20`（`kRmsPrefetchBlocks`）、`src/ops/launcher/rope.cu:16`、
  `src/ops/linear_attention/gated_delta_net/chunked/output.cu:9`（`kRtx5090SmCount`）、
  `src/ops/softmax_attention/dense/causal_cache/small_t.cu:66,230`、
  `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:33,42` → 运行时查询设备 SM 数（按设备号缓存，
  查询失败回退 170）。5060 Ti 为 36 SM ⇒ 170 块 ≈ 4.7 波，取整/预取/split-K 全部失配。
- 验证：目标 op 既有测试；rmsnorm / gdn gating / chunked output / small_t 在 5060 Ti 上的 A/B 计时。

**实施结果（2026-09-26）**

- 新增 `src/ops/common/device_sm_count.h::device_sm_count(fallback = 170)`：按设备索引缓存
  `cudaDevAttrMultiProcessorCount`，查询失败清错误并回退 170（与上游 `d7bbcf57` 的约定一致）。
- 三处代码改成运行时查询（170 SM 设备上取值不变，行为逐位不变）：
  `ops/launcher/rmsnorm.cu` 删除 `kRmsPrefetchBlocks = 170`，两处门限用 `device_sm_count()`；
  `ops/launcher/rope.cu` 的 `kLargeBlockWaveCapacity = 1020` 改为 `kLargeBlockCtasPerSm(6) *
  device_sm_count()`；`gated_delta_net/chunked/output.cu` 的 `kTargetCtas = 170 * 4` 改为
  `device_sm_count() * kCtasPerSm`。
- 两处**只改注释、不改代码**：`small_t.cu:66,230` 的 170 只出现在注释里，代码用的是与「一波」对齐的
  字面 CTA 目标（160/320）；`bf16_gdn_gating_proj_plan.cpp:33,42` 也只在注释里，且注明「launcher
  independently enforces actual-device residency」，SM 数已由调用方传入。把 160/320 改成 SM 派生值会
  改变 5090 上的既有行为，必须先有 5060 Ti 上 batch_size>1 的 A/B 才能定。
- 证据：构建 exit 0；18 个相关套件全绿（rmsnorm、rmsnorm_pack_tail、gated_rmsnorm、gated_delta_net、
  gdn_gating、gdn_gating_proj、rope、rmsnorm_rope、linear_tp2_split×3，61 s）。这些测试本来就跑在本机
  的两张 5060 Ti 上，因此新路径（36 / 216 / 144）已按 Op oracle 验收。
- 36 SM 路径的运行覆盖：2026-09-26 的双卡真实产物运行（append PASS、solo@q4all PASS）在 36 SM 的
  5060 Ti 上同时执行了两卡前向，新取值（36 / 216 / 144）已在真实 decode/verify 路径上跑通。
- 待补：5060 Ti 上的 A/B 计时（rmsnorm 门限、rope block 选择、chunked GDN 波目标），以及 small_t
  160/320 目标的测量。

**实施结果（2026-09-26）：3a 实测不支持，已回退；3b/3c 暂缓**

- **3a 取消 RDC（`ea74dca4`）——回退**。用上游同一指标（`cuobjdump --dump-resource-usage` 的 `STACK:`，
  即带栈帧的函数数）在本树对比 `apps/ninfer-serve.exe`：
  - 改前（`ninfer_cuda_archive` = `CUDA_SEPARABLE_COMPILATION ON` + `CUDA_RESOLVE_DEVICE_SYMBOLS ON`）：
    4405 个函数、231 个 `STACK>0`、栈帧合计 8328。
  - 改后（两者删除，全程序设备码）：4395 个函数、**269** 个 `STACK>0`（+16.5%）、合计 9200。
    分族：rope_fixed 有→无、gdn 8→0、q8 13→8（改善）；small_t_i8 90→150、q4 10→13（变差）。
  - 上游「1155→927」出自他们的树，本树净 +38，方向相反。TP-2 的 decode 是每 shard 权重流带宽 bound，
    核内指令效率不在关键路径；本机无 `.ninfer` 产物、做不了端到端 A/B，因而不在无证据时引入一个
    全量重建级的构建改动。已 `git checkout` 回退 `cmake/NinferTargets.cmake`、`src/ops/CMakeLists.txt`、
    `src/ops/linear/gguf/sources.cmake`，并全量重建恢复。
  - 同期一次全量 `ctest`（关闭 RDC 的构建，135 项）：4 项宿主侧失败（`chat_templates`＝Python 侧 GBK
    编码 emoji 报错、`resource_manager`、`engine_options`、`serve_options`，均与本次改动无关的既有失败），
    以及 `context_kv_materialize_test` 300 s 超时（单独复跑仍超时）。该超时未归因，也是回退理由之一。
  - 附注：`src/ops/CMakeLists.txt` 记载 nvfp4 warp-specialized kernel 因 `setmaxnreg` 被 RDC 破坏而单独用
    non-RDC 归档；一旦将来真的关闭 RDC，该归档与 `ninfer_cuda_archive` 将不再有区别。
- **3b attention gate 折进 reduce epilogue（`77c9cc39`）——暂缓**。本地
  `src/models/qwen3_5/execution/text.cpp` 有 3 处独立 `ops::sigmoid_mul`（613/762/1507）。折进需要给
  causal attention Op 增加可选 gate 形参并在各路由 epilogue 实现（跨 `include/ninfer/ops/softmax_attention.h`
  与多个 `.cuh`）。收益：每层少一次 launch，按 1009 次发射 ≈ 2.9 ms 间隙 ⇒ ~2.9 µs/次 × 28 层 ≈ 0.08 ms，
  占 38.2 ms round 的 0.2%。收益低于风险，暂不做。
- **3c INT8 系 attention scale 走 shared memory（`a941c9f4`）——暂缓**。落点为
  `prompt_i8.cuh:223/224/261/262/297/298/387` 与 `small_t_i8.cuh:331/332/440/441/540` 的 computed-lane
  `__shfl_sync`。暂缓两个理由：推荐配置是 `--kv-dtype fp8`（走 `*_fp8.cuh` 而非 i8 家族）；且本树关闭 RDC 后
  small_t_i8 栈帧反而增多，说明该家族在本树的 lowering 与上游树不同，需先用 SASS 归因是否真有 out-of-line
  call 再定改法。
  - **回退验证（2026-09-26）**：回退后全量重建（533 s，exit 0），`ninfer-serve.exe` 重新测得 4405 个函数 /
    **231** 个 `STACK>0` / 4486 条资源记录，与改前逐值一致 ⇒ 该指标在本树可复现，+38 确实由关闭 RDC 引起。
    另：`ninfer_context_kv_materialize_test` 在 RDC 构建下同样 300 s 超时 ⇒ 该超时是**既有问题**，与本次改动
    无关；135 项里另外 4 项宿主侧失败（chat_templates / resource_manager / engine_options / serve_options）
    同理，均为既有。

**3. 减内核/栈帧**
- `ea74dca4`：`cmake/NinferTargets.cmake` 的 `ninfer_cuda_archive` 取消 `CUDA_SEPARABLE_COMPILATION ON`
  与 `CUDA_RESOLVE_DEVICE_SYMBOLS ON`（上游实测带栈帧内核 1155 → 927）。
- `77c9cc39`：attention 输出 gate 折进 causal reduce epilogue（每层少一次 launch，bit-identical）。
- `a941c9f4`：INT8 系 attention 的 Q/K/V scale 从 shared memory 读，去掉 computed-lane shuffle。
- 验证：Op 测试 + 全模型贪心字节一致 + decode 计时。

**4. 权重字节**
- `1da31697`（W8 词表/输出头加载时转码，27B 每张 −644 MiB）、`f1982279`（`--embedding-q6`）、
  `f8c75edd`（`--mtp-experts-q4`）。§1 已点名约 1/4 per-token 流量是 1 byte/element，重打包省
  ~2.6 GB/shard/token。
- 验证：artifact 契约测试 + 困惑度 A/B + decode tok/s。
- **适用性复核（2026-09-26，读 `D:\LLM\qwen3_8_27b_swift15_dflash2_final.ninfer.conversion.json` 与
  `D:\LLM\w4a4_family_recipe.py`）**：上游 `1da31697` 的对象是「W8G32 的 token embedding / output head」，
  而本 artifact 的配方把 `text/token_embedding` 与 `text/output_head` 都分配为 **`fp8_e4m3fn_row_bf16`**
  （1 B/元素，head 带 `AllowA8`），其余矩阵是 NVFP4（128 个）或 FP8（130 个）。⇒ 上游那条「W8→Q4 载入期
  转码」在本 artifact 上**不成立**；等价动作是把 FP8 的这两张（以及每 token 要流的 FP8 attention/GDN 投影、
  后八层 FFN）改到 NVFP4/Q4，属**配方/转换层**改动 + 质量代价。
- 量级：`output_head` = 248320×5120 × 1 B ≈ 1.27 GB 总量、按 shard 切半 ≈ 0.64 GB/shard/forward，占 10.15 GB
  的 ~6%；改成 NVFP4 约省 3% 的 round ⇒ decode 约 +3%（embedding 只是容量，gather 不流全表）。质量代价与
  接受率代价需要按仓库规则单独 A/B，故此项应先做测量再决定是否改配方。

**5. verify 段**
- `ce2df46b` small-T tensor-core 内核（MTP verify / cohort，1–32 列）；`21df3069` GDN record 窗口
  `cp.async` staging。verify forward ≈30 ms / 38.2 ms round。
- 验证：Op oracle + verify graph 计时 + 贪心字节一致。

**实施结果（2026-09-26）**

- 归因（§5.4）：round 33.9 ms 里 verify 29.1（86%）、proposal 3.7（11%）；verify 的每 shard 权重流下限
  22.7 ms ⇒ 固定余量 3.9 ms（T=1 实测 26.6 ms），再加 2 列只多 2.5 ms（T=3 实测 29.1 ms）。
- **5b `21df3069` GDN record 窗口 staging：已移植并验证**（`recurrent.cuh`，+92 行；用 `cp.async` 预取
  窗口的 key/query/本 CTA 的 value 行/gate 到 shared memory，再在原位做同样的运算）。
  - Op 级（`ninfer_gdn_replay_bench --profile 27b --component recurrent`，B=1）：T=3 冷 L2
    22.1 → 16.2 µs（−27%），warm 16.1 → 16.3 µs（噪声内）；T=8 冷 L2 36.5 → 28.3 µs（−22%），
    warm 28.5 → 27.5 µs（−3%）。
  - 端到端：MTP 路线**测不出**——同一 workload 的 verify 相位 29.07 → 29.13 ms（0.2%）；decode tok/s
    从 70.4 → 76.4 是接受率漂移（committed/round 2.39 → 2.58，与 tok/s 同比例），不是速度。DFlash2
    路线按 T=8 的 ~8 µs/layer × 48 折算也只有约 0.4 ms，低于相位计时分辨率。故本项只主张 **Op 级** 结论。
  - 正确性：`gated_delta_net_test`（op oracle）、`gated_delta_net_replay_record_test`、
    `gdn_replay_fold_test`、`gdn_replay_records_test`、`gdn_input_proj_conv_record_test` 全绿；
    真实双卡件 `tp2_dflash_append_test`（含 proposal/accept 摘要）PASS ⇒ 逐位一致。
- **5a `ce2df46b` small-T tensor-core 内核：复核范围后判定不适用于 TP-2 主线**。该提交只落在 **q4/q5
  路线**（`linear/q4/q4_small_t_mma`、`linear/q5/q5_small_t_mma`、`linear_add/q5`、`linear_swiglu/q4`、
  `linear_topk/q4`、`attn_input_proj/q4_q5`、`gdn_input_proj/q4_q5`，约 25 文件 / 1500+ 行含新测试），
  而双卡件 `swift15` 的主体是 NVFP4（128 张）+ FP8（130 张），q4/q5 只覆盖草稿权重 ⇒ 最多碰到 3.7 ms
  proposal 段的一部分；而 1–32 列的「小 T 低效」在本机表现为 verify 的 2.5 ms 列扩展成本（占 round 7%）。⇒ 不做。
- 结论与顺序建议：唯一有两位数空间的是**权重字节**（22.7 / 33.9 ms = 67% 的流），即项目 4 的广义形式
  （把每 token 要流的 FP8 张量改 NVFP4/Q4；文档估 −2.6 GB/shard ≈ −5.8 ms ≈ 17%），代价是质量/接受率，
  必须先做困惑度 A/B 再决定是否改配方。

**6. 大件（需产品决策，先不改）**
- KV codec：`eb9f7a23` rk4v4-e8（280 B/token/head）、`ad26b362` rk2v4-e8（216 B）、`2ba10da2` rk8v4 G32、
  `9218b67b` 根码查表；当前 fp8 KV = 16.125 KiB/token/shard。
- 设备路线 profile：`0b2f8384` + `81861a07`（5090 profile 会开 FP16 P·V 与 fast prompt kernel；36 SM
  卡需重新校准）。
- fast prompt kernel：`4303e604`（`--fast-prefill-kernel`）+ `NINFER_PREFILL_ALIGN` 整波对齐。

### 5.3 明确不做

- MoE 五连（`5fd7c463`/`b2ba37d2`/`d63162c8`/`ba1284d4`/`f1806721`）：跑稠密 27B。
- T2 三元、INT8 激活路线（`--prefill-a8`/`--mlp-a8-decode`）、W4A8：decode 带宽 bound，属算力路线。
- sm_86/3090 调参全套；`82631f68` 的 crossing 字节流水（< 256 KiB 跳过，TP-2 每次 allreduce 只 ~20 KB）。
- 结构化输出 / vision / WebUI / disk KV / DirectStorage / D3D12 常驻（功能项，非本轮 perf）。
- 权重字节（本节 4 的广义形式：把每 token 要流的 FP8 张量改 NVFP4/Q4）：**2026-09-26 用户决定先放弃**，
  理由是需要配方/转换层改动 + 质量与接受率代价，收益（文档估 ~17%）不足以先进。
- NCCL（本机 `D:\nccl-windows` = SystemPanic/nccl-windows 的 NCCL 2.29.7 构建，含
  `build/bin/nccl.dll`、`nccl_static.lib`、`install/`）：**不集成**。无 P2P 时 NCCL 走 SHM transport
  （GPU 经 PCIe 写共享 host 内存、对端读回），与 NInfer 的 in-kernel mapped-pinned 机制同构；而
  `tests/test_tp_device_pair.cpp:425-429` 的 copy-engine ceiling 探针已确认分片 in-kernel 路径在链路原始
  下界的 2% 以内。NCCL 相对本路径的唯一结构优势是 P2P/NVLink，本机拿不到（打补丁 BAR1 才可开，外部测量
  只值 +2.67% / −5~7%），且每次 collective 要独立发射（128 次/token 光发射 ~0.2 ms），还要引入 53 MB DLL
  与硬依赖。仅保留作**外部参照计**（20 KB allreduce 落点是否也在 9–18 µs）。

### 5.4 TP-2 基线（2026-09-26，本机 2×5060 Ti，双卡件 swift15）

- 配置：`tools/win_port/serve.ps1 -Model D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer`（默认配方：
  `--devices 0,1 --kv-dtype fp8 --max-context 131072 --spec mtp --draft-tokens 2 --lm-head-draft --vision
  --reasoning-effort medium`，采样 0.7/20/0.80，`--host-kv-mib 32768`）。
- 启动：`engine ready | qwen3.8-27b-swift1.5 | total 45.6s | weights 9.66 GiB`；每 shard
  `weights+ctx 11050.6 / 10900.6 MiB`，KV 131,072（fp8）占 2322 / 2064 MiB，free 2446 / 2026 MiB。
- `tools/win_port/bench_serve.ps1`（一次性，非流式，数字取自服务端 timings）：

| workload | prompt | prefill | decode |
|---|---|---|---|
| prefill_2048 | 1,832 tok | 1,206 tok/s | — |
| prefill_8192 | 7,153 tok | 1,654 tok/s | — |
| prefill_32768 | 28,528 tok | 1,603 tok/s | — |
| decode_short | 23 tok | — | **75.2 tok/s**（119 tok） |
| decode_at_8192 | 7,154 tok | — | **71.2 tok/s**（128 tok） |

- 用途：作为项目 4/5 的 A/B 基线。注意：decode 走 MTP K=2 + `--lm-head-draft`，接受率随采样有轮次漂移，
  单次结果只能判 ≥5% 级别的差异；判定更小的改动需要重复多次（worklog 的接受率判决用 30-rep）。
- 产物格式（`*.conversion.json` + `w4a4_family_recipe.py`）：`text/token_embedding` 与 `text/output_head`
  均为 `fp8_e4m3fn_row_bf16`；128 个 NVFP4、130 个 FP8、567 个 BF16、88 个 Q4G64、54 个 Q5G64、
  1 个 Q6G64、7 个 Q8G32。

**verify 段归因（2026-09-26，`NINFER_TP2_TIMING=1` 的服务端阶段计时）**

| 段 | 每 round（ms） | 占比 |
|---|---:|---:|
| avg_round | 33.92 | 100% |
| **verify** | **29.07 / 29.90** | **~86%** |
| mtp（proposal） | 3.69 / 3.77 | ~11% |
| accept + copy + sync_wait + fold | 0.37 | ~1% |

（两段独立样本：`rounds=36 committed=86` 与 `rounds=24 committed=63`，即 2.39 token/round ⇒ 70.5 tok/s，
与 §5.4 的 bench_serve 数字一致。）

- 结论：项目 5 的方向正确——verify 是唯一值得动的段，proposal 只剩 11%。
- 但上限要算清楚：verify 的每 shard 权重流是 10.15 GB ⇒ 22.7 ms 带宽下限，实测 29 ms ⇒ **非带宽余量
  只有 ~6.3 ms（占 round 18%）**。small-T tensor-core 内核与 GDN staging 能抢的是这 6.3 ms 的一部分。
- 下一步要区分这 6.3 ms 里「小 T GEMM 效率 / 128 次 collective / 图内 kernel 间隙」各占多少，
  再决定 5a（`ce2df46b` 新内核族）与 5b（`21df3069` GDN staging）谁先做。

**AR 载荷字节的上界实测（2026-09-26，诊断开关 `NINFER_TP2_AR_PAYLOAD_DIVISOR`）**

新增诊断（`device_pair.{h,cu}`，默认 1 即逐位不变，`ninfer_tp_device_pair_test` 仍 PASS）：让每次
in-kernel collective 只交换 1/N 的 partial，其余区间保持目的缓冲原值（就地 allreduce 即本卡 partial）。
数值必然错，只为给「压缩 AR 载荷」定上界。同一 workload（prefill 8192、`bench_serve`）：

| divisor | prefill | 块时/1024 tok | decode verify | 结论 |
|---:|---:|---:|---:|---|
| 1 | 1,533.9 tok/s | 668 ms | 29.5 / 30.6 ms | 基线 |
| 2 | **2,025.7** (+32%) | 506 ms | 噪声内 | |
| 4 | **2,414.3** (+57%) | 424 ms | 28.1 / 28.9 ms | |

- 两次差值给出同一个解：**AR ≈ 324 ms、compute ≈ 344 ms**（÷1→÷2 省 162 ms = AR/2；÷2→÷4 再省 81 ms =
  AR/4）。即 AR 占 prefill 块时 **48%**，与旧成本模型一致。
- ⇒ **任何「AR 字节 → 0」的天花板 = 344 ms/块 ≈ 2.98k tok/s（+94%）**；÷2 拿 1/3，÷4 拿约 60%。
- **decode 无收益**：÷4 只把 verify 从 29.5 降到 28.1 ms（≈4%，且在轮次噪声内）——小载荷是延迟 bound，
  带宽只占每次 9–18 µs 里的 ~3 µs。
- 与硬件的对比：把卡 2 从 Gen4 x4 挪到直连 x8/x16 槽 = 链路 ×2~3.5 = **免费的 ÷2~÷3.5**，无质量代价、
  零代码 ⇒ 优先级高于压缩；两者不叠加（共同封顶在 344 ms 的 compute 地板）。
- 对「字典映射压缩」的判据：无损字典在 bf16 激活上拿不到 2×；要走到 ÷2/÷4 必须是有损量化（FP8 块 scale /
  INT4 码本），故它属于 PLAN §5.3 那条 S5「载荷量化」的范畴，需要重建 oracle 与质量/接受率 A/B。

---

## 6. FP8 PV：attention 计算路径效率（2026-09-27 启动 → 同日回退，CLOSED）

> **已回退（2026-09-27）**：实现通过全部质量门（G0–G3）但 op 级只到 1.179×（目标 ≥1.3×），按用户预授权
> 规则回退并恢复 fp8 判据 1.2e-2；补丁与其同处仓库根：`PLAN-fp8pv-v1b.patch`（`git apply`
> 即可还原整套实现）。**本轮最有价值的产出是瓶颈定位**：
> barrier stall 34.2% + math pipe throttle 25.2% ⇒ 余量在**相位结构**而非 PV 的 dtype（详见 §6 末尾）。
> 建议的后续「跨 tile 相位流水」（逐位精确、零质量代价）未立项。

**目标**：生产 prefill 的 PV 从 FP16 tensor 路径换成 FP8，并压缩非张量 FP32 遍数。作用域是深上下文
prefill（245k 处 attention 占每 token 成本 70%，§3.13 拟合）。预期 attention 项 **1.2–1.4×** ⇒
245k 端到端 **~1.1–1.3×**、50k ~1.05–1.1×。decode 小 T 内核与本杠杆无关。

### 6.1 依据（已定案，勿重复调研）

- L1 测量门（worklog §3.13/§3.14 + L1 实验记录）：KV 字节跨 3.5×、耗时只差 13% ⇒ **计算/ALU 受限**，
  不是 KV 带宽受限；「减 6× KV 冗余」因此被否证（DRAM 1.09%）。
- ncu（`causal_attention_prompt_k8v4_kernel`、W=1024、65,536 深度、h12-kv2）：**tensor 管线 63.5%
  （最忙资源）**、Compute 63.54%、DRAM 1.09%、occupancy 33.33%（1 block/SM；寄存器 100/线程 +
  动态 smem 85.12 KB 双重限制）、非张量 FP32 指令占 11%（ncu 提示 ~3.6% runtime）。
- tensor 管线预算（每 KV tile 每 Q block）：QK^T 2.10 MFLOP @FP8（85.9 TFLOP/s 天花板的 25%）+
  PV 2.10 MFLOP @FP16（~43 的 50%）⇒ **PV 占管线时间是 QK^T 的 2 倍**；PV 换 FP8 ⇒ 管线工作量 ×0.67。
- **生产 dtype 是 `--kv-dtype fp8`**（§1 推荐配置、`docs/tp2-dual-5060ti.md` 的 shipped recipe），生产
  内核是 `prompt_fp8.cuh`；k8v4 只是 245,760 扫描件的 dtype ⇒ **V 侧零新增舍入**：FP8 MMA 直接读今天
  的 E4M3 码，与今天「精确展宽到 FP16」等值。这是本方案相对 k8v4 视角的关键简化。
- 已试死（勿重做）：warps 16→32（−12%，寄存器 spill；1024 线程被 64K 寄存器文件顶在 ≤64 寄存器/线程）；
  Bc 64→128（消费级 Blackwell 单块动态 smem 上限 ~100 KB，`cudaFuncSetAttribute` 返回
  `cudaErrorInvalidValue`）；W=2048 chunk（只值 ~9%，W=1024 已在平台区）；GQA 减 6× KV 流量（DRAM 1%）。

### 6.2 设计（变体 V1a，先做）

1. **P 每 (row, tile) 一个 E4M3 scale**：`psc = P_tile_max/448 = exp2((bm − nm)·scale_l2)/448`。`bm`（tile
   max）与 `nm`（新 running max）生产者已在算 ⇒ **scale 免费**；tile max 映到 448，范围利用满。
   `p_s` 由 FP16 改 E4M3（8 KB → 4 KB）。
2. **累加器吸收 scale 比**：`alpha' = alpha · psc_prev/psc_cur`（每行一次，生产者算，随 `alpha_s` 广播；
   新增 `psc_s[Br]` 行数组）；末轮 `out = acc/(l · psc_last)`（每行一次乘，free）。
3. **V 直通**：8 个 V-worker warp 不再做 FP16 展宽，改为把 E4M3 码**转置/swizzle 写入 B 暂存**
   ——FP8 B 操作数（k32×n8）要求 `[n=d][k=key]` 行布局（`ldmatrix_x2` 非转置的行维＝n），而 cache 是
   `[key][d]`，故 worker 侧需一次 8×8 字节转置（寄存器内 shuffle + 带 swizzle 的写）。
4. **v_scale 归属（唯一开放选择）**：
   - **V1a（先做）**：worker 侧把 `code × v_scale` 重新量化为 E4M3（一次额外 2^-4 相对舍入）；P 与分母
     `l` 的逻辑完全保持现状（`l` 仍由未加权的 FP32 P 累加）⇒ 改动最小、无耦合。
   - **V1b（备选）**：把 `log2(v_scale)` 折进 score（`score + log2v_s[col]`，乘变加），tile max 自动含
     v_scale；但分母 `l` 必须用未加权的 exp2 另算 ⇒ **MUFU 翻倍**，只在 V1a 质量不足时考虑。
5. **非张量遍数压缩**（与 1–4 同批）：V 展宽遍整体消失；P 的 `__float2half_rn` 改 E4M3 转换；生产
   kernel 的 smem 由 92,416 B 降到 **~72 KB**（VStage 32 KB 删、P 减半、加 V 的 B 暂存与 psc 行）。

### 6.3 质量风险（本杠杆的主要否决点）

E4M3 只有 ~2^18 动态范围（2^-9 … 448），而 `P = exp2(score − m)` 的动态范围由分布决定。per-tile scale
把每个 tile 的 max 映到 448 后，**tile 内的小 P 仍可能落进 subnormal 或 0**：245k 上下文若 attention 接近
平坦（P ~ 4e-6），整行 P 可能被压到零 ⇒ 归一化失真。与 L1「减流量」不同，**本杠杆不是数值透明的**：
它必然改动贪心轨迹（先例：MTP cache 试 nvfp4 时 5 条 golden 有 3 条文本变化）⇒ 属「质量换速度」，
必须带声明的容差与采样式裁决，不能按「逐位一致」验收。

### 6.4 门禁与执行顺序（先测量后动代码；G1 不过则停止并记录否决）

| 门 | 内容 | 判据 |
|---|---|---|
| **G0** | ncu 重抓生产 fp8 kernel（d256-h12-kv2、W=1024、深度 ≥100k）+ fp8 基线深度曲线（8k→245k） | tensor 管线仍是最忙资源；occupancy/DRAM 与 k8v4 profile 同形 |
| **G1** | P 量化误差数值研究：按真实/合成分数分布在 per-tile E4M3 scale 下的 PV 相对误差，覆盖 245k 平坦/尖峰两端 | 尾部丢失在可接受范围（暂定 PV 输出相对误差与 fp8 KV 同量级） |
| **G2** | OP 级独立 FP32/FP64 naive oracle @ 生产几何深包络；声明 P 量化边界与容差（V1a 另记第二次 V 量化） | 生产路线直接对照 oracle |
| **G3** | 模型级 golden 摘要（`tp2_dflash_solo`/`append`）+ sessions + 30-rep 接受率 | 接受率无可测代价，或代价已由用户接受 |
| **G4** | 深度曲线 A/B + `bench_serve.ps1` prefill 行 | attention 项 1.2–1.4× 兑现 |

**文件**：`src/ops/softmax_attention/dense/causal_cache/prompt_fp8.{cuh,cu}`、
`tests/ops/softmax_attention/causal_cache.cpp`、`bench/ops/causal_softmax_attention_bench.cu`。
k8v4 / nvfp4 / bf16 / i8 的 prompt 内核保持 FP16 PV 不动（k8v4 只是扫描 dtype）。

**G0 结果（2026-09-27，PASS）**

- fp8 基线深度曲线（`ninfer_causal_softmax_attention_bench --entry cached --geometry d256-h12-kv2
  --kv-dtype fp8 --tokens 1024 --execution eager --cache cold`，单层 median）：8,192 → 2,645.3 µs；
  32,768 → 9,949.4；65,536 → 19,690.7；131,072 → 39,673.7；245,760 → **73,521.8 µs**（math 42.15 TFLOP/s，
  qk/pv 各 21.07 TFLOP/s）。线性度极好，与 k8v4 基线（73,880 µs）同档、略快。
- ncu 重抓 `causal_attention_prompt_fp8_kernel`（d256-h12-kv2、W=1024、65,536 深度、grid 192、block 512）：
  **Tensor 管线 63.7% 为最忙资源**、Compute (SM) 63.67%、**DRAM 2.23%**、L1 39.79%、L2 14.67%、
  IPC 1.26、Issue Slots 28.09%；**寄存器 102/线程 + 动态 smem 92.42 KB（配置 102.40）⇒ 1 block/SM**；
  最大 stall = 等数学管线 4.4/12.7 周期（**34.9%**，ncu 的 Est. Local Speedup 34.9%，原文「all active warps
  execute their next instruction on a specific, oversubscribed math pipeline」）。
- ⇒ 与 k8v4 profile **同形**：瓶颈是 tensor 管线 + 1 block/SM，不是带宽。前提成立，杠杆有效。

**G1 结果（2026-09-27，PASS + 设计改判）**

模拟（per-tile E4M3 scale、fp8 KV 的 V 行 scale、D=32、Bc=64、4 个 n×σ×spike 组合；误差＝输出 rel-L2）：

| 场景（P 分布） | V 重量化单独误差 | E4M3 per-tile | E5M2 per-tile | E4M3 global 1/448 | P 质量归零 |
|---|---:|---:|---:|---:|---:|
| n=8192 σ=1 无尖峰 | 2.610% | 2.967% | 6.073% | 4.813% | 0% |
| n=65536 σ=1 有尖峰 | 1.778% | **2.415%** | 4.194% | 3.059% | 0% |
| n=245760 σ=1 无尖峰 | 2.760% | **2.954%** | 5.965% | 3.805% | 0% |
| n=245760 σ=4 有尖峰 | 2.552% | **2.552%** | 2.551% | 2.650% | 0%（global 0.358%） |

- **结论 1（per-tile scale 必需）**：global 1/448 在长上下文明显更差（+0.9pp）且开始丢质量（0.358% 归零）；
  per-tile E4M3 的归零质量 ≈ 0%，**我原先担心的「长上下文尾部塌方」不成立**（每 64-key tile 自带 scale，
  tile 内动态范围有限）。
- **结论 2（E4M3 > E5M2）**：E5M2 的 2 位尾数（12.5% 相对误差）盖过其指数范围优势，长上下文差 ~2× ⇒ 用 E4M3。
- **结论 3（P 量化本身的代价很小）**：E4M3 per-tile 相对「V 重量化单独」只加 ~0–0.6pp；**真正的代价在 V 侧
  重量化（1.8–3.4%）**。
- ⇒ **设计从 V1a 改判为 V1b（V 保持逐位精确）**：把 `log2(v_scale)` 折进加权 score。重新推导后 V1b 反而
  **更便宜**：每元素只多 1 次加（`u+w`）+ 1 次 fmax，分母 `l` 用 `Σ P'_q·inv_scale`（每元素 1 次乘）在
  **同一移位**下恢复，无需第二次 exp2（MUFU 不翻倍——实测 SFU 占用仅 ~3%，翻倍也无妨）。
- **代数简化（重要）**：令 `u=score·scale_l2`、`w=log2(v_scale)`、`P'=exp2(u+w−m)`、`psc=exp2(bm'−nm')/448`，
  则累加器与分母共用同一 rescale **`alpha' = exp2(bm'_prev − bm'_cur)`**（只依赖相邻 tile 的加权最大值，
  running max 项完全抵消）⇒ 末轮 `out = acc / l`，**不需要额外的 psc 收尾乘法**。

**最终设计（V1b，实施中）**

1. worker 8 warp：不再做 FP16 展宽；把暂存的 V E4M3 码**转置写入 `[d][key]` 布局的 B 暂存**（FP8 B 操作数
   要求行维＝n=d、行内 16 字节＝16 个 key；`ldmatrix_x2` 非转置，k 步长 32 ⇒ `PVKs = Bc/32 = 2`），
   同时算每 key 的 `log2(v_scale)` 与 `inv_scale`（各 64 个，摊在 256 线程上可忽略）。
2. producer：score 循环里多算 `u+w` 的 tile 最大值 `bm'`（1 add + 1 fmax/元素）；`P'=exp2(u+w−nm')` 后按
   `psc` 量化成 E4M3 存 `p_s`（Br×Bc×1 B）；`l` 累加 `P'·inv_scale`。
3. consumer：`acc *= alpha'`（生产者按行算好广播，沿用 `alpha_s`）后走 FP8 PV mma（P 与 V 码都是 E4M3）。
4. smem：92,416 → **~72 KB**（删 VStage 32 KB、P 减半 4 KB、加转置 V 暂存 16 KB 与 log2/inv/psc 小数组）。

**实现结果（2026-09-27，方案 V1b 已落地；未提交）**

- `src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh`（+253/−94，唯一实质改动）：
  - smem **92,416 → 72,448 B**：删 `v_f16`（32 KB 展宽暂存）、`p_s` 由 `__half` 改 **E4M3 uint8**（8→4 KB）、
    新增 `v_t`（转置后 `[d][key]` 的 V 码 16 KB，原始 `v_fp8` 暂存保留）、`lv_s/iv_s[Bc]`、`bmp_s[Br]`，
    删 `running_m_s`；仍 1 block/SM（寄存器 102→96）。
  - producer：`bm'=max(u+w)`（1 fma + 1 fmax/元素，沿用 `partial_m_s` 合并两半）；
    `P=exp2(fma(score,scale_l2,lv−bm'))·448`（tile 最大值恰映到 448）；`code=__nv_cvt_float_to_fp8(P,SATFINITE,E4M3)`；
    `l += P·(1/v_scale)`（与码同一次 exp2，无第二次）；`alpha'=exp2(bm'_prev−bm'_cur)`；P 的 swizzle 改
    16 字节粒度（行 64 字节）。
  - worker 8 warp：不再展宽，改为**逐字节 8×8 寄存器转置**（三段 64-bit 掩码交换）把 V 码搬进 `v_t[d][key]`，
    落位 `((key_group>>1)^(j&3))<<4 | ((key_group&1)<<3)`。
  - consumer 16 warp：`PVKs=Bc/32=2`；A=P 用 `ldmatrix_x4`、B=V 用**非转置** `ldmatrix_x2` ⇒ `mma_fp8_e4m3`。
  - `prompt_fp8.cu` 未改（smem 常量在头文件，launch 自动同步）。
- **实现期的必要性修正**（均已写进代码注释）：
  (a) 分母要带同一 448 因子（mma 吃 `448·P` ⇒ `l += (448·P)·iv`；448 在 acc/l 中相消，末轮仍 `out=acc/l`）；
  (b) 直接用 per-tile `bm'` 作移位（不引入 running max）：存码等价且每元素少一次乘；
  (c) `v_scale<=0` 取 `lv=0,iv=1`：全零 V 行的 softmax 权重仍须进分母，不能取 `-inf/0`；
  (d) **空 tile（整块被 mask、`bm'=-inf`）必须 `alpha'=1` 且不更新 `bmp_s`** —— 实测真 bug：修前
      T=65/keys=128、dflash W=9/16 出 non-finite，修后 0 non-finite。

**G2（OP 级 oracle）：PASS，但判据必须放宽。** `ninfer_softmax_attention_test` 全量 PASS（0 fail /
0 non-finite）。P→E4M3 是唯一新增语义边界，实测最坏 **rel-L2 2.309e-2 / gross-abs 2.258e-2**（max_ref 1.0），
超原门限 1.2e-2 的 1.92× ⇒ fp8 判据改为 **3.2e-2 / 9.0e-3 / 2.4e-2**（约 1.4× 余量，低于 E4M3 单权重
2^-4=6.25e-2 的理论上界），`tests/ops/softmax_attention/causal_cache.cpp` 注释写明边界与实测数字。
**⇒ 这是相对旧 fp8 路线约 2× 的输出扰动，是 G3 必须裁决的代价。**

**G4（性能）：未达标。** `ninfer_causal_softmax_attention_bench --entry cached --geometry d256-h12-kv2
--kv-dtype fp8 --tokens 1024 --execution eager --cache cold`：

| 深度 | 改前 µs | 改后 µs | 加速 | math 前→后 (TFLOP/s) |
|---|---:|---:|---:|---:|
| 8,192 | 2,645.3 | 2,252.4 | 1.174× | 41.40 → 48.63 |
| 32,768 | 9,949.4 | 8,449.2 | 1.178× | 42.09 → 49.56 |
| 65,536 | 19,690.7 | 16,730.0 | 1.177× | 42.20 → 49.68 |
| 131,072 | 39,673.7 | 33,359.0 | 1.189× | 41.73 → 49.63 |
| 245,760 | 73,521.8 | 62,493.5 | 1.176× | 42.15 → 49.59 |

ncu（同 G0 配置）：Tensor 管线 **63.7% → 50.1%**、寄存器 102→96、动态 smem 92.42→72.45 KB、
DRAM 2.23→2.68%、L1 39.79→45.65%，仍 1 block/SM。⇒ PV 张量时间确实减了 1.5×（绝对 0.637→0.425），
但墙钟只快 1.18×：内核已从「张量独占」变成「张量与其它各半」，**剩下的 2–3% 缺口在非张量侧**——即
§6.2 第 5 条「压缩非张量 FP32 遍数」那一半，尚未做。


**G3（模型级裁决）：PASS，但覆盖有缺口。** 30-rep 接受率 A/B（`tools/tp_bootstrap/r62_sampling_ab.ps1`，
工件 `qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer`，dflash2 k=7，温度 0.7/top_k 20/top_p 0.8，3 类短 prompt ×
30 rep = 90 请求/臂，两臂 prompt 集合相同）：

| 臂 | 样本 | 接受率 | committed/round | drafts/round | 臂墙钟 |
|---|---:|---:|---:|---:|---:|
| change（FP8 PV） | 90 | **35.79%** | 3.4265 | 6.779 | 355.1 s |
| base（FP16 PV） | 90 | **35.50%** | 3.4083 | 6.783 | 366.8 s |

Δ=**+0.29pp**（远在 30-rep ±1.4pp / 15-rep ±2pp 噪声内，且方向有利于改动）⇒ **模型级无可测退化**。
golden：`tp2_dflash_solo`（q4all 件）**PASS**（digest `0x19047f8ccaf5707f`，43.3 s）；`tp2_sessions` dflash2 路
官方 r69 件 **PASS**（164.3 s）；`tp2_dflash_append` PASS 但**其 KV 是 BF16 ⇒ 不走 prompt_fp8，仅作回归**；
`sessions` 在 swift15 件上 FAIL 但**基线同样 FAIL**（stash 后重跑复现、失败点更早）⇒ 既有工件不匹配，非新内核
break。

**缺口（必须记住）**：接受率臂是 ~50 token 短 prompt、decode 主导；本杠杆的作用域是**深上下文 prefill**，
「深上下文下的 P 量化无质量代价」本门**未测**（G2 的 op 级上界 rel-L2 ≤2.309e-2 覆盖深包络，但只到 op 层）。

**G3 深上下文补测（2026-09-27）：作用域内同样无退化。** 长 prompt 变体 `build-win/g3_long_ab.ps1`
（gitignored，沿用 r62 的 serve 配方与汇总口径，仅把 3 条短 prompt 换成 1 条长 filler，**每次 rep 带新 nonce**
防 prefix cache ⇒ 每次全量 prefill），工件 `qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer`，dflash2 k=7，
实际 prompt_n ≈ **28.5k**（单请求 prefill ≈18.5–18.9 s ⇒ ≈1,550 tok/s）：

| 臂 | prompt_n | 样本 | 接受率 | committed/round | drafts/round |
|---|---:|---:|---:|---:|---:|
| change 10 | 28,537 | 10 | 39.96% | 3.7317 | 6.84 |
| base 10 | 28,538 | 10 | 38.57% | 3.6434 | 6.85 |
| change 30 | 28,533 | 30 | **38.09%** | 3.6025 | 6.83 |
| base 30 | 28,534 | 30 | **38.97%** | 3.6717 | 6.86 |

Δ(10)=+1.39pp、Δ(30)=**−0.88pp**（10→30 符号翻转 ⇒ 噪声；|Δ|<1.4pp 阈）；无 non-finite、四臂 exit 0
⇒ **在杠杆真正的作用域（深上下文 prefill）同样无可测质量代价**。

**G4 端到端（同服务会话，每格 1 样本，未达单样本可判阈）**：

| workload | change | base | Δ |
|---|---:|---:|---:|
| prefill_2048 (1,835 tok) | 1,288.0 tok/s | 1,299.9 | −0.9% |
| prefill_8192 (7,160 tok) | 1,560.4 | 1,594.0 | −2.1% |
| prefill_32768 (28,534 tok) | **1,550.5** | **1,529.4** | **+1.4%** |

PLAN §5.4 自述单次 `bench_serve` 只能判 ≥5% ⇒ 前两格在噪声内；深上下文 +1.4% 与 op 级 1.176× 同向，
但**未达可判阈，G4 正式判定仍需多 rep**。（decode 行随采样文本漂移，与改动无关。）

**总账（2026-09-27 收口）**：质量侧全绿——短 prompt 臂 Δ=+0.29pp、深上下文臂 Δ=−0.88pp，均在噪声内，
三个 golden 在有效工件上 PASS；性能侧 op 级 **1.176×**、端到端 28.5k **+1.4%**（单样本）。
**代价**：fp8 路由 op 判据 1.2e-2 → 3.2e-2（约 2× 输出扰动），这是**既有质量门的实质放宽**，
而换来的端到端收益目前只有 ~1.4% 量级 ⇒ 是否保留取决于后续非张量侧压缩能否把 op 级推到 ~1.3× 以上。

**用户裁决（2026-09-27）：选 A —— 保留改动，做一轮「有界的非张量侧压缩」**，目标把 op 级推到 **≥1.3×**
（端到端才够 ~+3%，交易才站得住）；**达不到则回退并恢复 1.2e-2 门限**。

本轮硬约束（否则刚通过的 G3 证据作废）：**只允许逐位精确的优化** —— 不重结合浮点、不改 fma/sub/exp2/cvt/mask
语义；逐位判据为 ① `ninfer_qwen3_5_tp2_dflash_solo_test`（q4all 件）digest 仍为 `0x19047f8ccaf5707f`、
② op 单测最坏误差数字（rel-L2 2.309e-2 等）逐字不变。占用率已被硬件锁死（96 寄存器 × 512 线程、72.45 KB
smem ⇒ 1 block/SM；warps 16→32 与 Bc 64→128 均已实测否决）⇒ 不碰结构级重构。

本轮起点：op 级 1.176×（245,760 深度 = 62,491.4 µs），ncu Tensor 50.1% / L1/TEX 45.65% / DRAM 2.68%。
待做：先 ncu 拿管线细分与 stall 明细定瓶颈；候选为 P 的 16 次 1 字节散写合成为 8 次 16 位写（同列相邻字节，
逐位精确）、地址预算、以及仅在 ncu 指向时才动的 worker 转置（V 从 global 直载省掉 raw `v_fp8` 16 KB 与一次
smem 往返，风险高）。

**有界压缩轮结果（2026-09-27）：未达 1.3×，已按用户预授权规则回退。**

ncu 细分（`causal_attention_prompt_fp8_kernel`、ctx=65536、reg 96、dyn smem 72.45 KB）：管线 elapsed 口径
Tensor(FP) **50.14%**、ALU 12.52%、FMA 7.59%；指令率口径 **LSU 37.39%**、Tensor 24.17%、ALU 12.27%。
Stall（cyc/issued inst，总 12.38）：**Barrier 4.23（34.2%）**、**Math Pipe Throttle 3.12（25.2%）**、
Wait 1.47（11.9%）、MIO 0.76、Short Scoreboard 0.69、Long Scoreboard 0.19。
⇒ **瓶颈是 tensor 管线的相位结构，不是非张量指令吞吐。**

- 本轮只做了唯一一组逐位精确优化（P 码 16 次 1 字节散写 → 8 次 16 位写，swizzle 地址计算 16→8）：
  solo digest 仍为 `0x19047f8ccaf5707f`、op 单测质量数字逐字不变 ⇒ 逐位精确成立；
- 但它**墙钟 0 收益**：`sm__inst_executed` −2.58%、LSU 同步下降，245,760 深度 62,481.0 → 62,358.7 µs
  （1.002×；相对原始 FP16 PV 仍 **1.179×**，距 1.3× 差 9.3%）⇒ 指令吞吐被隐藏，不是 limiter；
- 候选 2（worker 转置改 global 直载）与候选 3（`lv_s/iv_s` 冗余读合并）经 ncu 否定：Long Scoreboard 0.19、
  MIO 0.76，且每 tile producer ≈250 inst/线程 vs worker ≈100（worker 先到 barrier 干等）⇒ 加速它零收益。

**回退执行（2026-09-27）**：`prompt_fp8.cuh` 与 `tests/.../causal_cache.cpp` 已 `git checkout` 回 HEAD，fp8
判据恢复 **1.2e-2**；组合补丁归档到仓库根 `PLAN-fp8pv-v1b.patch`（23,614 B：FP8 PV 全部实现 + P 打包，
改 `prompt_fp8.cuh` 与 `causal_cache.cpp` 两个文件；`git apply PLAN-fp8pv-v1b.patch` 已验证可干净套用）。
回退后 bench 复测：8,192 = 2,645.9 µs；65,536 = 19,731.1；245,760 = **73,646.7**（回到 FP16 PV 基线，与
改动前 73,521.8 同档）。工作树只剩 `PLAN.md` 的改动。

**结论与建议的下一步**：本轮否决点不是「FP8 PV 不值」，而是**用质量门放宽换 1.2% 端到端是选错了杠杆**——
真正的余量在 **34.2% 的 barrier stall**，即 tile 循环的相位串行（producer QK+softmax ∥ worker 转置 → barrier
→ consumer PV → barrier；1 block/SM 下 producer 相位只有 8 warp 发 mma）。**跨 tile 相位流水**（double-buffer
`p_s`，把 tile t 的 PV 与 tile t+1 的 QK 重叠）**逐位精确、零质量代价**，且作用在已回退的 FP16 PV 基线上
⇒ 严格优于本次的 FP8 PV 交易。**已立项（2026-09-27，用户决定，见 §7）。**

---

## 7. 跨 tile 相位流水：吃掉 barrier stall（2026-09-27 立项 → 同日否决，CLOSED）

**目标**：把 `causal_attention_prompt_fp8_kernel` 的 tile 循环从相位串行改成跨 tile 流水，让 tile t 的 PV 与
tile t+1 的 QK^T/softmax 重叠，直接吃掉 §6 定位到的 barrier stall。**验收口径与 §6 相反：本杠杆不引入任何新的
数值边界，因此以逐位一致验收（solo digest + op 单测数字逐字不变），不设质量门、不放宽任何判据。**

**基线（2026-09-27，HEAD `d21febf6`，FP8 PV 已回退）**：`ninfer_causal_softmax_attention_bench --entry cached
--geometry d256-h12-kv2 --kv-dtype fp8 --tokens 1024 --execution eager --cache cold`（warmup 5 / repeat 30，
单层 median）：8,192 → 2,643.6 µs；32,768 → 9,949.2；65,536 → **19,689.3**；131,072 → 39,553.2；245,760 →
**73,517.7 µs**（math 42.15 TFLOP/s，qk/pv 各 21.08），与 §6 回退后复测同档。

### 7.1 依据

- §6 最后一轮 ncu（FP8 PV 版，ctx=65,536）：**Barrier 4.23 cyc/inst（34.2%）居首**、Math Pipe Throttle 3.12
  （25.2%）、Wait 1.47、MIO 0.76、Short Scoreboard 0.69；管线 elapsed 口径 Tensor(FP) 50.14%。
- 相位结构（`prompt_fp8.cuh:232-415`）：每个 KV tile 是
  `producer QK^T+softmax（8 warp）∥ worker V 展宽（8 warp）→ __syncthreads → 全 16 warp PV → __syncthreads`；
  producer 相位只有 8 warp 发 mma，PV 相位又必须等 producer 全部到齐；1 block/SM（96 寄存器 × 512 线程 +
  动态 smem）⇒ 没有第二个 block 填相位空洞。同文件 :379-413 显示 tile t+1 的 KV `cp.async` 预取**已经**压在
  tile t 的 PV 上 ⇒ 剩余缺口是 **QK/softmax 与 PV 两个计算相位的串行**，不是预取。
- **关键不确定点**：§6 的 34.2% 是 FP8 PV 版的数（PV 张量时间减半后才把 barrier 顶到首位）。回退后的 FP16 PV
  基线是否同样以 barrier 为首要 stall **必须由 G0 重测**——同类 k8v4 profile（worklog §3.13）当时是 math pipe
  34.4% 居首。这正是 G0 门存在的理由。

### 7.2 门禁与执行顺序（先测后改；G0 不过即否决）

| 门 | 内容 | 判据 |
|---|---|---|
| **G0** | 在当前 FP16-PV 基线（fp8 KV、d256-h12-kv2、ctx=65,536）重抓 ncu `--set full`，要 stall 分类 + 管线 elapsed 细分 | barrier 类 stall 居首或次席且 ≥15%（相位串行确实暴露可观时间）⇒ 进 G1；否则**记录否决、不动代码** |
| **G1** | 逐位精确的跨 tile 相位流水（具体形态由 G0 的 stall 归因决定，受 §7.3 约束） | 不重结合浮点、不改 fma/exp2/cvt/mask 语义与累加顺序；ncu 复测 barrier 显著下降 |
| **G2** | 逐位判据 | ① `ninfer_qwen3_5_tp2_dflash_solo_test`（`D:\LLM\qwen3_8_27b_w4a4_dflash2_q4all.ninfer`）digest 与**改动前 HEAD 构建**逐位相同；② `ninfer_softmax_attention_test` 全 PASS 且最坏误差数字逐字不变（fp8 判据维持 1.2e-2） |
| **G3** | 性能 | 深度曲线 8k→245k 相对基线无回退，且 245,760 上有可判提升 |

**文件**：`src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh`（唯一实质改动；`prompt_fp8.cu` 仅当
launch 常量需同步）、`tests/ops/softmax_attention/causal_cache.cpp`（仅当判据文字需改）。k8v4/nvfp4/bf16/i8 的
prompt 内核本轮不动。

### 7.3 设计约束（已实测，勿重复）

- **smem 预算硬约束**：当前动态 smem **92,416 B**，opt-in 上限 **101,376 B**（G0 实测；Bc=128 的
  151,808 B 直接 `cudaErrorInvalidValue`，worklog §3.13）⇒ 只有 **8,960 B** 的增量预算。双缓冲 `v_f16`
  （32 KB）、`k_fp8`/`v_fp8`（各 16 KB）全都放不下，设计必须先把这些缓冲区变小或换掉，而不是直接翻倍。
- **寄存器/占用率锁死**：102 寄存器 × 512 线程 + 92.4 KB smem ⇒ 1 block/SM（`regsPerSM = 65,536` ⇒
  ≤128 寄存器/线程）；warps 16→32 已实测 −12%（spill）。
- **可动点**：`v_f16`（32 KB，V 展宽暂存）是最大单项，且 §6 已论证 worker 相位每线程 ~100 inst 远早于
  producer 到 barrier ⇒ 它既占满预算又不是关键路径，是本轮最可能的腾挪来源。

**G0 结果（2026-09-27，PASS + 关键归因）**

- 复跑基线（HEAD `d21febf6`，ctx=65536）：Tensor 63.6%（最忙）、Compute 63.60%、DRAM 2.28%、L1 39.80%、
  寄存器 102/线程、动态 smem 92.42 KB、1 block/SM、Warp Cycles/Issued Inst **12.71**。
- stall 分类（cyc/issued-inst，口径同 §6）：**Math Pipe Throttle 4.44（34.9%）**、**Barrier 3.34（26.3%）**、
  Wait 1.68（13.2%）、Selected 1.00、Short Scoreboard 0.62、Not Selected 0.60、MIO 0.52、Long Scoreboard 0.12、
  No Instruction 0.16、Branch Resolving 0.14。⇒ **barrier 在回退后的 FP16 PV 基线上仍是第二 stall（26.3%）**，
  远超 15% 的 G0 门限 ⇒ 相位流水有效，进 G1。
- **PC 级归因**（`ncu --page source --print-source sass`，1,724,394 个 warp-stall 样本，其中 barrier 452,953）：
  | barrier 停在哪 | SASS PC（函数内偏移） | 样本 | 占比 |
  |---|---|---:|---:|
  | tile 末 `__syncthreads()`（0x5b00）之后 | UMOV（0x5b40，循环回边） | 262,286 | 58% |
  | producer/worker → PV 的 `__syncthreads()`（0x4bd0）之后 | BRA.U（0x4bf0） | 177,073 | 39% |
  | producer 内部第一个 `bar.sync 1,256`（0x3760）之后 | LDS（0x3770） | 9,422 | 2.1% |
  | producer 内部第二个 `bar.sync 1,256`（0x4740）之后 | BRA（0x4750） | 3,933 | 0.9% |
  ⇒ **97% 的 barrier stall 在两个全块 `__syncthreads()`（相位串行），producer 内部命名屏障只占 3%**。
  结论：要吃的就是这个相位边界，而不是 softmax 的两个 partial 归约。
- 说明：ncu 把 `.DEFER_BLOCKING` 的 barrier 等待记在屏障之后的第一条指令上，故上表 PC 均是 `BAR.SYNC` 的后继。
- 设备参数（`cudaDeviceProp`，sm_120）：`sharedMemPerBlockOptin = 101,376 B`、`sharedMemPerSM = 102,400`、
  `regsPerSM = 65,536`（512 线程 ⇒ ≤128 寄存器/线程）。当前动态 smem 92,416 ⇒ **余量仅 8,960 B**。

**G1 设计（2026-09-27，按 G0 归因定案）**：把 tile 循环拆成两个**常驻 warp 角色**，一个 `__syncthreads`/tile：
- **A（warp 0–7，producer）**：QK^T + softmax(t) → `p_s[t&1]`、`alpha_s[t&1]`、`running_m/l`；顺带预取
  tile t+1 的 K 码与 K scale。
- **B（warp 8–15，consumer）**：先做 **PV(t−1)**（读 `p_s[(t−1)&1]`、`alpha_s[(t−1)&1]`、`v_f16`），
  再做 **V 展宽(t)**（`v_fp8(t)`→`v_f16`），顺带预取 tile t+1 的 V 码与 V scale。
- 收尾：循环后再由 B 做 PV(K−1) 并写输出。
- **smem 增量为零以外的开销**：`p_s` 8,192→16,384（双缓冲）、`alpha_s` 256→512 ⇒ 92,416→**100,864**
  （余 512 B）；`k_fp8`/`v_fp8`/`v_f16`/`k_scale_s`/`v_scale_s` **保持单缓冲**——它们的预取分别由
  消费它的那个组在读完当前 tile 之后发出，靠 cp.async 的异步性跨一个相位隐藏延迟。
- **逐位精确性论证**：每个输出元素仍由唯一 warp 用同一组操作数（同一 P 片、同一 V 片）按同一顺序
  （`acc*=alpha` 后 k=0→3 四个 mma）累加，tile 顺序与 `running_m/l` 递推完全不变；唯一变化是
  **同一份工作换了个 warp 执行**（PV 由 8 warp 各覆盖 16 个 n-tile，而非 16 warp 各覆盖 8 个），
  以及 softmax 的 partial 归约仍由原 8 个 producer warp 用原 `bar.sync 1,256` 完成
  ⇒ 不重结合、不改运算语义。
- **已知风险**：① B 的 `acc[16][4]`=64 个累加寄存器（原 32）可能顶到 `__maxnreg__(120)`；② 5 个命名
  屏障（A 3 个 + B 2 个）与 1 个 `__syncthreads` 的相位关系必须精确，否则死锁或污染 `v_f16`。

**G1/G3 结果（2026-09-27）：否决。三配置全部比基线慢，lever CLOSED。**

（G1 自身的判据「barrier 显著下降」达成：3.34 → 2.23 cyc/inst；**否决发生在 G3**——三种实现的
65,536/245,760 全部回退到 0.71–0.80×，且指令数上升，属结构性代价而非调参问题，故停止并记录否决。）

实现（已回退）：`prompt_fp8.cuh` 拆成两个常驻角色 + 每 tile 一个 `__syncthreads`（`p_s`/`alpha_s` 双缓冲，
smem 92,416 → **100,864 B**，余 512 B）。**正确性无问题**：`ninfer_softmax_attention_test` 全 PASS（fp8 判据
1.2e-2 不变、0 fail / 0 non-finite）。否决纯粹是性能/结构原因。

| 配置 | 结构 | 65,536 µs | 相对基线 | 245,760 µs | 相对基线 |
|---|---|---:|---:|---:|---:|
| 基线（HEAD） | 8 producer ∥ 8 Vdeq → 16-warp PV | **19,689** | 1.00× | **73,518** | 1.00× |
| V1 | 8 producer + 8 consumer（PV 32→64 acc/线程，512 线程） | 24,527~24,640 | **0.80×** | 92,160~92,219 | **0.80×** |
| V2 | 24 warp（8 producer + 16 consumer，768 线程，`__maxnreg__(80)`） | 27,836 | **0.71×** | 104,193 | **0.71×** |

**V1 的 ncu 归因**（同 G0 配置，`--set full` + 逐 PC 采样）：
- **barrier 3.34 → 2.23 cyc/inst（目标确实吃到了）**；但 math pipe throttle 4.44 → **5.24**、
  long scoreboard 0.12 → **0.77**、short scoreboard 0.62 → 0.88、wait 1.68 → 1.85 ⇒ 总 latency 12.71 → 13.31。
- **执行指令数 +17.4%**（2,222.6 M → 2,608.9 M）：其中 `LDL` **0 → 40.4 M**（运行时 local 溢出）、
  IADD +54.8 M、LOP3 +37.3 M、IMAD +31.5 M、IMAD.SHL +29.7 M、BRA +34.8 M、S2R +23.8 M、
  `BSSY/NOP/BSYNC.RECONVERGENT` 各 +16.2 M（地址重算与分支管理）。`LDSM.16.M88.4`（P 操作数）
  −6.34 M 是唯一的实质节省。

**根因（结构性，可复现）**：整块 PV 累加器 = Br×D = 64×256 fp32 = **16,384 个寄存器 = SM 寄存器文件的 25%**。
- V1 把 PV 压到 8 warp（256 线程）⇒ 每线程 acc 32 → **64**；512 线程的硬上限是 128 寄存器/线程
  （65,536/512）。实测 `__maxnreg__(128)` 时 ptxas 用满 128 **仍溢出**：prologue 把 ~20 个跨循环不变的
  地址 `STL` 到栈，`run_pv` 每次调用再 `LDL` 取回（静态 48 条 STL/LDL、`REG:128 LOCAL:0` 是 cuobjdump
  的误报）。
- V2 把 PV 还原到 16 warp（32 acc/线程），代价是 768 线程 ⇒ 每线程上限 65,536/768 = 85，
  且寄存器按 8/线程粒度分配 ⇒ `__maxnreg__(84)` 直接 `cudaErrorLaunchOutOfResources`
  （24×32×88 = 67,584 > 65,536）；降到 80（61,440）后 producer 路径（`score[4][4]` + softmax ≈90 寄存器）
  被压出更重的溢出 ⇒ 0.71×。

⇒ **「腾出 8 个 warp 去发 QK」在这台机器上无解**：要么 PV 的 acc 翻倍撞 128/线程墙，要么加线程把上限
压到 ≤80 撞 producer 工作集墙。基线「8 producer ∥ 8 Vdeq → 16-warp PV」正是本机
「寄存器文件 + 100 KB smem + 1 block/SM」包络下唯一可行的 packing。另外 barrier 降到 2.23 后总 stall
反而上升说明：**相位串行的一部分是张量管线在相位尾部排空的表现**（math throttle 上升是同一现象的另一面），
不是可回收的空转。

**回退执行与 G2 复核（2026-09-27）**：`src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh` 已
`git checkout` 回 HEAD（`git status` 只剩 PLAN.md）；重建后 bench 复测 65,536 = 19,728.6 µs、
245,760 = 74,132.0 µs（与基线同档，math 42.13 TFLOP/s），`ninfer_softmax_attention_test` 全 PASS
（exit 0）。**digest 判据**：`tp2_dflash_solo`（q4all 件）= **`0xad284a4b774b1cc3`**，两个独立进程一致、PASS；
回退后源码与 HEAD 逐字节相同，故该值即基线值（§6 记的 `0x19047f8ccaf5707f` 是 **FP8 PV 版本** 的值——
§7.2 最初引用它属误记，已改为「与改动前 HEAD 构建逐位相同」；工作树历史值 `0x4bcc3994a5efba7d` 早于
`6e01537a`/b7d352e8 两个前端改动，三者互不矛盾）。

工作树只剩 `PLAN.md`。**本节 CLOSED，不再重开**——除非执行包络本身改变（例如 Br=32 把 acc/线程降到 16，
或出现 >100 KB 的块级 smem 预算让 `v_f16` 也能双缓冲）。
