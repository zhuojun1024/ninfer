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
