# NInfer TP-2 计划（2× RTX 5060 Ti · Qwen3.8-27B NVFP4）

> **唯一的活动计划**，也是跨上下文压缩的持久记忆。整理：2026-09-24（归档整理前全文入 worklog，本文件只保留
> 关键信息与未完成事项）。
> - 完整历史：`docs/tp2-dual-5060ti-worklog.md` —— 含三份 PLAN.md 全文逐字归档（2026-09-21 上游
>   cherry-pick 重规划版；2026-09-22 交付收尾版；2026-09-24 本次整理前全文，含 §3.5–§3.14 与 §4–§8
>   全部轮次记录）。
> - 交付说明、推荐配置与实测数据：`docs/tp2-dual-5060ti.md`；Windows 原生移植：`docs/windows.md`。

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
  超时失败语义）。
- **关键数字**（262,144 配置，WSL）：prefill 1,588 tok/s；decode 59.0 tok/s（MTP K=2）；Windows 131,072 配置
  decode 56.9 tok/s（与 Linux 持平）。DFlash2 K=7 在 4096 上下文贪心档约 71 tok/s；245,760/k8v4（draftall 件）
  扫描 K=5 96.4 / K=7 96.1 tok/s（交错复核 93.3 / 96.9）；roofline 修正上限 ≈108 tok/s（原归档 90–180 的
  180 不可达）。

**推荐运行配置**（WSL 8088 / Windows 8099 同配方，Windows 侧 `--max-context 131072`）：

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh` 是进程级默认思考强度（请求体优先）；`--chat-template` 随第一梯队摘取
可用（模板由内嵌 Jinja 执行）。2026-09-23 实测：DFlash2（K=5/7）decode 领先 MTP ~20%、prefill 代价 1% 内
⇒ 追求吞吐用 `--spec dflash2 --draft-tokens 7`（扫描配方为 245,760/k8v4，依据见归档 §3.9–§3.11）。

### 环境与运维要点

| 项 | 值 |
|---|---|
| 构建（WSL） | `bash tools/tp_bootstrap/r55_build.sh`（rsync + `cmake --build build_dyn -j 8`，成功标记 `BUILD_EXIT=0`，约 2–3 分钟；WSL 构建树现为 `build_dyn`（Ninja），`build_r35.sh` 仍指向已删除的 `build/`，不可用） |
| 构建（Windows） | `tools/win_port/configure.bat` + `build.bat`（VS2022 + CUDA 13.3，`-DCMAKE_CUDA_ARCHITECTURES=120a`） |
| 服务 | WSL 8088（`serve_supervise.sh`）；Windows 8099（`tools/win_port/serve.ps1`，默认前台；自测服务必须用 harness 后台 job，`Start-Process` 起的进程会随工具调用结束被杀） |
| 运行 PATH（Windows） | FFmpeg（`D:\ffmpeg-dev\…\bin`）与 libcurl（`D:\curl-dev\…\bin`）必须在 PATH，否则 `STATUS_DLL_NOT_FOUND`（三件套 solo/sessions 无它们会在加载期 `0xC0000135`） |
| artifact | `D:\LLM\qwen3_8_27b_nvfp4.ninfer`（23.7 GB，旧 artifact，模板 `c3cf9e34…`）；新官方 artifact（模板 `a497db9e…`）随第一梯队可用；实验量化件 `…_dflash2_draftall.ninfer`（r57，−715.6 MiB）与 `…_dflash2_q4all.ninfer`（r66，−173.05 MiB，w8a8 + w4a4 家族各一件） |
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
- TP-2 会合协议重构（2026-09-23，事故驱动）：设备侧自旋事故第二次复现（两卡 SM 100%、6 条请求被静默吞掉；
  杀进程后两卡仍自旋约 2 分钟）⇒ Phase 1 有界自旋（env `NINFER_TP2_AR_TIMEOUT_MS`，默认 2000）+ 超时失败
  语义（HTTP 503 + 会话毒化重新 prefill，服务存活、下一请求正常）；Phase 2 host 权威 64 位 id（每捕获图一条
  id 通道，id 经图内 memcpy node 送达——不可按值当 kernel 实参，陈旧槽永不满足未来自旋），已提交
  `de74d966`/`b9c5d11c`/`634df875`；Phase 3 大 payload 事件路径**实测否决**（copy engine 地板 3.32 ms vs 现
  sliced 2.70 ms，未计 event 栅栏已慢 23%）；Phase 4 partial 合并审计**定案形态不存在**（9 个调用点逐点
  分类）。回归矩阵零回归（3 项既有失败在改动前 HEAD 逐字节复现）。
- K 扫描与 prefill 成本归因（2026-09-23）：DFlash2 K=5≈K=7（交错配对 K=7 略优；**不要用 K=6**——三轮恒
  25.7 rounds/s、被两侧同时支配的可复现每轮成本凹陷）；纯 MTP K=3/4 平局（79.4/79.7）且同 K 落后
  DFlash2 ~20% ⇒ **生产继续 `--spec dflash2`**；开投机几乎不付 prefill 代价（1% 内），DFlash2 prefill 稳定
  慢 3–4%（确切来源未归因）；prefill 1,905→569 tok/s（0→245k）＝深度无关固定项（0.525 ms/token，其中
  AR ≈65% 已贴 Gen4 x4 链路地板、软件无空间）+ 深度项（245k 处 attention 占每 token 成本 70%）；GQA 6×
  请求冗余被测量门**否证**（KV 字节跨 3.5× 耗时只差 13%；ncu：DRAM 1%、张量管线 63.5% 最忙、占用被寄存器
  文件 + ~100 KB smem 锁在 33%）⇒ L1（GQA 感知 prompt 内核）取消，内核代码未动。
- DFlash2 draft 量化 r54–r57（2026-09-23，opt-in）：draft gate/up→Q4（shard 0 −450 MiB）、融合 finish 投影
  mlp/down + attention/output→Q5（共享 finish kernel + 新 q5 shape，−213.3 MiB）、feature_projection→Q5
  （−50.8 MiB）；合并实验件 `…_dflash2_draftall.ninfer` −715.6 MiB，K=7 接受率 24.59%→25.3%（无下降）、
  吞吐持平 ⇒ **决策：保持 opt-in**，不改官方 recipe。
- DSH/agent-loop 前缀复用（2026-09-24）：TP-2 路线 `--chat-template` 被静默忽略已修（frontend
  `chat_template_path` 透传，developer 角色 400→200）；模板 `|trim` 假设**实测否定**（维护模板覆盖后
  原样回放仍停在上一轮 prompt 长度）；新增常驻两轮用例与文档。
- 首个生成 token 不发布（2026-09-24）：TP-2 prefill 首 token 曾直接 `push_back` 落账、未经输出会话（唯一
  漏掉的路线）⇒ 现 `preview_model → commit → publish`，`max_tokens=1` 正常发布（reasoning_tokens=1）；
  回放 40/1400 token 回答直达 `src=live`（此前恒停在 prompt 长度）；新测试用例 `check_first_token_published`。
- 复用门控与弃稿（2026-09-24）：「未对齐复用必须弃 masked draft」的前提**被实测推翻**（同轮：弃稿 22.5 vs
  保留 87.1 tok/s，接受率几乎不变）⇒ 删除门控、`draft_context_declined`、`reuse_grid` 与会话均值定价，
  单趟扫描最深边界获胜；契约：网格对齐与 oracle 逐 token 一致、chunk 内只保证 boundary crossing。
- DFlash2 draft 二次量化 r62–r66（2026-09-24，opt-in）：feature_projection（r62）、finish 投影
  mlp/down + attention/output（r63，q4 复用普通 GEMM + 共享 finish，无需新融合 kernel）、kernel_projection
  （r64，复用普通 q4 GEMM + 新 reduce）→ q4；r66 合并件 `…_dflash2_q4all.ninfer` −173.05 MiB（字节级
  核账精确相等），采样接受率 K=5 +3.72pp / K=7 +1.15pp（双双通过）；贪心 K=7 −1.60pp 已证明为跨臂文本
  漂移混淆（7 条探针 6 条文本不同）；r65（context_key）实测后缓做；w8a8 + w4a4 家族件均已产出、跨家族
  66 个 dflash2 对象逐字节相同。**决策：保持 opt-in**——升级被 plain sessions 既有失败（§3.3）与门禁修订
  决策（§3.7）阻挡。

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
- `ninfer_qwen3_5_tp2_sessions_test` plain 路线 "a conversation behind a shared system prompt" 首采样分叉：
  既有失败（r57 基线件与 r66 q4all 件逐 token 复现，`got [2752 13 198 …]` vs `expected [365 2798 349 …]`；
  dflash2/mtp 路线通过）⇒ 阻挡「draft q4 升契约特性」（三件套须全绿）。

### 3.4 暂缓 / 可选（未排期）

- Round 15 剩余：工具调用约束真实模型端到端复测（75 条探针）；WARN 诊断；路线 B（完整 GBNF）；
  TP-2 `validate_licensed_tokens` 守卫。
- Round 14 候选（用户暂缓）：① system+tools 前缀末尾加锚点（修新会话/压缩后第一轮 46.9 s，可省约 11 s）；
  ② prefill 吞吐（suffix 0.95–1.13k tok/s、冷启 1.35–1.38k）；③ 每请求 ~0.3 s 固定开销。
- 归档 §11.6：MTP priming 对图片列用占位 embedding（~100 行，无收益证据）；单图上限 8,192 硬编码
  （让出 shard 1 可重拿更大 envelope）；视觉编码期 shard 0 空闲（可接受）。
- 归档 §6：KV dtype 扫描补全（nvfp4/k8v4 只验证了可启动与吞吐，无质量数据）。
- 若要从源权重真正复刻官方工件的 DFlash2 draft，需先获取其 BF16 源检查点（本机只有 FP8/EXL3/GGUF）。
- r65 context_key → q4（−79.7 MiB）：draft 的 5 个 attention QKV 融合对象全部别名同一 [6144,5120] 对象，
  decode（`attn_input_proj` 三输出）与 prefill（`context_kv_materialize` 7 路线融合 kernel）两处消费方均
  硬要求 Q8 ⇒ 需一族 4-bit 融合材质化 kernel + 三输出 q4 变体 + 两套 oracle；接受率风险本批最高
  （K 投影误差在 2048 窗口环内逐 token 累积）；失败回退＝保持 q8。
- L2：AR 与 MMA 重叠（环形 staging + 事件同步 + 子块流水）：浅层上限 ~1.5×（1,905 → ~4,760 tok/s 上限）；
  大重构，跨卡 rendezvous 死锁风险（worklog Round 35c 已评估）。
- 若仍攻 attention：只剩计算路径效率（FP8 PV + 压缩非张量 FP32 遍数），且需先解锁 profiler 权限
  （ncu 2025.4.1 已装，`ERR_NVGPUCTRPERM` 被拒；管理员 PowerShell 或 NVIDIA Developer Settings 允许
  GPU performance counters）。预期 attention 项 ~1.2–1.4×（245k 端到端 ~1.1–1.3×）。
- 硬件杠杆：卡 1 从芯片组 Gen4 x4 槽（~7 GB/s）移到 CPU 直连 Gen5 x8 槽（~20 GB/s）：零代码，
  浅/中上下文 1.78×、245k 1.15×；需主板有空槽。
- DFlash2 K=6 每轮成本凹陷（三轮恒 25.7 rounds/s，低于更窄窗口的 K=5（27.9）也更宽窗口的 K=7（26.6））：
  疑似按窗口宽度分桶/对齐的核函数边界效应；`NINFER_TP2_TIMING=1` 取逐相位每轮耗时排查。
- DFlash2 prefill 稳定慢 3–4% 的确切来源（未做逐相位归因；`NINFER_TP2_TIMING=1` 拆 prefill 每相位）。

### 3.5 环境受阻的未做项（已用替代路径覆盖）

- [ ] WSL 侧 op 测试 + 字节一致 + 前端夹具测试：**WSL2 CUDA 驱动崩溃**（`cudaGetDeviceCount()` 内 PTX JIT
  segfault，GPU 被 Windows 服务占用）；jinja 测试通过证明二进制无误，op 测试改在 Windows 侧跑。
  Windows 侧已过（NVFP4 A4/A16、frontend、jinja；`NINFER_OP_REPORT_STATS=1` 错误指标在容差内）。

### 3.6 DFlash2 上 TP-2：剩余验收与设计约束

- [ ] B7 性能验收：双卡吞吐对目标（roofline 修正 ≈108 tok/s）+ 与 MTP 对比 + plain/mtp 无回归，并记录
  `--spec dflash2` 的推荐 K。
- [ ] 待实测未知量：Windows 原生构建的真实 free（台账 `free` 取自 `cudaMemGetInfo`，WSL2 少报约 1 GiB）；
  草稿按 head/row 切分的数值等价性；K=15 的 workspace 峰值；目标 5 层残差跨 shard 汇聚的每步开销；
  草稿每层 allreduce 的延迟；稀疏拒绝采样在 TP-2 分片 logits 下的等价性。

**设计约束（勿重复调研）**：整份复制 draft 不可行——满上下文（262144 / fp8 / MTP K=2）下 free 仅
697/1217 MiB，而草稿 2.07 GiB/卡；切分后每卡约 1.04 GiB。提议条件是目标 5 个 block 的 residual 拼接投影，
TP-2 把 64 层切两卡 ⇒ 需跨卡 handoff（两卡 residual 逐位相同，只捕 shard 0 即可）。候选 selector 直读
全词表 codebook（pred/succ 各 248320×256 BF16，约 254 MiB），必须整份。DFlash2 与 MTP 互斥：选 DFlash2
可释放 shard 0 的 MTP 权重 430 MiB + KV 516 MiB。每轮成本构成、D1–D4 决策与各轮否定结论见归档。

### 3.7 draft 量化：升级门禁（保持 opt-in 直到满足）

- [ ] 修 plain 路线 `sessions` 既有失败（§3.3）——三件套全绿才可升级。r66 q4all 件已测：`append`
  K=7/K=5 PASS（金值重基线）、`solo` 双进程 digest 同 `0x4bcc3994a5efba7d`、`sessions` 仅 plain 失败；
  dflash2 路线定向跑 PASS（164.8 s，recall 71 token 逐字节）。
- [ ] 门禁修订决策：接受率裁决改以采样主导——字面门禁「贪心 |Δacc| ≤ 1.0pp」K=7 未过（−1.60pp），
  已证明为跨臂文本漂移混淆（7 条探针 6 条文本不同；唯一文本稳定的 idx 5 上 −3.75pp，属小样本保留）；
  pooled 采样 K=5/K=7 双涨（+3.72pp / +1.15pp）。门禁修订与升级分开决策。
- （Q4 op oracle：新 shape 5120×25600 / 5120×4096 / 5120×17408 / 1280×5120 已由
  `ninfer_linear_q4_a16_test` / `ninfer_linear_dynamic_grouped_conv_add_test` 按存储 scale 独立解码覆盖，
  含图重放。）

剩余杠杆（未排期）：
- selector codebook 量化（242.5 MiB BF16，非 GEMM）：需要新的 selector 路径（codebook codec + 保持
  top-k 域）；且参数是裸 `Tensor` 非 `Weight`，还要动 Tensor→Weight 管线 + TP-2 peer 路径。
- 全 draft NVFP4：**已做源码级可行性审计，判定不推进**（详见 `docs/tp2-dflash2-draft-nvfp4.md`）。
  要点：相对现役 r66 `q4all` 件只再省 202.3 MiB，其中 174.3 来自 codebook（非 GEMM）；NVFP4 每元素
  0.5625 B > Q4 的 0.53125 B，故已 Q4 化的 gate_up/output/down/kernel/feature 改 NVFP4 是增容；三个
  硬阻塞＝转换器无 float→NVFP4 量化器且 draft 源（动态激活 FP8）无 A4 激活除数标定、NVFP4 native Weight
  只收完整 parent（与融合 QKV 的 row-view 共享冲突）、各消费 op 的 NVFP4 几何/kernel 均为封闭集合
  （`linear_pair` 只服务 dflash v1/MTP，不在 DFlash2 路径）。仅当动机转为「统一 W4A4 家族 / prefill
  TMA」时重新立项。

### 3.9 DFlash2 draft q4 二次增量：②融合 QKV→q4 / ①codebook 量化（2026-09-24 开工）

> 依据：`docs/tp2-dflash2-draft-nvfp4.md` 审计（全 draft NVFP4 已否决，只留这两条真显存杠杆）。
> 决策：两项均保持 opt-in override + 专用 artifact，官方 recipe 与引擎默认路径不动。

**② 融合 QKV → q4（−79.7 MiB）——关键设计修正：不需要新融合 kernel**

审计发现 q4 GEMM 注册表**已经**含 `{4096,5120}`/`{1024,5120}`（`src/ops/linear/q4/q4_dispatch.cpp:14-15`），且 RowSplit q4 的
row-region 由 `native_weight` 全链路支持（`src/core/weight_view.cpp:271-284`）⇒ 融合 [6144,5120] QKV 对象降到 q4 后，
两个 Q8-only 消费方都能用既有算子绕开：

- decode（`draft.cpp:326`）：Q8 走融合 `attn_input_proj` 三输出；q4 改走 **3 次 `ops::linear`**
  （query/key/value 行视图，形状已在注册表）。
- append/prefill（`draft.cpp:157`）：Q8 走融合 `context_kv_materialize`；q4 改走**既有逐层通用分支**
  （2 次 `ops::linear` + `rmsnorm`/`rope`/`kv_cache_append_prefix`），并修正该分支对 dflash2 已裁剪 context 的复用
  （原实现只对 dflash v1 正确；v1 已在构造期拒绝）。代价：每轮多 ~20 次 launch，预估 ≪1%，由 A/B 门禁验证。

改动清单：`parameters.h`/`parameters.cpp`（DraftBlockParameters 增加 query/key/value 行视图；`query_key_value` 改
`std::optional`，仅 Q8 parent 构建）、`draft.cpp`（按 qtype 分支 + 通用分支修正）、override
`tools/tp_bootstrap/r67_draft_qkv_q4.py`（**恰好 5 个对象**计数校验，官方 recipe 的 share 使
query/key/value/context_key/context_value 同属一个对象）。
门禁：stock Q8 路线逐位不变；现有 q4 op oracle 已覆盖所用 shape（无需新 kernel oracle）；显存核账 −79.7 MiB；
贪心 K=5/7 + 采样接受率 A/B；DFlash2 三件套。

**② 进度（2026-09-24）**：代码落地，`BUILD_EXIT=0`；stock Q8 件（`…_w8a8_dflash2_q4all.ninfer`）
append 回归 **PASS**（ring/proposal digest 与基线逐位相同，Q8 路线零扰动）。r67 实验件已转换：
`D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_qkv4.ninfer`（1218 对象，379.5 s；payload **−79.69 MiB**，
q4 81→86 / q8 12→7，恰 5 个 `[6144,5120]` q4 对象；dry-run 选择 q4 46 / q8 10 / bf16 35）。
qkv4 件的 append 功能验证 **PASS**（`TEST_EXIT=0`：direct/split append 逐位一致、repeat 逐位一致、
K=7 proposal 确定，ring K 与 Q8 基线逐值相同、V 差在 q4 量化量级 ≤1.5%）；proposal cost
7.036 vs 6.854 ms/proposal（+2.7% 的 proposal 步，端到端约 +0.25%，待 A/B 实测）。
顺带修正 append 测试的 draft-weight 台账：融合 QKV parent 与其 context 行视图是同一分配，
改为按 parent 计一次（qkv4 臂 layers **843.9 MiB**；q4all 臂同口径 **923.6 MiB**，差 79.7 MiB
与 artifact 字节差一致——旧口径把 parent 记了 3 次，gross 虚高 318.7 MiB）。
待：贪心 K=5/7 / 采样接受率 A/B、solo digest 与 sessions 三件套，以及端到端吞吐 A/B。

**②-e 验收 A/B（2026-09-25，同一二进制；基线 = r66 `q4all` 件）**：

| 指标 | q4all（Q8 QKV） | qkv4 | Δ |
|---|---|---|---|
| 贪心 K=7 | 26.95%（720/2672） | 27.07%（722/2667） | +0.12 pp |
| 贪心 K=5 | 35.96%（712/1980） | 34.85%（703/2017） | −1.11 pp |
| 采样 K=7（15 run 池化） | 33.87%（2678/7907） | 33.23%（2663/8014） | −0.64 pp |
| 采样 K=5（15 run 池化） | 46.50%（2667/5735） | 44.34%（2626/5922） | −2.16 pp（差的标准误 ≈0.92 pp） |
| 贪心吞吐 | 59.0 / 59.2 tok/s | 59.4 / 57.9 tok/s | 噪声内 |

采样 K=5 > K=7 符合预期（草稿越短接受率越高）。q4all 采样臂的**首次** K=7 运行在第 14 个请求崩溃：
`tp2_generation_core.cpp:3119` 的 D2H `cudaMemcpyAsync` 报 `cudaErrorIllegalAddress`（sticky——故障在更早的
DFlash2 verify/accept kernel，报错点只是首个 API 边界）；同一臂**重跑 15/15 全部通过**，且该 artifact 的 draft
执行路径本次改动未触及（Q8 QKV 仍走融合 `attn_input_proj`/`context_kv_materialize`），故判定为
**既有 TP-2 竞态 flake**（worklog 历史同类：Round 36d `tp2_generation_core.cpp:443` D2H 同症状），
已记录待专项排查；本轮结论以重跑数据为准。

**三件套（qkv4）**：append-K7 / append-K5 / solo 全部 **exit=0**；solo 两次跨进程 digest 相同
（`0x4bcc3994a5efba7d`），且与 q4all **逐位相同**——贪心接受规则下草稿只改变每轮产出 token 数、不改变 token 序列，
所以这是预期；append 的 proposal digest 与 q4all 不同（`0xf9622ee5cd45709d` vs `0xda91572dd83980bd`）属预期。
`sessions` 在**默认（三路线全跑）**下报 `FAIL (plain): a conversation behind a shared system prompt diverged …`，
但该失败**与本次改动和草稿精度无关**：q4all 臂的输出 token **逐位相同**，且 `build-win/r66/` 里**改动前**的
`sessions.log`（baseline 件）与 `sessions-baseline.log` 是同一串 token；把路线收窄为 `NINFER_TEST_ROUTE=dflash2`
后 `sessions-dflash2-*.log` 一律 PASS。⇒ 本活动的 sessions 门禁一律按 dflash2 路线跑（r66 亦如此）。

**②-f 30-rep 高功效复测（2026-09-25，同一二进制，两臂相邻轮次）——推翻上面的 15-rep 结论**：

| 指标 | q4all | qkv4 | Δ | se(Δ) |
|---|---|---|---|---|
| 采样 K=7 | 35.13%（16234/46207） | 34.99%（16208/46326） | **−0.14 pp** | ≈0.31 pp |
| 采样 K=5 | 45.42%（15865/34927） | 45.25%（15852/35029） | **−0.17 pp** | ≈0.38 pp |
| proposal cost（CUDA events） | 6.851 ms | **7.044 / 7.031 ms** | **+0.19 ms（+2.8%）** | 确定性 |

接受率代价**不可测**（两档都 <0.5σ）；15-rep 的 "−2.2 pp" 与基线自身 0.6–1.1 pp 的轮次漂移同量级。
三件套在**当前二进制**下重跑仍全过（`SUITES_DONE failures=0`，5/5 exit=0；solo digest 与基线逐位相同；
append proposal digest K=7 `0x8103f572fb2d2f99` / K=5 `0x96582e7289dd2702`）。② 的真实代价只剩确定性的
proposal 步 +2.8%（+0.19 ms/轮；按 ~45 ms/轮折算 ≈ **+0.4% 端到端**）。

**② 结论（修正）**：② 是「**−79.69 MiB 换 ≈+0.4% 轮时、接受率无代价**」的中性偏正杠杆，而非上一轮判定的
「赔 1.3–3.2% 吞吐」。与 ①（−178.09 MiB、零代价）相比 ① 严格更优；② 是否进默认取决于显存压力是否值得
那 0.4% 轮时。（方法论：本活动的接受率判决必须用 30-rep；15-rep 的臂间噪声地板 ≈±2 pp。）

**① selector codebook 量化（−174.3 MiB，非 GEMM）**

现状：codebook 是裸 `Tensor`（`parameters.h:111-114`），唯一 bf16 路径（`src/ops/candidate_selector/bf16/`、
`wrapper/candidate_selector.cpp:110-113`），248320×256×2 = 242.5 MiB；TP-2 置于 shard 1
（`tp_split_spec.cpp:97-104`）并由 peer 代跑 selector（`text.cpp:973+`）。选择域是 256 维内积 top-16/248320。
设计：新增 codebook codec（目标档待定：Q4_G64 每元素最小，NVFP4 每 16 权重一 scale、排序保真更好），
保持 top-k 域；`SelectorParameters`（`parameters.h:116-119`）的 codebook 由裸 `Tensor` 升级为带 qtype 的表示，
再由 `candidate_selector_path` 的量化重载按存储 scale 独立解码。已定位的两个 ABI 约束：
① loader 目前用 `b.direct(..., BF16)` 精确钉住格式，`Binder::binding` 在 `exact_format` 不符时抛错
（`src/artifact/binder.cpp:53`）⇒ codebook 参数必须改为「不钉格式」并按 artifact 实际几何选择表示；
② TP-2 下 codebook 在 shard 1、由 peer 的 `dflash_propose_batch` 消费（`text.cpp:973+`），量化表示必须同样跨 peer。
先做 op 级 oracle（top-16 集合/分数 vs bf16 基线），再端到端 A/B（接受率为硬门禁——量化误差直接改提案集合）。

**① 进度（2026-09-25）**：op 层实现完成。格式无关的 plan 从 `bf16/` 上移为
`src/ops/candidate_selector/plan.{h,cpp}`（新增 `candidate_selector_path_q4_dispatch`，workspace 不重叠校验
同时覆盖 codebook 两平面）；新增 `src/ops/candidate_selector/q4/candidate_selector_path_q4.cu`，按
Q4_G64_FP16 RowSplit 独立解码——`code(r) = nibble (r&1) of byte (r>>1)`（nibble 语义 `(n^8)-8`），
scale = `fp16(scales + token*8 + (r>>6)*2)`（`padded_columns=align(256,128)=256`、`code_bytes_per_row=128`、
`scale_offset=align(n*128,256)=n*128` 已按 `weight_geometry` 核对）；公开头新增 `Weight` 重载，
wrapper 校验 qtype/layout/group/n/k/scale_dtype 与 payload 不重叠。`ninfer_ops` 与
`ninfer_candidate_selector_test` 均编译链接通过（exit 0）。测试新增 q4 oracle 臂：host 侧构造
`value = code * fp16_scale` 并打包 nibble，FP64 oracle 按**存储 scale 独立回解**，覆盖 K=3（Direct）/
K=7（Lattice）× B∈{1,3,8}。该测试已运行 **PASS**（`OK candidate_selector_path`，`TEST_EXIT=0`）——
q4 解码与 FP64 oracle（按存储 fp16 group scale 独立回解）在两条路由上逐位一致。
**①-b 进度（2026-09-25）**：loader / 参数 / peer 已打通。`load/dflash2.cpp` 的 codebook 由
`b.direct(..., BF16)` 改为 `b.parameter(...)`（不钉格式；`bindings` 里 `direct` 本就是
`parameter(..., {}, format)` 的特例，故只去掉格式钉死）；新增 `SelectorCodebook{dense, weight, quantized}`
表示与 `Prepare::codebook()`（按 artifact 几何在 BF16→`tensor()` 与 Q4→`prepare_linear_weight().weight`
之间选择，格式非法即抛错）；新增 `execution/selector.h::run_candidate_selector()` 作为**唯一派发点**，
`draft.cpp`（持有 selector 的 shard）与 `text.cpp`（TP-2 peer）都改走它，量化表示随 `SelectorParameters`
自然跨 peer。**stock BF16 codebook 回归**：q4all 件 append 测试 digest 与基线**逐位相同**
（proposal `0xda91572dd83980bd`、ring `0x371f934bfbd6c7ec`），`TEST_EXIT=0`。
r68 实验件 `D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_cb4.ninfer` 已转换（1218 对象，379.8 s；
payload **−178.09 MiB**，q4 81→83 / bf16 569→567，恰 2 个 `[248320,256]` q4 codebook；dry-run
q4 33 / bf16 33 / q8 25；不含 r67 的 QKV 改动）。加载期首个实现踩到 `Model::input(WeightId)` 要求
「恰好一个 Use」（`model.cpp:23-28`）而 codebook 是裸 direct 参数——改为直接 `native_weight(bound.view)`
从完整 parent 取 Weight（不需要 Use），仍是同一几何校验。**cb4 端到端验证**：append 测试
`TEST_EXIT=0`（direct/split append、ring repeat、proposal determinism 全过），且
`peer selector: on shard 1` 由 **245.0 MiB → 66.9 MiB**（−178.1 MiB，与 artifact 字节差一致），
proposal cost **6.854 ms/proposal**（基线 6.851，噪声内）。

**①-c 结果（2026-09-25，同一二进制；基线 = q4all）**：采样池化（15 run × 3 prompt 类）。

| 指标 | q4all | cb4 | Δ |
|---|---|---|---|
| 采样 K=7 | 34.62%（2697/7790） | 34.37%（2691/7829） | −0.25 pp |
| 采样 K=5 | 45.59%（2649/5810） | 43.40%（2609/6012） | −2.19 pp（se ≈0.91 pp） |

基线自身也在漂：同一 q4all 件在 r67 那轮是 K=7 33.87% / K=5 46.50%，本轮 34.62% / 45.59%
⇒ 15-rep 池化的臂间漂移约 ±1 pp，K=5 的 −2.2 pp 只有 ~2.4σ，**不足以定论**。三件套 **全过**
（append-K7/K5、solo×2、sessions-dflash2，`SUITES_DONE failures=0`）；solo digest 与基线逐位相同
（贪心序列不随草稿变）；append 的 K=5 proposal digest 与 q4all **相同**（`0x5a3629fb79be1cd3`）、
K=7 不同（`0x766be9d7643131e2`）⇒ codebook 量化对该 fixture 的候选选择影响很小。
**①-c 高功效复测（30-rep，90 run/臂，同一二进制）——15-rep 的 −2.2 pp 是噪声**：

| 指标 | q4all | cb4 | Δ | se(Δ) |
|---|---|---|---|---|
| 采样 K=7 | 35.77%（16329/45644） | 35.74%（16309/45629） | **−0.03 pp** | ≈0.32 pp |
| 采样 K=5 | 44.36%（15754/35510） | 45.95%（15928/34666） | **+1.59 pp** | ≈0.38 pp |

两臂各自的 15→30-rep 估计漂了 1.4–2.6 pp，说明 15-rep 池化的臂间噪声就有 ~±2 pp，此前 r67/r68 的
"−2 pp 接受率代价" 都是这个量级的噪声，不能作为否决依据。30-rep 下 K=7 无差异、K=5 方向反而为正
（+1.6 pp 更可能是残余臂效应而非真实提升，但至少**没有可测代价**）。

**① 结论（建议采纳）**：178.09 MiB（artifact **−0.75%**，运行时 peer selector 245.0→66.9 MiB）
换零可测接受率代价、零 proposal 开销；op oracle、三件套、stock BF16 逐位回归全过。**② 与 ① 的对比正是
A/B 噪声地板（~±2 pp）造成的误判**——两者对接受率的真实影响都小于该活动 15-rep 方案的分辨率，因此
"省显存赔吞吐" 的结论只对 ① 不成立（① 证伪），对 ② 则应重测后才可下结论（见下）。

**r69 提升为官方配方 + 最终件（2026-09-25，用户确认）**：`tools/convert/official_recipes.py::_optional`
现在承载完整 DFlash2 draft Q4 集合——r62-r66 的 MLP（gate/up/down）、attention output、两个动态卷积 kernel、
`feature_projection`，加 **②** 的融合 QKV（`dflash2/layers/*/attention/{query,key,value}`，一个
`[6144,5120]` 父对象/层，经行视图消费）与 **①** 的两个 selector codebook；全部限定在 `dflash2/`，
dflash v1 与 MTP 仍 Q8（`DFLASH2_Q4_NAMES`/`DFLASH2_Q4_ROLES`/`DFLASH2_CODEBOOKS`）。

最终件 `D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer`（**无 `--override`**，1218 对象，385.7 s，
`CONVERT_EXIT=0`）：与基线 `q4all` 逐参数对比，**恰好 17 个参数**格式变化（15 个 QKV + 2 个 codebook，
全部 → q4），其余 **1486 个参数的格式与形状完全一致** ⇒ 官方配方精确复现了 r66 件，最终件 = r66 + ① + ②。
payload **−257.77 MiB**（q4 81→88 / q8 12→7 / bf16 569→567）。

**r69 最终验收（30-rep，90 run/臂，两臂相邻轮次）**：

| 指标 | q4all（r66 基线） | final（r66+①+②） | Δ | se(Δ) |
|---|---|---|---|---|
| 采样 K=7 | 35.09%（16223/46228） | 34.79%（16188/46530） | **−0.30 pp** | ≈0.31 pp |
| 采样 K=5 | 44.99%（15828/35181） | 44.88%（15812/35233） | **−0.11 pp** | ≈0.37 pp |

两档都 ≤1σ ⇒ **无可测接受率代价**。三件套 **全过**（`SUITES_DONE failures=0`：append-K7/K5、solo×2、
sessions-dflash2）；solo digest 与基线逐位相同（`0x4bcc3994a5efba7d`）；运行时 **peer selector 66.9 MiB**
（基线 245.0），proposal cost **7.029 ms**（基线 6.851，即 ② 的 +0.18 ms ≈ +0.4% 轮时）。
⇒ **最终件可发布**：相对 r66 基线省 **257.77 MiB（−1.08%）**，接受率无代价，轮时代价 ~0.4%。

**过程中两次低频失败（与本次改动无关，待专项排查）**：① r67 那轮 q4all 臂 K=7 的
`cudaErrorIllegalAddress`（同臂重跑全过，判定既有 TP-2 竞态）；② 本轮 final 臂 K=5 首次运行第 14 个请求
`HTTP 503 service unavailable` 且无 FATAL 日志（重跑 90/90 全过）。两者都只出现在连续 ~90 请求的采样臂里，
约 15 个臂共 2 次。

**r69/r70 其余 Qwen3.8-27B 变体的最终件（2026-09-25）**：同一官方配方（无 override）+ 共用草稿源
`D:/LLM/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8`，三个变体各产一件，均含 `text,vision,mtp,dflash2`：

| 变体 | 源 | 最终件 | 大小 | 文本表示 | 转换 |
|---|---|---|---|---|---|
| W4A4 | `W4A16/NVFP4/W4A4` | `qwen3_8_27b_w4a4_dflash2_final.ninfer` | 18.42 GB | nvfp4 256 / fp8 2 | 336 s |
| Swift-1.5 | `Swift-1.5-Qwen3.8-27b-NVFP4` | `qwen3_8_27b_swift15_dflash2_final.ninfer` | 21.58 GB | fp8 130 / nvfp4 128 | 413 s |
| ThinkingCap | `ThinkingCap-Qwen3.8-27B-NVFP4A4-AWQ` | `qwen3_8_27b_thinkingcap_dflash2_final.ninfer` | 22.52 GB | fp8 146 / nvfp4 112 | 388 s |

四件（含 w4a4+w8a8）的草稿剖面完全一致（q4 88 / q8 7 / bf16 567），append 冒烟测试**全部 PASS**：
`peer selector on shard 1 = 66.9 MiB`、proposal cost 7.03–7.08 ms、K=7 proposal 确定性。

**ThinkingCap 的 tokenizer 坑（已修）**：该导出由另一个 `tokenizers` 版本序列化——`tokenizer.json` 的
Split 预分词器用 `\p{L}+`（无 `\p{M}`）且 `trim_offsets=true`，引擎的 Qwen tokenizer 拒绝
（`pre_tokenizer.Split is not supported`）；且 `tokenizer_config.json` 缺引擎必需的
`added_tokens_decoder`。该导出的 `vocab.json`/`merges.txt` 与规范导出**逐字节相同**，规范导出的 33 条
`added_tokens_decoder` 与其 `tokenizer.json` 的 added_tokens **完全一致**，故用规范 `tokenizer.json`
替换、并把缺失的 decoder 合并进该导出自己的 `tokenizer_config.json`（保留其模型专属字段）。
见 `tools/tp_bootstrap/r70_convert_thinkingcap.ps1`。

**执行顺序**：② 先行（改动集中、可用既有 oracle 验证），① 并行做设计/op oracle。

### 3.8 前缀复用：残留（2026-09-24）

- 文本协议回放无法保证逐 token 复现：首生成 token 问题已随「首个生成 token 不发布」修复（回放回答现直达
  `src=live`），但客户端重分词与采样边界不同、或边界落在 chunk 内（契约只保证 boundary crossing）时
  尾部复用仍会退化；保证逐 token 的唯一杠杆是携带 token id 的续写协议（未排期）。
- TP-2 prefill 的首个 sample 未应用工具语法掩码（prefill 与 decode 路径不对称，独立遗留项）。

### 3.11 TP-2 编排与握手：三项具体工作（2026-09-25 开工）

> 缘起：用户问「以'所有组件都支持低耦合 TP-2'为目标做整体重写是否值得」。结论是**不做整体重写**，
> 只做三件有明确收益的具体事。量化依据：分片计划已集中（`tp_split_spec.cpp` 175 行、shard mask 仅 2 处），
> 巨型物 `execute_walk`（1412 行/12 lambda）的大头是会话与检查点状态机而非 TP；瓶颈在 target verify。

**② 图谱接线：复核结论＝已完成，剩下的是 B7 验收**

我依据的 worklog 4128-4131（2026-09-22「B7 前置侦察（已完成，未实施）」）**已过期**。现役代码四处全部接线：

| 侦察所列 | 现役代码 |
|---|---|
| ① 总开关只对 MTP 生效 | `tp2_generation_core.cpp:374` = `(mtp_enabled_ || dflash2_enabled_) && in_kernel_allreduce()` ✓ |
| ② verify 桶只在 MTP 分支建 | `:387-397` 两条路线都建 `verify_window_host_`；`:398-409` profiles 走
`dflash_graph_profiles(DFlash2,…)` ✓ |
| ③ `capture_verify_graph` 不收 sink | `:954-958` 形参已含 `DFlashFeatureSink* sink` + `valid_columns` ✓ |
| ④ `:979-984` 的 throw | 已由捕获体 `:980-986` 的转发取代（注释即该结论）✓ |

**实测确认**：本 session 的 DFlash2 服务日志 `build-win/r69/sampling-r69-final-k7.log` 打印
`[tp2-graph] plain decode step: graph | verify step: graph | mtp chain: graph` ⇒ 生产已在图内运行。

**剩余工作 = B7 验收**（worklog 4133 的判据）：`NINFER_TP2_VERIFY_GRAPH=0/1` 成对 A/B，verify ≈30.6 ms、
比 eager 低 4~4.5 ms/轮、等输出吞吐 ≈+15%。gate：两臂吞吐 + proposal cost + 三件套。

**① 收口隐式不变量：两个具体目标（明确不重写 `execute_walk`）**

**1a 消灭「静默默认参数」这一类**——对应两个真实事故（worklog 4131 sink 漏接线、4135 `valid_columns`/
`active_valid_columns_` 未绑定导致的 paged-KV 同槽撕裂）：
`TextContext::forward_tp2_window`（`text.h:217-221`）末尾三个参数
`hidden_columns=nullptr` / `sink=nullptr` / `valid_columns=nullptr` **每一个都改变语义**——
头文件自己的注释（`:211-216`）就写着「nullptr leaves every column appending」，即漏传 = 静默退化成无掩码。
做法：**删掉三个默认值**，两个调用点（`:984` 捕获、`:1031` eager）显式传参；并把两侧实参收进同一个只读
struct，使「捕获与 eager 传不同参数」在类型上不可能（现在靠两处复制粘贴保持一致）。
gate：`BUILD_EXIT=0` + 三件套 digest 逐位不变。

**1b 让窗口内 give-up 可辨识**（§3.10(d)）：窗口内自旋 give-up 现在表现为语焉不详的 illegal address
（注入 700 已逐字节复现）。目标是失败时能判定「give-up in window」而非「illegal address」。
gate：注入 700 的报错可辨识，且干净臂零新增输出。

**③ 命名与声明载荷**

- shard mask 现在只剩 **2 处**（`tp_split_spec.cpp:101,109`），改为命名角色常量（如 `Shard::DraftHome`）。
- 4–6 种握手协议（candidate ids/scores 并集、MTP `pack_forward`/`back`、draft state、AR）的载荷与契约集中声明。

**顺序与 gate**：③ → 1a → ②B7 验收 → 1b。③/1a 只动结构与参数，不改数值 ⇒ 三件套 digest 必须逐位不变；
② 用 `NINFER_TP2_VERIFY_GRAPH` 成对 A/B；1b 用注入点。

**进度（2026-09-25，本批已实现并验证）**

- **1a 完成**：`TextContext::forward_tp2_window` 末尾三个语义参数删掉默认值
  （`hidden_columns`/`sink`/`valid_columns`），头文件写明「每个都改变语义、缺省即静默退化」；
  两个调用点（捕获 `tp2_generation_core.cpp:984`、eager `:1031`）本来就显式传参 ⇒ 现在是编译期约束。
  后续把两侧实参收进同一 struct 的动作未做（收益递减，见下）。
- **③ 完成（命名部分）**：`src/core/tp/tp_materialize.h` 新增 `kLocalShard=0x1`/`kPeerShard=0x2`，
  `tp_split_spec.cpp` 的两处裸 mask 改为命名角色 ⇒ 全仓 `0x1U/0x2U` 归零。**载荷声明部分未做**。
- **1b 完成并验证**：`ar_exchange` 的 give-up 路径**不再把输出留成上一轮的陈旧值**——新增
  `ar_abandon_output()`，三处 `stalled` 提前返回（入口检查/块序等待/arrival 等待）都先把本 block 的
  `out` 切片清零。理由：该 staging 缓冲逐轮复用，重放的捕获图复用同一 arena 内存，留旧值会让下游把
  陈旧数据当**索引**用 → 窗口内 illegal address，而 host 侧任何检查都来不及（§3.10 已证）。
  零是合法的 token id/candidate/logit，于是该轮能跑完、由轮收敛点的 `abort_if_ar_stalled()` 变成设计好的 503。
  **验证**：`r72_stall_injection.ps1 -Calls @(560,700)` ⇒ 两处都 `503,ok illegal=False`
  （修复前 700 是**逐字节复现**的 `tp2_generation_core.cpp:3119 cudaErrorIllegalAddress` + 进程退出）；
  健康路线零代价（填充只在超时路径执行）。
- **①②③ 的数值零影响**：`r66_suites.ps1`（dflash2 路线，final 件）**5/5 exit=0**，
  digest 与改动前**逐位相同**（K=7 `0x8103f572fb2d2f99`、K=5 `0x96582e7289dd2702`、
  solo `0x4bcc3994a5efba7d`、ring `0xb1a1fe192b36817a`）。
- **② B7 验收（`r81_verify_graph_ab.ps1`，final 件，K=7，6 reps × 3 prompt 类 = 18 请求/臂，4608 token/臂）**：

  | 统计量 | graph | eager | Δ |
  |---|---|---|---|
  | 逐请求 decode 均值 | 76.22 tok/s | 71.90 tok/s | +4.33（+6.0%，se=4.39，**t=0.99 不显著**） |
  | 逐请求 decode 中位数 | 72.2 | 70.5 | +1.7 |
  | **聚合吞吐**（Σtoken / Σ请求 total） | **72.68 tok/s** | **69.09 tok/s** | **+3.60（+5.2%）** |

  两臂 `[tp2-graph]` 行确证只差 verify 一项（`verify step: graph` vs `eager`，plain/mtp chain 相同）。
  ⇒ **图谱确实在跑且带来约 +5% 端到端 decode**（方向与 worklog 的 +15% 预估一致但幅度更小）；
  逐请求统计在本样本量下分辨不出，聚合口径可用但仍是单轮，若要定档需 ≥24 reps/臂。

**未做（本批有意留下）**：1a 的"两侧实参收进同一 struct"（当前两个调用点已显式且一致，编译期已够）；
③ 的握手载荷集中声明。

### 3.10 TP-2 传输低频失败专项排查（2026-09-25）

**目标**：定性并（若可行）根因定位采样臂连续长跑下的两次低频 engine 失败。

**证据（两个实例，全战役仅此两次；扫描 `build-win/**` 全部 `.out`/`.log` 确认）**

1. **IllegalAddress（2026-09-25 08:20:59，r67 q4all K=7 臂，req#14）**：
   ```
   tp2_generation_core.cpp:3119: CUDA_CHECK(cudaMemcpyAsync(licensed_host.data(),
     frame.licensed_tokens.data, sizeof(TokenId)*width, D2H, shard_a_.device.stream))
     failed: cudaErrorIllegalAddress
   ```
   这是**粘贴错误**：真正的越界 kernel 更早，报错点只是 round 末尾第一个 API 边界。日志无任何 503/停滞记录。
   同臂重跑 15/15、90/90 全过；同 artifact 在改动前也是这个行为。
2. **HTTP 503（2026-09-25 10:34:39，r69 final K=5 臂，req#14）**：请求体
   `{"code":"service_unavailable","message":"TP-2 allreduce stalled at rendezvous id 8590057052: the request
   was failed and the reusable context was discarded"}`。**不是崩溃**：这是有界自旋的设计行为
   （`tp2_generation_core.cpp:1410 abort_if_ar_stalled()`）。id = `2<<32 + 122460` ⇒ 通道 index 1
   （第 2 个创建的图），base 已累计 122460 次 collective，与「14 个请求 × K=5」的轮数×每轮 collective 数一致。

**已排除**
- 残留环境变量：HKCU/HKLM 与进程环境均**无** `NINFER_*`；故障注入
  `NINFER_TP2_AR_FAULT_SKIP_PEER_CALL` 未开（`device_pair.cu:327-331` 在 0 时不计数、零开销），
  超时为默认 **2000 ms**（`device_pair.cu:481`）。
- 时钟：`ar_now_ns()` 用 `%globaltimer`（ns，设备内一致），deadline 语义正确（`device_pair.cu:113-123`）。
- 边界：`ar_exchange` 的 group/tail 循环、parity 槽（`(token&1)*slot_bytes`）、stall 标志
  （`stall_host_` 128 B = 2×64 B，`ar_stalled()` 读 flags[0]/flags[16]）与 store helper 的 `lane<=k`
  逐项核对无越界；`count_bytes <= slot_bytes` 恒成立（小路径 64 KiB / 大路径 24 MiB 与各自守卫一致）。

**机制判断**：一次失败的自旋会让**那一轮的本地偏和未被对端更新**（内核提前 return），
而 host 只在**每轮收敛点**检查 trip 标志（`abort_if_ar_stalled()` 调用点 2751/3317）。
若损坏数据在检测前被消费成**索引**（candidate/anchor/frontier/argmax），就会变成粘性
illegal address ⇒ 两种表象可能是**同一事件在两个时刻被观察到**（干净 503 vs 先崩）。

**验证结果（2026-09-25）**

1. **长跑未复现**：`tools/tp_bootstrap/r71_ar_stress.ps1`（final/K=5、看门狗开启、600 请求、
   首个失败后继续收集）**0 次失败** ⇒ 自然发生率低于 ~1/1350 请求，不能靠长跑排查。
2. **注入可确定性复现，并证实「同一事件」**：`tools/tp_bootstrap/r72_stall_injection.ps1` 用既有注入点
   `NINFER_TP2_AR_FAULT_SKIP_PEER_CALL=N` 在指定 collective 序号跳过一次对端启动：

   | 注入序号 | 落点 | 结果 |
   |---|---|---|
   | 20 | warmup（eager） | `FATAL warmup failed … stalled at rendezvous id 1<<62+129`，引擎退出，无崩溃 |
   | 280 / 300 / 330 / 360 / 400 / 440 / 480 | warmup 之后、prefill（eager） | **干净 503 + 下一请求 ok**（7/7） |
   | 560 | captured 图内（`graph-ch2`） | 干净 503 |
   | **700** | verify 窗口内 | **`tp2_generation_core.cpp:3119 cudaErrorIllegalAddress`**，与生产事故**逐字节同一条报错**，进程退出、连接被掐断 |

   看门狗 dump 同时给出基线事实：warmup 结束于 `calls=268`（真实请求从 269 起），最后一个健康
   collective 两侧 `arrA==arrB`（6 slices，大 payload 路径）。
3. **二分定位：故障发生在窗口内部**。在失败分支的 `run_verify_window(...)`（`3100`）之后临时插入
   `abort_if_ar_stalled()` 探针后重跑注入 700：探针**确实捕获了 trip**，而报错从 3119 变为
   **`tp2_generation_core.cpp:1419`（`abort_if_ar_stalled` 内的 `cudaStreamSynchronize`）**
   ⇒ 非法访问**已经在 verify 窗口（一个 captured 图）内部发生**，任何 host 侧检查都来不及。探针已回退。
4. **工具边界（负面结论，重要）**：`CUDA_LAUNCH_BLOCKING=1` 与本传输**不兼容**——A 侧的 launch 变成阻塞，
   而 B 侧内核要等它返回后才 launch ⇒ 第 1 次 collective 必然 give-up（自死锁）。`compute-sanitizer`
   同理（串行化内核；把超时抬到 120 s 后 warmup 仍 give-up）⇒ **本传输无法用 launch 串行化工具做 kernel 级
   归因**，只能靠源码级探针二分。

**缺陷定性（已证）**：有界自旋 give-up 后，本轮各 collective 的偏和**只在本地**（对端贡献缺失），而 round
会继续跑。phase 1b/2 的缓解只在**轮收敛点**（`abort_if_ar_stalled()`，调用点 2751/3317）检测，
当 give-up 落在 **captured 窗口内部**时**来不及**：窗口自己的 kernel 把不一致的数据当**地址**用就会崩
（已复现）。只有落在 eager prefill 的 give-up 才会走成干净 503 + 下一条请求恢复。

**触发源（未决）**：自然 give-up 无法按需复现（600 请求 0 次；全战役 ~2700 请求 2 次）。生产那次 503 的
rendezvous id（`2<<32+122460`）也在**captured 通道**内，与复现一致。需要 >2 s 的 host/device 延迟，
或一次真实 desync（id 复用/错配）。已排除环境残留注入与默认超时被改。

**处置建议（待决策）**
- (a) **抓到触发源（已实施）**：采样 A/B（`r62_sampling_ab.ps1`）、三件套（`r66_suites.ps1`）、
  长跑（`r71_ar_stress.ps1`）一律开 `NINFER_TP2_AR_WATCHDOG=1`；失败时 `tools/tp_bootstrap/ar_watch.ps1`
  把日志**另存为不可被复跑覆盖**的 `*-FAILED.log`，并把解释后的 dump 写到 `*-FAILED.arwatch.txt`。
  判读规则按 **id 差值的量级**分类（数组保留的是各侧**最后写入**的值）：某侧整组为 0 ⇒ 该侧
  **从未到达 arrival 写入**（内核没跑）；差值 1 ⇒ **OFF-BY-ONE**（两侧对同一次 collective 编号不同）；
  差值达一个整块（数百）⇒ **BLOCK-LEVEL**（两侧跑在不同 id 块上，正是「captured 图读到了 host 已
  重新 arm 的 cell」的形态）；其余为一般 SKEW。同时改进了 `device_pair.cu` 的看门狗：只在**设备真的 trip**（读两个 stall flag，与 `ar_stalled()`
  同源）时才 dump——请求之间的空闲同样会让 host 计数停住，原来会被误当成 stall；轮询从 500 ms 收紧到
  **25 ms**，因为 trip 之后往往只有几毫秒就 fault 并退出，慢了就抓不到。
  **验证**：干净臂 0 条 ar-watch、无 FAILED 文件；注入臂（`FAULT_SKIP_PEER_CALL=700`）exit=1 并留下
  `FAILED.log` + 判读 `OFF-BY-ONE: A=1<<62+439 B=1<<62+440`；三件套在看门狗开启下
  **5/5 exit=0**，proposal digest 与改动前逐位相同（K=7 `0x8103f572fb2d2f99` / K=5 `0x96582e7289dd2702`）；
  门控两点验证：空闲（3 请求 + 4 s 间隔）**0 条 dump**，注入 trip **2 条 dump 且判读正确**。

  **自然长跑仍未抓到触发源**：看门狗开启的 900 请求长跑（`r77`）**900/900 成功、0 次失败**，累计
  自然请求 1500 次（`r71` 600 + `r77` 900）零事件，佐证自然发生率 ≲1/1500。（另注：`r76` 那次
  「请求 3 连接被掐断」是**残留在跑的旧 harness 与本次抢同一端口**造成的假象——server 日志显示两个
  客户端同时在跑，且真正把 serve 杀掉的是旧 harness 的收尾；确认环境干净后重跑得到 `r77`。）
- (b) **让 give-up 安全**：host 侧检查已被证伪（来不及）；要么让传输在 give-up 后不再继续消费
  （毒化/中止语义），要么让窗口内做索引的算子对输入做范围约束——都是架构级改动，需要先知道确切算子。
- (c) **止损**：默认超时 2000 ms 是相对**内核时间**（最慢 2.7 ms）的 700×，但对**host 侧停顿**并非安全余量；
  抬高默认值可显著降低自然发生率，代价是真实 desync 的检测变慢（仍会失败，只是更晚）。
- (d) **可诊断性**：窗口内 give-up 现在表现为一个语焉不详的 illegal address；即使不修安全性，也应让它可辨识。
