# NInfer TP-2 计划（2× RTX 5060 Ti · Qwen3.8-27B NVFP4）

> **唯一的活动计划**，也是跨上下文压缩的持久记忆。整理：2026-09-22（上游 cherry-pick 与 DFlash2 各轮完成后
> 重写为交付前收尾版）。
> - 完整历史记录：`docs/tp2-dual-5060ti-worklog.md` —— 含两份 PLAN.md 全文逐字归档（2026-09-21 上游
>   cherry-pick 重规划版；2026-09-22 本版整理前的全文）。
> - 交付说明、推荐配置与实测数据：`docs/tp2-dual-5060ti.md`；Windows 原生移植：`docs/windows.md`。

---

## 1. 关键信息

- **分支**：`feat/windows-native-port`（Windows 原生 + WSL2 双树）。origin/master 领先的 20 个提交已全部
  cherry-pick 完成（前端 jinja 链 + NVFP4 性能三项），端到端 A/B 无退化。
- **硬件**：2× RTX 5060 Ti 16G（SYS 拓扑、无 P2P、448 GB/s/卡）。WSL2 构建树 `/home/zhuojun/ninfer`；
  Windows 工作树 `D:\Documents\workbench\ninfer`（构建树 `build-win`）。
- **主负载**：Qwen3.8-27B NVFP4，TP-2（权重按 shard 切半，本地自建半几何 shape）。草稿后端：`--spec mtp`
  （K=2、`--lm-head-draft`）或 `--spec dflash2`（K=7）；`--spec dflash`（v1）在构造期拒绝。
- **现状**：交付目标 ①–⑤ 全部完成；vision、Windows 移植、多会话 KV 池、decode graph + AR 策略均已落地实测；
  DFlash2 已在 TP-2 放行（B1–B6、verify CUDA Graph、MTP draft 链图化、弃稿守卫修复）。
- **关键数字**（262,144 配置，WSL）：prefill 1,588 tok/s；decode 59.0 tok/s（MTP K=2）；Windows 131,072 配置
  decode 56.9 tok/s（与 Linux 持平）。DFlash2 K=7 在 4096 上下文贪心档约 71 tok/s（与 MTP K=2 持平）；
  roofline 修正上限 ≈108 tok/s（原归档 90–180 的 180 不可达）。

**推荐运行配置**（WSL 8088 / Windows 8099 同配方，Windows 侧 `--max-context 131072`）：

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh` 是进程级默认思考强度（请求体优先）；`--chat-template` 随第一梯队摘取
可用（模板由内嵌 Jinja 执行）。

### 环境与运维要点

| 项 | 值 |
|---|---|
| 构建（WSL） | `bash tools/tp_bootstrap/r55_build.sh`（rsync + `cmake --build build_dyn -j 8`，成功标记 `BUILD_EXIT=0`，约 2–3 分钟；WSL 构建树现为 `build_dyn`（Ninja），`build_r35.sh` 仍指向已删除的 `build/`，不可用） |
| 构建（Windows） | `tools/win_port/configure.bat` + `build.bat`（VS2022 + CUDA 13.3，`-DCMAKE_CUDA_ARCHITECTURES=120a`） |
| 服务 | WSL 8088（`serve_supervise.sh`）；Windows 8099（`tools/win_port/serve.ps1`，默认前台；自测服务必须用 harness 后台 job，`Start-Process` 起的进程会随工具调用结束被杀） |
| 运行 PATH（Windows） | FFmpeg（`D:\ffmpeg-dev\…\bin`）与 libcurl（`D:\curl-dev\…\bin`）必须在 PATH，否则 `STATUS_DLL_NOT_FOUND` |
| artifact | `D:\LLM\qwen3_8_27b_nvfp4.ninfer`（23.7 GB，旧 artifact，模板 `c3cf9e34…`）；新官方 artifact（模板 `a497db9e…`）随第一梯队可用 |
| 进程纪律 | 全机同一时刻只有一个模型进程（WSL 8088 与 Windows 8099 互斥）；Windows 重新链接 exe 前先停服务（LNK1104） |
| 工具链 | 嵌套 `pwsh` → `wsl -e bash -lc` 吞 `$var` 与重定向 ⇒ 命令写成脚本文件放 `tools/tp_bootstrap/` 再执行；单次阻塞调用上限 600 s ⇒ 长任务用后台作业 |
| GPU/WSL 状态（2026-09-23 起） | 机器可见 3 卡：nvidia-smi 序 0/2＝5060 Ti、1＝Tesla T10；`--devices 0,1` 账本两 shard 均 16310.6 MiB ⇒ CUDA 序数取到两张 5060 Ti（与 T10 的 nvidia-smi 序错位），TP-2 配对不受影响。WSL 侧 CUDA 当日全线段错误（cast/q4/q6/q8/q5 未改动测试全挂，疑似 T10 可见后 CUDA init 崩）⇒ 测试改走 Windows `tools/win_port/test.ps1`，WSL build_dyn 只作编译验证 |

---

## 2. 已完成摘要（细节见归档）

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
- 官方 NVFP4 工件接受率调查：合成工件（官方 mtp/dflash2/vision + 自转 text）与自转件一致、官方件最低
  ⇒ 差异来自 text 权重，与组件无关。
- TP-2 显存不平衡：方案 A（reduced proposal head 按词表切分，−170 MiB/卡）与方案 B（selector 移至 shard 1，
  −246/+246 MiB）均已落地，输出逐字节一致。

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

### 3.4 暂缓 / 可选（未排期）

- Round 15 剩余：工具调用约束真实模型端到端复测（75 条探针）；WARN 诊断；路线 B（完整 GBNF）；
  TP-2 `validate_licensed_tokens` 守卫。
- Round 14 候选（用户暂缓）：① system+tools 前缀末尾加锚点（修新会话/压缩后第一轮 46.9 s，可省约 11 s）；
  ② prefill 吞吐（suffix 0.95–1.13k tok/s、冷启 1.35–1.38k）；③ 每请求 ~0.3 s 固定开销。
- 归档 §11.6：MTP priming 对图片列用占位 embedding（~100 行，无收益证据）；单图上限 8,192 硬编码
  （让出 shard 1 可重拿更大 envelope）；视觉编码期 shard 0 空闲（可接受）。
- 归档 §6：KV dtype 扫描补全（nvfp4/k8v4 只验证了可启动与吞吐，无质量数据）。
- 若要从源权重真正复刻官方工件的 DFlash2 draft，需先获取其 BF16 源检查点（本机只有 FP8/EXL3/GGUF）。

### 3.5 环境受阻的未做项（已用替代路径覆盖）

- [ ] WSL 侧 op 测试 + 字节一致 + 前端夹具测试：**WSL2 CUDA 驱动崩溃**（`cudaGetDeviceCount()` 内 PTX JIT
      segfault，GPU 被 Windows 服务占用）；jinja 测试通过证明二进制无误，op 测试改在 Windows 侧跑。
      Windows 侧已过（NVFP4 A4/A16、frontend、jinja；`NINFER_OP_REPORT_STATS=1` 错误指标在容差内）。

### 3.6 DFlash2 上 TP-2（主体已完成；余项待排期）

**状态**：`--spec dflash2` 已在 TP-2 放行（`--spec dflash` v1 仍构造期拒绝）。已完成：加载/分片（B1）、
feature sink 上下文物化（B2a/B2b）、masked 提议前向（B3）、组装轮次 `program/dflash_round`（B4）、整轮接通
（B5）、draft ring 随会话/检查点搬运（B6）、verify 窗口 CUDA Graph（C 轮）、MTP draft 链图化（E 轮）、
reduced proposal head 按词表切分（方案 A）、selector 移至 shard 1（方案 B）。新增 `DevicePair::sendrecv`
做逐字节跨卡交换，修掉方案 A 用 BF16-add allreduce 搬 I32/FP32 的 sNaN 位型风险。金值：append K=7
`0xbad27a494a9bc853` / K=5 `0xbee487264ca8ffb8`；ring `0x1a53fd824cd4e360`；solo digest
`0x4bcc3994a5efba7d`。

**根因定案（A 轮，2026-09-22）**：`forward_tp2_window` 从未绑定 `active_valid_columns_` /
`active_sequence_batch_` ⇒ 窗口每列都写 KV；而 DFlash2 的预算钳位窗口把尾列钉在同一绝对位置 ⇒ 同一次前向
多 warp 并发写同一 paged-KV 槽，内容运行间不确定 ⇒ 近似并列处 argmax 翻转（此前的 S1「彻底同步」只是压低
概率）。修复：窗口接受 `valid_columns`（钳位列禁写 KV、其 logits 置零），仅 DFlash2 传入，MTP/plain 路由
逐位不变。修复后召回逐 token 复现 from-scratch，保留属性已实证并开放（见归档 A/B/C 轮证据）。

**召回契约**：网格对齐边界（`reuse % reuse_grid == 0`）做全 token 对比；非对齐回退（declined 边界）只钉
边界首采样，尾部可复现性由 solo digest 跨进程保证；弃稿状态经 `GenerationResult::draft_context_declined`
暴露。

**弃稿守卫修复**：非网格回退（模板前缀 ~37–57 token 恒命中）曾让 `dflash_draft_declined_` 每轮置位 ⇒
`extent=0` ⇒ 每轮只提交 1 token（~22 tok/s）。现对 DFlash2 加门：`position <= max(reuse_grid, 16×remaining)`
时不取非对齐边界、改用对齐扫描重放 clip ⇒ 草稿全程在线；实测每轮 3.17 committed、约 71 tok/s。归档里
20–25 tok/s 的合成负载数字早于该修复。

**余项**：

- [ ] B7 性能验收：双卡吞吐对目标（roofline 修正 ≈108 tok/s，90 这端相符、180 不可达）+ 与 MTP 对比 +
      plain/mtp 无回归，并记录 `--spec dflash2` 的推荐 K。
- [ ] 待实测未知量：Windows 原生构建的真实 free（台账 `free` 取自 `cudaMemGetInfo`，WSL2 少报约 1 GiB）；
      草稿按 head/row 切分的数值等价性；K=15 的 workspace 峰值；目标 5 层残差跨 shard 汇聚的每步开销；
      草稿每层 allreduce 的延迟；稀疏拒绝采样在 TP-2 分片 logits 下的等价性。

**设计约束（勿重复调研）**：整份复制 draft 不可行——满上下文（262144 / fp8 / MTP K=2）下 free 仅
697/1217 MiB，而草稿 2.07 GiB/卡；切分后每卡约 1.04 GiB。提议条件是目标 5 个 block 的 residual 拼接投影，
TP-2 把 64 层切两卡 ⇒ 需跨卡 handoff（两卡 residual 逐位相同，只捕 shard 0 即可）。候选 selector 直读全词表
codebook（pred/succ 各 248320×256 BF16，约 254 MiB），必须整份。DFlash2 与 MTP 互斥：选 DFlash2 可释放
shard 0 的 MTP 权重 430 MiB + KV 516 MiB。每轮成本构成、D1–D4 决策与各轮否定结论见归档。

### 3.7 DFlash2 draft 量化实验（2026-09-23；已决定：保持 opt-in）

**目标**：验证「只改转换 recipe、不动算子」能把 TP-2 DFlash2 draft 变小多少、接受率掉多少。

**结论**：能，但只能动一块——draft 的 fused gate/up SwiGLU 可在 Q4 下运行，**shard 0 省 450 MiB**，K=7 接受率
不变（24.77% vs 24.59%），K=2 约 −2.5pp（1.6σ，样本不足以定论），探针吞吐持平。`mlp/down` 与
`attention/output` 不能改：draft 的 fused dynamic grouped-conv finish 硬编码要求 Q8（见下）。

**决策（2026-09-23，用户确认）**：保持为 **opt-in override**，不改官方 recipe（`official_recipes.py:35-46`
仍把 mtp/dflash/dflash2 钉在 Q8）；新 artifact 只作实验件，产品路线继续用 Q8 draft。

**分布一致性结论**：本引擎的投机解码是分布精确的——贪心分支发出的每个 token 都是 target 自己的 argmax
（`src/ops/kernel/speculative_round.cuh:120-142`），采样分支是带掩码残差校正的拒绝采样（`:144-214`，算法说明
`:216-228`），钳位列不写 KV（`docs/tp2-dual-5060ti.md:251-256`）⇒ **draft 权重（含其量化）只改变提议分布 q 与
接受率，不改变模型输出分布**。但「输出轨迹逐位一致」被打破：窗口形状差异在近似并列处翻转（既有性质，先例
`docs/tp2-dual-5060ti-worklog.md:1364-1366`），实测两个 K 的贪心文本长度均已变化。

**脚本与产物**：

- override：`tools/tp_bootstrap/r54_draft_mlp_override.py`（`dflash2/layers/*/mlp/gate|up` → `q4_g64_fp16`，
  `grouped_absmax`）；配方仍为 `D:/LLM/w4a4_family_recipe.py`，draft 源 `W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8`。
- 转换：`tools/tp_bootstrap/r54_convert_mlp4.ps1`（332 s，1218 objects，1 file）。
- 新 artifact：`out/qwen3_8_27b_w4a4_w8a8_dflash2_gateup4.ninfer`（24,338,661,380 B，比基线 −451.6 MiB）。
- A/B 臂：`tools/tp_bootstrap/r54_dflash2_arm.ps1`（起服 → greedy_probe 7 提示 × 160 token → 汇总 → 停服）；
  检查工具 `tools/tp_bootstrap/r54_draft_quant_inspect.py`、dry-run `tools/tp_bootstrap/r54_draft_quant_dryrun.py`。

**账本（`--max-context 131072`、DFlash2 K=7 + vision + host KV 32 GiB，同一二进制）**：

| arm | shard 0 `weights+ctx` | shard 0 `free` | shard 1 |
|---|---:|---:|---:|
| 基线（gate/up q8） | 13462.6 | 102.0 | 12112.6 |
| gate/up q4 | **13012.6** | **552.0** | 12112.6 |

⇒ shard 0 上限 +450 MiB ≈ **+28.5k token**（16.125 KiB/token/card）。

**接受率（greedy，每臂独立冷启服）**：

| arm | K=7 drafted/accepted | K=7 acc | K=7 tok/s | K=2 drafted/accepted | K=2 acc | K=2 tok/s |
|---|---|---:|---:|---|---:|---:|
| 基线 | 2843 / 699 | 24.59% | 54.9 | 1012 / 603 | **59.58%** | 52.0 |
| gate/up q4 | 2826 / 700 | **24.77%** | 57.2 | 1035 / 591 | 57.10% | 51.8 |

（tok/s 为 7×160 token 的探针用时折算，含 prefill，±5% 噪声。）

**失败记录（重要）**：第一版把 `mlp/down` 一并降到 Q5，加载正常但 warmup 即报
`FATAL warmup failed | linear dynamic grouped conv add: invalid projection_weight`。根因：
`src/ops/wrapper/dynamic_grouped_conv.cpp:61-73` 的 `require_finish_projection_weight` 硬性要求
`Q8_G32_FP16 / RowSplit / group 32`——该 finish 把 down/output 与动态卷积 delta 融合，draft 的
`mlp/down`（5120×17408）与 `attention/output`（5120×4096）没有 Q5/Q4 变体。该中间 artifact 已删除。

**后续杠杆（未排期；按性价比排序）**：

- [x] **fused dynamic-conv finish 的 Q5 变体**（放开 `dynamic_grouped_conv.cpp` 的 Q8-only 校验）：
      已落地（r56，2026-09-23）。`mlp/down` −172 MiB、`attention/output` −42 MiB（5 层 draft 合计）。
      实现为「Q5 GEMM 物化 + 共享 finish kernel」路线：**复用已合规的 `select_q5_a16_launch` Q5 A16
      GEMM，无新 GEMM kernel**（原"需要新 kernel"预估偏高）——新增 q5 形状 `n5120_k4096`
      （`src/ops/linear/q5/shapes/n5120_k4096.cu`，selector 镜像 n5120_k17408）+ dispatch 注册；
      新增 `src/ops/dynamic_grouped_conv/q5/`（plan/materialized，路线名
      `dynamic_grouped_conv_add.q5.*.materialized_bf16`）；finish kernel 从 q8 materialized 抽出为共享
      `src/ops/dynamic_grouped_conv/dynamic_conv_finish.cu`（q8/q5 共用，算术逐位不变）。wrapper
      校验与非重叠检查按 qtype（Q8_G32_FP16 / Q5_G64_FP16 RowSplit，Q5 含 qhigh 高位面检查）分派；
      容量 API `linear_dynamic_grouped_conv_add_workspace_capacity_bytes` 增加 qtype 首参（对齐
      `linear_workspace_capacity_bytes` 惯例，`draft.cpp`/`dflash_round.cpp` 调用方跟进）；公共契约
      （`include/ninfer/ops/dynamic_grouped_conv.h`）同步双 codec 语义；bench 增加 `--qtype q8|q5`。
      **验证**：op oracle `ninfer_linear_dynamic_grouped_conv_add_test` 通过（4.38 s：Q8/Q5 ×
      C∈{4096,17408} 全 W/B 域 120 形状 × 图重放，FP64 oracle 按存储 scale 独立解码）；
      `ninfer_linear_q5_a16_test` 通过（3.11 s，新增 5120×4096 全路由边界 vs FP64 oracle）；
      WSL build_dyn 与 Windows build-win 编译通过。
      **实验件已转换**（2026-09-23，CONVERT_EXIT=0，285.8 s，1218 objects）：
      `D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_finish5.ninfer`（23,449.4 MiB，**−213.3 MiB** vs 基线
      23,662.7，与精确计算一致；恰 10 对象 `q5_g64_fp16`：`dflash2/layers/*/attention/output`
      [5120,4096] ×5、`dflash2/layers/*/mlp/down` [5120,17408] ×5，其余 draft tensor 不变）。
      override `tools/tp_bootstrap/r56_draft_finish_override.py`，转换脚本
      `tools/tp_bootstrap/r56_convert_finish5.ps1`（r54 配方，`--device cpu`）。
      **r56 实测**（2026-09-23，Windows 8099，K=7 贪心探针 7×160，2×5060 Ti，日志 `build-win/r56/`）：
      账本 shard 0 free **102.0 → 314.0 MiB**（weights+ctx 13462.6 → 13250.6 = −212.0 MiB 设备侧，
      工件 −213.3，1.3 为分配粒度吸收），shard 1 不变 810.0 ✓ 本地性成立；
      接受率 base **24.59%**（699/2843）→ finish5 **25.13%**（703/2798），**+0.54pp**（p0/p2/p3 与
      base 逐轮全同，p1/p4/p5/p6 文本轨迹漂移使分母变小；无下降迹象）；吞吐 56.4 → 56.2 tok/s（−0.4%）。
      确定性：base 臂跨重建逐字节复现 r55 基线（2843/699、文本 len 全同）——共享 finish 重构后
      Q8 路线零漂移。判定：**维持 opt-in**，升契约特性前仍欠 §3.7 末尾三件套验证。
      **已知 flake（既有问题，用户确认偶发多次遇到）**：首次 finish5 臂在 req#2 流中途概率性挂死
      （两卡 SM 100% 自旋等 handoff、/health 200、日志无报错；桌面进程已排除），重跑即过；
      挂死机理待专项排查（用户指示暂缓，日志留存 `build-win/r56/serve-finish5-hang1.log`）；**该专项已由 §3.8 接手**（2026-09-23 第二次复现后定案为设备侧自旋，处置见 §3.8）。
- [x] **`feature_projection` 新 profile**（5120×25600，132.8 MiB）：已落地（r55，2026-09-23）。
      更正：该 tensor 走 plain `linear` op（`project` → `ops::linear`），不是 `linear_add`。
      新增 `src/ops/linear/q5/shapes/n5120_k25600.cu`（selector 镜像 n5120_k17408：T=1 simt_r8_c4、
      T=2–6 ksplit、T≤24 simt_r8_c8、其余 mma_r64_c128）+ dispatch 表注册 + 合规测试用例
      （`tests/ops/linear/test_q5_a16.cpp` 5120×25600，seed 183U，FP64 oracle）。opt-in override
      `tools/tp_bootstrap/r55_draft_feature_proj_override.py`（`dflash2/feature_projection` → Q5）。
      收益精确值：该对象为 shard 0 本地全量（tp_split_spec 的 dflash2/* Replicated shards=0x1），
      Q8 132.81 MiB → Q5 82.03 MiB（row_split_k128_v1：400 组/行 × 42 B + scale），**−50.8 MiB**
      （原估 −63 偏高）。转换脚本 `tools/tp_bootstrap/r55_convert_featproj5.ps1`（r54 配方 +
      r55 override，`--device cpu`，无需 GPU）。WSL build_dyn 编译通过（BUILD_EXIT=0）。
      **实验件已转换**（2026-09-23，CONVERT_EXIT=0，299 s，1218 objects）：
      `D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_featproj5.ninfer`（23,611.9 MiB，
      实测 −50.8 MiB vs 基线，与精确计算一致；`dflash2/feature_projection` 已确认
      `q5_g64_fp16:[5120,25600]`，其余 draft tensor 保持 Q8）。A/B 基线：
      `D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2.ninfer`（23,662.7 MiB）。
      **r55 实测**（2026-09-23，Windows 8099，K=7 贪心探针 7×160，2×5060 Ti，日志 `build-win/r55/`）：
      op 合规 `ninfer_linear_q5_a16_test` 通过（5120×25600 全路由边界 vs FP64 oracle，2.9 s）；
      账本 shard 0 free **102.0 → 152.0 MiB**（weights+ctx 13462.6 → 13412.6，设备侧 −50.0 MiB，
      分配粒度吃掉 0.8），shard 1 不变 810.0 ✓ 本地性成立；接受率 base **24.59%**（699/2843）→
      featproj5 **24.40%**（696/2853），**−0.19pp**（与 r54 gate/up Q4 的 ±0.2pp 同带宽 ⇒ 不变）；
      吞吐 56.7 → 56.4 tok/s（−0.5%）。探针确定性：base 跨重建逐字节复现（2843/699、文本 len 全同）；
      改 draft 必然微漂文本（FP 平局随轮切移动，r54 gateup4 同现象：834→837、795→784；
      featproj5 漂 4/7 条），目标贪心流仍是验证 oracle。判定：**维持 opt-in**，
      升契约特性前仍欠 §3.7 末尾三件套验证。
- [ ] **selector codebook 量化**（242.5 MiB BF16，非 GEMM）：需要新的 selector 路径（codebook codec + 保持 top-k 域）。
- [ ] **全 draft NVFP4**（约 −0.95 GiB）：先给转换器加 BF16/FP8→NVFP4 方法（现只有 `import_encoded` 能产 NVFP4），
      再补上面三条的 NVFP4 profile；`linear_pair` 与 draft 版 `attn_input_proj` 仍是 Q8-only，全量化必须动它们。

**组合实验件（r57，2026-09-23）**：三个已验证量化合并为最终实验件
`D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer`（22,947.1 MiB，**−715.6 MiB** vs 基线
23,662.7，= 451.6+50.8+213.3 精确求和；21 对象：gate/up ×10 `q4_g64_fp16`、
feature_projection/down/output ×11 `q5_g64_fp16`；`attention/{query,key,value,context_*}`
[6144,5120]、draft 版 `attn_input_proj`/`linear_pair` 等仍 Q8——属杠杆#4 范围）。
override `tools/tp_bootstrap/r57_draft_all_override.py`（含计数校验）+
`tools/tp_bootstrap/r57_convert_draftall.ps1`（CONVERT_EXIT=0，290 s）。
**r57 实测**（Windows 8099，K=7 贪心探针 7×160，日志 `build-win/r56/`）：账本 shard 0
weights+ctx 13462.6 → **12748.6**（−714.0 MiB，粒度吸收 1.6；三杠杆单独测量之和 712.0 的 2 MiB
差为各自粒度取整方式不同），free 102.0 → **816.0 MiB**（≈ +45k token 上限）；shard 1 不变 810.0 ✓；
接受率 **25.3%**（706/2791）vs 基线 24.59%（无下降）；吞吐 56.9 tok/s（+0.9%）。
命名 `qwen3.8-27b-w4a4-w8a8-draftall`。

**升级为受契约保护的特性前必须补的验证（本次未做）**：

- [ ] DFlash2 三件套针对新 artifact 重基线：`ninfer_qwen3_5_tp2_dflash_append_test`（K=7/K=5 金值）、
      `ninfer_qwen3_5_tp2_dflash_solo_test`（digest）、`ninfer_qwen3_5_tp2_sessions_test`（保留断言）。
- [ ] 采样模式（temperature>0）接受率/吞吐 A/B：分布精确性由算法保证，但 q 变化会影响采样下的接受率。
- [ ] Q4 gate_up 在该形状上的 op oracle，以及「按存储 scale 独立解码」检查（AGENTS.md 的数值契约）。

---

### 3.8 TP-2 会合协议重构（2026-09-23；事故驱动，进行中）

**事故（第二次复现，且已定位到设备侧自旋）**：2026-09-23 21:02:56，Windows 3456，`qwen3.8-27b-w4a4-draftall`
（245760 / k8v4 / DFlash2 K=5 / 4 会话 host KV）。req#34 在 prefill 2,189 tok + decode 158 tok 后冻结：
两卡 SM 100%、显存 12,962/12,558 MiB 恒定、`/health` 200、host 恰好 1 个核连续跑满、日志零输出；
req#35–#39（客户端约每 5 分钟重试）被接受但永不推进 ⇒ 6 条请求被静默吞掉（`--max-pending-requests 16`）。
日志留档 `build-win/serve-hang-2026-09-23.log`（30,559 B），现场快照 `build-win/live-serve-snapshot.log`。
**杀进程后两卡仍报 100% / 2805 MHz / 22–25 W / 0 MiB 且无 compute 进程** ⇒ 卡住的是设备侧自旋内核
（`__nanosleep` 轮询的低功耗签名）。进程终止不会立刻回收：21:45 杀进程，21:47 两卡仍 100%/2805 MHz/~25 W，21:49 自行回落到 180 MHz/0%/4–7 W ⇒ 设备侧自旋的又一佐证；测量前先确认两卡回到 0%（本次已确认，无需重启）。

**与下午修复的关系（结论：不是漏打补丁）**：部署件 `C:\ninfer\ninfer-serve.exe` = `build-win/apps/ninfer-serve.exe`
（202,497,024 B，14:23:32），二进制内含 `NINFER_TP2_AR_WATCHDOG` 与 `graph queue[` 字面量 ⇒ 14:50–14:51
那批提交（`12760cb8` 看门狗 → `f8d33f13` cache-line 修复 → `fe96c13d` 图重放测试）在链接时已在树内；
14:24–14:30 的 r57e（图压力 60/60 PASS）与 r57f（serve 复检）跑的就是它。⇒ 本次是**同类现象的另一条路径**：
`src/core/tp/device_pair.cu:191-197` 只堵掉「两计数器同 cache line 互相写回覆盖」这一条漏 bump。

**三条设计缺陷（根因判断）**：

- **D1 会合标识＝两卡各自 RMW 的自增计数器**（`device_pair.cu:208`、`:116-119`）：把"两卡永久锁步"当成
  不可验证、不可恢复的分布式不变量。且本平台无法用原子修——`atomicAdd_system()` 需要
  `hostNativeAtomicSupported`，PCIe 消费卡没有（llama.cpp `ggml/src/ggml-cuda/allreduce.cu:56-58`）。
- **D2 等待严格相等且无上界**（`:153-158`）：唯一失败模式是活锁，且能带走整个服务。
- **D3 会合落在最不可控的一致性域**（两块 GPU 写同一块 mapped host memory），每 token 约 128 次。

**关键决策（对照 llama.cpp ar3-opt 与上游 `allreduce.cu`；细节见 `docs/tp2-dual-5060ti-llamacpp-notes.md`）**：

1. **借"谁来发号"，不借架构**。采用 host 权威 id，但**不**采用 llama.cpp 的图级 meta 架构
   （每 decode step 约 81 段 host lockstep）：同机 3 卡实测 44.5 tok/s vs 本引擎 2 卡约 100 tok/s，
   且违反本仓库 "Models own finite execution composition" 的 ownership。
2. **id 送达复用本引擎已有机制**：host 每轮写 pinned `base` → 图内 memcpy node 拷进 device 标量 →
   内核算 `id = base + call_index`（`call_index` 为捕获参数，跨 replay 恒定）。与 `valid_columns` 同路
   （`src/models/qwen3_5/execution/text.h:203-216`、`tp2_generation_core.cpp:998-1006`）。
   **必须走 memcpy node，不能按值当 kernel 实参**——llama.cpp 的形态一旦入图会重放冻结值，自旋条件被
   上一轮残留值满足 ⇒ 不等待直接读，静默算错（比挂死更难查）。
3. **大 payload 事件路径只作 A/B 候选**，不预设替换：sliced 已在链路地板（2.70 vs 2.66 ms）；上游
   copy-engine 固定开销约 80 µs 对 SM kernel stage A 约 30 µs，故其 1 MB 阈值。
4. **补两边都没有的**：有界自旋 + 失败语义 + 退化路径。partial 合并（`ggml-backend-meta.cpp:2126-2180`，
   已核实）BF16 下非结合、与逐位摘要契约冲突，本轮不做。

**改动清单**：

**Phase 1 —— 止血（不动协议，可独立上线）**
- [ ] `device_pair.cu`：自旋加 `clock64()` 上界（env `NINFER_TP2_AR_TIMEOUT_MS`，默认 2000，0=关），超时后
      `__threadfence_system()` + 置 mapped `stalled` 标志并让所有 block 退出；扩展现有 `ArWatchState`
      （`:236-244`）承载该标志。
- [ ] `allreduce`/`sendrecv` 返回状态；超时后 host 排空两条流、复位 arrival/order/parity，失败该请求并
      **毒化该会话上下文**（要求重新 prefill，避免用半写 KV 继续），服务保持存活。
- [ ] host 侧每轮不变式检查：`token_a_ == token_b_`（两个 mapped 计数器 host 可直读），不等即 dump 并失败。
- [ ] 验收：故障注入（人为让一侧少一次合算／丢一次发布）必须表现为"该请求 5xx + 服务存活 + 下一请求正常"；
      真实 agent 流量连续 ≥1 h 无永久挂死；prefill/decode A/B 回退 ≤1%。

**Phase 2 —— id 权威化（删掉整类 desync）** ✅ 已完成（2026-09-23）
- [x] `device_pair.{h,cu}`：删除 `token_host_/token_a_/token_b_`、`bump_ar_token`、`fuse_bump` 与
      `ar_size_keyed_`/`NINFER_TP2_AR_STRATEGY`。落地形态比原计划更严，三处偏离都是有意的：
      - **每个捕获图一条 id 通道**（`create_ar_channel()`，通道 k 占 `[(k+1)<<32, …)`），而不是"每设备一格 pinned cell"：
        每轮三个 window 各 launch 一次，共用一格会在上一个 launch 的 memcpy node 执行前就被下一次写掉而串值。
        每通道 = 一格 pinned cell + 每设备一个 device 标量（图内 memcpy node 的源与目标）。
      - **id 全程 64 位**（arrival/order 槽改 `unsigned long long`）：32 位截断会让不同通道的 id 段撞值，
        那样陈旧槽又能满足未来的自旋——正是要根除的那一类。
      - **eager 走 64 位 kernel 实参**（从 `1<<62` 起、独立区间），不用 mapped cell：eager 调用会背靠背入队，
        共用一格同样会读到后一个 id。内核只做 `token = base_dev ? *base_dev + call_index : value`，
      `call_index` 是捕获期常量，烘进实参是安全的（烘 id 不安全）。
      - 钩子为 `begin_capture(channel)/end_capture()/arm_round(channel)`：`end_capture` 自记 `calls`（少一处真相）；
        `ar_token_skew()` 删除（计数器概念消失）→ `ar_last_id()` 供错误消息与看门狗；
        `clear_ar_stall()` 只清 trip 标志——**id 永不重复 ⇒ 陈旧槽不可能满足未来自旋 ⇒ 停滞那一轮原地可恢复**，
        这是 Phase 2 相对 Phase 1 的实质收益。
- [x] `src/core/decode_graph.{h,cpp}`：**未改**（钩子由调用方显式调，未使用 `cudaStreamIsCapturing` 隐式判定）。
- [x] `src/runtime/engine/tp2_generation_core.{h,cpp}`：`WindowGraph` 增 `ar_channel`；三个捕获点接钩子
      （`capture_verify_graph`/`capture_decode_graph`/`capture_mtp_chain_graph`）；`launch_window_graph` 先 `arm_round`。
      9 个模型侧 `allreduce/sendrecv` 调用点确实**一行未改**。
- [x] 验收（2026-09-23）：
      `ninfer_tp_device_pair_test` PASS——含改造后的 `check_graph_queue`：**逐次重放换新操作数**（否则 id 复用根本测不出来，
      旧操作数会让陈旧槽照样"通过"）+ 每轮 `arm_round`；
      `tp2_dflash_solo`(solo digest)、`tp2_dflash_append`(K=7/K=5)、3× `linear_tp2_split` 全 PASS；
      `tp2_sessions` 仍以**改动前逐字节相同**的序列失败（既有 ulp 非契约，未被扰动）；
      服务级注入 3000：`HTTP 503 | TP-2 allreduce stalled at rendezvous id 4294968076`
      （`= (1<<32) + 1548` ⇒ 正是通道 1 的段，证明 base 确实经图内 memcpy node 送达）⇒ 下一条请求 HTTP 200；
      同轮 `decode 120.4 tok/s / dflash2 accepted 782/1,200 (65.2%)` 与 Phase 1 的 120.0/65.2% 完全一致（无性能回退）；
      看门狗 dump 同时出现 `12884914887`(通道 3) 与 `4611686018427390359`(eager `1<<62`+) ⇒ 多通道与双区间并存互不干扰。

**Phase 3 —— 大 payload 事件路径（A/B 决定）** ❌ 已否决（2026-09-23，实测定案）
- [x] 决定测量：**不写完整实现**，因为候选路径的**地板**已经超过现路径的**全部成本**。事件路径与 sliced 路径搬的是同样的字节
      （每设备 D2H 本地 delta + H2D 对端 delta，都过同一条 PCIe），所以先量 copy engine 在该 payload 形态下的天花板即可判定。
      探针：`NINFER_TP2_AR_COPY_BENCH=1 build-win/tests/ninfer_tp_device_pair_test.exe`
      （`check_copy_engine_ceiling`：按计划的 `chunk = clamp(nbytes/4, 512 KiB, 2 MiB)` 分块，**每设备两条 copy stream 让双向同时在飞**；
      串行单流版本实测 3.315 vs 双向 3.316 ms ⇒ copy engine 不会因双向并发变快，故该数字是可靠地板）。
      实测：**10 MiB → 3.32 ms/次（两向合计 6.33 GB/s）；24 MiB → 8.86 ms（5.68 GB/s）**。
      现 sliced 路径同一 10 MiB delta 实测 **2.70 ms**（≈7.4 GB/s，链路地板 2.66 ms 的 98.5%）。
      ⇒ 候选路径尚未计入 event 栅栏（约 80 µs/次固定）与 add 输入所需的额外设备往返，就已慢约 23%：
      "prefill 吞吐 ≥ 现 sliced 路径"不可能满足。第二个条件（删除 8-slice write-order 链）随之不成立——write-order 链正是现路径
      吃到链路地板的原因，删掉它只会更慢。
      范围说明：prefill 每层 payload 被 `--prefill-chunk 1024` 钉在 10 MiB，与提示长度无关，所以按 payload 形态做微观测量即可判定
      （现路径已在链路地板 1.5% 以内 ⇒ 端到端不可能有 >1.5% 的收益，而候选在传输层就差 23%）。
- [x] 结论：照计划 **"否则不做"**——不实现事件路径，不动 sliced 路径与 write-order 链。若日后换平台（PCIe 双向带宽更不对称、
      或 SM 访问 mapped host 更慢的机型）可重跑该探针复核。

**Phase 4 —— partial 合并可行性审计** ✅ 已审结（2026-09-23）：**形态不存在，门槛未触发，不做**
- [x] 审计范围：9 个模型侧调用点（`text.cpp:409/475/822/893/950/952/2288`、`text.h:496/509`）。判据：是否存在
      **两个独立 partial 各自归约后送进同一个 elementwise ADD**——只有这种形态才能压成"本地两 partial 先相加 → 一次 collective → 一次 add"。
      逐点分类：
      | 调用点 | 实际形态 | 可合并 |
      |---|---|---|
      | `text.h:496` mixer delta → residual_add | 行并行 partial → 一次 allreduce → `x += delta` | **否**：紧随的 MLP（`text.h:505`）读的是**就地更新后**的 `x`，两次 add 严格依赖 |
      | `text.h:509` MLP delta → residual_add | 同上 | **否**：同上；下一层 mixer 又依赖本层结果 |
      | `text.cpp:409 / 475 / 822 / 893 / 2288` | 一侧**零填充**、另一侧为本地数据拷贝 ⇒ 实为 replicate/gather（add-over-zeros） | **否**：不存在两个独立 partial；且 `__hadd(x,0)==x` 逐位成立，这里本就不是部分和归约点 |
      | `text.cpp:950 / 952` | 背靠背两次 `sendrecv`（ids + scores） | **否**：结果是 top-k **选择式合并**，不是 elementwise ADD |
- [x] 结论：**形态不存在** ⇒ 按门槛"存在才评估"不进入评估，**FP64 oracle 不触发**，无实现改动。附带记录（审计副产品，均判定不做）：
      - replicate 类 5 处改为 `sendrecv` 逐位等价（对端全零），但成本由搬运字节主导、内核内 add 几乎免费 ⇒ 无可测收益。
      - `950/952` 合并为一次需打包 I32/FP32 两种 dtype，省下的只是一次握手（每轮几十 µs / 37 ms 轮时 ⇒ ~0.1%）。
      - 推论：每 decode round 约 128 次 collective 里，96 次是 48 层 × 2 次**真**部分和归约，且每个结果都有依赖它的消费者
        ⇒ **在不改数学的前提下不可约减**；这也解释了为什么 transport 的优化空间只在每次 collective 的成本上（Phase 1–3 已封口）。

**进度**

- [x] **Phase 1a 传输层（2026-09-23）**：`device_pair.{h,cu}` 加 `%globaltimer` deadline（`NINFER_TP2_AR_TIMEOUT_MS`，默认 2000，0=关）、每侧 mapped `stall` 标志、
      协作式退出（thread 0 经 shared 发布、全 block 一起 return，避免 `__syncthreads` 死锁）、`ar_stalled()/ar_token_skew()/clear_ar_stall()`
      （复位=两计数器取大值 + 清零 arrival/order 槽）、以及测试用故障注入 `set_ar_fault_skip_peer_call()`。
      新用例 `tests/test_tp_device_pair.cpp::check_ar_timeout`（跳过一次对端启动）实测：**gave up after 2000 ms, token skew 1 →
      复位后逐位正确**，`ninfer_tp_device_pair_test` PASS（其余 allreduce/sendrecv/图重放/move 用例全过）。
      ⚠️ **1a 不能单独上线**：超时内核在 phase 3 之前返回，调用方只拿到本地偏和 ⇒ 会把"响亮挂死"变成"静默算错"；已与 1b 同批完成。
      另修一个 1a 自身的严重缺陷：内核入口先读**两侧** trip 标志，已 trip 立即退出——否则首次超时后 token 永久错位，同一轮后续 ~128 次合算会各烧 2 s（≈4 分钟/轮）才等到 host 检测。
- [x] **Phase 1b 引擎层（2026-09-23）**：`TP2GenerationCore::abort_if_ar_stalled()`（`tp2_generation_core.cpp`，声明在 `tp2_generation_core.h`）在**每轮收敛点**
      （`request.budget.remaining()`/output policy 之前、任何 token 送达客户端之前）与 prefill 首个 token 采样后检测；触发条件**只用 trip 标志**——
      plain decode 路径只排空 shard A，镜像 shard 落后 1~2 次合算属正常，用 token 偏差判会误报 503，偏差只写进错误消息当取证；
      触发后排空两卡 → `clear_ar_stall()` → `invalidate_host_checkpoints()` + `session_invalidate_active()` → `RequestError(Unavailable)` ⇒ **HTTP 503 + 下一请求重新 prefill**。
      故障注入入口：`NINFER_TP2_AR_FAULT_SKIP_PEER_CALL=<n>`（env，构造期读；诊断用）。
      **服务级验收**（新件已部署 `C:\ninfer\ninfer-serve.exe`，验收时 hash `001446D4…`；注入 3000）：`[ar-watch] calls=3114 … tok=[34054,34053]`（偏差 1）
      → `WARN req#2 failed during generation | HTTP 503 | service unavailable` → 错误体 `TP-2 allreduce stalled (token skew 1)…`
      → 复位后 dump `tok=[34054,34054]`、arrival/order 全 0 → **后续两条请求 1 s 内 HTTP 200**。Phase 1 完成。
      记录教训：本次验收原文只留在本节与当次会话——`C:\ninfer\serve-win.log` 被随后的干净重启覆盖；后续验收先 `--request-log-jsonl` 或先复制日志。
      **回归矩阵**（artifact `D:/LLM/qwen3_8_27b_w4a4_dflash2_draftall.ninfer`；每项都做了"改动后 / git-stash 改动前 HEAD"两次）：
      `ninfer_tp_device_pair_test`（含新注入用例）**PASS**；`ninfer_qwen3_5_tp2_dflash_solo_test`（solo digest）**PASS**；`ninfer_qwen3_5_tp2_dflash_append_test`（K=7/K=5 哈希）**PASS**；3× `ninfer_linear_tp2_split_*` **PASS**；
      `tp2_sessions_test`(plain recall 序列分叉)、`dflash2_real_test`(1.7 s cudaMalloc OOM)、`dflash_real_test`(0xc0000409 快败) 三项**在改动前 HEAD 上逐字节/同码复现** ⇒ 既有问题与本次无关
      （sessions 即 worklog 记的"热引擎从零重放不保证与冷 oracle 逐 token 相同"的 ulp 非契约；另两项为 artifact/环境类，docs 未收录这两个用例名）。
- [x] **Phase 2（2026-09-23）**：见上方 Phase 2 清单（id 权威化完成，已提交 `de74d966`/`b9c5d11c`/`634df875`）。
- [x] **Phase 3（2026-09-23）**：见上方 Phase 3 清单——**实测否决，不做**（copy engine 地板 3.32 ms vs 现路径 2.70 ms）。
- [x] **Phase 4（2026-09-23）**：见上方 Phase 4 清单——**形态不存在，门槛未触发，不做**（9 个调用点逐点分类，FP64 oracle 未触发）。
      **部署注意**：`build-win/apps/ninfer-serve.exe` **不会**自动同步到 `C:\ninfer\`，需手动复制；PE 时间戳使每次链接的 hash 都不同，比对以"部署件与 build-win 产出 `Get-FileHash` 相等"为准（本轮已同步，`108AD767…`）。
- [x] 停掉卡死进程并留档日志（2026-09-23；显存立即清零；残留自旋内核约 2 分钟后自行回落）
- [x] 测量前置确认：两卡空闲占用已归零（2026-09-23 21:49，无需重启）
- [ ] Phase 1 · Phase 2 · Phase 3 · Phase 4
