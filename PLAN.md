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
### 3.9 DFlash2 DraftTokens（K）扫描（2026-09-23；结论：**维持 K=5**）

**问题**：`qwen3_8_27b_w4a4_dflash2_draftall.ninfer` 在 MTP + DFlash2 路线下 `--draft-tokens` 取多少最优。
**方法**：`tools/tp_bootstrap/r59_draft_tokens_sweep.ps1`——每个 K 用**完全相同的生产参数**启动 ninfer-serve
（245760 / k8v4 / `--spec dflash2` / `--lm-head-draft` / temp 0.7 / top-k 20 / top-p 0.8 / vision / preserve-thinking），
只改 `--draft-tokens`；每 K 先 warmup 一次，再对 3 类 prompt（reason 推理链 / prose 描述文 / code 代码）各 5 次、`max_tokens=256`。
指标取自请求日志的 `decode X tok/s` 与 `dflash2 accepted a/b`；tokens/round = output ÷ (drafted/K)，rounds/s = decode ÷ tokens/round。

| K | decode tok/s | 接受率 | tokens/round | rounds/s | reason | prose | code |
|---|---|---|---|---|---|---|---|
| 3 | 83.8 | 58.4% | 2.78 | 30.2 | 93.4 | 76.0 | 82.2 |
| **5** | **96.4** | 48.5% | 3.45 | 27.9 | 113.9 | 81.6 | **93.6** |
| 7 | 96.1 | 36.9% | 3.62 | 26.6 | **117.8** | **83.2** | 87.4 |
| 9 | 89.1 | 30.1% | 3.74 | 23.8 | 113.1 | 72.9 | 81.3 |

（粗扫 8 点的其余值：K=1 → 60.3、K=11 → 84.6、K=13 → 85.4、K=15 → 81.8 tok/s，均低于 K=3~7 的平台。）

**K=6 补测 + 交错复核**（5/6/7 轮流启动 3 轮、每 K 共 18 样本；同一轮内比较可消掉机器状态漂移——K=5/7 都随机器转热从 c1 升到 c3）：

| K | 池化 decode | 接受率 | tokens/round | rounds/s（三轮） | 推理链 | 描述文 | 代码 |
|---|---|---|---|---|---|---|---|
| 5 | 93.3 | 46.3% | 3.34 | **27.9 / 28.0 / 27.9** | 112.1 | 77.4 | **90.4** |
| 6 | 89.7 | 40.9% | 3.48 | **25.7 / 25.7 / 25.7** | 111.8 | 75.7 | 81.6 |
| 7 | **96.9** | 37.4% | 3.65 | 26.6 / 26.4 / 26.6 | **123.2** | **78.2** | 89.4 |

同轮配对（c1/c2/c3）：K=5 88.1 / 93.8 / 98.0，K=6 87.6 / 94.0 / 87.5，K=7 94.4 / 96.8 / 99.6 ⇒ **K=7 三轮均不低于 K=5（+7% / +3% / +2%，差距随机器转热收窄）**，K=6 三轮均低于 K=7。

**K=6 是一个可复现的每轮成本凹陷**：它的 rounds/s 三轮都是 25.7（±0.0），既低于窗口更窄的 K=5（27.9）也低于窗口更宽的 K=7（26.6）；接受率与 tokens/round 都正常插在两者之间 ⇒ 不是噪声，也不在启动台账里（`[mem]` 的 `record` 随 K 单调 5.1/6.0/6.8 MiB，`[tp2-graph]` 三个 K 相同）。疑似按窗口宽度分桶/对齐导致的核函数边界效应，**待查**（线索：`NINFER_TP2_TIMING=1` 取逐相位每轮耗时）。

**机制**：K 增大提高 tokens/round（2.78 → 3.74），但接受率按位置快速衰减（58% → 30%），同时窗口变宽使每轮更贵
（rounds/s 30.2 → 23.8）⇒ 净收益在 K=5~7 到顶。
**结论**：**K=5 与 K=7 相差只在几个百分点内，K=7 略优**；**不要用 K=6**（被两侧同时支配）；K≤3 与 K≥9 明显更差。
- 交错配对里 K=7 三轮均 ≥ K=5（+7% / +3% / +2%，热机后收窄到 ~+1.6%），驱动项是**推理链 +10%**（123.2 vs 112.1）；
  code/tool-call 类反过来 K=5 略高（90.4 vs 89.4），描述文几乎相同。
- 单轮粗测时两者是 96.4 vs 96.1 的平局 ⇒ 差异 ≤ 几个百分点，在样本波动内；要拍板需要更长的配对测量。
- 倾向：客户流量以思考为主 ⇒ **建议把 `--draft-tokens` 从 5 调到 7**（也正是二进制对 dflash2 的默认值 7，与 `append K=7` 金值一致）；
  工具调用占比若上升，5 与 7 的差距会进一步缩小。
**边界**：单机单 artifact、3 类 prompt、temp 0.7、256-token 生成；接受率与内容强相关，换语料结论可能移动。
显存不是区分项（K=3→9 每卡 free 仅差 ~10 MiB；K=15 在 245760 下也能正常启动）。
### 3.10 纯 MTP 后端（`--spec mtp`）的 K 扫描（2026-09-23；结论：**K=3/4 平局，K=5 明显差；仍用 DFlash2**）

**方法**：同 §3.9 的生产配方与交错方法（3 轮 × K=3/4/5 × 每 K 18 样本），只把 `--spec` 换成 `mtp`：

| K | decode tok/s | 接受率 | tokens/round | rounds/s（三轮） | 推理链 | 描述文 | 代码 |
|---|---|---|---|---|---|---|---|
| 3 | 79.4 | 52.7% | 2.58 | **30.8 / 30.8 / 30.8** | 88.5 | **73.8** | **75.9** |
| 4 | **79.7** | 45.0% | 2.80 | 28.5 / 28.5 / 28.5 | **92.4** | 71.0 | 75.7 |
| 5 | 74.5 | 37.2% | 2.86 | 26.1 / 26.1 / 26.1 | 88.3 | 67.7 | 67.5 |

**结论**：MTP 的最优落在 **K=3~4（79.4 vs 79.7，统计平局）**，K=5 低 6.5%；分类看 K=4 优于长推理链（92.4 vs 88.5），K=3 优于描述文（73.8 vs 71.0），代码打平。
但**同 K 对比 DFlash2 全面落后**：K=5 时 74.5 vs 93.3（-20%），且 K 越大差距越大（MTP 的 tokens/round 2.58→2.86 只微增，DFlash2 是 3.34→3.65），
加上 MTP 上限 K=5 无法再换收益 ⇒ **继续用 `--spec dflash2`**。
**顺带澄清**：启动脚本第 60-61 行注释"TP-2 只支持 mtp（dflash/dflash2 会被拒绝）"**已过期**——本轮与 §3.9 都在 TP-2 上实测跑通了 dflash2。
**方法论旁证**：两个后端的 rounds/s 在各自三轮里都只变 ±0.1（MTP 30.8 / 28.5 / 26.1；DFlash2 27.9 / 25.7 / 26.6），
说明"每轮成本"是 K 的确定性函数，decode 的轮间波动来自接受率/内容，交错配对是这类比较的正确做法。

### 3.11 prefill 对比：DFlash2 K=7 / MTP K=3 / 无投机（2026-09-23；结论：**投机几乎不付 prefill 代价，DFlash2 慢 ~3-4%**）

**方法**：`tools/tp_bootstrap/r60_prefill_spec_compare.ps1`——三个配置用**同一套生产参数**，只差 spec 相关开关
（`--spec dflash2 --draft-tokens 7 --lm-head-draft` / `--spec mtp --draft-tokens 3 --lm-head-draft` / 三者都去掉）；
每个配置一个**新进程**（不跨配置复用），两个长度用**不同文本**，日志确认 `cache 0%`（6481 那 0.6% 是共用的 chat 模板前缀 ~39 tok）。
为排除"同一轮里越晚跑越热"的顺序伪影，除 2 轮正序（dflash2→mtp→plain）外又跑 1 轮**反序**（plain→mtp→dflash2）。

| prompt | DFlash2 K=7 | MTP K=3 | 无投机 | TTFT（dflash2 / mtp / plain） |
|---|---|---|---|---|
| ~1,673 tok | 1,473 tok/s | **1,593** | 1,567 | 1.10 / 1.05 / 1.10 s |
| ~6,481 tok | 1,743 tok/s | 1,807 | **1,817** | 3.70 / 3.60 / 3.55 s |

（每组 3 次测量，配置内重复性 ±1%；反序轮里 dflash2 排在最后仍最低 ⇒ 顺序伪影已排除。prompt 变长速率上升 = 每请求固定开销被摊薄。）

**结论**：开投机**几乎不付 prefill 代价**（1% 内），MTP 与无投机基本打平；**DFlash2 稳定慢 3-4%**（扣掉 0.6% 的前缀命中差后仍 ~3%）。
整请求上可忽略：decode 侧 DFlash2 领先 MTP ~20%（§3.10），远超这 3-4% ⇒ **生产继续 DFlash2（K=5 或 7）**。
**未定位**：DFlash2 这 3-4% 的确切来源（未做逐相位归因；要查可用 `NINFER_TP2_TIMING=1` 拆 prefill 每相位耗时）。

### 3.12 DFlash2 K=7 的 prefill 深度曲线（0 → 239k；2026-09-23）

**方法**：`tools/tp_bootstrap/r61_prefill_depth_curve.ps1`——单进程、单一逐步加长的文档（每次请求发送 1..i 段），
靠前缀复用让每步**只 prefill 新增段**，于是日志里 `prefill X tok/s (N tok)` 的 N 就是"新增 token 数"（已验证 `N = prompt − cache`），
即**该深度区间内的瞬时速率**。K=7 / dflash2 / 生产参数，共 17 步、每步 ~14.1k token，max_tokens=1 以隔离 prefill。

| 深度区间 | 速率 | 拟合 | 累计耗时 |
|---|---|---|---|
| 0-14,116 | 1,810 | 1,785 | 7.8 s |
| 13,312-28,180 | 1,610 | 1,590 | 17.0 s |
| 27,648-42,244 | 1,440 | 1,428 | 27.1 s |
| 41,984-56,308 | 1,310 | 1,297 | 38.1 s |
| 55,296-70,372 | 1,200 | 1,191 | 50.7 s |
| 69,632-84,436 | 1,090 | 1,098 | 64.3 s |
| 83,968-98,500 | 1,000 | 1,018 | 78.8 s |
| 98,304-112,564 | 931 | 949 | 94.1 s |
| 111,616-126,628 | 908 | 891 | 110.6 s |
| 125,952-140,693 | 822 | 838 | 128.5 s |
| 140,288-154,758 | 797 | 791 | 146.7 s |
| 154,624-168,823 | 738 | 749 | 166.0 s |
| 167,936-182,888 | 700 | 712 | 187.4 s |
| 182,272-196,953 | 684 | 678 | 208.9 s |
| 196,608-211,018 | 653 | 647 | 231.0 s |
| 210,944-225,083 | 622 | 618 | 253.7 s |
| 224,256-239,148 | 597 | 593 | 278.7 s |

**拟合**：每 token 成本 `1/rate = 5.25e-4 + 5.01e-9 · depth` 秒（17 点，全部误差 <3%）
⇒ 深度 0 外推 **1,905 tok/s**、245,760 处 **569 tok/s**；**满上下文时 attention 已占每 token 成本的 70%**（浅层由固定开销主导）。
**实用结论**：灌满 239k 上下文实际耗时 **278.7 s**（其中 248,072 token 被真正计算——比 239,148 多 3.7%，是 1024-chunk 检查点粒度导致的尾部重算），
有效速率 890 tok/s（首段 1,810 → 末段 597，降 3.0×）。
**复现与旁证**：独立重跑的前 16 点逐点吻合（±1%）；`cache` 数字显示前缀复用在 239k 深度仍然近乎完美（缓存边界 = 上一 prompt 减 ~1 个 chunk），
即上下文缓存环在满深度可用。

### 3.13 深上下文 prefill 的成本归因与优化空间（2026-09-23；分析，未改代码）

**问题**：§3.12 测到 prefill 速率 1,905 → 569 tok/s（3.3×），是否有优化空间。

**架构事实**（`D:/LLM/W4A16/NVFP4/W4A4/config.json`）：Qwen3.5 是**混合注意力**——64 层里只有 **16 层 full attention**
（24 Q / 4 KV / head_dim 256；TP-2 单卡 12 Q / 2 KV），另 **48 层 linear attention（GDN）成本与深度无关**。
⇒ 整条斜率只来自 16/64 层；若 64 层全是 full attention，同一深度处 ≈185 tok/s（当前 569，混合结构已买到 3.1×）。

**实测分解**（§3.12 拟合）：每 token 成本 = **0.525 ms（深度无关，1,905 tok/s 上限）+ 5.012e-9·d**；
在 245,760 处两者分别占 **30% / 70%** ⇒ 这 70% 就是"任何 attention 侧优化"的**收益预算**：完美内核最多 3.3×，现实约 1.3–1.8×。

**生产 prefill 走哪个内核**（已核实）：`causal_softmax_attention.cpp:382-385` 对 `(q_heads==12||16) && width<=16` 才给 ChunkedSmallT，
宽度 1024 的 prefill 因而落到 **Prompt** 路线 ⇒ `causal_attention_prompt_k8v4_kernel`（flash 风格：Br=64 行、Bc=64 KV 列、16 warps、
FP8 tensor core 做 QK^T、FP16 做 PV），但 **grid = (tokens/64, 12 Q 头)**：**每块只处理一个 Q 头**，而单卡只有 2 个 KV 头
⇒ 6 个 Q 头共享 1 个 KV 头却各自独立遍历同一段 KV（请求级 6× 冗余；L2 可能吸收一部分，需 profile 判定）。

**推算的交通量与效率**（以引擎自己的 `[mem]` 账本为锚：kv 3015 MiB/shard @245760 ⇒ 12.87 KB/token/shard，
其中 full-attention 部分 16 层×2 KV 头×256 dim×1.5 B = 12,288 B ✓）：
245,760 深度、1024-token chunk 下 KV 流量 ≈ 16 行 tile × 12 Q 头 × 384 B × 245,760 ≈ **290 GB/chunk/卡**；
实测 attention 时间 ≈1.26 s/chunk ⇒ **~230 GB/s（峰值的 51%）**；attention 计算 ≈4.95e13 FLOP/chunk ⇒ **~39 TFLOP/s（FP8 roofline 的 ~40%）**。
⇒ 两个 roofline 都未饱和，且存在 6× 的请求级冗余 ⇒ 确有空间。

**优化清单（按收益排序；均为估算，需 profile 定夺）**：
1. **GQA 感知的 prompt 内核**：让共享同一 KV 头的 Q 头共用一个 KV 数据流（如 6 头×16 行/块），流量 290 GB→48 GB/chunk；
   若转为带宽受限（~400 GB/s）⇒ attention 0.12 ms/token，245k 处端到端可达 **~1,550 tok/s（2.7×）**；
   若被 FP8 计算下限拦住（~60–70% roofline）⇒ 更现实 **~830–1,000 tok/s（1.5–1.8×）**。50k 处约 1.3×，10k 处约 1.05×。
2. **提高 Prompt 内核已达到的带宽/占用**（KV tile 双缓冲、prefetch、occupancy）：对 attention 项最多 ~2×。
3. **通用性说明**：收益集中在长上下文；**短 prompt 的瓶颈是 0.525 ms/token 的固定项**（48 层线性注意力 + MLP + W4A4 GEMM），那是另一条线。
4. **不是杠杆**：更小的 KV dtype（已 1.5 B/元素，且有质量代价）、TP 切分、GQA 比例（已 4 KV 头）、chunk 上限（`kPrefillChunkMaximum=1024` 是 TP-2 既定设计）。

**下一步（建议先做）**：`ncu` 抓 `causal_attention_prompt_k8v4_kernel`（深度 ≥100k 时）看 **DRAM 吞吐 / L2 命中率 / tensor pipe 利用率**
——据此判定 6× 冗余是否真落到 DRAM（决定做"减流量"还是"提占用"）。注意 worklog 已修过同族的 12 头路由漏洞（仅覆盖 width≤16 的 verify 窗口），
大 chunk 这条是遗留。

**补：PCIe 拓扑（Gen5 x8 + Gen4 x4）已考虑，且是"深度无关项"的主因（2026-09-23 交叉验证）**

worklog Round 35b/35c（`docs/tp2-dual-5060ti-worklog.md:636-691`）早已实测本机拓扑并定位：

- 卡 0（`pci 01:00.0`）**Gen5 x8，实测 ~20 GB/s**；卡 1（CUDA device 1，`pci 08:00.0`）**芯片组 Gen4 x4，实测 ~7 GB/s**（`nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max` 复现：device 0 = 5/x8，device 2 = 4/x4）。
- nsys 归因：`ar_inplace_bf16` 占内核总时间 **48%**（1024 宽块内约 60%）；每层 2 次 allreduce、载荷 [hidden 5120, T] bf16、写+读往返 = 4·hidden·T 字节/层/卡
  ⇒ **2.6 MB/token**，在 7 GB/s 上 = **0.368 ms/token ≈ 2.67k tok/s 的 AR 硬件上限**（实测边际斜率正好贴上、与块宽无关）。
- 关键否证：这条 Gen4 x4（走芯片组）链路**双向合计只有 ~7.9 GB/s**（不是每方向）；21 MB 往返物理下限 2.66 ms，8 切片有序流水已达 **2.704 ms = 98.5%**。
  （本机 Phase 3 独立复现同一数字：10 MiB 载荷 2.70 ms ✓）⇒ **AR 侧软件已无空间**；多 block AR 无差异、copy-engine 会合更慢且结果不一致，均已实测否决。

**与本节拟合的交叉验证**：我测的深度无关项 `a = 0.525 ms/token`（1024 块 = 538 ms）与 worklog 在 ~1.3k 深度的 nsys 归因（AR ~350 + MMA ~188 + 其它 ~35 = 573 ms）**一致**
⇒ 该固定项的构成是 **AR ~65% + MMA ~34% + 其它 ~6%**。注意 worklog 那次深度只有 ~1.3k，所以其归因自然只覆盖深度无关部分，与本节 attention 项不冲突。

**由此修正 §3.13 的措辞**：固定项**不是**"另一条可优化的计算线"，而是**已经贴死链路地板的 PCIe 开销**。
浅层上限 1,905 tok/s = 1/(0.34 + 0.18 + 0.03) ms，其中 0.34 ms/token 已无软件空间。两条杠杆：

| 杠杆 | 作用域 | 收益 | 代价/风险 |
|---|---|---|---|
| 卡 1 移到 CPU 直连 Gen5 x8 槽 | 硬件 | 浅/中上下文 **1,905 → ~3,390 tok/s（1.78×）**；245k 处 569→655（1.15×）；14k 处 1,810→2,740（1.51×） | 零代码；需主板有空槽 |
| AR 与 MMA 重叠（环形 staging + 事件同步 + 子块流水） | 软件 | 上限 ~1.5×（AR 隐藏后 ~4,760 tok/s 浅层上限） | 大重构，跨卡 rendezvous 死锁风险 |
| GQA 感知 prompt 内核（见上） | 软件 | 深上下文 1.5–1.8× | 内核工程 |

三者互补：硬件换浅层、内核换深层。P2P 不可用的根因也是这个槽位——`canAccessPeer=0`（worklog:200）、`nvidia-smi topo` 显示两卡为 **SYS**（不同 root complex），所以 peer 流量必须过主机桥，这也是 host-staging 设计的由来（`device_pair.h:114`）。

### 3.14 待实施的两个软件杠杆（2026-09-23；L1 已开工）

按 §3.13 的账，prefill 两个成本项各有杠杆，且**互补**（一个作用于 `b·d`，一个作用于 `a`）：

| # | 杠杆 | 作用项 | 预估收益 | 前置/风险 |
|---|---|---|---|---|
| **L1** | **GQA 感知 prompt 内核**：让共享同一 KV 头的 6 个 Q 头共用一份 KV 数据流（现在 `grid=(tokens/64, QHeads=12)`，每块只处理 1 个 Q 头，单卡 2 个 KV 头被各读 6 遍） | `b·d`：深上下文 attention，245k 处占 **70%** | 深上下文 **1.5–1.8×**（245k: 569 → ~830–1,000 tok/s） | 先过测量门确认 6× 冗余是否真落到 DRAM；只改调度与数据复用，**不动物理累加顺序 ⇒ 可与现内核逐位一致**。**（2026-09-23 测量门否证 ⇒ 已取消，见下）** |
| **L2** | **AR 与 MMA 重叠**（环形 staging + 事件同步 + 子块流水） | `a`：深度无关项，其中 AR ≈ 65% | 浅层上限 **~1.5×**（1,905 → ~4,760 tok/s 上限） | 跨卡 rendezvous 死锁风险，属大重构（worklog Round 35c 已评估） |

> 硬件杠杆（卡 1 从芯片组 Gen4 x4 槽移到 CPU 直连 Gen5 x8）不在表内：零代码但需主板槽位，收益 1.78×（浅/中上下文），与 L1/L2 可叠加。

**L1 执行顺序**：
1. **测量门（先做，未过则不动代码）**：ncu 抓 `causal_attention_prompt_k8v4_kernel`（生产几何 12q/2kv/d256/k8v4，KV 深度 ≥100k），看
   `dram__bytes_read.sum`、`lts__t_sector_hit_rate.pct`、tensor pipe 利用率、achieved occupancy。
   - L2 命中率高且 DRAM 已近峰值 ⇒ 6× 冗余已被 L2 吸收 ⇒ 杠杆变小，改走"提占用/降延迟"；
   - DRAM 流量 ≈ 6× 唯一 KV 字节且吞吐近峰值 ⇒ 冗余真实 ⇒ 进入第 2 步。
2. **设计**：块的行维度由「1 头 × 64 行」改为「6 头 × R 行」（R 由 smem 预算反推，Q/K/V/P/stats 各项已在 `prompt_k8v4.cuh` 里算过），
   KV tile 仍按 Bc=64 顺序流过；**每行看到的 KV tile 顺序与累加顺序不变** ⇒ 目标是与现内核逐位一致。
3. **实现**：只改 `prompt_k8v4`（生产 dtype）；其余四种 dtype 的 prompt 内核暂保持现路径（同构，后续同步）。
4. **验证**：OP 级 FP32/FP64 oracle（12/2/d256 深包络）+ 同步 `causal_attention_workspace_capacity_bytes` + 模型级金值（`tp2_dflash_solo`/`append` 摘要）+ 复跑 §3.12 深度曲线作性能对照。

**L1 测量门结论（2026-09-23，已否证 ⇒ 不改内核）**：用现成的 op 级基准 `bench/ops/causal_softmax_attention_bench`（本次为它补上生产分片几何 `d256-h12-kv2`——op 与测试本就支持，bench 落后）在 W=1024（Prompt 路线）、h12-kv2 下实测，245,760 深度单层：

| KV dtype | 耗时 | 达成 math | 唯一 KV 字节 |
|---|---|---|---|
| bf16 | 82,927 µs | 37.4 TFLOP/s | **482.0 MB** |
| int8 | 83,207 µs | 37.2 | 248.5 MB |
| fp8 | **73,782 µs** | 42.0 | 242.9 MB |
| nvfp4 | 82,270 µs | 37.7 | 135.6 MB |
| k8v4（生产） | 73,910 µs | 41.9 | 189.2 MB |

**KV 字节跨 3.5×，耗时只差 13%**：fp8 比 k8v4 多 28% 字节却同耗时；nvfp4 少 28% 字节反而慢 11%；bf16 字节 2.55× 只慢 12%
⇒ **内核不是 KV 带宽受限，而是计算/ALU 受限** ⇒ 去掉 6× 请求冗余不会有时间收益，**L1 按原设计取消**（内核代码未动，符合门禁约定）。

**顺带确定的天花板**：同机 T=1024 下 `ninfer_fp8_linear_add_bench`（A8, K=17408）**85.9 TFLOP/s**、`ninfer_q4_linear_swiglu_bench`（W4A4）**44.1 TFLOP/s**；
attention 的 QK^T 与 PV 各约 21 TFLOP/s（合计 41.9）。与纯 FP8 GEMM 比每个 attention GEMM 只到 24%，
但 k8v4 与 fp8 同耗时说明**差距不只在 MMA dtype**，而在内核结构（k8v4 的 PV 走 FP16、Hadamard 旋转/行缩放、softmax、smem 流量）。
⇒ 若仍要攻 attention，唯一形态是**计算路径效率**（FP8 PV + 减少 ALU 遍数），且**必须先有 profiler**：本机**未安装 Nsight Compute（ncu）**，黑盒计时已到极限。
预期：attention 完全消除 ⇒ 245k 处 569 → ~1,900 tok/s（3.3× 上限）；现实减半 ⇒ ~1.4×。

**基线留档**（W=1024、h12-kv2、k8v4、单层 median）：8,192 → 2,637 µs；32,768 → 9,924；65,536 → 19,630；131,072 → 39,602；245,760 → 73,880。
线性度极好，且 ×16 层 = 1.18 s/chunk，与 §3.12 模型级 1.26 s/chunk 只差 6% ⇒ 两条独立测量互证。
复跑：`ninfer_causal_softmax_attention_bench.exe --entry cached --geometry d256-h12-kv2 --kv-dtype k8v4 --tokens 1024 --context 8192,32768,65536,131072,245760 --execution eager --cache cold`。
**构建注意**：`-DNINFER_BUILD_BENCHMARKS=ON` 在 Windows 上会让默认目标失败（`bench/context_cost/model_context_fixture.cpp:400` 用了 MSVC 不支持的 `__int128`），
本次只按目标构建（`cmake --build build-win --target ninfer_causal_softmax_attention_bench`）；该缓存开关已复位为 OFF，不影响常规构建。

**L1 后续（2026-09-23）：profiler 已就绪，但被驱动权限挡住**

- 已安装 Nsight Compute 2025.4.1（winget `Nvidia.Nsight.Compute`），`ncu --version` 正常；但非管理员进程读性能计数器被拒：
  `ERR_NVGPUCTRPERM`（target device 0）。解锁二选一：① 用**管理员** PowerShell 运行 ncu；
  ② NVIDIA 控制面板 → Desktop → Enable Developer Settings → Developer → Manage GPU Performance Counters → 允许所有用户访问 → 应用（持久生效）。
- 无计数器可得的归因（已完成）：
  * **W 扫描**（深度 65,536、k8v4、h12-kv2）：W=256 → 31.7、512 → 41.8、1024 → 41.8、2048 → 45.4 TFLOP/s
    ⇒ **W=1024 已在平台区**；把 prefill chunk 上限从 1024 提到 2048 只值 ~9%，不是杠杆。
  * **同机 GEMM 天花板**（T=1024）：FP8 A8 **85.9**、W4A4 **44.1** TFLOP/s；attention 平台约 45（QK+PV 合计），
    而 QK^T / PV 各约 21–23 ⇒ QK^T(FP8) ≈ 其 dtype 天花板的 25%，PV(FP16) ≈ 50%。
  * 4-bit K 路径（nvfp4-K vs fp8-K）贵 +11%；bf16（完全无反量化）反而慢 12% ⇒ **MMA dtype 比反量化更主导**。
- **候选改动（待 profile 确认后再动，遵守门禁）**：把 prompt 内核的 KV tile 宽度 `Bc` 从 64 提到 128
  （QK^T 的 N 维翻倍、KV 循环次数减半；smem 由 ~99 KB 增至 ~165 KB，仍可单块容纳）。

**L1 实验记录（2026-09-23）：两条低成本路径均被硬件墙挡死，内核已回退**

ncu 对 `causal_attention_prompt_k8v4_kernel`（W=1024、65,536 深度、k8v4、h12-kv2）的实测：

| 指标 | 值 | 含义 |
|---|---|---|
| Compute (SM) Throughput | **63.54%** | 计算为主 |
| Tensor 管线 | **63.5%（最高）** | 张量管线是最忙资源 |
| **DRAM Throughput** | **1.09%** | KV 流量基本全被 L2 吸收 |
| L2 / L1 吞吐 | 10.10% / 40.43% | 都不是瓶颈 |
| Issue Slots Busy / IPC | 30.89% / 1.39 | 不是发射受限 |
| 最大 stall | 等数学管线 4.0/11.5 周期（34.4%） | 张量管线排队 |
| Occupancy | 理论=实际 **33.33%**（16/48 warp） | 1 block/SM |
| Block Limit | 寄存器 1 / smem 1 | 双重限制 |
| 寄存器/线程 · 动态 smem | **100** · 85.12 KB（配置 102.40 KB） | — |

⇒ **"减 6× KV 冗余"彻底证伪**（DRAM 1%）；真瓶颈是张量管线 + 33% 占用。

两次实验（均先过 oracle 正确性）：
1. **Warps 16→32**（同 smem 内翻倍 warp，寄存器上限随之压到 64）：正确性 **PASS**，性能 **−12%**（65k: 19.63→22.17 ms；245k: 73.88→82.43 ms）。
   原因结构性：1024 线程的块被 64K 寄存器文件硬顶在 ≤64 寄存器/线程，而内核需 ~100 ⇒ spilling。**死路**。
2. **Bc 64→128**（加宽 KV tile 改善 MMA 形状，smem 85,120→151,808 B）：**无法启动**，
   `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` 返回 `cudaErrorInvalidValue` ⇒ 消费级 Blackwell **单块动态 smem 上限 ~100 KB**。**死路**。

**结论**：该内核已贴住本机硬件边界（张量管线最忙、DRAM 空闲、占用被"寄存器文件 + 100 KB smem"同时锁死）。
剩余可做的都更大且收益有限：Br=32/256 线程的 tile 重构争取 2–4 block/SM（受寄存器支配）、PV 改 FP8、压缩非张量 FP32 指令（占指令 11%，ncu 提示 ~3.6%）。
上限：attention 项约 1.2–1.4× ⇒ 245k 端到端 ~1.1–1.3×、50k ~1.05–1.1×。
⇒ **性价比排序更明确：硬件移槽（浅/中 1.78×，零代码）> L2（AR/MMA 重叠，浅层 ~1.5×）> attention 重构（深上下文，收益最小、风险最大）。**

**顺带修复**：`bench/context_cost/model_context_fixture.cpp` 的 `__int128` 改为等价的 uint64 溢出检查（两个累加项各自不溢出，只需检查其和），
`-DNINFER_BUILD_BENCHMARKS=ON` 现在在 Windows 可正常构建（已用默认目标 `cmake --build build-win` 验证通过）。

---

## 4. DSH/agent-loop 前缀复用修复（2026-09-24，进行中）

**问题**：DSH 接入 ninfer 的 agent loop 里，某轮生成大 tool call（写大文件，24k token）后，
下一轮必须把这 24k token 重新 prefill（实测 req#3：prompt 34,050 / cache 9,216 / prefill 24,834 tok / TTFT 17.0s）。

**已确证（NINFER_TP2_REUSE_TRACE=1 / NINFER_TP2_SESSION_TRACE=1 实测）**：
1. 引擎保存了上一轮完整历史（`cached` = prompt+generated），但 `shared` 等于上一轮 **prompt** 长度
   （332/331/315 三例），即**第一个生成 token 就不匹配** → 只能从上一轮 prompt 末尾重启。
2. 用 `/v1/chat/completions` 原样回放 `reasoning_content` 复现同样结果 → 不是 DSH 特有。
3. artifact 内嵌模板对 `reasoning_content` 做 `|trim`，把模型生成的首个空格吃掉（实测
   `"ABCD"` 与 `"  ABCD  "` 渲染出的 prompt 长度完全相同）；而模型 reasoning 首个 token 带前导空格，
   因此任何文本回放都无法逐 token 复现。
4. TP-2 路线自建 frontend 时**漏传 `chat_template_path`**（`tp2_generation_core.cpp:472-480`），
   所以 `--chat-template` 在该路线被静默忽略（实测 developer 角色仍报 `artifact:chat_template.jinja`）。
5. 复用扫描里"对齐边界优先"会遮蔽更深的未对齐 live frontier，只有在 `reuse == 0` 时才走兜底；
   而 DFlash2 的兜底带代价门控（放弃 draft ≈ 3× 解码变慢 vs 省下的 prefill）。

**结果（2026-09-24）**：
- [x] A `tp2_generation_core.cpp`：TP-2 frontend 透传 `chat_template_path`。实测：部署新件并带
      `--chat-template` 重启后，`developer` 角色由 400 变 200（覆盖真正生效）。
- [x] C `tp2_generation_core.cpp`：兜底边界扫描改为无条件第二轮，门控改用"边际节省"
      （`position - reuse`），对齐边界仍是最低优先级。`dflash2`/`mtp` 两路 `ninfer_qwen3_5_tp2_sessions_test` PASS。
- [x] D `tests/models/qwen3_5/test_tp2_sessions.cpp`：新增常驻会话连续两轮用例（1000 token prompt；
      二轮期望 plain/mtp = 1007、dflash2 = 768）。没有 C 时 plain/mtp 会停在 768，用例能抓住该改动。
- [x] E 构建（`nm`/ninja，测试目标 + `ninfer-serve` 并同步 `C:\ninfer\ninfer-serve.exe`）+ 测试 +
      在线复测；证据：`profiles/sessions-{dflash2,mtp,plain-trace,all}.log`、`profiles/reuse-trace*.log`。
- [x] F 文档：`docs/tp2-dual-5060ti.md` 增补常驻连续两轮、门控与 `--chat-template` 说明。
- [~] B 模板 `|trim` → `rstrip('\n')`：**已回退（实测否定）**。覆盖为维护模板后原样回放
      `reasoning_content` 仍得 `shared == 上一轮 prompt 长度`，与 artifact 模板结果一致。

**残留根因（新发现，未修）**：文本协议回放无法逐 token 复现。`max_tokens=1` 时实测
`completion_tokens=1 / reasoning_tokens=0 / reasoning_content 为空`；k=2 得 `" need"`、k=3 得
`" need answer"`，而回放一律停在 `shared=72`（上一轮 prompt 长度）。即**第一个生成 token 不在已发布的
reasoning 文本里**（或回放重分词与采样边界不同），因此 DFlash2 上"复用上一轮回答尾部"对任何文本客户端
都不可达。再叠加门控（放弃 draft ≈ 3× 解码 vs 省下 prefill），32k 输出预算下重算 24.8k token（17 s）
是代价模型下的**理性选择**。

**可选杠杆**：客户端输出预算（`remaining` 需 < 节省量/16 ≈ 1.5k token 才触发复用）；让 DFlash2 在
未对齐复用下保持 draft（需要先解决 `extent=0` 钳位轮的可复现性）；或走携带 token id 的续写协议。

**既有失败（与本次改动无关）**：`NINFER_TEST_ROUTE=plain` 在 `shared_a_first` 处 `reuse=0/src=none`
的从零 walk 与冷 oracle 首 token 不同（`[2752 13 198 …]` vs `[365 2798 349 …]`）。trace 证明该场景及
其之前所有复用决策（0/71/512）与测试期望一致，新扫描路径在 `shared_prefix=0` 时不可能取任何边界。

## 5. TP-2 首个生成 token 不发布（2026-09-24，已修）

**现象**：`max_tokens=1` 时 `completion_tokens=1 / reasoning_tokens=0 / 无文本`；`max_tokens=k` 只发布 k−1 个 token
的文本；逐字节回放上一轮回答（chat-completions 原样回传 `reasoning_content`+`content`，带/不带 tools、换模板都一样）时
`shared` 永远等于上一轮 prompt 长度（实测 67 / 306 / 72）。于是 TP-2 上"复用上一轮回答尾部"对任何文本客户端都不可达，
`cache` 在服务日志里恒为 `floor(上一轮 prompt/1024)×1024`。

**根因**：`src/runtime/engine/tp2_generation_core.cpp` 的 prefill 末尾用 `request.generated.push_back(first)` 直接落账，
**没有经过输出会话**（`preview_model`/`commit_preview`）。单卡路线会把 prefill 采样的首 token 经
`commit_pending`→`preview_model` 提交（`src/runtime/engine/engine_core.h`），TP-2 是唯一漏掉的路线；因此
`generated`（KV/状态/会话目录索引的序列）比"已发布文本"多一个 token。

**修复**：prefill 采样后 `preview_model({first}, budget.remaining(), limit_reason())` → `budget.commit` →
`publish_preview(false)`；若该决策已终止请求（stop token 或预算用尽）则记入 `first_token_finish` 并跳过 decode 循环。
这也顺带修掉"首 token 是 stop token 时引擎仍继续生成"和首 token 未受语法约束（前者已修，后者记录为独立遗留项）。

**验证**（证据：`C:\ninfer\serve-win.log` 的 `[tp2-reuse]` trace、`profiles/fixtok-*.log`）：
- `max_tokens=1` → `reasoning_tokens=1`、`R="We"`（修复前 0 / 空）。
- 回放 40 token 回答：`shared=107`（= 67 prompt + 40 生成；修复前 67）。
- 回放 1400 token 回答：`reuse=1470 src=live` = 整个上一轮回答（DFlash2 按门控对该 1-token 请求弃稿，代价可忽略）。
- `ninfer_qwen3_5_tp2_sessions_test`：dflash2 / mtp 均 PASS（既有断言无回归）。
- 新增用例 `check_first_token_published`：thinking 开、输出预算 1 token 时 `reasoning_tokens` 必须为 1。

**残留（未修）**：门控仍按 `max(reuse_grid, 16 × 剩余预算)` 决定是否取未对齐边界，所以 DSH 的 32,768 预算下长回答仍会整段重算
（代价模型下的理性选择）；prefill 的首个 sample 仍未应用工具语法掩码（TP-2 prefill 与 decode 路径的不对称，独立遗留项）。

**附带观察（未修，独立）**：在服务实例上以 `temperature: 0` 连发同一请求 3 次，前 4 个 token 出现两种结果
（`"The user wants me"` / `"We need to respond"`）。`src/serve/translate.cpp:38-88` 会把请求里的 temperature 覆盖到服务默认值上，
所以温度确实被应用；DFlash2 的稀疏接受（`speculative_accept_sparse_drafts`）配合每次请求随机 seed（`translate.cpp:51-57`）
使解码在温度 0 下也不保证逐位 argmax——这正是 `--greedy`（"force temperature 0 (exact argmax)"）存在的原因。
需要可复现 A/B 时应加 `--greedy`；这条独立于本次修复，未改动。

## 6. 复用门控改按"会话自身轮次"定价（2026-09-24，已改）

**问题**：门控 `position - reuse <= max(reuse_grid, 16 × request.budget.remaining())` 里的 budget 是客户端设的上限
（DSH 固定 32,768）。预算 ≥ 16K 时 `16 × budget >= max_context`，未对齐回退分支**在数学上不可达**——而它是尾部复用
的唯一入口（网格上没有任何 decode 检查点），也就是说该机器生产配置下永远不会触发，即使节省量是整个上下文。

**改动**：
- `SessionEntry` 增 `generated_tokens_total` / `generated_turns`；每轮结束后按 `request.generated.size()` 累加
  （只统计生成 >=1 token 的轮次，保证均值非零）。
- 门控改用 `expected = min(budget, total/turns)`；没有已完成轮次（首轮请求、或关闭 retention）退回 budget（原最坏情况）。
- 语义上只放宽（`min(budget, mean) <= budget`），任何原先接受的边界仍被接受。

**验证**：
- 测试 `ninfer_qwen3_5_tp2_sessions_test`：dflash2 / mtp EXIT 0。新增场景 = 5 短轮 + 1 长轮（60 token）+ 1024 预算
  续轮，期望复用直达 frontier；另加两条场景自检，若某天不再落在 take 分支会带数字明确失败。证据 `profiles/gate-{dflash2,mtp}.log`。
- 在线（serve + trace）：20 短轮（均值 2）+ 1 个 2000-token 长轮 → 续轮 **budget 32,768** 下 `reuse=2442 src=live`、
  `cache 2,442 (99.3%)`、**TTFT 57.9 ms**（prefill 仅 17 token）；旧规则阈值 16 x 32,768 = 524,288 必然跳过。
  同一请求按门控弃稿，decode 22.5 tok/s、输出 37 token ⇒ 代价约 1.2 s < 省下约 1.6 s，本注是净赚。

**已知边界（故意保留）**：定价用的均值包含突发轮，所以"突发之后紧接的短轮"能否复用取决于该会话的均值——
均值 > 节省量/16 时仍然跳过（风险中性结论，不是 bug）。当前 DSH 会话均值约 1,735，其 24.8k 尾部仍会被跳过；
要翻转只能把比率 16 做成可配选项，或走第 3 步（未对齐复用下保住 masked draft）。
