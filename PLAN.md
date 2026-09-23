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
      挂死机理待专项排查（用户指示暂缓，日志留存 `build-win/r56/serve-finish5-hang1.log`）。
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
