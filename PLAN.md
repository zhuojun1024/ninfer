# NInfer TP-2 计划（2× RTX 5060 Ti · Qwen3.8-27B NVFP4）

> **唯一的活动计划**，也是跨上下文压缩的持久记忆。整理：2026-09-27（整理前全文入 worklog，本文件只保留
> 关键信息与真正要处理的任务）。
> - 完整历史：`docs/tp2-dual-5060ti-worklog.md` —— 含五份 PLAN.md 全文逐字归档（2026-09-21 上游
>   cherry-pick 重规划版；2026-09-22 交付收尾版；2026-09-24 会合协议重构后版；2026-09-25 r67–r70 草稿量化
>   与传输/编排工作后版；2026-09-27 本次整理前全文，含上游性能项移植、全架构精读、FP8 PV 与跨 tile 相位
>   流水两轮已关闭项目）。
> - 交付说明、推荐配置与实测数据：`docs/tp2-dual-5060ti.md`；Windows 原生移植：`docs/windows.md`。
> - 定案决策速查（否决方案、根因、方法论规则、worklog 指针）：`docs/tp2-decisions.md` —— 重复调研已定案
>   话题前必读。

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

## 2. 已关闭项目（勿重开）

> 细节见 `docs/tp2-dual-5060ti-worklog.md` 的 2026-09-27 全文归档与 `docs/tp2-decisions.md`；本文件不再展开。

- **上游 ninfer-all 性能项移植**：per-device `cudaFuncSetAttribute`（10 处函数局部 static + 4 处重复实现
  收敛）与 170 SM 硬编码（3 处改运行时设备 SM 数）已落地；3a 关闭 RDC 实测不支持已回退；5b GDN record
  窗口 staging 已移植（Op 级 −22~27%，端到端测不出）；5a small-T tensor-core 经复核只覆盖 q4/q5，不适用
  于 TP-2 主线；项目 4（把要流的 FP8 张量改 NVFP4/Q4）由用户决定放弃。剩余项见 §3.8。
- **全架构精读**：唯一「确认 latent bug」（`zero_pages` HeadMajor 越界写）经字节级复核**撤销**（`ne[0]`
  最内层，两种 plane order 都正确）；全架构精读核实旧 §3.3 的已知缺陷 ①②③④⑧ 当前代码均已修复；host-staging allreduce 性能
  悬崖已由 S1+S2 落地（回退快 5–8×、`kInKernelArBytes` 48 MiB、chunk 钳位由 staging 容量反推）。余下
  架构隐患见 §3.7。
- **FP8 PV（2026-09-27）**：实现通过全部质量门（短 prompt Δ=+0.29pp、深上下文 Δ=−0.88pp，均在噪声内，op
  级输出扰动约 2×），但 op 级只有 1.176×（目标 ≥1.3×）⇒ 按用户预授权规则回退，fp8 判据恢复 1.2e-2；
  补丁存仓库根 `PLAN-fp8pv-v1b.patch`。**关键产出**：barrier stall 34.2% + math pipe throttle 25.2% ⇒
  余量在相位结构而非 PV 的 dtype。
- **跨 tile 相位流水（2026-09-27）**：G0 证实 barrier 在回退后的 FP16-PV 基线上仍是第二 stall（26.3%，
  其中 97% 落在两个全块 `__syncthreads()`）；但两种常驻角色实现都回退到 0.71–0.80×——整块 PV 累加器
  16,384 个寄存器 = SM 寄存器文件的 25%，拆 warp 必然撞 128 寄存器/线程墙或 producer 工作集墙 ⇒ 当前
  「8 producer ∥ 8 Vdeq → 16-warp PV」packing 在 1 block/SM 包络下已最优。**CLOSED，不再重开**，除非
  执行包络改变（如 Br=32 把 acc/线程降到 16，或出现 >100 KB 块级 smem 预算）。
- **TP-2 跨会话稳定块复用：浅召回吞掉真实分歧锚（2026-09-27 修复 + 回归场景）**：故障现象是「新会话系统
  提示词缓存不生效」。根因链：DSH 每个新会话并行发 title 请求，title 与长会话只共享 thinking 前导（≈41
  tok），于是长会话被留下一个浅镜像（anchor=33），它成为后续主请求唯一的可召回边界；而 `session_restore`
  把被召回 entry 的 `tokens` 截断到召回边界、`cached_prompt_tokens_` 随之缩短 ⇒ `shared_prefix` 被压在
  边界上、`anchor_position > reuse` 永不成立、真实分歧点（系统提示词末尾 ≈9,292）从不写锚 ⇒ 前三个新会话
  各付一次全量 prefill。修复（`src/runtime/engine/tp2_generation_core.{h,cpp}`）：召回不再截断历史
  （`frontier` 只管设备 KV 覆盖范围）、新增 `SessionEntry::host_kv_end` 记录 slab 有效范围（浅召回不缩短
  slab）、`session_capture_shared_state` 允许写仍持有该前缀 slab 的 device-resident entry、锚归属从「被换出的
  上一个会话」改为「复用扫描比对的 entry」（`anchor_session_`）、store 时丢弃 slab 已覆盖不到的旧镜像。
  证据：`ninfer_qwen3_5_tp2_sessions_test` 新增「浅镜像召回 → 稳定块末尾锚」场景，`dflash2`/`mtp` 路线在
  `qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer` 上全绿——W1 只复用 8（浅镜像）而 `shared=500`，同一次 walk
  把 492 锚进被召回 entry 自己的 slab（`[tp2-session] anchor entry=0 position=492 resident=1 tokens=636`），
  W2 复用 492；同场景在修复前 `tokens=8`/`cached=8`、W2 复用 8（A/B 已复现，见 23:45 那次运行）。
  注：`plain` 路线仍停在 §3.3 既有的「shared system prompt 首采样分叉」，与本次改动无关（修复前后逐字节
  相同的失败）。
- **TP-2 跨会话稳定块复用：第二次会话仍慢（prompt 自带块边界锚，2026-09-28 修复 + 服务端复现）**：上一轮的
  「浅召回锚」只覆盖「第二个之后」——第一个共享稳定块的会话本身就是必须走完块的那次，观察永远晚一步，所以用户
  日志里三个主请求仍是 `cache 33`、TTFT 6.2s。本轮两处修复：
  1. **锚归属**（`tp2_generation_core.cpp` 的 `execute_walk`）：不再只认复用扫描比对的 entry，改为扫描所有
     host-resident entry，选与入站 prompt 共享最深前缀、且 slab 覆盖 `shared - margin`（`host_kv_end` 校验）
     的那个，排除 `anchor_session_` 自身；浅镜像落在 title entry、深历史在主 entry 时，锚因此落到深 entry 上。
  2. **块边界由 prompt 自带**：`Frontend::prepare_context_cache` 无条件把 leading instruction block 的 frontier
     写进新字段 `PreparedContextCache::leading_instruction_frontier`（与 `allow_engine_automatic_shared_prefixes`
     无关，单卡上下文缓存的 mark 策略不变）；TP-2 core 在 prefill 中把 `frontier - margin` 强制为 chunk 边界并写
     入新增的 block ring slot（`kReuseBlockCheckpointCount = 1`，与 divergence slot 一样在 `--host-state-slots`
     之外），记录 `block_anchor_position_/block_anchor_prefill_id_`；会话换出时（`session_store_active`）按
     position+prefill_id 找到该 checkpoint，把 state（含 DFlash2 draft 镜像）拷进该 entry 的 `host_shared_state`，
     成为下一次同族会话的可召回边界（`host_shared_end`）。
  3. 证据：服务端复现（`build-win/apps/ninfer-serve.exe`，`--max-context 65536 --host-state-slots 32`，swift15
     artifact，`--prefill-chunk 1024`，三个会话各带一个 title 请求）：主请求 1 = 7,037 tok 全量、TTFT 4.4s；主请求 2
     复用 **6,145/7,039 (87.3%)、TTFT 817ms**（锚点落在 1024 对齐的 chunk 边界上，因此冻结的是「从头走也会得到
     的状态」，代价是最多一个 chunk 的复用深度）；主请求 3 复用 7,020 (99.8%)、TTFT 172ms（走上一轮 walk 留下的
     观察式锚）。轨迹：`[tp2-session] anchor entry=2 position=6145`（换出时转移）→
     `[tp2-reuse] prompt=7039 shared=7028 -> reuse=6145 src=live`。回归：`ninfer_qwen3_5_tp2_sessions_test` 新增「prompt 自带块边界 → 换出转移 → 同族会话复用 ≥90%」场景，`dflash2`/`mtp`
     全绿，`plain` 仍停在既有的首采样分叉（与本次无关）。
  4. 运维：用户的 `--max-context 262144` 把显存吃到 `free 0.0 MiB`；本机当前（桌面/浏览器占 GPU）连**旧二进制**
     同样启动 OOM（把 `--host-state-slots` 降到 31、使 pinned 与旧版一致也照样 OOM），与本改动无关，需要时降到
     `--max-context 131072` 或释放 GPU 占用。
  5. **短会话不占名额（2026-09-28，接续修复）**：DSH 每个会话会产生两个 entry（title ~160 tok + main ~10k tok），
     `--max-private-continuations 3` 实际只留得住约一个会话，于是会话 A 的续写（req#7）`cache 0`。新增
     `ContextCacheOptions::session_retention_floor_tokens`（引擎默认 0 = 全留）+ 服务端默认 2048 +
     `--session-retention-floor` 覆盖；`session_publish` 对短于阈值的历史不建/不更新 entry（请求照常服务）。
     坑：TP-2 归一化在 `model_instance.cpp` 里**重建** `ContextCacheOptions` 白名单字段，新字段当时被丢掉
     （表现为规则不生效）——已补进该初始化列表。验证（全量重建 + 服务端 A/B）：warmup/title 请求后
     `entries=0`（不占名额），topic 请求 `reuse=3` 走 ring；req#4 `cache 5,121/5,609`(91.3%)/TTFT 468ms、
     req#6 `cache 5,590/5,606`(99.7%)/TTFT 126ms。另：本仓库增量依赖扫描不可靠，改 `include/ninfer/types.h`
     或引擎头后必须 `--clean-first` 全量重建（本轮两次假故障均由此引起）。
  6. **会话 entry 生命周期（2026-09-28，同轮修复）**：第 5 条的阈值规则去掉了每个新会话旁边那条 title entry，
     而"驻留 entry 原地更新"分支原本依赖它当牺牲品——于是下一条会话直接覆盖上一条会话的 entry，A 的历史被
     B、C 依次抹掉（服务端复现：`entries=1`、回到 A `reuse=0`）。修三处：① `session_publish` 只在
     `history` 以上一条 entry 的 tokens 为前缀（同一会话的延伸）时才原地更新，否则新建 entry；② 新增
     `mark_device_resident` 强制"最多一条 entry 声称驻留"（此前两条同时驻留 → 淘汰扫描只能选到那条刚冻结
     边界的 entry，引擎测试因此红）；③ `session_capture_shared_state` 成功后刷新 `lru_clock`（写入即使用，
     刚冻结的边界不会被同轮淘汰）。验证：`dflash2`/`mtp` 全绿；服务端 A→B→C→回到 A 复现 `entries` 0→1→2、
     回到 A `reuse=5590/5625 (99.4%)`（修复前 `reuse=0` 全量 prefill）。

- **Vision 聚合上限 32,768 → 131,072 merged tokens（2026-09-28，用户要求）**：现象是用户 1080p 截图会话在
  「历史已有 16 张 + 本轮新增 8 张 = 24 张」时被 `HTTP 400 media_budget_exceeded: vision budget exceeded` 拒绝。
  根因：整请求聚合上界 `kMaximumPromptVisionTokens`（`src/models/qwen3_5/frontend/prepared_prompt.h`）为
  32,768 merged token（= 131,072 raw patch = 33.5 MP 对齐像素），而客户端每轮把整个会话历史重发、历史里的图片每轮
  重新计入 ⇒ 实际等价于「每会话 1080p 最多 16 张 / 720p 最多 37 张」，且超限后该会话后续请求全部 400。
  改动只抬聚合上界（单项 ceiling 16,384 与编码 workspace 规划不变 ⇒ **设备显存占用不变**）：
  `kMaximumPromptVisionTokens = 131'072` 并把注释写明它只决定「保留 patch 预算 / 媒体 live 下限 / 准入检查」；
  同步 `docs/maintainer/qwen3_5-model.md`（524,288 raw patch / 131,072 merged token）、`docs/serving.md`
  （新 envelope + `--media-live-mib` 需 ≥ 12,288 B/merged token = 1.5 GiB，默认 2048 满足）、
  `tests/test_request_log.cpp` 的 fixture。证据：`--clean-first` 全量重建（769 步）后
  `ninfer_request_log_test` / `ninfer_qwen3_5_frontend_test` / `ninfer_public_api_test` 通过；
  `ninfer_qwen3_5_vision_workspace_test` 在本机 16 GB 卡 `cudaMalloc` OOM（worklog 已记「本机不适用」，且其断言
  只涉单项上界，与本次改动无关）；端到端（`build-win/apps/ninfer-serve.exe`，1920×1080 图 = **2,040** merged
  token/张）：24 张 → HTTP 200（prompt 49,022）、64 张 → HTTP 200（prompt 130,702）、65 张 → HTTP 400
  `media_budget_exceeded`（"vision raw patches exceed processor budget"）⇒ 边界精确落在 131,072。
  代价：host 侧媒体 live 下限 384 MiB → 1.5 GiB、单请求最坏保留 1.5 GiB BF16 patch（2 GiB live 预算里 media cache
  由 ~1 GiB 缩到 ~512 MiB）；KV/上下文仍由 `--max-context` 决定。已部署 `C:\ninfer\ninfer-serve.exe`
  （SHA256 `29D8758E548E581B4BAF38EDD6C56E5451F43CDC3E2F1615EA6DF4523708F2C1`，与 `build-win` 产物一致）。

---

## 3. 待处理任务

> 按优先级/可执行性排列；已关闭项目见 §2。

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
  的 chunk-split 不变量在 N=2/N=4 仍 exact）。三条前置证据：2026-09-27 归档 §5.4 的 divisor 臂（AR 是
  严格串行的 48% 块时）、`tools/tp_bootstrap/bench_ar_overlap.cu`（AR 不被占满 SM 的 compute 饿死，引擎粒度
  下只膨胀 7–18%）、以及 Round 12 的 −5% 只是其 host 侧调度形态的问题。**剩余**：默认值/N 的选择、与 decode
  路径无关、以及跨卡 rendezvous 的失效语义（见 §3.6）。
- 若仍攻 attention：FP8 PV 与跨 tile 相位流水均已试并关闭（见 §2）⇒ 现执行包络下无未试的 attention
  杠杆；继续需要先改包络（更大 smem 预算 / 不同 Br）或解锁 profiler 权限（ncu 2025.4.1 已装，
  `ERR_NVGPUCTRPERM` 被拒；管理员 PowerShell 或 NVIDIA Developer Settings 允许 GPU performance counters）。
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

### 3.5 环境受阻项（已由替代路径覆盖，无需再处理）

- WSL 侧 op 测试 + 字节一致 + 前端夹具测试：**WSL2 CUDA 驱动崩溃**（`cudaGetDeviceCount()` 内 PTX JIT
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

### 3.7 架构级隐患（长期，未排期）

- **双执行核心漂移风险**：TP-2 走独立 `TP2GenerationCore`，绕过 EngineCore 调度器/CUDA Graph/
  ResourceManager（单请求 lockstep）。请求生命周期、容量、取消、前缀复用、checkpoint ring 等逻辑在两条
  路径重复实现，须长期保持语义一致——已多次产出 prefill/decode 不对称类缺陷（工具掩码、首 token 时序、
  MTP ids 交换皆实例）。最大维护性隐患。
- **TP-2 in-kernel allreduce give-up 安全性开放**：host 侧检查已被证伪，见 §3.6(b)；「让 give-up 本身
  安全」是架构级改动，需要先确定确切算子。
- **host-staging CPU allreduce 回退**：S1+S2 已把它从性能悬崖降为链路受限；图降级（EagerExact）保留
  ——host 同步是回退固有属性，非 WSL2 环境的退化只剩「无图」。
- **TP-2 前缀复用容量硬上限**：device reuse snapshot 固定 `kReuseSnapshotCount=2`（每槽 73 MiB/shard），
  更深分歧靠 host checkpoint ring（一次 PCIe 往返）；深分歧场景复用成本被结构性抬高。

### 3.8 上游 ninfer-all 移植剩余项

- **上游 `origin/master` `e31bc99b`（2026-09-26）/ `dev` `a012e2bc`（2026-09-28）性能项（2026-09-29 源码评估）：无 TP-2 合并价值**——linear 统一+sliced-K、nvfp4 W4A4 TMA、q4/q5 dispatch 均无收益（decode weight-stream bandwidth-bound；主 GEMM 半几何走 MMA 不走 TMA；主线 NVFP4 非 q4/q5）；Jinja chat 模板已合并（含本地增强）。定案与 commit id 见 [tp2-decisions.md](docs/tp2-decisions.md) 的 "Upstream merge-value assessments"；上游未越过 `e31bc99b` 触及主干前不重评估。

- **3b attention gate 折进 reduce epilogue**（上游 `77c9cc39`）：暂缓——每层少一次 launch ≈0.08 ms，
  占 38.2 ms round 的 0.2%，收益低于风险。
- **3c INT8 系 attention scale 走 shared memory**（上游 `a941c9f4`）：暂缓——生产走 `--kv-dtype fp8`；
  本树关 RDC 后 small_t_i8 栈帧反而增多，需先用 SASS 归因是否真有 out-of-line call。
- **SM 派生值的 5060 Ti A/B 计时**：rmsnorm 门限、rope block 选择、chunked GDN 波目标，以及 small_t
  160/320 目标（后者改派生值会改 5090 既有行为，必须先有 5060 Ti batch_size>1 的 A/B）。
- **项目 6 大件（需产品决策，先不改）**：KV codec（`eb9f7a23`/`ad26b362`/`2ba10da2`）、设备路线 profile
  （`0b2f8384`+`81861a07`，36 SM 卡需重新校准）、fast prompt kernel（`4303e604`）。
- **发射期无条件 `cudaFuncSetAttribute` 收敛**：`context_kv_materialize/materialize.cu`、
  `linear/gguf/ggml_bridge_mmq.cuh`、`linear/nvfp4/nvfp4_w4a4_tma.cu` 等仍在每次发射调用 driver；
  正确但值得在有计时证据时一并收敛。
---

## 4. Round 54：TP-2 会话缓存四项改进（实施计划）

> 来源：anthropic 0 命中诊断（stream-500 → session_invalidate_active 链条）+ 单卡 vs TP-2 缓存机制对比分析（上游 4201b5d2）。
> 基线分支：feat/windows-native-port @ 92d7ab57（运行服务器 C:\ninfer\ninfer-serve.exe 同源；本地 master 仅 1102 行早期 TP-2，勿用）。
> 状态：**计划已整理，未实施。** ④ 可最先独立落地（生产痛点）。

### 4.1 环境约束

- 工作分支 feat/windows-native-port；实施前 git checkout 该分支。
- 双 5060 Ti 被生产服务器（:3456）占用 → 集成测试需停服时间窗（先与用户确认）：测试二进制占 :3456 → 探针 → 恢复。
- 探针脚本 _temp/ 时间戳前缀，成功即删。

### 4.2 ④ count-tolerant anthropic stream 编码器（最小，独立）

目标：adoption splice 改 prompt count 不再 500，会话状态不再被摧毁。
- src/serve/anthropic_messages_response.cpp L225-230：start(generation) 删 throw，改 input_tokens_ = generation.prompt.prompt_tokens;（streaming_start_usage 的 input - reused 算术自动正确）。
- 审计最终 usage（L91/L352 message_delta 用 outcome.*）：start 与 final chunk 一致（均引擎 count）。
- 审计 openai_chat_response.cpp L287-295 prompt_progress 同类 throw：return_progress_ 默认关 → 记录为已知限制，改动量小则同样宽容化。
- 范围外（记录）：serve 渲染失败不销毁引擎状态的更深层修复。
- 测试：1620 场景探针（divergent echo stream → 200 + cache_read = splice 后 reuse；随后 non-stream 命中自己的边界）；message_start 与 message_delta usage.input_tokens 一致。

### 4.3 ① session_key + 保留权重（引擎层地基）

目标：TP-2 catalog entry 携带 session key 与保留类；驱逐按权重。
- src/runtime/engine/tp2_generation_core.cpp：
  1. SessionEntry + std::optional<CacheSessionKey> session + 权重（有 key=LiveSession 16，无=RecentPrivate 4；常量与通用 cache private_retention_weight 对齐）。
  2. session_recall：匹配保持纯 token 扫描；命中后按 incoming hints 更新绑定。
  3. session_publish：从 data.context_cache().session_key 取 key（plumbing 已存在）；同 key 他处已绑定 → 重绑定 + 旧 entry 降 4（对齐通用 cache publish_session/demote_replaced_session 的 publication_order 规则）。
  4. session_evict_one()：权重感知 LRU（先逐最低权重最老；全 16 时逐最老）。
- serve 层无改动（openai-responses 已传 key；anthropic 合成 key 为独立后续项）。
- 测试：单元（驱逐顺序 16 vs 4 / 同权最老 / 全 16 最老）；集成（openai A 2 轮 + 压力 B..G，A 存活且下一轮命中）；回归（anthropic 全 4 行为不变）。

### 4.4 ② 容量感知准入（消除静默 drop）

目标：host KV 不足时显式拒绝（503 + Retry-After），而非静默丢弃会话。
- session_store_active()（L1751）：D2H 前 feasibility 检查。
- 切换路径（drop_previous）：失败先经 ① 权重驱逐腾 slab 重试；仍失败 → 新异常 ContextCapacityUnavailable 从 submit()/execute_walk 抛出——incoming 拒绝，**当前 resident entry 不销毁**（关键不变式）。
- generation_service：捕获 → HTTP 503 + Retry-After: 1，标签 "context capacity temporarily unavailable"。
- 启动容量核对：session_capacity_ > host 满尺寸会话数 + 1 → operational log WARNING。
- 不加新 flag。协议契约变化 → schema 测试 + docs/serving.md 同步。
- 测试：N（>host 容量）大会话后第 N+1 并发切换 → 503（非静默）；空闲可逐后恢复；拒绝不销毁 resident（日志验证）。

### 4.5 ③ 共享前缀 entry（两阶段）

目标：多会话共享稳定前缀（系统提示+工具）缓存一次并受保护。
- 阶段 A（仅保留保护，先做）：
  1. SharedPrefixEntry：token 前缀（digest key）+ refcount + host slabs 双 shard 副本；refcount>0 不可逐，==0 按最老可逐。
  2. session_publish 检测：与现有 entry 共享 >= --shared-prefix-floor（新 flag，默认 4096，0=禁用）→ 分歧边界建/更新共享 entry，refcount += 1。
  3. 驱逐释放引用；refcount → 0 后可逐。
  4. recall 不变（各 entry 自身 slab 物化）；共享 entry 兼作新会话 recall anchor。
- 阶段 B（存储去重，可选）：entry 只存 suffix KV；recall = 共享前缀 H2D + suffix H2D；grid 覆盖 suffix；snapshot 在分歧点。仅当多 agent 部署证明 A 不够。
- 测试（A）：两会话同 30k 系统+工具 → 1 共享 entry；驱逐压力后第三同系统会话可站共享边界；refcount 生命周期。

### 4.6 顺序与依赖

④（独立，serve-only）→ ①（引擎地基）→ ②（依赖 ① 权重驱逐）→ ③A（依赖 ①②）。

### 4.7 风险

- ② 改 serve 错误面 → 协议契约更新（schema 测试 + 文档）。
- ① 重绑定语义必须与通用 cache publish_session 一致，保持跨模式行为一致。
- ③A publish 增加有界 token 比较（catalog 规模内，可忽略）。
- TP-2 文件本地独有（上游无 TP-2）→ 无上游合并冲突；serve 层文件上游在演进 → 实施前对 serve 层做 rebase 检查。

### 4.8 进度与实测结果

- [x] **④ stream 编码器宽容化** —— 已实施（`anthropic_messages_response.{h,cpp}`）+ 单测用例（`test_anthropic_schema.cpp`）+ 实机 A/B。
- [x] **① session key + 保留权重** —— 已实施（`tp2_generation_core.{h,cpp}`：SessionEntry.session/retention_weight、bind_entry_session/demote_session_owners、权重感知 session_evict_one）。
- [x] **② 容量感知准入** —— 已实施，但**未采用计划中的 503 拒绝**，见下。
- [ ] **③A 共享前缀 entry（仅保留保护）** —— 未实施（设计见 4.5；规模较大，见下）。
- [ ] ③B 存储去重（可选）—— 未实施。

#### ④ 实机 A/B（决定性复现）

确定性触发：模板渲染时 trim `reasoning_content` 但不 trim 正文，而 `same_rendered_turn` 两侧都 trim；因此**给回显的 assistant 正文加尾部空白**即让同一轮在引擎看来等价、但 token 化与生成字节不同 → 触发 `adopt_generated_turn` 拼接 → prompt count 变化。

| 轮次 | 旧二进制（C:\ninfer\ninfer-serve.exe.bak-20260929-1722） | 新二进制 |
|---|---|---|
| turn 1 | 200, cache 0 | 200, cache 8375 |
| turn 2（splice） | **SSE error: "Anthropic stream prompt count differs from Engine start"** | 200, cache 8441（99.7%） |
| turn 3 | 200, **cache 0**（entry 被 500 摧毁 → 全量重 prefill） | 200, cache 8375 |

即：生产 0 命中循环的根因链条被完整复现，且修复后消失。单测 `ninfer_anthropic_schema_test` 通过（含新增 splice 用例）。

#### ① 实机验证（命名会话 vs 匿名压力）

12 个匿名 anthropic 会话（无 session key，权重 4）+ 1 个命名 openai-responses 会话（store:true，权重 16，clock=1）填满 catalog（9 槽）：

- 驱逐轨迹 4 次全部 `weight=4`；clock=1 的命名会话从未被选中（权重生效的直接证据）。
- 命名会话 turn2 召回 `cached=8460/8485 = 99.7%`（在压力之后仍存活并可深召回）。

#### ② 实机验证（`--host-kv-mib 2048`，1024 MiB/shard）

- catalog 自限到 host 实际可容纳的 7 个条目（而非目录上限 9）。
- 13 个请求全部 200；驱逐全部 `weight=4`；**零** `[tp2-session] drop ... alone exceeds the host session budget`。
- 命名会话仍 99.7% 命中。

#### ② 与计划的偏差（需记录）

计划原定：store 失败 → 权重驱逐重试 → 仍失败则抛 `RequestError(Unavailable)` → 503。实施中发现：**驱逐循环会把所有可驱逐条目清空**；循环结束后仍失败 ⇔ 只有 resident 自己剩下 ⇔ 它自身就超出 host 预算（如 `--host-kv-mib` 小于单会话容量）。此时拒绝会造成**每次切换都 503 的永久性故障**（因为没有任何切换能成功、也就没有任何条目会被逐出）。因此改为：

1. store 失败 → 按权重驱逐并重试（覆盖真正常见的"host 被弱会话占满"场景，这是静默 drop 的主因）；
2. 仍失败（resident 自身超预算）→ 保持原有 drop，但**无条件打印** `[tp2-session] drop ...` 到 stderr（进入服务端运维日志），不再静默。

另修正一处实施中发现的索引缺陷：`session_evict_one()` 在循环中擦除条目会移动 vector，捕获的 `previous` 索引会失准；改为循环后重读 `active_session_`（`session_drop` 会维护它）。

#### ③A 未实施的原因

- 现状已有**部分覆盖**：TP-2 的 per-entry divergence image（`host_shared_state`/`host_shared_end`）已提供跨会话共享前缀的 recall anchor，实测轨迹中可见 `reach=8448 via=shared`；③A 的增量价值是"会话全部被逐出后共享前缀仍存活"（会话无关的共享条目）。
- 该增量需要新的存储与召回分支（共享条目及其 host slabs、分歧点快照、以共享条目为源的 recall 分支、引用计数生命周期），属较大改动；在 ④①② 已修复生产 0 命中与静默 drop 之后，单独排期更稳妥（避免半成品特性进入正在服务的引擎）。
- 建议下一步：先按 4.5 阶段 A 实现"会话无关的共享前缀 checkpoint 池"，复用现有 host checkpoint 的 restore-then-prefill 路径，而不是复制 SessionEntry 的整套 host 状态。

#### 既有失败（与本轮改动无关，已归因）

`ninfer_resource_manager_test`（candidate-stratified reuse closure）与 `ninfer_serve_options_test`（server reasoning-effort default）在本分支上**基线即失败**：将本轮全部改动 `git stash` 后重建并运行，二者输出完全相同的失败信息。前者正是单卡 context cache 驱逐/复用闭包选择的问题，可作为单卡侧独立缺陷线索。

### 4.9 TP-2 每隔一轮复用塌陷：adoption 的坐标位移（已修复）

**现象（生产日志 `C:\ninfer\serve-win.log`，两个 anthropic 会话来回切两次）**：31 个请求全部成功、无 0 命中，但同一会话增长期**严格交替**：`private endpoint`（98-100%）↔ `long anchor`（83-93%），浅的那轮多 prefill 4,000–9,500 token。

**复现（用导出会话的真实 systemPrompt + 30 个工具重建，`_temp/20260929-1955_tool_replay.mjs`）**

| 轮次 | 修复前 cache | 修复后 cache |
|---|---|---|
| 1 | 0（全量） | 0（全量） |
| 2 | 51,917 | 51,852 |
| 3 | **51,200** | **51,969** |
| 4 | 52,151 | 52,109 |
| 5 | **51,910** | **52,212** |
| 6 | 52,381 | 52,386 |

**根因（trace 证据：`shared`、`replay_split`、`adopted`、`prev_prompt`）**

隔轮出现 `adopted=1`：`adopt_generated_turn` 把「客户端重渲染的同一个 turn」替换成本 lineage 生成的 token。当客户端渲染**已经覆盖 entry 的全部历史**（`shared == cached`）时，这次替换**不会加深任何可复用边界** —— 扫描的上限就是 `shared`，而最深边界（frontier = `turn_end - 1`）本来就在 `shared` 之内。它唯一的效果是把 prompt 缩短、让 entry 的坐标整体前移。

于是下一轮客户端按自己的坐标重发，公共前缀只能到 `replay_split`（例：52,152，比上一轮记录的 `prompt_end` 52,177 更浅）→ `resident_continues` 判定失败 → 走 `switch`（把会话 D2H 到 host、`live_state_valid_=false`）→ 从更浅的 host checkpoint 重新 prefill。**`switch` 只是症状，不是病因。**

**修复**：`adopt_generated_turn` 只在替换能加深可达前缀时才做 —— `if (shared + 1 >= turn_end) { return adoption; }`。分歧落在回答内部（`shared < turn_end - 1`）时照旧替换，此时它确实把可达前缀从 `shared` 推进到 `turn_end`。

**实测**：trace 全程 `continue`、`adopted=0`、每轮 `src=live`（直接命中 frontier），`shared == cached`（entry 保存的就是客户端自己的 token，不再被位移）。

**被否决的替代方案（记录以免重走）**：把续会判定阈值从 `prompt_end` 换成「客户端渲染一致点」（`adoption.divergence`，每轮都已计算）。实测它确实让每轮都走 `continue`，但**复用深度一点没变**（浅轮仍是 51,200 / 51,906，因为扫描上限仍是 `shared`），并且引入隐患：首轮 entry 的 `divergence` 可能只有 1，判定退化成「任何 prompt 都算续会」，于是另一个会话到来时**不会**把 resident 存到 host；而 entry 一旦失去 resident 又没有 host 副本（`host_kv_end=0`、`frontier>0`），`session_recall` 仍会把它的 frontier 当候选，`session_restore` 会解引用空的 `host_kv`。已回滚。

**已知残留**：当分歧确实落在回答内部（长回答场景，adoption 有真实收益）时，替换仍会让 entry 坐标前移，下一轮匹配被 cap 在分歧点；而分歧点通常**没有 snapshot**（walk 从不经过它，它位于回答内部），于是下一轮只能落到它之前最近的 checkpoint。这是「本轮多复用一段、下一轮少一段」的权衡，是否净收益需要长回答实测才能定；若要消除，需要让 walk 在 splice 分歧点留下 checkpoint（或把续会判定与匹配整体搬到客户端坐标），属独立设计。

**证据与测试**：`_temp/20260929-*_replay_*.out.txt`（对照重放）、`C:\ninfer\serve-guard.err.log`（trace）、`build-win/adopt_guard.log`（构建）；`ninfer_turn_replay_test` 通过。**测试空缺**：`ninfer_qwen3_5_tp2_sessions_test` 走裸 token 路径（不产生 `message_boundaries`），`adopt_generated_turn` 在该测试下从不触发 —— adoption 路径目前没有自动化覆盖。

