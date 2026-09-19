# NInfer TP-2 计划（2× RTX 5060 Ti · Qwen3.8-27B NVFP4）

> **唯一的活动计划**，也是跨上下文压缩的持久记忆。整理时间：Round 48 收尾。
> - 完整历史记录（原 PLAN.md 全文，Round 1–48 逐轮证据与失败尝试）：`docs/tp2-dual-5060ti-worklog.md`
> - 交付说明、推荐配置与实测数据：`docs/tp2-dual-5060ti.md`
> - 状态：**交付范围 ①–⑤ 全部完成。**① 的判据已从「与 plain 逐 token 一致」（不可达，见 §5.5）改为
>   「无退化 + 质量同档 + 有加速」，并已达成（§5）。

---

## 1. 目标（用户原始要求）

1. 修好 MTP 一致性问题；
2. 做基准测试；
3. 查找 prefill/decode 优化空间；
4. 优化显存占用，开启 KV cache fp8/q8_0 量化，调整对性能/质量影响不大但占用资源的参数，以开启更大上下文；
5. 交付一个针对本机双卡 5060Ti TP-2 极致优化的推理框架。

原始计划的补充完成条件（记录在归档 §0）：TP=1 路径零回归、切分 Op 有独立 oracle、端到端质量确认、
性能下限（MTP0 ≥ 28 tok/s、MTP3 ≥ 70 tok/s）、C=1/2/4 稳定性。

---

## 2. 完成状态

| 目标 | 状态 | 证据 |
|---|---|---|
| ① MTP 可用（无退化 / 质量同档 / 有加速） | **完成**（§5） | 6 次采样 content 1519/1318/1294/1195/1683/0（中位 1306、`longest0` 全 0）vs plain 1165/1452/1546/672/224/1237（中位 1201）；稳态 decode 46–49 vs 31.6 tok/s |
| ② 基准测试 | 完成 | prefill 279 token / 0.453 s（615 tok/s）；decode 端到端（含 prefill）256 token：plain 8.153 s（31.4 tok/s）、MTP K=3 6.532 s（39.2 tok/s）；纯 decode：plain 31.5、K=1 42.0、K=2 50.5–54.0、K=3 47.9–48.0 tok/s |
| ③ 显存 / KV 量化 | 完成 | 五种 dtype（bf16/int8/fp8/nvfp4/k8v4）在 TP-2 全部可用；fp8 质量与 bf16 同档（1365/859/1382/1602 vs 1233/763/1469/1456/1430/1621，均无 0 复读）；显存 plain 13,888 MiB/卡 |
| ④ 更大上下文 | 完成 | 65,536 → **131,072 token**（fp8 KV，13,904 MiB/卡，吞吐不变）；端到端 118,869-token 提示 HTTP 200 / 96.8 s，19,869-token 提示 19.5 s；262,144 OOM |
| ⑤ 文档与推荐配置 | 完成 | `docs/tp2-dual-5060ti.md`（推荐配置、基准表、KV dtype 表、MTP 限制与根因、修复清单） |

**未达成的原有门槛**（诚实记录）：MTP3 ≥ 70 tok/s 未达到（实测纯 decode 上限 K=2 50.5–54.0）；
TP-2 路径未跑 perplexity 评测（质量证据改用同提示词多采样 A/B）；per-shard arena 的
`memory_summary()` 仍报 `pages 0/0`、`runtime 0 B`（仅显示口径问题，未影响功能）。

---

## 3. 推荐运行配置（8088 当前按此运行）

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 131072 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2
```

相比 bf16 KV @65,536：上下文翻倍、显存与吞吐不变、质量同档。MTP（`--spec mtp --draft-tokens 2`）已在
bf16 与 fp8 KV 两条路线验证无退化、质量同档，稳态 decode 46–49 vs plain 31.6 tok/s；去掉这两个 flag 即回到 plain。

---

## 4. 已完成的关键工作

- **prefill/decode 优化（目标 3）**：批量 prefill（TTFT 19×）、块宽扫描与 allreduce 真因定位（Round 35b）、
  AR 有序切片流水（+15%，Round 35c）、allreduce 6 sync → 2 sync 与 in-kernel allreduce + mixer 头切分
  （Round 33/34，追平 llama.cpp）。细节见归档。
- **KV 量化修复（目标 4 的关键前提）**：量化 KV 原本在 TP-2 上完全无法启动。根因是
  `prompt_fp8.cu` / `prompt_nvfp4_non_rdc.cu` / `prompt_k8v4.cu` 用**函数内 `static`** 做
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`（92,416 B > 48 KiB 默认值）的 opt-in；该属性是
  **每 device** 的函数属性，第二个分片所在 device 从未 opt-in → 每次 launch 报 `cudaErrorInvalidValue`。
  改为按 device 缓存后五种 dtype 全部可用（`tools/tp_bootstrap/fix_per_device_attr.py`）。
- **更大上下文（目标 4）**：fp8/int8 KV 使 131,072 token 在 13.9 GB/卡内可用，并以 118,869-token 提示端到端证明。
- **MTP 状态机修复**：每轮 verify 前快照 GDN 状态到新 scratch slot（`kRoundScratchSlot`），fold 前先 restore，
  消除「verify 推进活状态 + fold 二次推进」的双倍前进（首分叉 4 → 10）。
- **临时插桩清理**：`NINFER_TP2_ROUND_DETAIL/PARITY/TOKEN_TRACE/SKIP_FOLD/VERIFY_PHASE` 全部删除
  （`grep NINFER_TP2_ ` 为 0 命中）。

---

## 5. 已完成：① MTP 输出退化（根因：分片未写 replay 记录）

### 5.1 现象（修复前）
MTP 可用且更快（端到端 +25%、纯 decode +50%），但输出退化：temp0.7/top-k20/top-p0.80 下 6 次采样
content 中位数 0（6/6 出现 "0" 复读，`longest0` 达数百），plain 中位数 1443、0/6 退化。贪心同样复现。

### 5.2 根因（已确认）
TP-2 的 head-split 分片**根本不产生 replay 记录**：`gdn_mix` 里两条记录分支都以 `shard_config_ == nullptr`
为条件（融合的 width>1 记录算子只注册了全量 16384 行 fused parent，分片拿不到）。而每轮 MTP 仍然执行
「恢复轮前快照 → 用 `commit_columns` 跑 `gdn_replay_fold`」，fold 消费的是**从未写过的记录平面**：
48 个线性注意力层的 recurrent 状态每轮都被回放成空日志（等于不前进），conv 历史也被同一份空日志重建；
16 个全注意力层照常工作 ⇒ 前几个 token 局部连贯、随后塌缩成 "0" 复读。与现象完全吻合。

### 5.3 修复（`src/models/qwen3_5/execution/text.cpp` · `gdn_mix`）
- 分片在自身的 decomposed 路线上记录：conv 记录就是该路线已物化的 `[q|k|v]` 原始卷积输入，直接拷贝到
  记录平面（通道序与布局一致，fold 需要的就是 `conv_record[0:committed]`）；
- recurrent 改用 `ops::gated_delta_net_replay_record`：它显式注册了分片几何（qk 8 heads / value 24 heads），
  且契约规定其输出与归一化 `gated_delta_net` **逐位相同** ⇒ 窗口逐列 logits 与接受判定不变，只有状态转移
  变成可回放（正是 fold 需要的）；
- `causal_conv1d_silu.h` 的 row profile 文档补上分片 profile `(1024,1024,3072) / C=5120`（实现早已支持）。

### 5.4 证据
- 端到端 A/B（`r36f_stats.sh`，同一提示词、temp0.7/top-k20/top-p0.80、各 6 次）：MTP content
  1519/1318/1294/1195/1683/0（中位 1306），**6/6 `longest0 = 0`（零复读）**；plain 1165/1452/1546/672/224/1237
  （中位 1201）。唯一那个 0 是 reasoning 预算 artefact，plain 同样出现（两例把预算全花在 reasoning）。
- 吞吐无回归：MTP K=2 稳态 44.3/46.2/45.5/46.1/49.2 tok/s，plain 31.6–31.8 tok/s。
- 第二个缺陷（同属"per-device 属性用进程级 static"）：`small_t_fp8.cu`/`small_t_k8v4.cu`/`small_t.cu` 的小 T
  核也需要 >48 KiB 动态 smem 的 opt-in，但单 token decode 用 32 KiB tile 恰好低于 48 KiB 默认值，缺陷一直潜伏；
  MTP 窗口（T=K+1）切到 64 KiB tile 才暴露 ⇒ `small_t_fp8.cu:56 cudaErrorInvalidValue`。四路已统一到
  `src/ops/common/cuda_smem.h` 的按 (kernel, device) opt-in。验证：fp8 KV @131072 + MTP K=2 三次采样
  content 1530/1481/1708、`longest0` 全 0（`r49_fp8mtp_smoke.sh`）。
- 回归测试全 PASS：`gated_delta_net_replay_record`、`gdn_replay_fold`（含 48×8×24×5120 分片几何）、
  `gdn_input_proj_conv_record`、`gdn_input_proj_conv_snapshot`、`gdn_input_proj`、`tp_device_pair`、
  `qwen3_5_tp2_load --artifact`、`qwen3_5_tp2_forward --artifact`。

### 5.5 判据修正（重要）
「与 plain 逐 token 一致」是**不可达判据**，已废弃：verify 窗口与单 token decode 本来就是不同执行形状
（本引擎走 chunked attention/conv，llama.cpp 走 chunked GDN 核），且被接受 token 的 KV 由该窗口写入后不回改。
llama.cpp 的 Qwen3.5 MTP 结构相同，其文档也只要求「需要精确一致时用 greedy」。W1 判据因此改为
**无退化 + 质量同档 + 有加速**，并已达成。

### 5.6 被否定的两条诊断（留档，避免重走）
- **相位说**：曾判定「verify 走 `Phase::Prefill`、decode 走 `Phase::Verify`，两核舍入不同」。错误：`Phase`
  在 head-split 分片上没有语义——`attn_mix` 完全不用它的 phase 形参，`gdn_mix` 的两处 `ph == Phase::Verify`
  都被 `shard_config_ == nullptr` 排除——所以改相位不会改变任何数值。原 §5.5 计划的「rank-3 → rank-4
  引擎级布局改造」随之作废。
- **`rope.cu:189` 说**：那是 `CUDA_CHECK(cudaGetLastError())` 检查点而非出错核；多设备交错下异步非法地址
  会在**第一个**检查点浮现，报错点不必等于故障核。

---

## 6. 可选后续（非阻塞）

- **W2 负载均衡**：MTP 开启时 GPU0 100% / GPU1 ~72% 属结构性（MTP 层 + 全部 argmax/accept/fold/采样/策略只在
  shard A；B 在 text mixer lockstep 后空等）。方向：①两卡 lockstep 冗余跑同一份 MTP 层（墙钟不变）；
  ②按 vocab 切 MTP 提案的 lm_head + 窗口 argmax（真正缩短关键路径）；③重扫 K=1/2/3。
- **KV dtype 扫描补全**：fp8/int8 已有质量与显存数据，nvfp4/k8v4 只验证了可启动与吞吐。

---

## 7. 交付前收尾清单

- [x] 删除运行时插桩（`NINFER_TP2_*`）
- [x] 完整记录归档：`docs/tp2-dual-5060ti-worklog.md`
- [x] 删除只向 PLAN.md 追加历史文本的 `tools/tp_bootstrap/r36b_plan_append.sh`、`r36c_plan_append.sh`
- [x] W1（①：MTP 退化已修复、判据已修正为「无退化 + 质量同档 + 有加速」，见 §5）
- [ ] 用推荐配置复测并发 C=1/2/4 与流式/stop 冒烟（Round 31/34 的结论基于 bf16 配置）
- [ ] 交付时按 AGENTS.md 移除本文件（PLAN.md）；`tools/tp_bootstrap/` 下有约 300 个一次性诊断脚本，
      需决定保留/清理范围（其中 `build_r35.sh`、`serve_*`、`r37+r38+r39+r4x` 系列与 `docs/tp2-dual-5060ti.md`
      引用为可复现流程）

---

## 8. 本机环境与运维要点

| 项 | 值 |
|---|---|
| 编译/运行 | WSL2；Windows 工作树 `D:\Documents\workbench\ninfer`，构建树 `/home/zhuojun/ninfer` |
| 构建 | `bash tools/tp_bootstrap/build_r35.sh`（rsync `src tests apps include bench tools` 后 `cmake --build build -j 8`；成功标记 `BUILD_EXIT=0`，约 2–3 分钟） |
| 服务 | 8088；`serve_stop.sh` → 后台 `wsl -e bash .../serve_supervise.sh <ctx> 8088 '<extra args>'` → 轮询 "listening on http://127.0.0.1:8088"（加载 30–37 s） |
| 首请求 | 付 CUDA graph 捕获（实测 14.4 s），稳态数字才有意义 |
| 约束 | 本会话不终止/重启/新启动任何 llama.cpp 进程；只管理 ninfer 进程 |
| 工具链注意 | 嵌套 `pwsh` → `wsl -e bash -lc "..."` 会吞 `$var` 与重定向 ⇒ 把命令写成脚本文件放 `tools/tp_bootstrap/` 再执行；单次阻塞调用上限 600 s ⇒ 长任务用后台作业 + ≤420 s 轮询 |

---

## 9. 证据索引

- 脚本：`build_r35.sh`（构建）、`serve_supervise.sh`/`serve_stop.sh`（服务）、`r37_colparity.sh`（逐列 parity）、
  `r36h_trace.sh`（逐 token 分叉）、`r38_kvsweep.sh`/`r42_fp8check.sh`/`r44_fp8quality.sh`（KV 量化）、
  `r39_bench.sh`/`r43_bigctx.sh`/`r48_longctx.sh`（基准/长上下文）、`r45_verify.sh`/`r46_recconfig.sh`/`r47_verify.sh`（验收）、
  `fix_per_device_attr.py`/`strip_tp2_probes.py`/`strip_tp2_probe_comments.py`（源码批改）、
  `verify_gdn_replay_fix.sh`/`verify_tp2_artifact.sh`（① 修复的回归）、`r36f_stats.sh`（质量 A/B）、
  `r49_fp8mtp_smoke.sh`（MTP + fp8 KV 冒烟）
- 日志：`/home/zhuojun/prof/`（`serve_supervised.log`、`kvsweep-out.log`、`r39/bench-out.log`、`r42_fp8check.log`、
  `r43_bigctx.log`、`r44_fp8quality.log`、`r45_verify.log`、`r48_longctx.log`、`kvdiag-out.log`、`ab_stats.log`、`r49_smoke.log`）
- 源码改动清单：`docs/tp2-dual-5060ti.md` 的 "Fixes applied on this branch" + 本文件 §4/§5.3；逐轮文件引用见归档

> 说明：① 的修复改动为 `src/models/qwen3_5/execution/text.cpp`（`gdn_mix`）与 `include/ninfer/ops/causal_conv1d_silu.h`（文档）；其余历史改动清单见 `docs/tp2-dual-5060ti.md` 的 "Fixes applied on this branch"。
