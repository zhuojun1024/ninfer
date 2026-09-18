# NInfer TP-2 计划（2× RTX 5060 Ti · Qwen3.8-27B NVFP4）

> **唯一的活动计划**，也是跨上下文压缩的持久记忆。整理时间：Round 48 收尾。
> - 完整历史记录（原 PLAN.md 全文，Round 1–48 逐轮证据与失败尝试）：`docs/tp2-dual-5060ti-worklog.md`
> - 交付说明、推荐配置与实测数据：`docs/tp2-dual-5060ti.md`
> - 状态：**交付范围 ②③④⑤ 已完成；① （MTP 与 plain 逐 token 一致）未完成、阻塞、待决策。**

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
| ① MTP 与 plain 逐 token 一致 | **未完成（阻塞，见 §5）** | 逐列 parity：col0 228 同 / 24 异（9.5%，首差 pos=80）、col1 200/52（21%）、col2 214/38（15%） |
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
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088
```

相比 bf16 KV @65,536：上下文翻倍、显存与吞吐不变、质量同档。开启 MTP 请用 `--spec mtp --draft-tokens 3`，
但它是实验特性（见 §5）。

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

## 5. 未完成：① MTP 与 plain 逐 token 一致（阻塞）

### 5.1 现象
MTP 可用且更快（端到端 +25%、纯 decode +50%），但提交的 token 流与 plain 不一致：贪心下首个分叉在 index 3–25；
temp0.7/top-k20/top-p0.80 下 6 次采样 content 中位数 0（6/6 退化出 "0" 复读），plain 中位数 1443、0/6 退化。

### 5.2 根因（已定量确认）
校验窗口走 `forward_tp2_prefill`（`Phase::Prefill`，T=K+1 的 chunk 核），而 decode/plain 走
`forward_tp2`（`Phase::Verify`，T=1 核）；两核求和顺序/舍入不同 ⇒ 逐列 logits/argmax 有 9.5–21% 翻转
（逐列 T=1 重放探针：col0 24/252、col1 52/252、col2 38/252 不一致）。更关键的是：**被接受 token 的 KV 行由 chunk
核写入**，bf16 KV 下的这点差异永久留在 cache 里，两条轨迹约 50–80 个位置后必然分叉并退化。
⇒ 只要 verify 不是 decode 等价核，W1 的 token-for-token 判据就无法满足。

### 5.3 已实现并保留（默认关闭，备用）
- `text.h`/`text.cpp`：`forward_tp2_prefill(..., TextPhase phase = Phase::Prefill)`，phase 透传到 `run_layers_tp2`。
- Verify 分支补齐 sequence 绑定：`active_sequence_batch_=1`、`active_sequence_width_=tokens`、
  每卡 `active_linear_state_source_slots_`、`active_valid_columns_`（I32[1]=tokens）、
  `active_backend_kv_table_rows_`（I32[1]=0）、`verify_positions`（I32[tokens,1]）+ `ScopedPositions`（cache/rope）。
- `batched_verify` 与 GDN 的 `ph == Phase::Verify` rank-4 分支加 `shard_config_ == nullptr` 门控
  （head-split 分片没有全量行数的 fused record workspace，须走分片 decomposed 路线）。
- `tp2_generation_core.cpp` 的 verify 调用点**仍用 `Phase::Prefill`**（附 TODO）；快照 + fold 逻辑保留。

### 5.4 阻塞点
启用 Verify 后故障点逐层推进并在同一点稳定复现（两次）：
`tensor dimensions must be positive` → `gdn_input_proj_conv_record workspace: unsupported single-parent profile`
→ `residual_add.cu:27 cudaErrorIllegalAddress` → `text.cpp:91 cudaMemcpy2DAsync cudaErrorIllegalAddress`
→ **`src/ops/launcher/rope.cu:189 cudaErrorIllegalAddress`（稳定）**；`CUDA_LAUNCH_BLOCKING=1` 下 warmup **挂死**。

根因判断：Verify 相位要求 rank-4 batched 布局（`[.., width, batch]`）贯穿全图，而 TP-2 prefill 前向按 rank-3
（`[.., T]`）推导形状与工作区；这是引擎级的布局/工作区改造，而非再补一处绑定。

### 5.5 下一步（若决定投入，按序）
1. 从 `rope.cu` 的维度契约入手，把 TP-2 前向的形状/工作区推导从「1 列」改为「`width × batch` 列」
   （residual/window buffer、mixer/mlp 侧、GDN record workspace profile 逐层核对）；
2. 每改一层跑 `tools/tp_bootstrap/r37_colparity.sh`，目标是 col* diff = 0；
3. `tools/tp_bootstrap/r36h_trace.sh` 判定：400 token 内无分叉；
4. 6 次采样统计 A/B（判据：content 与 plain 同量级、无 0 复读）；达到则更新 `docs/tp2-dual-5060ti.md` 的 MTP 章节。

### 5.6 备选/兜底（成本低、可立刻判定）
混合数值修正：chunk verify 只决定 draft 是否被接受，**最后一个 licensed token（bonus）改用 T=1 的
`forward_tp2`（天然 `Phase::Verify`）从同状态重算**再提交。观测到的错误正是「chunk 的 bonus argmax 与 decode
不一致」（pos=79: target[0]=13 vs decode 15）。代价：每轮多一次 T=1 前向（+30–50% 目标算力），收益是 bonus 与
plain 一致（draft 误接受仍可能分叉）。当前默认口径：**MTP 标注 experimental，一致性优先时用 plain**。

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
- [ ] W1（阻塞，待用户决定是否投入 §5.5 的改造）
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
  `fix_per_device_attr.py`/`strip_tp2_probes.py`/`strip_tp2_probe_comments.py`（源码批改）
- 日志：`/home/zhuojun/prof/`（`serve_supervised.log`、`kvsweep-out.log`、`r39/bench-out.log`、`r42_fp8check.log`、
  `r43_bigctx.log`、`r44_fp8quality.log`、`r45_verify.log`、`r48_longctx.log`、`kvdiag-out.log`）
- 源码改动清单：`docs/tp2-dual-5060ti.md` 的 "Fixes applied on this branch" + 本文件 §4/§5.3；逐轮文件引用见归档

> 说明：本工作树相对 HEAD 整体未提交（大量文件），`git diff` 不能作为本次改动清单；上表按会话实际编辑与验证记录整理。
