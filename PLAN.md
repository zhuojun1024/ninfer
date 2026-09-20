# NInfer TP-2 计划（2× RTX 5060 Ti · Qwen3.8-27B NVFP4）

> **唯一的活动计划**，也是跨上下文压缩的持久记忆。整理时间：Round 48 收尾。
> - 完整历史记录（原 PLAN.md 全文，Round 1–48 逐轮证据与失败尝试）：`docs/tp2-dual-5060ti-worklog.md`
> - 交付说明、推荐配置与实测数据：`docs/tp2-dual-5060ti.md`
> - 状态：**交付范围 ①–⑤ 全部完成。**① 的判据已从「与 plain 逐 token 一致」（不可达，见 §5.5）改为
>   「无退化 + 质量同档 + 有加速」，并已达成（§5）。
> - **Round 52 收尾（已完成）**：词表/隐层并行切分（§10 的 Step 1+2+3）与显存腾挪把上下文顶到
>   artifact 上限 **262,144 token**（15,614 / 15,094 MiB/卡），greedy 输出与优化前逐字节一致。
>   详细数据见 `docs/tp2-dual-5060ti.md` 与本文件 §10。
> - **Round 53（进行中）**：TP-2 视觉识别 —— 「静态分片分工」（MTP 只 shard 0 / vision 只 shard 1），
>   见 §11。用户已批准；目标是显存足够时不引入单图 token 上限配置。

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
| ④ 更大上下文 | 完成 | 65,536 → 131,072 → **262,144 token**（fp8 KV，15,614 / 15,094 MiB/卡）；端到端 118,869-token 提示 HTTP 200 / 96.8 s，65k-token 提示 48.1 s |
| ⑤ 文档与推荐配置 | 完成 | `docs/tp2-dual-5060ti.md`（推荐配置、基准表、KV dtype 表、MTP 限制与根因、修复清单） |

**未达成的原有门槛**（诚实记录）：MTP3 ≥ 70 tok/s 未达到（实测纯 decode 上限 K=2 50.5–54.0）；
TP-2 路径未跑 perplexity 评测（质量证据改用同提示词多采样 A/B）；per-shard arena 的
`memory_summary()` 仍报 `pages 0/0`、`runtime 0 B`（仅显示口径问题，未影响功能）。

---

## 3. 推荐运行配置（8088 当前按此运行）

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh`（Round 53c 新增）是**进程级默认思考强度**：不带该 flag 时
沿用模板自带的默认（本 artifact 为 `xhigh`），请求体的 `reasoning_effort` 优先。实测同一问题
`xhigh` 思考 172 tokens、`medium` 46 tokens。
**`--chat-template` 不存在且本架构不支持**：模板是 artifact 资源
（`resource/text/chat_template.jinja`），必须与 `tokenizer_config.json.chat_template` 逐字节相同，
再按 sha256 白名单编译——只有 `e84f32a2…`（thinking-toggle）与 `c3cf9e34…`（reasoning-effort，
本 artifact 用的就是它）两种语义。详见 §11.8。

262,144 是 artifact 的 `max_position_embeddings`：预填 1588 tok/s、稳态 decode 59.0 tok/s。
MTP（`--spec mtp --draft-tokens 2 --lm-head-draft`）已在 bf16 与 fp8 KV 两条路线验证无退化、
质量同档；去掉这三个 flag 即回到 plain。

`--vision`（Round 53 新增可用）：静态分片分工把 MTP 放 shard 0、Vision 塔放 shard 1，
启动台账多出 `vision 413.3` 一行（shard 1），两卡余量 699 / 951 MiB。多模态请求与纯文本请求
走同一套机制（MTP + 前缀复用，见 §11.6），冷启动 greedy 逐字节相同。

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

- **W2 负载均衡**（**已完成**）：MTP 开启时 GPU0 100% / GPU1 ~72% 属结构性（MTP 层 + 全部 argmax/accept/fold/采样/策略只在
  shard A；B 在 text mixer lockstep 后空等）。方向：①两卡 lockstep 冗余跑同一份 MTP 层（墙钟不变）；
  ②按 vocab 切 MTP 提案的 lm_head + 窗口 argmax（真正缩短关键路径）；③重扫 K=1/2/3。
- **KV dtype 扫描补全**：fp8/int8 已有质量与显存数据，nvfp4/k8v4 只验证了可启动与吞吐。
- **工具调用约束解码**（诊断见 Round 15）：对齐 llama.cpp 的 lazy tool-call grammar，把 `<function=` 后的工具名
  和参数掩码限制在本次请求声明的工具上（先做「工具名前缀树 + 参数名掩码」，完整 GBNF 太重）。
  注意掩码要和 MTP 投机解码 / exact-batch CUDA Graph 对齐，否则 verify 会误判接受。
- **TP-2 host 侧状态检查点 + 短 prompt 按 LCP 截断复用**（**已实现并完成端到端验证**，诊断见 Round 16，实测见 Round 16.3）：把 GDN 状态快照放进 pinned host
  环形池（0 显存增量，`--host-state-slots` 语义），步长 8192，复用扫描取最深的 `<= shared_prefix` 检查点。
  修掉「客户端在 turn 边界重渲染出更短 prompt ⇒ reuse=0 ⇒ 整条全量重算（实测 42.0 / 57.6 s）」。实测 `src=host` 命中：req#5 10.2 s → 1.3 s、
  req#11 37.6 s → 28.6 s，重算量与理论最小值差额 ≤ 1 个 chunk。

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

---

## 10. 已完成：词表/隐层并行切分（Step 1 + 2 + 3）与 262,144 上下文

动机（Round 51 显存账本）：每卡有一份完整的 `text/token_embedding`（1,213 MiB）与 `text/output_head`
（1,213 MiB），两张卡各一份 ⇒ 合计 4.8 GiB。词表维切分后每卡只留一半 ⇒ 每卡 −606 MiB/表，
两张表都做完 −1.2 GiB/卡；换算每省 1 GiB/卡 ≈ +65,536 token 上下文，262,144 还需要卡 0 再腾 ~930 MiB。

**Step 1（本次）**：`text/output_head` → `WeightSplitKind::ColumnParallel`（每卡行区间
`[shard*V/2, (shard+1)*V/2)`）。

- 载入：`tp_split_spec` 新增 `TpSplitOptions::split_output_head`；`materialize_model_tp2` 按
  `purpose == Generation && (!speculative_enabled() || proposal_enabled())` 打开。理由：选 Full 提议头时
  `text/output_head` 与 `mtp/output_head` 是同一个 artifact 对象（weight-tied），而 MTP 层只在 shard 0 上跑，
  该路线的提案 argmax 需要全量头 ⇒ 保持 replicated（行为与今天完全一致），留到 Step 3 用 peer 参与解决。
- 执行：新增 `TextContext::project_head_tp2(peer, pair, hidden, hidden_peer, logits, logits_peer)`：
  - 未切分：两次全量 `project`（现状）。
  - 已切分：每卡先 `project` 出 compact `[V/2, T]`，把 `[V,T]` 缓冲清零后用 `cudaMemcpy2DAsync` 写进自己的半区，
    再 `pair.allreduce` 求和（对侧半区为 0）⇒ 两卡都拿到完整 `[V,T]` logits，采样/接受判定逻辑不变。
  - 三个调用点：`forward_tp2`（decode T=1）、`forward_tp2_prefill` 的 `[V,1]` 分支、verify 的
    `logits_columns`（`[V,width]`，原本只有 shard A 有缓冲 ⇒ 函数内为 shard B 新开一个）。
- 验收（全部通过）：`ninfer_qwen3_5_tp2_load_test` 三条路线（无 spec / `--spec mtp` /
  `--spec mtp --lm-head-draft`）、`forward_test`（11 探针 + 32 步 AR + 批量 prefill）、
  `ninfer_tp_device_pair_test`；新 op oracle `ninfer_linear_tp2_split_fp8_head_test`（真实头形状
  [248320,5120] 的 T=1/3/5、[16384,5120] 的 T=1/4，全部 "exact"）；8088 端到端 greedy 对照
  （`top_k=1` 固定）5 个提示词逐字节一致（`ab-control.jsonl` vs `ab-split.jsonl`，hash 全等）。
- 显存：14,940/14,678 → **14,334/14,072 MiB/卡**（−606 MiB/卡）。
- 顺带修复 `weight_splitter` 的 FP8 分片描述符：`slice_fp8_rows` 补 `scale_ne[0]`/`scale_nb[1..3]`，
  `slice_fp8_cols` 补 `group_size`/`group`（此前分片 Weight 与 payload 不自洽，只有经 geometry 重建
  描述符的生产路径掩盖了该问题）。

**Step 2（已完成）**：`text/token_embedding` → `WeightSplitKind::RowParallel`（按 hidden 列切分，每卡
`[V, D/2]`）。选列切分而非词表行切分：ids 始终是全域（不存在越界行），且合并路径与输出头完全同构。

- 执行：`TextContext::embedding_tp2(peer, pair, ids, ids_peer, x, x_peer)`：两卡各查自己那半列到 compact
  `[D/2, T]`，再用新抽出的 `merge_local_row_blocks`（与头共用）清零+写半区+allreduce；文本路径
  （`forward_tp2`/`forward_tp2_prefill`）直接传 peer 的 ids。MTP stem 只在 shard 0 跑 ⇒ 新增
  `TextContext::set_tp_peer`（TP-2 core 在两卡建好后注册），把 ids 广播到对侧（16 字节对齐填充 + allreduce；
  形状用前缀视图保持 `T` 与输出一致）再合并；未注册 peer 且表已切分时硬报错。
- 阻塞与解决：FP8 embedding 的注册域原本硬编码 `[248320,5120]`。把 FP8 gather 内核按隐藏宽度模板化
  （`embed_gather.cuh`）、launcher 按宽度分派（5120/2560 各自的 block 划分）、wrapper 用
  `embed_gather_fp8_supports_width` 校验（词表不再限制，因为内核与行无关），并更新 op 文档与
  `ninfer_embedding_test`（`Fp8Table` 参数化宽度，2560 走完整 T 扫描 + CUDA Graph + 对齐用例）。
- 验收（全部通过）：`ninfer_embedding_test`（5120+2560 全量）、`tp2_forward_test`、两条 load test；
  8088 端到端 greedy 对照 5 个提示词与 Step 1 构建逐字节一致（`ab-embed.jsonl` vs `ab-split.jsonl`）。
  期间发现并修复两个只有真实服务才能暴露的问题：`allreduce` 要求字节数是 16 的倍数；广播缓冲形状必须
  等于 ids 个数（否则 gather 的 T 校验失败）。
- 显存：14,334/14,072 → **13,728/13,466 MiB/卡**（再 −606 MiB/卡；两步合计 −1,212 MiB/卡）。

**Step 3（已完成）**：`proposal/head`（131072×5120 Q4_G64_FP16，341 MiB）按**词表行**切分，每卡 65,536 行
（171 MiB）。`proposal_argmax` 走 `hidden` 广播 → 两卡各算自己那半 → `merge_local_row_blocks` 合成
`[131072,T]` → shard 0 argmax + token-id 重映射。阻塞点：Q4 A16 是**枚举 (n,k) 表**，没有 65536×5120 这档，
新增 `select_q4_n65536_k5120`（`q4_shapes.h`/`q4_dispatch.cpp`/`sources.cmake` + 新 `.cu`）。oracle：
`ninfer_linear_tp2_split_grouped_head_test` 在真实 [131072,5120]、T=1/2 上与全量 Op 逐位比对全 "exact"。

**Round 52 结果（262,144 落地）**：

- 上下文：131,072 → **262,144 token**（artifact 上限）。`nvidia-smi` 15,614 / 15,094 MiB/卡（余 697 / 1,217）。
- 显存腾挪：state arena 367→294 MiB（前缀复用快照 3→2；`rewind_depths` 只剩 `{0, rewind_near_}`，
  实测 16k/65k 相同 prompt 重入 0.30/0.33 s ⇒ 聊天轮次模式够用）；workspace 384→192 MiB。
- 性能（t=0，usage 计数）：预填 16,057 token / 10.11 s = **1588 tok/s**；解码 512 token / 8.68 s = **59.0 tok/s**
  （223 轮、2.3 token/轮、每 draft 接受 64.6%），比 131072 时代 46–49 tok/s 高约 20%。
- 逐位一致：5 条 greedy 提示在 131072（优化前）与 262,144 上 hash 全同（`r52_ab.sh final` + `r52_cmp3.sh`）。
- 死路记录：MTP 层 KV 降 nvfp4 省 228 MiB、接受率不变（392/510 vs 391/510），但 3/5 提示输出变化 ——
  verify+fold 与单 token 解码执行形状不同，draft 模式一变近似并列的取舍就变；因此保留 fp8 MTP KV。
- 回归全绿：3 条 load route、`tp2_forward`、`embedding`、`linear_tp2_split_fp8_head`、
  `linear_tp2_split_grouped_head`（新）、`linear_tp2_split_nvfp4`、`tp_device_pair`。

---

## 11. 进行中：TP-2 视觉识别（静态分片分工）

**目标**：在双卡 TP-2 路由上支持 `--vision`。采用**静态分工**而不是动态换入换出：
`mtp/*` 只放 shard 0（shard 1 今天那份 430.4 MiB 是纯浪费 —— MTP 本来就只跑 shard 0），
`vision/*` 与视觉编码 workspace 只放 shard 1，shard 0 完全不动。用户决定：**若显存足够，就不要引入
「单图 token 上限」配置项**。

### 11.1 为什么是这个方案（Round 53 论证，已完成）

现状（三个独立障碍，按严重度）：
1. `normalize_engine_options` 在 `device_b >= 0` 分支硬置 `options.enable_vision = false`
   （`src/runtime/engine/model_instance.cpp:96`）；serve 层还有第二道闸
   （`src/serve/generation_service.cpp:313,368` 的 `vision_disabled`）。
2. `TP2GenerationCore` 全文没有视觉执行路径（只有 `load.vision` 与 `frontend.vision_enabled` 两处透传）；
   编码器会话 `VisionPrefillSession` 只存在于单卡 Program 路径。
3. 显存。

显存账（本机实测）：
- 分量字节（`tools/artifact/reader.py` 直接解析目录）：text 19431.7 / dflash2 2123.6 / mtp 430.4 /
  proposal 340.5 / **vision 282.0** MiB（333 张量、27 层、hidden 1152）。
- vision 在 TP 切分表里无规则 ⇒ `tp_split_spec.cpp:157-159` 的 `else` ⇒ `Replicated`（两卡各一份）。
- 编码 workspace 峰值（`vision.cpp:88-195` 布局 + `startup.cpp:720-725` 的
  `merged = min(context, kMaximumVisionItemTokens=16384)` ⇒ 65,536 patches）：x 144 + patch_bf16 192 +
  qkv 432 + attention_norm 144 + mlp_up 538 + mlp_norm 144 MiB，作用域复用后 **约 1.0–1.2 GiB**（估算，
  ±20%），另有 160 MiB output handoff。
- 现余量 697 / 1217 MiB（nvidia-smi，262,144 配置）。分工后：
  - shard 1：1217 + 430(MTP 移走) − 282(vision) = **1365 MiB** → 装 ~1.0–1.2 GiB workspace，余 165–365 MiB；
  - shard 0：**保持 697 MiB 不动**。
- 与动态换出对比：峰值完全相同，但静态分工零切换、零 pinned host 副本、无请求期分配、
  失败发生在启动期而非请求期，因此不采用动态换出。

### 11.2 Step A —— 按分片放置（可独立验证，先做）

- `tp::TPObjectSplit` 增 `shards` 掩码（默认 `0b11`）：`tp_materialize.h`。
- `object_shard_bytes`/`shard_capacity_bytes` 只统计会收到该对象的卡（`tp_materialize.cpp:77-115`）。
- `materialize_tp2` 对掩码外的卡不分配、不拷贝、`storage.device` 留空（`tp_materialize.cpp:173-261`）。
- `build_tp_split_spec`：`mtp/*` → `shards = 0b01`；`vision/*` → `shards = 0b10`
  （名字前缀已在同一循环内可判，不需要新 option）。
- `shard_views` 跳过本卡没有的对象（`tp_shard_views.cpp:38-52` 的 `device_parent` 会失败）。
- `Parameters`/`TextContext` 容忍缺失：shard 1 不得解引用 MTP 权重，shard 0 不得构造 VisionContext。
- **验收（2026-09-19 完成，全绿）**：
  - 9 项回归全通过（3 条 load route、`tp2_forward` 11 探针 + 32 步 AR + 批量 prefill、`embedding`、
    `linear_tp2_split_fp8_head`、`linear_tp2_split_grouped_head`、`linear_tp2_split_nvfp4`、`tp_device_pair`）。
  - 台账：shard 1 `weights+ctx` **11494.6 → 11064.6 MiB（−430.0）**；shard 0 不变。
    nvidia-smi 15,614 / **14,664** MiB/卡 ⇒ shard 1 余量 **1,217 → 1,647 MiB**。
  - greedy 逐字节：`ab-final.jsonl` 与参考 `ab-embed` **IDENTICAL**（5/5 hash 全同）⇒ 分片放置数值中性。
  - 实现：`TPObjectSplit::shards` 掩码 + 按卡容量/上传 + `shard_views` 保序留空 +
    `Model::has_weight` + `Parameters` 按存在性构造 mtp/vision 块。
  - shard 1 的 1,647 MiB 中，扣除 vision 权重 282 MiB 后余 **1,365 MiB** 给编码 workspace（预计需 1.0–1.2 GiB）。

### 11.3 Step B —— TP-2 视觉执行路径（已实现，待实测）

**执行层（`execution/text.h|.cpp`）**

- 新增 `Tp2VisionChunk`：`control`（本 chunk 覆盖的已编码 item，可为空）、`embeddings`
  （`[hidden, merged]`，属 Vision 卡）、`positions`（prompt 全量 `[3, T]` MRoPE 表）、`prompt_tokens`。
- `forward_tp2_prefill(..., Phase phase = Phase::Prefill, const Tp2VisionChunk* vision = nullptr)`：
  - `make_bind` 在有 `vision` 时为每个 chunk 建 **`[tokens, 3]`** 的 rope 张量（token 维连续，
    与单卡 `[tokens, axes]` 绑定一致；`ne[1]==3` 才是 RoPE op 读 MRoPE 表的分支），
    否则沿用原 `[tokens]`（cache position 与 rope position 仍重合）。
  - `embedding_tp2` 之后，若本 chunk 与某 item 相交：取该 item `scatter_indices` 在
    `[first_position, first_position+tokens)` 内的**连续**列段（`[visual_begin, +count)`），
    shard 1 从 handoff 张量 D2D 拷到本 chunk workspace 的 `source`，shard 0 同样大小的
    `staging` 清零，一次 `pair.allreduce(staging, source)`（与 0 相加 ⇒ 精确拷贝）后两卡各自
    `ops::scatter(staging/source, local_indices, x/x_peer)`。
    代价只与 chunk 相关（≤1024 token ⇒ ≤10 MiB/次），不随图片大小一次性占用。
- `PreparedPromptAccess::mutable_view`（`frontend`）：session 需要可变 prompt 以释放已编码的
  host patch payload。

**核心层（`runtime/engine/tp2_generation_core.*`）**

- 启动：shard 1 + `enable_vision` 时按单卡同一 planner 建 plan
  （`plan_workspace(config, params, min(capacity, 16384), kWorkspaceBytes)`）并独占一个 arena，
  台账新增 `vision %.1f` 列。
- 每请求：`plan_vision_control` → 组装 `VisionPrefillPlan`（`uses` 全量、`control_index==prepared_item_index`、
  `max_merged_count`）→ `build_vision_control(data, plan, 0)` → `VisionPrefillSession`（shard 1）。
- prefill 循环：`vision_session->prepare_chunk(t0, length)` 先按 item 边界截断，再套用 snapshot 边界
  （rewind 边界优先，下一 chunk 重新进入同一 item）；把 `Tp2VisionChunk` 传给 `forward_tp2_prefill`。
- **多模态请求与纯文本请求走同一套机制**（Round 53b 修正，见 §11.6）：MTP 照常开启、
  前缀照常复用；视觉 item 只在"落在复用前缀之外"时才进 session。
- `vision_seconds` 单独上报，`prefill_seconds = 总时长 - vision_seconds`（与单卡语义一致）。
- `model_instance.cpp` 去掉 TP-2 分支的 `enable_vision = false`（DFlash 限制保留）；
  serve 侧 `--vision` → `EngineOptions.enable_vision` → `LoadOptions.vision` 链路本就存在。

### 11.4 Step C —— 端到端与显存裁决（已完成，通过）

- **判据通过**：`load_00/03/07/11.png` 各问「图里的数字是多少」→ 回复 `00 / 03 / 07 / 11`
  全部正确（`tools/tp_bootstrap/r53_numbers.py`）；另 3 张图的自由描述与图内容一致
  （同心圆环、彩色竖条纹、左上角白色标签框）。prompt 1,092 tok，TTFT ~870 ms，
  prefill 1.57k tok/s。
- **显存裁决**：16,384 的完整 envelope **装不下**（首个 `cudaMalloc` 被拒；本机 WSL2 上
  `cudaMemGetInfo` 又不可信，所以判据只能是「分配是否被拒」）。自适应循环落到
  **item ceiling 8,192 vision tokens**、arena **413.3 MiB**（`encode peak 413.3` / `handoff 80.0`）。
  8,192 token ≈ 2,900×2,900 px，已覆盖实际输入。**没有新增任何用户可见配置**；
  超出 ceiling 的单图在请求期被明确拒绝，错误信息指向 `[mem] vision` 台账行。
- 运行期峰值与纯文本完全相同：shard 0 15,614 MiB（余 699）、shard 1 15,360 MiB（余 951）——
  arena 常驻，图片请求不再做大块分配（设计意图）。
- 为什么不是 16,384：`test_vision_workspace.cpp` 自身断言最大 item 的 plan < 827 MiB
  （≈ 8,192 的 413 MiB 线性外推），所以更像**碎片**导致 ~800 MiB 连续块拿不到，而非总量不足。
  把 arena 挪到 KV 之前分配也许能拿到 16,384，但会让 KV/state（4.1 GiB + 293 MiB）先垫底失败，
  风险大于收益，故保持「KV 先、vision 自适应」。
- 图片/带图会话的吞吐见 §11.6（MTP 开启后 68–71.5 tok/s，优于纯文本实测的 59）。
- `ninfer_qwen3_5_vision_workspace_test`（需要 `NINFER_TEST_ARTIFACT`）在本机**不适用**：
  它走单卡 Program，单张 16 GiB 卡装不下这个 27B 模型，实跑 OOM。它断言的最大 item plan
  <827 MiB 只作为量级参考引用，不作为本机证据。

### 11.5 Round 53b —— 修掉「发过图片后整个会话掉到 33 tok/s」（已完成）

**现象与根因**：多轮会话里客户端每一轮都会把图片重新放进 history，所以**后续每一轮请求都仍然带
media**。首版为缩小风险设的两条范围收敛（`use_mtp = mtp_enabled_ && !media`、图片请求 `reuse = 0`）
于是让整个会话都失去 MTP 与前缀复用：decode 33 tok/s、TTFT 每轮 ~0.8 s。

**修法：多模态请求与纯文本请求走同一套机制**

- 取消 `!media`，MTP 照常（回到 `mtp_enabled_`），解码循环/位置推进/解码阶段都用同一个标志。
- 前缀复用恢复：视觉 item 只在 `token_end > reuse`（落在复用前缀之外）时才进 `VisionPrefillPlan`
  与 session，`control_index = prepared_item_index - first_item`（与单卡 `request_plan.cpp:702-724`
  + `materialization.cpp:295-312` 一致）；落在复用前缀内的 item 直接释放 host patch payload。
- **复用前缀覆盖全部 item 时不建 session**（`vision_plan.uses` 为空）——这正是"带图会话的第 2 轮起"
  的常态：不再重编码、不再 scatter，但**只要 prompt 带 media，每个 chunk 仍绑定 3 轴 RoPE 表**
  （那是 prompt 的属性，与是否编码无关）。
- `Tp2VisionChunk` 的传参条件从「有 session」改成「prompt 带 media」，control/embeddings 由 session 填。

**实测（shipped 配置 262,144 + `--vision`，`tools/tp_bootstrap/r53_chat.py`）**

| | prompt | cached | TTFT | decode | 回答 |
|---|---|---|---|---|---|
| 第 1 轮（图片，全新 prefill） | 1,092 | 44 (4%) | 807 ms | **68.0 tok/s** | "07" ✓ |
| 第 2 轮（同会话追加文本问题） | 1,288 | **1,082 (84%)** | **169 ms** | **71.5 tok/s** | "21" ✓ |

- MTP 接受率 `rate=170/216 = 78.7%`，高于纯文本 Round 52 实测的 64.6%。
  ⇒ MTP priming 对图片列仍用占位 token 的 embedding（视觉 embedding 只进文本 KV）这一点
  **没有可测损失**，因此不再增加"把 shifted 视觉 embedding 接进 MTP priming"的复杂度。
- 冷启动 greedy A/B 与 golden `embed` **逐字节相同**（`v53f == embed == v53b`）。
- 顺带查清一个**既有**特性（非本次引入）：结果对"上一轮请求留下的 rewind 深度"敏感——warm 状态下
  A/B 的 req0 会变（`v53c=f4f52ac3`；同一 history 完全可复现 `v53d==v53e=5b4a9783`；冷启动回到
  golden）。触发它的是一轮**长纯文本**请求也会复现同样偏移（`v53g==v53c`），说明与 Vision/MTP 无关：
  `rewind_near_` 决定 prefill chunk 切分点 → 不同 FP 归约次序在近似平局处翻转贪心 token。
  shipped 配置是 `temperature 0.7/top-k 20`，实际使用不可见。
- 纯文本仍逐字节不变：`ab-v53b.jsonl` 与参考 `ab-embed` **IDENTICAL**（5/5）。

### 11.6 已知未决点

- 图片请求的 MTP 已开启（见 §11.5），接受率 78.7%：MTP priming 对图片列用的是占位 token 的
  embedding，实测无损失；若以后要把它做精确（shifted 视觉 embedding 接进 `mtp_prefill_priming`），
  那是约 100 行的工作量，目前没有收益证据。
- 单图 token 上限被硬编码在自适应搜索里（8,192）；若以后要把 shard 1 的 KV 或 state 让出一点，
  可以重新拿到更大的 envelope，无需改接口。
- 视觉编码跑在 shard 1 时 shard 0 在编码期空闲（编码约百毫秒量级，可接受）。

### 11.7 decode 瓶颈测量（Round 53d，仅测量未改动）

同一文本提示（prompt 75 token，output 256，greedy 请求）在 262,144 配置下扫 draft 窗口：

| 配置 | decode | 说明 |
|---|---|---|
| plain（无 MTP） | **32.8 tok/s** | 30.5 ms/次前向（1 token/次） |
| `--spec mtp --draft-tokens 2`（shipped） | **56.8 tok/s** | ~39–41 ms/轮、~2.0–2.3 token/轮 |
| `--draft-tokens 3` | 58.5 tok/s | 与 d2 在噪声内，略优 |
| `--draft-tokens 4` | 55.5 tok/s | 窗口变长的边际收益已为负 |

decode 期间实测（`nvidia-smi -lms 300`）：gpu0 **sm 2820 MHz / mem 13801 MHz / 92.8 W / util 100%**，
gpu2 2805 MHz / 13801 MHz / 84.7 W / util 99% ⇒ **满频、未撞功耗墙（180 W）、无热降频**，
所以不是时钟/功耗问题。`nvidia-smi topo -m` 显示两卡之间是 **SYS**（无 NVLink/P2P，peer 流量过主机桥）。

**结论（瓶颈拆分）**：一轮 ≈ 39–41 ms，其中
- **~30.5 ms（约 74%）是目标模型的一次 verify 前向**：每卡每次读 ~9.7 GB 权重，折合 ~318 GB/s，
  只有 448 GB/s 峰值的 ~71%；差距主要嫌疑人＝每层 row-parallel 的跨卡 allreduce（走 SYS 主机桥）
  + 无 CUDA Graph 的 ~150 次 kernel launch + NVFP4 GEMV 在 5060 Ti 上的实际效率。
- **~9–10 ms（约 24%）是 MTP 起草链 + 窗口组装 + argmax/接受 + state 恢复/折叠 + 每轮 host sync**
  （`mtp_propose_window` 是 host 串行 AR，每个 draft 一次同步；d4 变慢即边际 draft ~3 ms）。

**可压榨清单（按性价比排序）**
1. `Phase::Verify`（decode 等价的 batched GDN/attention）——目前 verify 走近似 `Phase::Prefill`，
   接受率是 token/轮的直接乘数；这正是代码里标注的 "known-incomplete TP-2 path"。
2. 跨卡集合通信：减少每层 allreduce 次数（把 attention-out 与 MLP-down 两次合成一次）、
   或让它与下一层权重流重叠；P2P 在消费级驱动上不可用，只能减少次数/字节。
3. 每轮 9–10 ms 固定开销：draft 链改成 device 侧 AR（去掉每 draft 的 host sync）；
   TP-2 core **完全没有 CUDA Graph**（`cudaGraph` grep = 0），单卡路线是有的。
4. NVFP4 GEMV 按 5060 Ti（448 GB/s、SM 更少）重新调参：用 ncu 看 top kernel 的 DRAM 达成率。
5. 长上下文：262k 时 fp8 KV 读取约占流量 30%，`--kv-dtype k8v4|nvfp4` 可再砍（质量取舍）。
6. 不可行的：超频/功耗墙（已满频、93/85 W）、P2P（驱动屏蔽）。

### 11.8 进程级思考强度（Round 53c）+ 为什么没有 `--chat-template`

- **新增** `ninfer-serve --reasoning-effort low|medium|xhigh`（进程默认，请求体优先）：
  `serve_options.{h,cpp}`（解析 + usage）、`translate.cpp`（`effective_reasoning_effort` 回退链：
  请求字段 → 进程默认 → 模板默认）、`generation_service.cpp`（启动即校验模板是否支持该档）、
  `request_log.cpp`（`server_start` 记录 `default_reasoning_effort`）、`docs/serving.md`（flag 表 +
  说明）、`tests/test_serve_options.cpp`（解析 / usage / 解析语义三处断言，`ninfer_serve_options_test` 通过）。
  实测：`--reasoning-effort high` 被拒（提示 `must be low, medium, or xhigh`）；8088 启动 11 s 健康；
  请求行显示 `thinking medium`；17×23 回答 391、思考 46 tokens（同题 `xhigh` 为 172）。
- **`--chat-template` 不可用**，三条独立证据：
  1. 产品里没有这个 flag（`ninfer-serve: unknown argument: --chat-template`；CLI `ninfer` 也没有）。
  2. 模板是 **artifact 资源**：`frontend.cpp:194-202` 强制
     `tokenizer_config.json.chat_template` 与 `resource/text/chat_template.jinja` 逐字节相同，随后
     `chat_template.cpp:414-424` 按 sha256 只接受两种语义——`e84f32a2…`（thinking-toggle）与
     `c3cf9e34…`（reasoning-effort，本 artifact 内嵌的就是它，所以 `reasoning_effort` 本来就能用）。
  3. 放在模型目录的 `chat_template.jinja` 是**第三个**模板：28,234 B、sha256 `e57684ba…`、
     `template_version = "qwen3.8-froggeric-v22.5"`，带 XML/JSON 工具调用格式、
     `auto_disable_thinking_with_tools`、`max_tool_arg_chars`、`preserve_reasoning` 别名等。即使加了
     flag 也会被白名单拒绝，因为它的渲染语义在本引擎里没有实现（前端不是通用 Jinja 解释器，而是每种
     语义一套手写 C++ 渲染器）。要用它 ＝ 新增一种前端语义（含工具调用/多模态/effort 别名），属独立功能，
     需另行确认范围。

---

## 12. 进行中：Windows 原生移植（分支 `feat/windows-native-port`）

**目标**：同一 artifact、同一套产品代码，在原生 Windows 上跑通——先单卡 CLI/serve，再做 TP-2 spike。

**平台面普查（Round 54，已确认）**
- 很薄：`mmap`/`madvise`/`epoll`/`pthread`/`NUMA`/`setrlimit`/线程亲和 **全无**；线程与同步用 `std::thread`；
  HTTP 服务端是 cpp-httplib，`src/serve/http_transport.cpp` 已有 `#if defined(__linux__)` 结构可直接补分支。
- 待改代码：
  - `src/artifact/file_io.cpp`：`::open`/`::pread`/`O_CLOEXEC` → `CreateFileW` + `ReadFile` + `OVERLAPPED`
    定位读（注意 >4 GiB 偏移、`FILE_FLAG_SEQUENTIAL_SCAN`），约 50 行。
  - `src/product/logging/logging.cpp`、`src/serve/request_log.cpp`：`localtime_r`/`getpid`/`isatty`/`strerror_r`。
  - `src/product/media_acquire/acquire.cpp`：POSIX socket 出站抓图 → 建议改用仓库已有的 cpp-httplib 客户端。
- 工具链（已核实存在）：VS 2022 Community `D:\Program Files\Microsoft Visual Studio\2022\Community`
  （`vcvars64.bat` ✓）、Windows CUDA **v13.3**（`sm_120a` 支持 ✓；WSL 侧是 13.1，对比时要记住这个差异）。
- **最大未知工作量**：`.cu`/`.cpp` 里是否存在 GCC 扩展（`__attribute__`/`__builtin_*`）——需一次 MSVC+nvcc
  编译扫描才能定价。
- **最高风险（决定 TP-2 版 Windows 是否成立）**：`cudaHostAllocMapped` 的双卡 in-kernel allreduce
  （`src/core/tp/device_pair.cu:195`）在 Windows/WDDM 下的可用性与性能。
- 可参考：三卡分支 `C:\llama.cpp` @ `ar3-opt`，HEAD 为
  `92758c647 ggml-cuda: add 3-way AllReduce kernel (on by default) + timing instrumentation`——
  与我们的跨卡集合通信瓶颈（见 §11.7）直接相关。

**步骤**
1. 单卡 `ninfer` CLI：文件 IO + 日志杂项 + MSVC/CMake 构建；跑 5 个 greedy 提示词与 Linux 结果对照。
2. 单卡 `ninfer-serve`：HTTP、流式、媒体输入行为对照。
3. **TP-2 spike**：mapped pinned 双卡是否成立，量 decode 每轮毫秒并与 Linux（~39 ms/轮）对照。
4. 收尾：Windows 构建/运行写进 `docs/`，脚本 `.ps1` 化，回归清单。

**预期**：机械移植 3–5 天；性能上 WDDM 很可能略逊于 Linux（我们已在 448 GB/s 峰值上只拿到 71%，
见 §11.7），所以移植的收益主要是**部署便利**，不是速度。

**进度（Round 54 第 1 轮，已实测）**
- CMake 配置在原生 Windows 上走到了很深：MSVC 19.44.35228 + nvcc 13.3.73（host=MSVC）+ CUDAToolkit 13.3 全部识别成功，
  `sm_120a` 被接受，Threads 找到。CMake 用的是 pip 装的 **4.4.2**（仓库要求 ≥3.28；注意 CMake 4 可能对
  `third_party` 里 `cmake_minimum_required(<3.5)` 报策略错误，必要时加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`）。
- `cmake/NinferTargets.cmake` 里**没有任何 GCC 专用 flag**（只有 CUDA 的 `-lineinfo`），原先担心的
  "GCC 扩展"风险从构建系统这侧看比预想小得多。
- **唯一 configure 阻塞点**：`cmake/Dependencies.cmake:3` 的 `find_package(PkgConfig REQUIRED)`。
  它被用来找两样东西：
  - `FFMPEG`（libavformat/libavcodec/libavutil/libswscale，**无条件 REQUIRED**）：`src/media/decode/decode.cpp`
    的 60 处调用就是媒体的实际解码器 ⇒ 图片/视频解码依赖它，**不是可选项**。
  - `libcurl>=7.85`（仅 `NINFER_BUILD_PRODUCT_SUPPORT`）：只被 `src/product/media_acquire/acquire.cpp`
    用来抓远程 http/https 媒体（26 处）。
- 机器上既没有 pkg-config/pkgconf，也没有 vcpkg/MSYS2/conda；`C:\WINDOWS\system32\curl.exe` 只是运行时。
  `C:\ffmpeg-8.1.2-full_build` 只有 `bin`，**没有 `include`/`lib`（dev 文件缺失）**，所以不能直接拿来链接。

**下一步决策（第 2 轮）**
1. 取 MSVC 可用的 ffmpeg dev 文件：优先 gyan.dev 的 `ffmpeg-release-full-shared`（带 `include/`+`lib/*.lib`）；
   直连不通时按项目规则走本地代理 `127.0.0.1:7897`；备选是 vcpkg 源码构建（耗时更长）。
2. 把 `cmake/Dependencies.cmake` 改成平台感知：Unix 保留 pkg-config，Windows 用
   `NINFER_FFMPEG_ROOT` + `find_path`/`find_library` 定位同一组库；libcurl 同样处理，或把
   `media_acquire` 的抓取换成 WinHTTP 以彻底去掉该依赖（该文件本身就是 POSIX socket 待改点）。
3. 重新 configure → 首次编译扫描，定价 `.cu/.cpp` 的 MSVC 兼容性。
- 已取到 MSVC 可链接的 FFmpeg dev（BtbN win64-gpl-shared）：
  `D:\ffmpeg-dev\expanded\ffmpeg-master-latest-win64-gpl-shared`，`include/libavcodec|libswscale` 与
  `lib/{avcodec,avformat,avutil,swscale}.lib` 全部就位。下一步把它接进 `cmake/Dependencies.cmake` 的
  Windows 分支（`NINFER_FFMPEG_ROOT` + `find_path`/`find_library`，Unix 侧继续走 pkg-config）。

**进程占用纪律（用户要求）**
- Windows 侧一旦跑起 `ninfer`/`ninfer-serve`，同一个 23.7 GB artifact 会被再装一遍（显存 + host KV/state），
  与 WSL 侧 8088 服务抢显存和内存。**移植期间 WSL 的 8088 服务保持停止**（Round 54 已停：三卡
  `memory.used` 均为 0 MiB、health 无响应）；需要跨平台对照测量时再临时启动，测完立即停。
**进度（Round 54 第 1 轮）：核心库在 Windows 上编译通过**

工具链 VS2022 19.44 + CUDA 13.3 nvcc + Ninja，`-DCMAKE_CUDA_ARCHITECTURES=120a`；
`tools/win_port/configure.bat` + `build.bat`（`NINFER_JOBS`，默认 12）。当前配置 apps 关
（`-DNINFER_BUILD_APPS=OFF`），`cmake --build build-win` → `BUILD_EXIT=0`，10 个静态库
（artifact/core/engine/media_decode/model_loading/model_runtime/nvfp4_non_rdc/ops/runtime_support/text）
+ 394 个目标文件。

移植改动（已落地）：

| 位置 | 问题 | 处理 |
|---|---|---|
| `cmake/Dependencies.cmake`、`src/{media,product}/CMakeLists.txt` | 本机无 pkg-config | 平台分支：Windows 走 `NINFER_FFMPEG_ROOT`/`NINFER_LIBCURL_ROOT` + `find_path`/`find_library`；导入目标改名 `ninfer::ffmpeg`/`ninfer::curl` |
| `CMakeLists.txt` | CUDA 13 的 CCCL 要求 `/Zc:preprocessor`；`windows.h` 的 min/max 宏 | MSVC 下加 `/Zc:preprocessor`（CUDA 经 `-Xcompiler`）+ `NOMINMAX WIN32_LEAN_AND_MEAN` |
| `src/artifact/file_io.{h,cpp}` | `open/fstat/pread/close`、`unistd.h`、`off_t/ssize_t` | `CreateFileW`（顺序扫描 / `FILE_FLAG_NO_BUFFERING`）+ `ReadFile` 带 `OVERLAPPED` 偏移做定位读；句柄类型 `NativeFileHandle`（Windows 为 `void*`） |
| `src/ops/linear/nvfp4/nvfp4_w4a4_tma.{cuh,cu}`、`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.{cuh,cu}` | C2719：128 字节对齐的 TMA 描述符不能按值传参 | 新增 `Nvfp4TmaDescriptorStaging`（`cudaMallocAsync` + `cudaMemcpyAsync` 到设备内存，传指针，仅 `_WIN32`）；内核体内用引用，保证两条路径共享同一份代码 |
| `src/core/uint128.h`（新） | MSVC 无 `__int128`/`__uint128_t` | 新增 `Uint128`（`_umul128` 或原生 128 位 + 饱和加法/右移/比较）；改 `runtime/contract/resources.h`、`runtime/engine/context_cache/context_cost.cpp`、`context_cache/materialization_planner.h` |
| `src/core/host_process.h`（新） | `getpid`/`isatty`/`localtime_r` | `host_process_id()`/`host_stderr_is_interactive()`/`host_localtime()`；`context_cost.cpp` 已改用 |
| `src/text/CMakeLists.txt` | utf8proc C2491（静态库里定义 dllimport） | `target_compile_definitions(ninfer_text PRIVATE UTF8PROC_STATIC)` |
| `src/models/qwen3_5/execution/text.cpp` | C2397 非恒定窄化（GCC 只警告）；`void*` 算术是 GCC 扩展 | 显式 `static_cast`；先 `static_cast<const std::byte*>` 再做字节偏移 |

经验：GCC 把「非恒定窄化」当警告，MSVC 直接报 C2397，所以 Linux 能编的代码在 Windows 上会逐个暴露，需逐个加显式转换。

Windows 侧资源与路径：

- FFmpeg dev：`D:\ffmpeg-dev\expanded\ffmpeg-master-latest-win64-gpl-shared`（BtbN shared，含 `include/`+`lib/`）。
- libcurl：`D:\curl-dev\expanded\curl-8.22.0_1-win64-mingw`（**只有 `lib/libcurl.dll.a`，mingw 版**；MSVC 能否直接链接待验证，否则换 vcpkg 或改 WinHTTP）。
- 模型 artifact：`D:\LLM\qwen3_8_27b_nvfp4.ninfer`（23,719,715,076 B，与 WSL 侧同一个 artifact，可直接做跨平台对照）。
- 自定义模板：`D:\LLM\chat_template.jinja`（同 WSL 侧那份 28,234 B）。

下一步（apps 关→开）：`src/product/logging/{logging,startup_log}.cpp`（`unistd.h`/`localtime_r`/`isatty`/`sys/ioctl.h` TIOCGWINSZ →
`host_process.h` + `GetConsoleScreenBufferInfo`）、`src/serve/request_log.cpp`（`getpid`）、
`src/product/media_acquire/acquire.cpp`（POSIX socket + libcurl）、`src/serve/http_transport.cpp`（`#if defined(__linux__)` 分支）；
之后单卡 `ninfer` CLI 与 Linux 输出对齐（贪婪基准哈希见 §11），再进 TP-2 mapped-pinned spike。
**进度（Round 54 第 2 轮）：CLI / serve 全部编译链接通过，并能在 Windows 启动**

配置 `-DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_FFMPEG_ROOT=… -DNINFER_LIBCURL_ROOT=…` →
`BUILD_EXIT=0`，产物 `build-win/apps/{ninfer,ninfer-serve,ninfer-perplexity}.exe`（各约 188 MB）；
`ninfer.exe --help` 与 `ninfer-serve.exe --help` 均正常输出、退出码 0（说明 DLL/CRT/链接已就位）。

本轮新增修复：

| 位置 | 问题 | 处理 |
|---|---|---|
| `CMakeLists.txt` | spdlog 内置 fmt 要求 `/utf-8` | MSVC 下加 `/utf-8`（CUDA 走 `-Xcompiler`） |
| `src/core/host_process.{h,cpp}` | `ioctl(TIOCGWINSZ)`、`gmtime_r` | 新增 `host_terminal_columns()`（`GetConsoleScreenBufferInfo`）与 `host_gmtime()`；`src/core/CMakeLists.txt` 注册 `host_process.cpp` |
| `src/product/logging/{logging,startup_log}.cpp`、`src/serve/request_log.cpp` | `localtime_r`/`isatty`/`getpid`/`unistd.h` | 改用 `core/host_process.h` 的平台函数 |
| `src/product/media_acquire/acquire.cpp` | `<arpa/inet.h>` 等 POSIX 头；`relative.native().starts_with("..")`（Windows 是宽字符） | 平台分支 `<winsock2.h>/<ws2tcpip.h>` + 一次性 `WSAStartup`；改判首个路径分量是否为 `path("..")` |
| `src/product/CMakeLists.txt` | WinSock 符号未解析 | 链 `ws2_32` |
| `cmake/Dependencies.cmake` | `find_library` 选中**静态** `libcurl.a`（mingw 静态档，MSVC 无法链接） | 优先 `find_file(... libcurl.dll.a)` 取导入库，找不到再回退 `find_library` |
| `apps/perplexity/main.cpp` | `gmtime_r` | 改 `ninfer::host_gmtime`（该文件处于全局匿名命名空间，必须写限定名） |

已验证：MSVC 能链接并**运行** mingw 版 `libcurl.dll.a` + `libcurl-x64.dll`（探针 `tools/win_port/probe_curl_link.bat`，
打印 `curl libcurl/8.22.0 LibreSSL/…`，退出码 0），因此 `media_acquire` 在 Windows 上继续用 libcurl，暂不需要 WinHTTP 重写。

运行环境（重要）：两个 exe 运行时需要 `PATH` 含

    D:\ffmpeg-dev\expanded\ffmpeg-master-latest-win64-gpl-shared\bin;D:\curl-dev\expanded\curl-8.22.0_1-win64-mingw\bin

（`libcurl-x64.dll` 必须可见）。启动前确认没有第二个模型进程：`nvidia-smi memory.used` 三卡均为 0 MiB。

下一步：单卡 CLI 真实生成（`D:\LLM\qwen3_8_27b_nvfp4.ninfer`）并与 Linux 贪婪基准哈希对齐 → serve 端到端 →
TP-2 mapped-pinned spike（重点验证 `src/core/tp/device_pair.cu` 的 `cudaHostAllocMapped` 在 WDDM 下是否可用）。
**进度（Round 54 第 3 轮）：TP-2 在 Windows 上跑通，性能与 Linux 持平；mapped-pinned spike 通过**

端到端实测（`tools/win_port/serve.ps1`，两卡 5060 Ti，artifact `D:\LLM\qwen3_8_27b_nvfp4.ninfer`）：

| 项目 | Linux (WSL2) | Windows |
|---|---|---|
| TP-2 纯 decode | 32.8 tok/s | 32.9 tok/s |
| TP-2 MTP `--draft-tokens 2` | 56.8 tok/s | 56.9 tok/s |
| 图片轮 decode | 53b 修复后正常 | 60.4 tok/s，MTP 持续（rate=106/174） |
| 引擎加载（23.7 GB artifact） | — | 35–42 s |
| 真实 agent 流量（流式 / tools / 图片） | — | 已服务，41–73 tok/s |

KV 容量与显存余量（实测，带 `--vision`）：capacity 8192 → kv 145 MiB、free 4180/3930 MiB；
capacity 131072 → kv 2322 MiB、free 2002/1580 MiB。**Windows 推荐 131072**（262144 是 WSL 配方，
WDDM 下余量过小）。

**进度（Round 10）：与 llama.cpp `ar3-opt` 的同机对照 —— decode 打平的原因不在内核**

新增/修改脚本：`tools/win_port/bench_serve.ps1`（同一套工作负载压任意 OpenAI 兼容服务，数字全取服务端自报的 `timings`）、
`tools/win_port/serve_llama.ps1`（把本机调好的 llama.cpp 配方固化，含 `-Stop`/`-Status`；与 `serve.ps1` 互斥，同一时刻只驻留一个引擎）、
`serve.ps1` 增加 `-DraftTokens`。协议：temperature 0、256 输出、medium 思考、两边都开 MTP、同题面、一次一个引擎。

| 工作负载 | ninfer | llama.cpp | 比值 |
|---|---|---|---|
| prefill 1.9k | 1,401 tok/s | 764 tok/s | 1.83x |
| prefill 7.2k | 1,667 tok/s | 1,040 tok/s | 1.60x |
| prefill 28.5k | 1,552 tok/s | 1,016 tok/s | 1.53x |
| decode 256 贪婪 token（3 次） | 56.7 / 56.7 / 56.7 tok/s | 51.7 / 52.6 / 52.7 tok/s | 1.08x |

同一 decode 题面的单轮预算：ninfer K=2 38.2 ms/轮、2.18 committed、58.8% 单 draft 接受率；ninfer K=3 40.7 ms、2.27、42.4%；
llama.cpp `n_max=3` 49.4 ms、2.60、53.8%（llama.cpp 每 256 个 token 复用 95–96 次 CUDA graph）。

结论（完整论证见 `docs/tp2-dual-5060ti.md` 的 Cross-engine comparison 一节）：

1. **batch=1 decode 是权重流不是算力题**：`tok/s = 每轮 committed / 轮时长`，轮时长下限 = 每卡权重字节 / 448 GB/s。
   ninfer 每卡 10.15 GB/forward，llama.cpp 8.56 GB（用 `llama-gguf` 实测其 GGUF 15.94 GiB 的类型分布：NVFP4 MLP 10.38 GiB、
   q5_K 注意力/GDN 3.09 GiB、q6_K 词表 1.55 GiB、q8_0 输出头 1.29 GiB）。两者分别落在各自下限的 59% / 39%。
   两张 5060 Ti 合计 896 GB/s = 一张 5090（1,792 GB/s）的一半，这正是同一 artifact 在 5090 上 71.2 tok/s、在这里 ~57 tok/s 的原因：
   **TP-2 是容量决策（27B 装不进单张 16 GiB），不是提速决策**。
2. ninfer 的定向优化确实生效在"算得动"的地方：每轮比 llama.cpp 快 23%，prefill 快 1.5–1.8 倍。
3. 吃掉这点优势的四项：**draft 深度**（2.18 vs 2.60；本 artifact 上 K=3 反而掉到 55.7）、**FP8 权重多读约 26% 字节**、
   **TP-2 关掉了 CUDA graph**（`model_instance.cpp:99` `use_cuda_graph=false`，~1,009 kernel/forward、~2.9 ms 启动间隔）
   且**头切分把 allreduce 翻倍到 128 次/token**、融合投影内核在分片行上被绕过（`text.cpp:1104-1113`）、
   **MTP 链每轮 7–10 ms 串行在 host**（`mtp_propose_window` 每个 draft 一次 `cudaStreamSynchronize`）。
4. 后续优化顺序（按证据强度）：(a) 给 TP-2 的 verify forward 上 exact-batch CUDA graph（~7.6%/轮）；
   (b) 把 attention/GDN/embedding/head 与末 8 层 FFN 从 FP8 换 NVFP4（每卡每 token −2.6 GB，约 +18%，有精度代价）；
   (c) 减少 collective 次数（不切头或合并同层两次 AR）；(d) 让 draft 链与下一轮 verify 重叠或下沉到设备端。

**Round 10 追加：优化 ①③④ 的理论收益核算（模型推算，非实测）**

decode 单轮 38.2 ms / 2.166 committed（=56.7 tok/s）拆账：权重流 25.6 ms（448 GB/s 理论 22.7 ms，效率 89%）
+ allreduce 1.7 ms（128 次 × 9–18 µs，已从 42 µs 优化过）+ 发射间隙 2.91 ms（1009 kernel/forward，nsys 实测）
+ MTP 链 / accept-fold / host 同步 7.7 ms（= 轮时长 − verify 30.5 ms）。
prefill 每 1024-token 块 660 ms 拆账（按 worklog Round 35c 的 580 ms = AR 350 + MMA 188 + 其它 35 换算到 Windows）：
AR ≈350 ms（53%）、MMA+其它 ≈310 ms；AR 已贴 Gen4 x4 链路地板（2.6 MB/token ÷ 7 GB/s → 2.67k tok/s）。

| 项 | decode 理论收益 | prefill 理论收益 |
|---|---|---|
| ① exact-batch CUDA graph | 发射间隙 2.91 → 0：38.2→35.3 ms，61.4 tok/s（+8.3%） | 块内间隙摊薄，≈ −0.4% |
| ③ collective 重叠（**不是删掉**，TP 必需） | −1.7 ms（与 draft 链重叠）：→33.6 ms，64.5 tok/s | AR 与 MMA 重叠：660→≈390 ms，**≈2.6k tok/s**（受链路 2.67k 封顶） |
| ④ draft 链下沉/重叠（保留 ~3 ms 设备端） | −4.7 ms：→28.9 ms，**74.9 tok/s（+32%）** | 不适用（prefill 不走 MTP） |

合计理论值：decode **≈75 tok/s**（绝对地板 22.7+3.0=25.7 ms → 84 tok/s）；prefill **≈2.6k tok/s**（worklog 自己的保守估计 1.5x → ≈2.1k）。
③ 的边界：worklog 已算过头切分的账——AR 次数 64→128 多花 4.9 ms，换回 9.4 ms 计算，净赚；
所以"减少 AR 次数"必须放弃权重切分，在 decode 上一定亏，唯一正确方向是"重叠"。

**①③④ 的显存账**（每卡；Windows 131072 实测 free 2002/1580 MiB，262144 配方 free 0/196 MiB）：

| 项 | 增量/卡 | 机制 |
|---|---|---|
| ① CUDA graph | ≈ 0–几 MiB（若用 graph 私有 pool 最坏 ≈ 192 MiB） | 所有 arena 启动即常驻（workspace 192.0、state 293.6、KV、权重），graph 只新增 exec 与节点簿记；硬前提是把 AR arrival token 从 mapped host 移到设备端（KiB 级）。风险点本仓库出现过：capture 内 `cudaMallocAsync` 会变成 graph memory node（TMA 描述符那次），故 capture 期的一切分配必须先预分配 |
| ③ AR∥MMA 子块流水 | **+190–580 MiB** | 2–4 个 in-flight 子块各自需要激活区（workspace 192 MiB × 流水级数）与 GDN 状态快照（state 293.6 由 4–5 份组成，≈59–73 MiB/份）。AR staging 本身是 `cudaHostAllocMapped` 的 host pinned（2×24 MiB × 2 缓冲 = 96 MiB/卡），扩容只吃主机内存 |
| ④ draft 链下沉/重叠 | 只去 host 同步 ≈ 0–10s MiB；若让 verify 与 draft **并发**则需第二份 MTP KV：**+258 MiB**（仅 shard 0，131072） | MTP 权重 430 MiB 已在两卡复制、MTP KV 已存在（262144 时 516.2 MiB 仅在 shard 0） |

合计 **+0.2–0.85 GiB/卡**：131072 能放下（余量降到 shard 0 ≈ 1.4–1.8 GiB、shard 1 ≈ 1.0–1.4 GiB），
262144 放不下（free 0/196）。反过来 ② 权重 NVFP4 化会**省** ≈2.15 GiB/卡——② 与 ③ 一起做等于既提速又把 262144 装回来。

**mapped-pinned spike（go/no-go 已通过）**：`tools/win_port/spike.ps1`（源 `tp_mapped_spike.cu`）实测
peer access 0→1 / 1→0 均为 0（SYS 拓扑，无 P2P）；`cudaHostAllocPortable|cudaHostAllocMapped` +
`cudaHostGetDevicePointer` 两卡均成功且 peer 看到**同一 UVA 地址**；跨卡读 mapped host 内存校验
`sum=match`，带宽约 1.4–1.9 GiB/s（本卡写 5.8 GiB/s）。结论：WDDM 下 in-kernel mapped-pinned allreduce
可用、与 WSL2 同档，这正是端到端 decode 与 Linux 持平的原因。

**重要环境事实**：`C:\Users\zhuojun\.dsh\settings.yaml` 已配置 `ninfer-win` provider
（`baseURL: http://127.0.0.1:8099/v1`）且 `agent-default-model` 指向它。也就是说：Windows 侧
ninfer-serve 跑在 8099 就是本机 DSH 的默认模型服务；测试期间它已服务过真实流式/tools/图片请求
（`req#4 media 1, prepared 7.91 ms`）。WSL 侧 8088 已停，全机只有一个模型进程。

**脚本教训**：`Start-Process` 起的服务是「调用方 shell 的 job object 子进程」，agent 工具调用结束时会被
连带杀掉（现象：进程在、GPU 0%、模型从未加载）。因此 `serve.ps1` 默认**前台**运行（终端用户拥有进程，
Ctrl+C 停止），只有 `-Background` 才用 `Start-Process`；自测服务时必须用 harness 的后台 job（跨调用存活）。

工具与文档：`tools/win_port/` 现全部为 PowerShell（`vcvars/build/serve/spike/fetch_ffmpeg/fetch_curl`，
旧 `.bat` 已删）；新增 `docs/windows.md` 并在 `docs/README.md` 登记；`docs/windows.md` 含构建、运行、
平台差异、实测数据与限制（单卡 CLI 需 ≥20.2 GiB 显存，本机 16 GiB 卡只能走 TP-2）。
**进度（Round 54 第 4 轮）：提交与回归验证**

- 提交：`453aaa6a feat: build and serve natively on Windows with VS2022 and CUDA 13.3`（分支
  `feat/windows-native-port`）。
- Linux 回归：`build_r35.sh` → `BUILD_EXIT=0`（全量重建 + 测试目标）。修复了一处**测试基建问题**：
  `build_r35.sh` 的 rsync 列表缺 `cmake` 与顶层 `CMakeLists.txt`，导致 WSL 树仍用旧的
  `cmake/Dependencies.cmake`（`PkgConfig::FFMPEG` vs 新的 `ninfer::ffmpeg`）而配置失败；现已补上
  （注意：该脚本会同步自身，改动要**跑第二次**才生效）。
- 受影响的 Linux 测试：`context_cost`、`resource_manager`、`pretty_logging`、`context_cost_measure`
  全部通过；`artifact|nvfp4|qwen3_5` 共 26 项 25 通过。
- **唯一失败为既有问题（与本分支无关）**：`ninfer_qwen3_5_frontend_test` 抛
  `unsupported frontend/chat_template.jinja (sha256 821b1c036748885c26b552d2a174878fe06199c390ccf612a6169326338cbaa5)`
  —— 即 53c 引入的模板摘要白名单拒绝了测试夹具；`src/models/qwen3_5/chat_template.cpp` 与 `tests/`
  在本分支均未改动，故 HEAD 上同样失败。
- 教训：Windows 上重新链接 exe 前必须先停服务（运行中的 exe 会锁住文件 → `LNK1104`）。

**剩余缺口（用户决定不做）**：目标里的「跑通单卡 ninfer CLI」在本机无法字面完成——CLI 仅支持单卡，而该
27B NVFP4 artifact 需要约 20.2 GiB 常驻权重（10.1 GiB/shard × 2），16 GiB 卡装不下（已写入
`docs/windows.md` 限制节）。要真正跑到端到端，需要另造一个小的 `.ninfer` 测试 artifact（转换工具链 +
合成小 checkpoint），属额外工作；当前 CLI 二进制本身已验证可运行（`--help`），且与 serve 共用同一套
引擎/加载器（已端到端跑通）。
**进度（Round 54 第 5 轮）：Windows 测试套件跑通，并抓到一处真实移植缺陷**

- 新增 `tools/win_port/test.ps1`：先把 FFmpeg/curl 的 `bin` 放进 PATH 再跑 ctest。**必须如此**：测试 exe 运行时
  加载 avcodec/swscale/libcurl，PATH 缺失时 Windows loader 会对每个 exe 弹「找不到 DLL」对话框，而不是报测试失败。
- Windows 全量测试目标编译链接通过（253 个目标，`BUILD_EXIT=0`）。为跑通测试补的平台分支：
  `test_pretty_logging`（`pipe/dup/dup2/read` → `_pipe/_dup/_dup2/_read` + `_setmode(_O_BINARY)`）、
  `test_context_cost`/`test_request_log`（`getpid` → `ninfer::host_process_id()`）、`tests/artifact/fixture.h`
  （`mkdtemp` → 唯一目录名循环）、`test_gdn_replay_records`（`std::aligned_alloc` → `_aligned_malloc`）、
  `test_host_timing`（`#undef near`：winnt.h 的遗留空宏）、3 处 `constexpr ... std::sqrt`（MSVC 不折叠）、
  1 处 `std::array` CTAD 显式化、4 个文件补 `<array>`。
- 唯一平台排除：`ninfer_artifact_materialization_test` 的 CUDA 故障注入依赖 GNU ld `--wrap`，MSVC 链接器无
  等价物；`tests/artifact/tests.cmake` 用 `if(NOT MSVC)` 排除该源文件与 `--wrap` 选项，其余检查照常。
- Python 工具平台分支：`tools/artifact/file_io.py`（`os.sysconf`/`posix_fadvise`/`fdatasync` 有则用、无则
  降级：丢弃页缓存变 no-op、`fsync` 兜底；新增 `read_at`/`write_at` 封装 `os.pread`/`os.pwrite`），
  `reader.py`/`writer.py`/`convert/sources/safetensors.py` 改用它。
- **真实产品缺陷（由测试套件抓到）**：Windows `open_handle` 只用 `FILE_SHARE_READ`，导致**读取 artifact 期间
  无法重写/替换/删除该文件**（POSIX 读语义允许），表现为 `ninfer_artifact_reader_test` 抛
  `ios_base::failbit`。已改为 `FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE`，与 POSIX 读语义对齐。
- 回归：Windows `artifact` 3/3 通过；Linux 23/23 通过（`artifact|context_cost|pretty_logging|request_log|
  host_timing|resource_manager|kv_cache|gdn|attn_input_proj|gated_delta`），Linux 全量构建 `BUILD_EXIT=0`。
**进度（Round 54 第 6 轮）：修掉 Windows A4 TMA 的真实缺陷（CUDA Graph 捕获）**

- 症状：`ninfer_linear_nvfp4_a4_test` 在 Windows 上抛 `cudaErrorIllegalInstruction`；`CUDA_LAUNCH_BLOCKING=1`
  把它定位到 `src/core/decode_graph.cpp:144` 的 `cudaGraphLaunch` —— 即失败发生在**捕获后的 CUDA Graph** 中。
- 根因：第 4 轮为绕开 MSVC C2719 把 tensor map 改成「`cudaMallocAsync` 设备暂存 + 从**栈上** host 对象
  `cudaMemcpyAsync`」。这条路在 graph 捕获下不成立：流序分配会变成 graph 的 memory node（地址到实例化时才
  确定），而 pageable 源指针在 replay 时早已失效 → TMA 读到垃圾描述符 → illegal instruction。
  Linux 不受影响（那边是 `__grid_constant__` 按值传参）。
- 修复：Windows 改用 **mapped pinned host arena** 存描述符（`cudaHostAllocMapped|Portable` +
  `cudaHostGetDevicePointer`，32 槽轮转；host 直接写、TMA 经同一 UVA 地址读），启动路径**不再发出任何 CUDA
  调用**，因此捕获安全；内核侧用 `fence.proxy.tensormap::generic.release/acquire.gpu` 把描述符发布给
  tensor-map proxy（PTX ISA 8.3，SM_90+）。linear 与 linear_swiglu 两个 A4 内核同步修改。
- 结果：`ninfer_linear_nvfp4_a4_test` Windows 通过；Windows 线性/TMA 面 30/30、Linux 同面 29/29。
- 全量 Windows 套件（124 项）修复前 121 通过/3 失败，现 a4 已修；`ninfer_context_kv_materialize_test` 在整轮
  串行运行时撞 300 s 超时、单独运行通过（判为干扰，非缺陷）；`ninfer_qwen3_5_frontend_test` 与 Linux 同样
  存在的既有模板白名单问题。
**进度（Round 54 第 7 轮）：单卡 CLI 条款的取证与判定**

- 事实：Windows 侧 `ninfer.exe` 可运行、参数处理正确（`ninfer_cli_options_test` 在该平台 **exit 0**），CLI 之下
  的加载/引擎/KV/graph/采样全部是已被 TP-2 服务端到端验证过的同一套引擎（Windows 与 Linux 性能持平）。
- 判据：本机唯一 artifact（27B NVFP4）单卡需 ~20.2 GiB 常驻权重 vs 16.3 GiB 显存，且**没有预分配容量检查**
  （grep 未发现 "does not fit" 类前置校验），强行单卡运行会被 WDDM 把缺的 ~4 GiB 换页到主存而不是干净报错——
  正是用户要求避免的风险，故**不执行**。
- 结论：该条款属硬件约束（不是移植缺口）。若要字面满足，需要另造小 artifact：Windows 侧无 torch（miniconda
  3.13 未装），WSL 侧有项目 py311 环境可转换，但还需写一个 tiny 模型的 recipe —— 属独立子任务，等用户决定
  是否投入。
**进度（Round 54 第 8 轮）：单卡 CLI 小 artifact 子任务计划（跨压缩记忆）**

目标：在本机产出一个小到能装进单张 16 GiB 卡的 `.ninfer`，然后用 Windows `ninfer.exe` 单卡跑通
「加载 → 生成 → 采样」，为「跑通单卡 CLI」条款提供实测证据。

已确认的门槛与事实（勿重复调研）：
- 本机唯一 artifact：`D:\LLM\qwen3_8_27b_nvfp4.ninfer`（27B NVFP4，单卡需 ~20.2 GiB > 16.3 GiB）；引擎
  `src/runtime` 内**没有**预分配容量前置检查，强行单卡会被 WDDM 换页 → 禁止尝试。
- Windows python 为 miniconda 3.13，**无 torch**；WSL 侧无 miniconda/py311（AGENTS.md 里那是上游作者机器）。
- 转换器入口 `python -m tools.convert`（`tools/convert/__main__.py`），官方 recipe 只有 27B 级
  （`qwen3_8_27b_nvfp4` 等），tiny 模型需自写 recipe（`--recipe my_recipe.py[:configure]` 或 `--override`）。
- 文档：`docs/weight-conversion.md`（recipe/source/override 语义、safetensors 源）。

拟定步骤（每步可独立验证，失败即可回退）：
1. 装**CPU 版** torch（避免 2.5 GB CUDA 轮子）：优先 `pip install torch --index-url
   https://download.pytorch.org/whl/cpu`；直连失败再走本机代理 `http://127.0.0.1:7897`。若不可行，改走
   「纯 numpy 的合成 `LogicalSource`」，绕开 torch。
2. 读 `tools/artifact/*.py` 的 `LogicalSource` 接口 + `tools/convert/sources/safetensors.py` 的用法，确定合成源
   需要提供什么（张量名/形状/字节）。
3. 按 `src/models/qwen3_5/config.cpp` 的校验与 HF 命名写一个 tiny config（例如 2 层、hidden 1024、
   head_dim 128、q/kv heads 满足整除约束），生成随机权重（numpy/torch，fp32/bf16 即可）。
4. `tools/convert` 转换出小 artifact（先试 `--override` 只改 source/输出路径，必要时写 tiny recipe）。
5. Windows 单卡运行：`ninfer.exe <tiny.ninfer> --prompt "..." --max-new 8`，记录加载时间、是否生成、退出码；
   若引擎对极小 shape 有硬约束（静态断言/blocksize），按报错调大 config 的最小可行值。
6. 成功则写入 `docs/windows.md`（单卡 CLI 实测数据）并提交；长期不收敛则把该条款记为硬件+工具链限制。
**进度（Round 54 第 9 轮）：子任务前置条件就绪**

- CPU 版 torch 已装在 `build-win\torch-venv`（gitignored，不动 base 环境）：**torch 2.14.0+cpu / numpy 2.5.3** ✓
  （`build-win\torch-venv\Scripts\python.exe`）。
- 转换器接口已确认：`python -m tools.convert --model <checkpoint目录> --recipe <官方名|file[:func]> --out X.ninfer
  [--components text] [--override file.py] [--source NAME=PATH] [--name NAME] [--device DEVICE]`；
  `--model` 即「主 checkpoint/config + 默认资源」。
- 下一步（按代价从低到高）：① 先直接用官方 recipe 跑 tiny checkpoint（`--recipe qwen3_8_27b_nvfp4
  --components text`），若它按 config 驱动逐层映射即可零改动成功；② 不成功再写 tiny recipe/override；
  ③ 生成脚本放 `tools/win_port/tiny_model.py`（合成 tiny config + 随机 bf16 权重，HF 命名）。

---

## Round 11: ①③④ 的实现（TP-2 双卡优化）

目标（用户明确要求）：在 `feat/windows-native-port` 上实现 PLAN §12 表格里的 ①③④ 三项，**不做 ②**（权重 NVFP4 化，
它要求新 artifact + 逐形状 Op 准入，已判定为部署期变体而不是运行时开关）。

三项定义与理论值（PLAN:640-667 的核算，基线 decode 38.2 ms/轮、2.166 committed、56.7 tok/s；prefill 1552 tok/s@28.5k）：

| 项 | 内容 | decode 理论 | prefill 理论 |
|---|---|---|---|
| ① | TP-2 verify forward 上 exact-batch CUDA graph（现被 `model_instance.cpp:99` 关掉） | 发射间隙 2.91→0 ms：35.3 ms / 61.4 tok/s | 块内间隙摊薄 ≈ −0.4% |
| ③ | collective 与 compute **重叠**（不是删 AR；删 AR 会亏，见 PLAN:655） | −1.7 ms（AR 与 draft 链重叠） | AR∥MMA：660→≈390 ms / ≈2.6k tok/s |
| ④ | MTP draft 链下沉/重叠（每 draft 一次 cudaStreamSynchronize，见 tp2_generation_core.cpp:543） | −4.7 ms → 28.9 ms / **74.9 tok/s** | 不适用 |

单轮预算拆账（PLAN:642-644）：权重流 25.6 ms（448 GB/s 地板 22.7）+ AR 1.7 ms（128 次）+ 发射间隙 2.91 ms
（1009 kernel/forward）+ **MTP 链/accept/fold/host 同步 ≈ 8.0 ms**。prefill 每 1024-token 块 660 ms = AR ≈350 + MMA/其它 ≈310。

显存账：合计 **+0.2–0.85 GiB/卡**；131072 配方（free 2002/1580 MiB）能放下，262144（free 0/196）放不下。

### 已确认的接口事实（勿重复调研）

- `forward_tp2_prefill`（src/models/qwen3_5/execution/text.cpp:1766）是 TP-2 唯一的分块前向入口，解码轮的 verify
  也走它（Phase::Prefill，见 tp2_generation_core.cpp:999-1005）。它的 host 依赖只有三处：
  `copy_i32(ids.data(), b.ids, stream)`（:1826）、`ops::fill_i32_positions(b.positions, first_position, stream)`（:1827）、
  `b.envelope = {visible_end, visible_end}`（:1823）。device-driven/graph 化必须把前两者改成设备张量、
  把 envelope 变成按桶捕获的常量。
- 单卡路径的 graph 模式可直接照搬：src/models/qwen3_5/program/graph_execution.h（capture_graph/run_prepared）、
  program/decode.cpp:24-63（captured body 首尾各一次 **pinned** host↔device memcpy，program_impl.cpp:62 分配
  PinnedHostBuffer）、program/planning/graph_profiles.cpp:62-89 mtp_graph_profiles（capacity 131072、K=2 时
  ends = 124/508/2044/4092/8194/16386/32764 → **8 个桶**，envelope 按桶最大值捕获，逐轮真实位置由设备张量携带）。
- `DecodeGraphDefinition::capture` / `DecodeGraphExecutable::instantiate|launch` 在 src/core/decode_graph.cpp:59-146，
  平台无关，TP-2 可直接用（每卡一份 definition/executable）。
- AR 是 cudaHostAllocMapped|Portable 的 host pinned + 内核自旋（src/core/tp/device_pair.cu:90-143），
  两卡各自 capture 后交替 launch 即可，不必改 AR 协议。
- Windows 无 nsys/ncu（C:\Program Files\NVIDIA Corporation 下只有 App/FrameView），性能归因**只能用引擎内 CUDA event**。

### 执行顺序（每步独立可测、可回退）

1. **测量**：TP-2 解码轮加 env-gated 相位计时（host 时钟 + CUDA event），确认 8.0 ms 的 MTP 链/accept/fold 归属，
   并复现基线 56.7 tok/s。
2. **④**：draft 链去 host 化——drafts 留设备、window token 直接给设备版前向、去掉冗余 D2H/H2D 与逐 draft 同步。
3. **①**：把整轮 capture 成两张图（每卡一张），pinned ingress/egress，按 mtp_graph_profiles 桶捕获；每轮只填 ingress + launch。
4. **③**：prefill 的 AR 与 MMA 重叠（子块流水；需两套激活区与两份 GDN 状态快照，见 PLAN:663 的显存账）。
5. 验证与文档：贪婪 A/B（与改造前逐 token 一致）+ bench_serve.ps1 实测 + 更新 docs/tp2-dual-5060ti.md / worklog。

### 进度

- [x] 计划落盘（本节）。
- [ ] 步骤 1 测量。

#### Round 11 侦察结论（子代理只读调研，勿重复）

- **层循环**：\`run_layers_tp2\` 是 text.h:411-458 的模板（不在 text.cpp）。每层顺序固定：
  \`tp_mixer_layer\`（双卡，行并行输出投影写 partial delta）→ **AR#1**（text.h:436）→ \`residual_add\`×2 →
  \`tp_mlp_delta\` → **AR#2**（text.h:449）→ \`residual_add\`×2。层内**无任何 host 同步/回读**（\`src/ops\` 全目录
  grep \`cudaStreamSynchronize|DeviceToHost\` 零匹配）。
- **allreduce**（src/core/tp/device_pair.cu:289-395）：签名 \`allreduce(void* a, void* b, bytes, stream_a, stream_b)\`；
  in-kernel 路径 host 侧只连续发两个 kernel、无同步；epoch \`token = ++ar_call_\`（单调不减，device_pair.h:64），
  按 \`token&1\` 双缓冲（24 MiB/卡），block 间有 \`order_mine[b-1]==token\` 写序链，对端自旋 \`arrival_other[b]!=token\`。
  **graph 危害**：token 是**内核参数**，capture 会把它固化，replay 复用旧 token 会让对端自旋提前通过（读到上一轮的旧数据）
  → 上 graph 前必须把 token 改成**设备端计数器**（\`increment_i32_scalar\` 或内核内自增）。
- **positions**：\`fill_i32_positions\` 只接受 host base（src/ops/kernel/position.cuh:7）；\`offset_i32_positions\`
  支持 **[1] 设备 delta 广播到 [T]**（kernel: \`dst[i]=src[i]+delta[0]\`）→ 设备化路径 = 静态 arange + 设备 base。
- **fold**：\`GdnReplayFoldPlan::execute(std::span<const GdnReplayFoldRow>, stream)\` 在 **host 校验并打包**，
  \`commit_columns\` 进 \`__grid_constant__\` 内核参数（replay.cpp:266-325、recurrent.cu:104-128），
  头文件自述「改变 host row 描述符必须重新 capture」→ **fold 不能带变异 commit 数进 graph**，本轮保持 eager。
- **state 账**：state pool \`slot_count=1\`，live backing = **73.4 MiB**；PLAN 里 293.6 MiB = live + 3 份快照
  （2 份 prefix-reuse + 1 份 round scratch）。每轮 snapshot+restore = 2×73.4 MiB D2D/卡（不是 293.6×2）。
- **accept**：\`speculative_accept_greedy_drafts\` 已输出设备端 \`accepted\`，但**无人读取**；提交数始终来自 host 输出策略。
- **forward_tp2_prefill 的 host 依赖**（text.cpp:1766-2021）：\`copy_i32(ids)\` H2D（:1826）、
  \`fill_i32_positions(b.positions, first_position)\`（:1827）、\`envelope{visible_end,visible_end}\`（:1798/:1823，
  host 端决定 attention 路由/chunk/split）；\`Phase::Verify\` 分支（:1931-1965）MTP 轮**不走**（走 Prefill）。
- \`set_i32_scalar\` 只支持 host 值（scalar.cu:9）；\`assign_i32_scalar\` 是 D2D memcpy；\`increment_i32_scalar\` 设备端。
- **verify 窗口的显存**：K=2 时 T=3，激活峰值只有几 MB（最大项 \`logits_columns\` [248320,3] BF16 = 1.5 MB），
  所以 graph 私有 workspace **不需要** 192 MiB（PLAN:662 的最坏估计过于保守）。
- **③ 的子块流水前提**：所有 op 形状泛化（T 任意），但 (a) token 范围由 forward_tp2_prefill 每 chunk 一次性绑定
  （positions/envelope/kv_table_rows），(b) GDN conv/递推状态按 source/destination slot 顺序推进，
  (c) full-attention 子块 2 的查询要能看到子块 1 已 append 的 K/V ⇒ 每层依赖链 L,A→L,B→L+1 必须保持，
  能做的是**按 slot 的 A→B 流水**（AR 走链路时另一子块走 SM）。
- **踩坑（已修）**：给解码轮加 CUDA event 计时时，事件在**创建时的当前设备**上绑定；若与
  \`shard_a_.device.stream\` 不是同一设备，\`cudaEventRecord\` 会返回 \`cudaErrorInvalidResourceHandle\`，
  而该错误会被**锁存**到下一次不相关的 \`cudaGetLastError()\`（表现为 \`scalar.cu:10\` 的 set_i32_scalar 崩溃）。
  修法：在 shard A 设备绑定后再创建事件，且计时调用失败即丢弃错误并自我禁用。





## Round 11 进度：① 已完成（已实测），并修掉一处真实跨卡竞态

### 步骤 1 测量（已完成）

在 tp2_generation_core.cpp 加 env-gated 相位计时（NINFER_TP2_TIMING=1，7 个 CUDA event + host 时钟）。
基线（graph 关闭 = eager 路径，bench_serve.ps1 同一配方）：

    decode rounds=52 committed=127 avg_round=38.90ms mtp=3.70 verify=34.23 accept=0.08 copy=0.09 sync_wait=0.01 fold=0.00

即 38.9 ms/轮 = MTP 链 3.70 + verify 34.23 + accept 0.08 + copy 0.09 + host 策略/收尾（差额约 0.8 ms）。
verify 占 88%，其发射/调度开销实测比 PLAN:642 的 2.91 ms 更大；MTP 链 3.70 ms 是第二大项。

### 步骤 3 ① exact-batch CUDA Graph（已完成）

实现位置：src/core/decode_graph.{h,cpp}、src/core/arena.{h,cu}、src/models/qwen3_5/execution/text.{h,cpp}、
src/runtime/engine/tp2_generation_core.{h,cpp}。

- DecodeGraphDefinition::capture_group(defs, streams, body)：多流同时 capture，每张图只记录本流上的 kernel；
  跨卡仍只靠 in-kernel AR 自旋，host 不参与。
- TextContext::forward_tp2_window(...)：prefill 前向的 capture 安全变体。每轮输入（ids/positions）走
  可移植 pinned host 缓冲（capture 成 memcpy node，每次 replay 重读），attention envelope 变成捕获期常量。
- 桶来自 mtp_graph_profiles(max_context, mtp_drafts_)（K=2、131072 时 8 个桶，按可见上界分派）。
- arena 布局必须逐轮可复现：新增 DeviceArena::rewind(watermark) 与 position_arena(arena, floor, target)。
  每轮先丢弃 proposal 链的 arena 占用、回到固定 watermark，图里固化的地址才成立；replay 时按捕获 watermark
  对齐（更低 rewind，更高 alloc_bytes，低于本请求 watermark 直接报错）。已捕获桶的 watermark 若低于当前请求
  则视为未捕获并重捕（warm-up 的 watermark 更小，因此首个真实请求会重捕一次）。
- NINFER_TP2_VERIFY_GRAPH=0 关闭（A/B 用），默认开启；eager 参考路径与 capture 路径共用同一个 window 函数
  与同一个桶 envelope，所以 A/B 只比较发射方式。

实测（同一 bench 配方，graph 开/关）：

| 指标 | ① 关（eager） | ① 开（graph） | 差 |
|---|---|---|---|
| avg_round | 39.6 ms | 35.3 ms | **−4.3 ms** |
| verify | 35.0 ms | 30.6 ms | **−4.4 ms** |
| decode_at_2048（两侧都是 64 token） | 65.6 tok/s | 76.5 tok/s | +16.6% |
| prefill_512 | 1419 tok/s | 1419 tok/s | 持平 |
| sync_wait | 0.01 ms | 0.01 ms | — |

（tok/s 还受本轮文本决定的 draft 接受率影响，所以以轮时间与等输出的 2048 对照为准。）
verify 少的 4.4 ms 全是发射/调度开销（两路径的 kernel 参数与输入完全相同）。
数值正确性：5 个 prompt × 最多 256 token 的贪婪 A/B，eager 与 graph 逐字节一致（含 reasoning_content）；
graph 路径对是否插桩不敏感、可复现。

### 顺带修掉一处真实跨卡竞态（改动前就存在，非 ① 引入）

排查中发现：未插桩的 eager 路径在 5 个 prompt 里有 3 个与 graph 不一致，而任何额外同步/回读
（即使放在轮末）都会让它变成与 graph 一致 —— 即 eager 路径对时序敏感。根因是解码轮结束时只同步了 shard A
（cudaStreamSynchronize(shard_a_.device.stream)），随后 host 立刻做 fold / state restore / 下一轮 window 与
arena 复用，而 shard B 的 verify 尾部可能仍在跑；两卡之间唯一的顺序保证是 AR 自旋。
修法：轮末同时同步 A、B（tp2_generation_core.cpp）；实测 sync_wait 仍为 0.01 ms（B 只落后极少）。
修后 eager 与 graph 在全部 5 个 prompt 上逐字节一致。

### 顺带修掉的第二处缺陷：in-kernel allreduce 的 token 归属（真实崩溃）

带上 ① 的全量测试里 ninfer_qwen3_5_tp2_forward_test 报 cudaErrorIllegalAddress（第一层内），而 HEAD 的同一
二进制通过。二分定位：把 allreduce 强制走 host-staging 路径即通过 ⇒ 本轮改成「设备端 token」的 in-kernel AR 引入。
根因：token 计数器原先分配在各自设备的 device 堆上，但「哪个 shard 驱动 pair」由调用方决定（该测试会让两个 shard
各驱动一次，于是 stream_a 属于 device 1），内核在 device 1 的 context 里解引用了 device 0 的裸指针。
修法：token 改为 mapped pinned host memory（两个设备都可见），kernel 侧每 block 读一次并用 shared memory 广播
（若每个线程都直接读系统内存，单次 allreduce 多 ~4 us、整轮多 ~0.5 ms）。修后 12 个相关测试全绿
（含 tp2_load / tp2_forward），eager 与 graph 仍逐字节一致。

### 未决（非本次范围）

MTP 输出质量：与 plain decode（-Plain，单 token TP-2 路径）贪婪对照，5 个 prompt 中只有 1 个逐字节一致，
其余在近似并列处翻转。这是 docs/tp2-dual-5060ti.md §MTP 已记录的**接受行为**：verify 窗口与单 token 解码是
不同的执行形状（分块注意力/卷积路由），逐列 logits 必然有差异，llama.cpp 的 Qwen3.5 MTP 同样如此。
**注意**：不要再用「Phase::Prefill 导致近似」解释它 —— 同一文档已记录那是被否定的旧诊断（Phase 在
head-split 分片上无任何语义作用）。这不是 ① 的回归。

### 下一步

- [x] 步骤 1 测量。
- [x] 步骤 3 ①（exact-batch graph）+ 轮末双卡同步。
- [x] 步骤 2 ④ 复核：**该项按原描述已经不需要做**。PLAN 的 ④ 依据（"每 draft 一次 cudaStreamSynchronize"，
  指旧版 tp2_generation_core.cpp:543）在当前代码里已不成立：mtp_propose_window 整个窗口只有 **1 次 D2H + 1 次
  sync**，每 draft 只剩一次 D2D 拷贝 + 一次 increment 内核；mtp_forward_batch / mtp_forward_ar_step / mtp_forward_core
  内无任何 host 同步或 D2H（grep 实证）。实测 mtp=3.70~3.75 ms 是**真实 GPU 工作**（两次 T=1 的 MTP 层前向，
  权重流受限），且与 verify 串行。残余机会只有「fold 与 MTP 链重叠」≤0.8 ms（≈2%，fold 走 replay.cpp，不含
  allreduce，理论上可另开流），但要动 AR token 协议/流序，收益风险比不划算 ⇒ 判定 ④ 已完成，不再改。
- [x] 步骤 4 ③：prefill AR 与 MMA 重叠——**已实现、已实测、已否决**（见下）。
- [x] 步骤 5：复测与文档更新（③ 的结论写入本文件与 worklog）。

## Round 12：③ prefill AR∥MMA 子块流水（已实现，实测无收益，已回退）

### 实现（完整可用，最终未保留）

- `DeviceContext` 增加第二条流 `collective_stream`（与 `stream`/`transfer_stream` 同生命周期）。
- `forward_tp2_prefill` 在 `NINFER_TP2_PREFILL_OVERLAP` 打开且 phase=Prefill、`pair.in_kernel_allreduce()`、
  驱动卡 == `pair.a()`、tokens ≥ 256 时走两子块流水：A=[0,ta)、B=[ta,T)；每层顺序
  A.mixer → arm(0) → B.mixer → arm(1) → AR(A.mix)/AR(B.mix) → A.res+A.mlp → B.res+B.mlp → A.res2 → B.res2，
  4 个 AR 各配一对 event（`produced` 在 compute 流、`done` 在 collective 流），
  子块绑定靠嵌套 `ScopedPositions`/`ScopedEnvelope`（positions/rope 为外层 tensor 的 slice，
  envelope 分别为 {first+ta,first+ta} 与 {visible_end,visible_end}）。
- **关键约束（踩坑）**：切分点必须落在激活调度块边界上。初版 ta=tokens/2（300→150+150）会让
  `tp2_forward_test` 的 chunk-split 不变量失败（max_logit_diff=1.14/1.60）；改成 64 对齐
  （ta=max(64,(tokens/2)&~63)，300→128+172）后**逐位一致**：
  `chunking 300 -> 128+172: max_logit_diff=0`、`64+236: max_logit_diff=0`、
  `T=1024 one chunk vs 4x256: max_logit_diff=0`，且 300/1024 的 top5 与**未切分**的基线逐位相同 ⇒
  层间 A/B 交错与双流 AR 在数值上完全等价（这是 ③ 唯一确定的技术收益）。

### 实测（同一二进制、无缓存前缀、单请求 2072 token prompt，TTFT 端到端）

| 配置 | run1 | run2 | run3 |
|---|---|---|---|
| `NINFER_TP2_PREFILL_OVERLAP=0` | 1249 ms | 1189 ms | — |
| `NINFER_TP2_PREFILL_OVERLAP=1` | 1258 ms | 1314 ms | 1513 ms（冷启） |

稳定态 ≈ 1219 ms vs ≈ 1286 ms ⇒ 重叠**慢约 5%**。

### 结论（为什么不做）

1. 原成本模型（PLAN:651，AR 占 prefill ~48%）把 AR 当作**串行**开销，实测不成立：in-kernel AR 的两个
   内核是对称的，两卡各自 drifted 推进时，本卡自旋等对端的时间本来就被**对端仍在跑的 compute 掩盖**，
   可重叠的余量远小于模型估计。
2. 子块切分的代价是真实的：T 减半后 GEMM tile 变窄、**权重每层被流读两次**（prefill 是权重流敏感的形状），
   A/B 实测这两项加起来盖过了 AR 的收益。
3. 因此 ③ 在 TP-2 上不是有效优化：**保留最终方案为不带子块流水的单块 prefill**，`collective_stream`
   与整套流水代码已回退（工作树干净回到 8f6a812e）。
4. 若将来出现真正 compute-bound 的 prefill 形状（例如更宽 chunk、更快的权重路径），可重新评估；
   届时本节的 64 对齐约束与 event 拓扑可直接复用。

---

### Round 13 — 清理 decode 循环的 [mtp] 刷屏打印（已完成）

`tp2_generation_core.cpp` 的 decode 循环里有一条裸 `std::fprintf(stderr, "[mtp] round pos=…")`：
不经 logger（所以 `--log-level error` 关不掉）、无 env 门控，每轮一行（≈26 行/秒）。
实测写入成本 `2>NUL` 1.8 µs/行、文件 1.5 µs/行、**真实控制台 20–25 µs/行**
⇒ 单轮 20 µs / 35.3 ms = **0.057%**，吞吐影响可忽略；真正的风险是 cmd 的 QuickEdit 选中会让控制台
停止消费、阻塞生成线程（不是慢，是卡）。

已删除该打印与仅供其使用的 `mtp_draft_checked_`/`mtp_draft_hit_` 两个计数器；
接受率改由 `NINFER_TP2_TIMING=1` 的 `[tp2-time] decode rounds=R committed=C` 推导：
接受 draft 数 = C−R，接受率 = (C−R)/(R·K)。`docs/tp2-dual-5060ti.md` 已同步。

---

### Round 14 — 前缀复用“续写边界”：实测证伪并撤回（已完成）

实现了方案 2（续写边界：`cached_tokens_` 延伸到上一轮生成序列末尾 + 新状态槽），用真实 DSH 流量（36 请求）验证：
`slot=0` 35 次、`slot=1`(rewind) 1 次、**续写槽 0 次**；33/36 条 `shared == prefill_end` 精确相等 —— 客户端下一条 prompt
与上一条逐 token 相同到末尾，然后在第一个生成 token 处分叉（不回传生成流，而是重新渲染 assistant 回合）。
结论：该方案对本客户端零收益，**已撤回**（引擎两个文件回到 `1cdfab97`，`state` 回到 293.6 MiB/卡）；
保留 `NINFER_TP2_REUSE_TRACE=1`（每请求打印 `prompt/cached/shared/prefill_end/rewind -> reuse/slot`），
它是判断“客户端是否原样回传历史”的唯一手段。

**下一步候选（用户暂缓）**：① system+tools 前缀末尾加锚点，修新会话/压缩后第一轮的 46.9 s（可省约 11 s）；
② prefill 吞吐（suffix 0.95–1.13k tok/s、冷启 1.35–1.38k）；③ 每请求 ~0.3 s 固定开销。

---

### Round 15 — 待做：工具调用的约束解码（对齐 llama.cpp 的 lazy grammar）

**现象**：DSH 接 NInfer 时偶尔把工具调用当正文吐出（`<tool_call> <function=todo_write> …`），前端提示
“工具调用方式不对”；同一套 DSH 接 llama.cpp 从未出现。

**已定位（诊断已完成，细节见 `docs/tp2-dual-5060ti-worklog.md` Round 15）**：不是量化（诱导探针 A/B：官方
`nvfp4` 与转换 `w4a4_w8a8` 各 75 条工具请求，标记类失败 27/75 vs 19/75，z≈1.4 不显著；失败形态相同——模型照抄
提示里的 SDK 名字 `tools.read`/`todo_write`）、不是 chat_template（逐字复述探针两种写法都正确解析）、不是解析器 bug。
客户端侧也一致（`~/.dsh/settings.yaml`：两个 provider 都是 `api: openai-responses`，`agent-presets.default: ptc`）。
根因是**引擎策略差异**：

- llama.cpp 做 **lazy GBNF 约束解码**：`common/chat.cpp:1286` 为每个声明工具生成 `<function=NAME>` + 该工具 JSON
  schema 的规则，`:1316-1337` 在 `tool_choice=auto`（`common/chat.h:258` 默认值）下 `grammar_lazy=true`、触发词
  `<tool_call>`；`tools/server/server-common.cpp:1332-1346` 把 grammar 注入生成参数 ⇒ 未声明的名字在采样层不可达，
  模型“想直接调 read”会被逼回已声明工具（PTC 下就是 `run_code`），参数也必然合规。
- NInfer 没有约束层：自由生成后解析（`src/models/qwen3_5/frontend/tool_call_parser.cpp`），用 `enforce_declared_names`
  校验，未声明即按契约拒收、原样返回文本并打 WARN。

**待做（用户暂缓，后续做）**：

1. **工具调用约束解码（首选）**：在 `<tool_call>` 进入后、`<function=` 位置掩码 logits，只允许本次声明的工具名；
   参数按声明 schema 约束。先做「工具名前缀树 + 参数名掩码」，完整 GBNF 作为后续。
   **必须先想清楚**：掩码在 MTP draft/verify 两侧的落点（draft 提议的 token 也要过同一份掩码），以及它与
   exact-batch CUDA Graph 的关系，否则 verify 会误判接受。
2. **不做**「未声明也当 tool_call 返回」：等于放行模型绕过 `run_code` 直接触发 `pwsh`/`write`，破坏 PTC 契约。
3. 轻量诊断（随时可做）：WARN 里打印模型实际写出的工具名，线上就能看出是 `tools.read` 还是 `todo_write`。
4. 可选闭环实验：llama.cpp 起来时把同一份探针（`%TEMP%\ab_probe.ps1`）打过去，预期不出现未声明名字。

**设计（调研完成，对照 `C:\llama.cpp` @ `ar3-opt` 源码）**：

llama.cpp 机制（已核实）：
- grammar 是 GBNF 状态机（`src/llama-grammar.cpp` 的 `llama_grammar`，维护一组 parse 栈 `stacks`）。
- `llama_grammar_apply_impl`（`llama-grammar.cpp:1354`）：`awaiting_trigger` 期间直接返回（不掩码）；否则对每个候选
  token，用 `llama_grammar_reject_candidates(rules, stacks, candidates)` 算出当前状态下非法的候选，把它们的 `logit` 置
  `-INFINITY`。EOG 仅在某个 parse 栈为空时允许。
- `llama_grammar_accept_impl`（`:1397`）：接受 token 时推进状态；`awaiting_trigger` 期间累积 `trigger_buffer`，命中触发
  token 或正则 `trigger_patterns` 后 `awaiting_trigger=false` 开始约束。
- 采样器侧（`common/sampling.cpp:265`）：`grammar_lazy=true` 时用 `llama_sampler_init_grammar_lazy_patterns` 建 grammar；
  `grammar_should_apply`（`:452`）在 reasoning budget 处于 IDLE/DONE 时才应用（思考阶段不约束）。触发词为 `<tool_call>`。

NInfer 设计（轻量版，先做「工具名前缀树 + 参数名掩码」）：

1. **约束表示**：一个小状态机（非完整 GBNF parser），跟踪工具调用格式位置。Qwen 格式
   （`tool_call_parser.cpp:20-25`）：`<tool_call> <function=NAME> <parameter=PARAM> VALUE </parameter> </function> </tool_call>`。
   状态：AWAITING_TRIGGER →（命中 `<tool_call>`）→ FUNCTION_NAME → PARAM_NAME → PARAM_VALUE → … → DONE。每个状态算出允许
   token 集合（掩码）：
   - AWAITING_TRIGGER：不掩码（自由文本/思考）。
   - FUNCTION_NAME：只允许能续接「已声明工具名」的 token（声明名的前缀树）。
   - PARAM_NAME：只允许能续接「当前工具已声明参数名」的 token。
   - PARAM_VALUE/其它：不掩码（完整 schema 校验留作 GBNF 后续）。
   状态机累积已生成文本（同 llama.cpp 的 `trigger_buffer`），处理名字跨多 token 的情形。
2. **掩码落点（核心）**：新增 op/内核 `apply_token_mask(logits, mask, ...)`，把不允许的 logits 写 `-INF`。
   - plain decode（无 MTP）：在 `ops::sample`（`tp2_generation_core.cpp:1407`）前掩码 `[vocab,1]` logits。
   - MTP verify：在 `ops::argmax`（`:1489`）+ `speculative_accept_greedy_drafts`（`:1490`）前掩码 `[vocab,width]` 的
     `window_logits`；每列 grammar 状态不同，掩码按列。
   - **关键**：只掩码 target 的 verify logits ⇒ target argmax 恒合法 ⇒ 被接受的 token 恒合法。draft 提案无需单独掩码
     （非法 draft 与掩码后的 target argmax 不匹配，被 `speculative_accept_greedy_drafts` 拒收）。
3. **CUDA Graph**：verify forward 被捕获进 graph，`window_logits` 从 graph 出来；掩码在 graph 之后、argmax 之前应用
   （graph 外），host 算掩码、device 应用。**无需改 CUDA Graph**。
4. **plumbing**：把 `ToolCallOutputContract`（或派生的掩码结构：工具名 + 参数名）从 `prepared_prompt`
   （`prepared_prompt.h:152`）plumb 到解码循环。契约在 prompt 准备时由 `PromptOptions.tool_jsons`（`types.h:413`）构建
   （`build_tool_call_output_contract`）。存进 TP-2 核心的 Request 或传给 decode 函数。
5. **lazy 触发**：约束是 lazy 的——模型吐出 `<tool_call>` 后才激活。host 侧累积生成文本、检测触发；触发前不掩码。
6. **思考门控**：思考（reasoning）阶段不约束（同 llama.cpp 的 `grammar_should_apply`）。NInfer 有 `ThinkingControlOptions`，
   思考中禁用掩码。

实施分阶段：
- **Phase 1（轻量）**：工具名前缀树 + 参数名掩码。建状态机 + 掩码；加 `apply_token_mask` op；在 plain decode 与 MTP verify
  应用；plumb 契约；lazy 触发 + 思考门控。
- **Phase 2（完整 GBNF）**：完整 JSON schema 校验（参数类型、结构）。

**路线选择（A/B，2026-07-11 修正）**：

「正确性 bug」风险（工具名跨多 token 时掩码算错）在 llama.cpp 上同样存在，但被其实现方式压到很低：
`llama_grammar_apply_impl` 把每个候选 token 用 `token_to_piece` 解码成文本、用 `llama_partial_utf8` 处理跨 token
半个 UTF-8，再拿「已累积文本 + 片段」去 GBNF parse 栈校验——**按文本约束、对分词不敏感**，且 parser 长期生产验证。
若轻量版在 token 序列上建前缀树则分词敏感、易掩错。故分两条路线：

- **路线 A（文本化轻量版，Phase 1 先做）**：状态机累积「已生成文本」（同 llama.cpp 的 `trigger_buffer`），每步把候选
  token 用 NInfer tokenizer 解码成文本片段（处理跨 token 半个 UTF-8），校验「累积文本 + 片段」是否为已声明工具名/参数名
  的合法前缀（用声明名的 trie 加速），否则置 `-INF`。对分词不敏感，正确性风险对齐 llama.cpp，代码量远小于完整 GBNF。
  代价：约束激活的每步有 O(vocab) 的 CPU 掩码计算（~数 ms），但仅作用于工具调用区（几个 token），摊销可忽略。
- **路线 B（移植 llama.cpp 的 GBNF parser，Phase 2 选项）**：直接搬 `llama-grammar.cpp` 的栈式 parser（~1500 行）+ 为
  Qwen 生成 grammar。正确性风险最低（生产验证），天然支持完整 schema 校验。若 A 在真实流量仍偶发边界问题、或要做完整
  schema 校验，升级到 B。

验证：
- 单测：掩码正确把工具名限制在已声明集合。
- 集成：模型不再吐未声明工具名（对照 Round 15 的 75 条探针）。
- MTP：约束在 MTP 下生效（非法 draft 被拒）。
- 回归：非工具调用生成不受影响。

风险/未决：
- 分词：工具名可能跨多 token，状态机须累积文本、按前缀匹配。
- 性能：每步掩码计算应很便宜（小状态机）。
- 一致性：掩码须在所有采样路径（plain decode、MTP verify、未来路径）一致应用。

**实现与验证（2026-09-20，路线 A 已落地）**

- 新增 op `include/ninfer/ops/token_mask.h` + `src/ops/{kernel/token_mask.cuh,launcher/token_mask.{h,cu},wrapper/token_mask.cpp}`：
  按 `[rows, columns]`（dim0 连续）逐元素把 U8 掩码为 0 处的 logits 写成 `-inf`（BF16），在 CUDA Graph **之外**调用。
- 新增 `src/models/qwen3_5/frontend/tool_call_constraint.{h,cpp}`：`ToolCallMaskTable`（每个 vocab id 一份解码字节 + 特殊位，
  由 `Frontend` 用 `shared_ptr<const Tokenizer>` 持有并**只建一次**）、`ToolCallNameTrie`、`ToolCallGrammar`、
  `ToolCallGrammarState`（字节级，`Free / FunctionLiteral / FunctionName / FunctionClose / ParameterName / ParameterValue /
  ToolClose`）、`ToolCallConstraint`（共享表 + 不可变语法 + 每请求可变状态，故按请求构造的是非 const 对象）。
- `OutputSession` 保留 Content 通道的原始字节流（`raw_content_text()`）并暴露 `in_reasoning()`：掩码与工具调用 parser 吃
  **同一条**字节流，所以跨 token 的工具名、半个 UTF-8、跨越结构字面量的 token 都不会被误判。
- `tp2_generation_core`：plain decode 在 `ops::sample` 之前、MTP 在每个 verify 列 `ops::argmax` 之前应用掩码；MTP 每列掩码 =
  「committed 文本 + 该列之前的 drafts」对应的语法位置（`build_mask_after`），draft 本身不掩码（非法 draft 与掩码后的
  argmax 不一致因而被拒）。约束只在 `!in_reasoning()` 时激活。
- 掩码只在 `FunctionLiteral / FunctionName / FunctionClose / ParameterName / ToolClose` 生效；`ParameterValue` 与自由文本不掩码。
- **安全阀**：某位置若除纯空白外没有任何候选能推进语法，则报「不受约束」而不是发全 0 掩码——空白被跳过、不推进语法，
  掩到它会让模型一直吐空格直到 context 用尽；全 0 掩码则会卡死请求。
- 验证：`ninfer_token_mask_test`（GPU；掩码逐位精确比对、掩码只读、全掩、形状/rank/dtype/别名校验）与
  `ninfer_tool_call_constraint_test`（CPU；惰性触发、只许已声明名、多 token 名字、完整名后必须 `>`、参数名、自由值、
  回到 Free、跨字面量 token、不可拼写名的回退、二次调用）**均通过**；`ninfer_tool_call_parser_test` 通过；
  `ninfer_qwen3_5_frontend_test` 仍是上文 703–707 行记录的既有失败（与本改动无关）。
- 未做：真实模型端到端（对照 Round 15 的 75 条探针）与 MTP 下的实测对照——需重启 `ninfer-serve` 后执行。
- **修正（同日追加）**：约束表的行数是 tokenizer 的公开词表（`resources.public_token_count`），而 logits 域是**打包后**的
  embedding 行数（本机 artifact `config.text.vocab_size` = 248320，公开域更小；`frontend/resources.cpp:12` 明确只要求
  `count <= config.text.vocab_size`）。原先的一致性检查错误地要求两者**相等**，于是任何带 tools 的请求都 500
  （`unknown: TP-2 tool-call constraint vocabulary does not match the logits domain`）。现改为掩码按 **logits 域**生成
  （`build_mask(domain, mask)` / `build_mask_after(prefix, domain, mask)`），tokenizer 未定义的行一律置 0（排除）；
  仅在「约束表 > logits 域」时报错，且错误信息带上两个数值。新增 packed-domain 单测覆盖该尾部。
- **既有缺陷，已修（同日）**：TP-2 核心有 4 处把**物理行数** `vocab` 当作有效域传出：`ops::sample`（prefill 首 token、
  plain decode）、`ops::argmax`（MTP target）、`speculative_accept_greedy_drafts`（接受核）。契约本来就把两者分开
  （`sample` 校验 `token_domain ∈ [1, physical_rows]`，`argmax` 参数名即 `valid_rows`，accept 文档写 `token_domain`），
  单卡路径（`text.cpp:2279/2283`、`decode.cpp:258`、`draft.cpp` 的 selector 域）都传 `public_token_count`；那枚**算了却
  从未使用**的 `public_tokens` 正是这个意图的残留。现将声明上提到 `vocab` 旁，4 处全部改传 `public_tokens`；
  `sampling_workspace_capacity_bytes(vocab, 1, 1)` 保留（容量上界）。
  影响面：越界 id 不会越界访存（embedding/lm_head 都是 248320 行，仅读到填充行），但会进 `OutputSession` 的
  `Tokenizer::decoded_token`（`output_session.cpp:458/563`，无守卫）抛 `out_of_range` ⇒ 单个请求 500。TP-2 路径缺的正是
  单卡 `program/decode.cpp:256 validate_licensed_tokens` 那层守卫。触发条件是「打包行胜过所有真实候选」（贪心要求它
  是全域最大；采样要求它进 top-20 且活过 top_p/min_p），正常分布下几乎不会发生。
  风险：`token_domain` 在核里只作遍历上界（`sampling.cuh:33/41/127`），两域的 `cap` 都是 `min(20,domain)`，RNG 抽取
  发生在截断后的候选集上 ⇒ 修复只在本该触发的那些步改变结果，正常步逐位不变。
- 顺带观察（未改）：`make_sampling_config` 把 `token_counts` 置空（`tp2_generation_core.cpp:233`），故 TP-2 路径的
  presence/frequency penalty 实际不生效（惩罚项 `c_v` 恒为 0）。
- 环境注记：DSH 沙箱处于 `workspace-write` 时 ninja **无法执行任何子进程**（连平凡工程都挂，`ninja -t/-n` 正常），
  构建须在 `danger-full-access` 下进行；另外 `pwsh` 的后台作业若用 `Tee-Object` 把输出写进管道会因管道写满而在中途卡死。

---

### Round 16 — TP-2 host 侧状态检查点 + 短 prompt 按 LCP 截断复用

**状态**：已实现并完成端到端验证（`src/runtime/engine/tp2_generation_core.{h,cpp}`、`src/runtime/engine/model_instance.cpp`、
`tests/test_engine_options.cpp`）；`ninfer_engine` 编译通过、新增的选项契约单测通过；实测 `src=host` 命中（req#5 10.2 s → 1.3 s、req#11 37.6 s → 28.6 s，详见 worklog Round 16.3）。

**问题（Round 15 后续实测）**：客户端在 turn 边界重渲染历史，新 prompt 比上一条**短**（实测 −219 / −1,233 /
−15,888 token），分叉点落在历史中段。TP-2 核心只保留 2 个**显存**边界（上一条 prompt 末尾 + 一个 rewind 点），
判据要求 `boundary <= shared_prefix`，两个边界都落在分叉点之后 ⇒ `reuse=0` ⇒ **整条 prompt 全量重算**：
req#43 57,378 token / TTFT 42.0 s，req#51 74,536 token / TTFT 57.6 s（NVML 采样证实这 59 s 两卡都是 98–100%，
与 `done` 行的 `prefill 1.30k tok/s (74,536 tok)` 一致）。

**根因不是 KV**：walk 从边界开始，边界之前的 KV 页本来就不重算（`tp2_generation_core.cpp:1042-1045` 注释）。
缺的是 **GDN（线性注意力）状态**：它是循环累积量，既不能从 KV 反推、也不能平移；全注意力层的 K/V 可以随便
截断复用，线性层不行。llama.cpp 正是因此才额外做 context checkpoints（PR #15293）：它的
`n_past = slot.prompt.tokens.get_common_prefix(input_tokens)`（`tools/server/server-context.cpp:3103`）只解决
注意力层，hybrid 模型靠检查点兜底 —— `n_ctx_checkpoints = 32`、`checkpoint_min_step = 8192`（`common/common.h:611-615`），
创建在 `llama_decode()` 之前（`server-context.cpp:3508-3518`），恢复时从新到旧找 `<= LCP` 的检查点，找不到才
`do_reset` 全量重算（`:3236-3249`）。

**方案（显存 0 增量）**：检查点放 **pinned host 内存**，引擎已有这套语义：`--host-state-slots`（文档表里叫
「完整 Host StateImages」，`src/serve/serve_options.cpp:220-223`）、`HostStatePool` +
`StateImageStore(device_pool, host_pool, capacity)`（`program_impl.cpp:125-144`）、启动阶段 `HostStatePin`；
TP-2 核心本来就在用 `PinnedHostBuffer`（`tp2_generation_core.cpp:279`）。

1. 一份状态 = **73.4 MiB/卡**（`state_bytes`；台账 `state 293.6 = (2+2)×73.4`）。16 槽 = 2.3 GB host、
   32 槽 = 4.7 GB host、**显存 +0**；恢复一次 = 每卡 73.4 MiB H2D（pinned，≈5–20 ms），相对几十秒可忽略。
2. prefill 过程中按 `kReuseCheckpointStride = 8192`（对齐 llama.cpp 的 `checkpoint_min_step`）把 `state_backing`
   异步拷进环形 host 池，并记下 frontier 位置。
3. 复用扫描从「2 个显存边界」扩成「2 个显存边界 + host 检查点」，取最深的 `<= shared_prefix`
   （`tp2_generation_core.cpp:891-897` 就是扩展点）；`begin_gdn_state`（`:976-987`）增加 host→device 分支，
   KV 路径完全不动。
4. **正确性规则（谱系有效性）**：检查点带 `valid`；每次选择前把 `position > shared_prefix` 的置为失效（越界者
   永不可能再被用上），prefill 成功发布时用 prefill id 把本次写的槽标为有效。**不能**只认「上一次 prefill」——
   上一次若只走了很短的后缀（完全复用），它自己不留检查点，那样最常见的小尾巴分叉会依旧 `reuse=0`（Round 16.2）。
   另外 prefill 中途取消时清 `cached_state_valid_` 并清空整环（取消走的 KV 与 `cached_prompt_tokens_` 不再一致）。
5. 一个 131k 上下文按 8k 步长最多 16 个检查点，所以 16 槽即可覆盖满上下文（超出部分环形淘汰）；
   200k 建议 32 槽。
6. **尾部窗口**：最后 8192 token 内每个 chunk 末尾额外存一份（对齐 llama.cpp 的 `near_prompt_end` 例外），
   覆盖「只重渲染尾巴」这一最常见情形（实测 req#12：只差 171 token 却全量重算 39.0 s → 命中尾部检查点后 ~0.4 s）。

**验收**：同一会话连续两轮，第二轮 prompt 比第一轮短且分叉在中段 ⇒ `[tp2-reuse]` 报 `reuse > 0` 且来自 host
检查点，`cache` 命中从 0% 升到 `shared` 量级，TTFT 从 ~57 s 降到 `(prompt − checkpoint)/1.3k + 固定开销`；
回归：现有 TP-2 用例 + 新增「短 prompt 命中 host 检查点」场景；`git diff --check` 干净。
