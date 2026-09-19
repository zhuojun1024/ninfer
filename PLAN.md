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

