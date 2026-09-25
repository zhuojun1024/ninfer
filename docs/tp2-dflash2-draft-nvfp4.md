# DFlash2 全 draft NVFP4：源码级可行性分析

> 结论性分析记录（2026-09-24），服务于 [PLAN.md](../PLAN.md) §3.7 的剩余杠杆「全 draft NVFP4」。
> 方法与证据：消费 op 的 qtype 校验器、weight 绑定/布局实现、转换器方法注册表、真实 r66 artifact
> 的 conversion 报告，以及本机 `D:/LLM/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8` 源与
> `qwen3_8_27b_w4a4_w8a8_dflash2_q4all.ninfer`（现役最好件）。

## 0. 结论（TL;DR）

**全 draft NVFP4 在源码层「可做」，但不值得做；作为显存杠杆它已被现役 Q4 件击败。**

1. **收益被高估**：相对官方 Q8 draft 全量 NVFP4 省 **−1090.9 MiB（−1.065 GiB）**，与 §3.7 的「约 −0.95 GiB」
   同量级；但相对现役 r66 `q4all` 件只剩 **−202.3 MiB**，其中 **−174.3 MiB 来自 codebook**（非 GEMM，
   且与「NVFP4 profile」无关），其余 GEMM 项合计只再省 ~28 MiB。
2. **逐张量看 NVFP4 比 Q4 更大**：NVFP4 = 0.5625 B/元素，Q4_G64_FP16 = 0.53125 B/元素。draft 的
   gate_up / output / down / kernel_projection / feature_projection 现役已是 Q4，改 NVFP4 是**增容**；
   仍在 Q8 的融合 QKV 用 Q4 化（r65 路线）比 NVFP4 还省 4.7 MiB。
3. **三个硬阻塞**：(a) 转换器没有 float→NVFP4 量化器，且 draft 源是 compressed-tensors **动态激活 FP8**，
   没有 A4 所需的 `activation_input_divisor` 标定来源；(b) NVFP4 native Weight **只接受完整 parent**，
   与 draft 融合 QKV 的 row-region 共享拓扑冲突；(c) 各消费 op 的 NVFP4 变体/几何注册表是封闭集合，
   需要新写 4 族 kernel（attn_input_proj 3 输出、context_kv_materialize、dynamic grouped conv add/prepare、
   candidate_selector codebook）。
4. **可能的唯一非显存动机**是让 draft 与主模型统一 W4A4 家族、吃到 NVFP4 TMA 张量核；但那需要独立的
   激活标定 + 性能论证，且 draft decode 的小 T 路径本就走 SIMT，收益不确定。

因此建议：**不把「全 draft NVFP4」作为显存杠杆推进**；若目标是显存，按性价比排序为
① codebook 量化（独立杠杆，174 MiB，最深的改动）② r65 融合 QKV→Q4（79.7 MiB，RowSplit 天然支持 region）
③ 其余不动。

---

## 1. draft 张量清单与现役格式

配置来源 `D:/LLM/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8/config.json`：hidden 5120、intermediate 17408、
5 层、32 q heads / 8 kv heads / head_dim 128、sliding_window 2048、conv_kernel_size 2 / conv_group_size 16、
selector_rank 256 / selector_top_k 16、target_layer_ids [5,19,33,47,61]。绑定代码：
`src/models/qwen3_5/load/dflash.cpp:5-45`、`src/models/qwen3_5/load/dflash2.cpp:5-29`。

真实 r66 artifact 报告 `qwen3_8_27b_w4a4_w8a8_dflash2_q4all.ninfer.conversion.json` 中
`method: grouped_absmax` / `layout: row_split_k128_v1`，共 66 个 dflash2 对象：

| 张量 | 形状 | 现役格式 | 个数 |
|---|---|---|---|
| attention QKV（query+key+value 融合；context_key/context_value 是 key/value 的 share） | [6144,5120] | q8_g32_fp16 | 5 |
| mlp gate/up（共享一个存储对象） | [34816,5120] | q4_g64_fp16 | 5 |
| attention output | [5120,4096] | q4_g64_fp16 | 5 |
| mlp down | [5120,17408] | q4_g64_fp16 | 5 |
| attention_conv / mlp_conv kernel_projection | [1280,5120] | q4_g64_fp16 | 10 |
| attention_conv / mlp_conv base_kernel | [2,2,5120] | bf16 | 10 |
| input/post_attention/query/key norm | [5120]/[128] | bf16 | 20 |
| feature_projection | [5120,25600] | q4_g64_fp16 | 1 |
| context_norm / final_norm | [5120] | bf16 | 2 |
| candidate_selector hidden_projection | [256,5120] | bf16 | 1 |
| candidate_selector predecessor/successor codebook | [248320,256] | bf16 | 2 |

（context_key/context_value 不单独出对象：官方配方 `tools/convert/official_recipes.py:47-54` 对
`key/value` 调 `recipe.share(prefix+"context_"+role, prefix+role)`，二者 Binding 即 key/value 的 Part，
位于同一 [6144,5120] 对象内。）

## 2. 收益重算（MiB）

三列：官方 Q8 draft（所有投影 Q8、kernel/codebook/norm BF16）、现役 r66 `q4all`（r62–r66 已落地）、
全 NVFP4（本分析的目标）。

| 组件 | 官方 Q8 | r66 q4all | 全 NVFP4 | Δ vs Q8 | Δ vs r66 |
|---|---:|---:|---:|---:|---:|
| QKV [6144,5120]×5 | 159.4 | 159.4 | 84.4 | −75.0 | −75.0 |
| gate_up [34816,5120]×5 | 903.1 | 451.6 | 478.1 | −425.0 | **+26.6** |
| output [5120,4096]×5 | 106.3 | 53.1 | 56.3 | −50.0 | **+3.1** |
| down [5120,17408]×5 | 451.6 | 225.8 | 239.1 | −212.5 | **+13.3** |
| kernel_projection [1280,5120]×10 | 125.0 | 33.2 | 35.2 | −89.8 | **+2.0** |
| feature_projection [5120,25600] | 132.8 | 66.4 | 70.3 | −62.5 | **+3.9** |
| codebook ×2 [248320,256] | 242.5 | 242.5 | 68.2 | −174.3 | −174.3 |
| hidden_projection [256,5120] | 2.5 | 2.5 | 0.7 | −1.8 | −1.8 |
| **合计** | **2123.1** | **1234.5** | **1032.2** | **−1090.9** | **−202.3** |

注：官方 Q8 列合计 2123.1 MiB = 2.073 GiB，与归档「草稿 2.07 GiB/卡」逐位吻合，可视为清单/字节公式的交叉验证。
§3.7 的「约 −0.95 GiB」应为不含 codebook（−916.6 MiB）或 GB/GiB 取整的估计。

**关键**：NVFP4 每元素 0.5625 B 大于 Q4_G64 的 0.53125 B，所以 r62–r66 之后所有已 Q4 化的 GEMM 改
NVFP4 都是增容；唯一真正的大额节省（codebook，−174.3 MiB）与 GEMM/profile 无关。

## 3. 逐 op 的 NVFP4 支持矩阵（源码证据）

| draft 张量 | 消费 op | 现状 | 结论 |
|---|---|---|---|
| QKV [6144,5120] decode | `ops::attn_input_proj` 三输出（`execution/draft.cpp:326-328`） | 硬要求 Q8_G32 RowSplit、n=6144（`ops/wrapper/attn_input_proj.cpp:243-261`）；NVFP4 变体只为 4 输出 [14336,5120] 存在（`nvfp4_attn_input_plan.cpp:19-88`、`nvfp4_geometry.h:29`） | 需新写 3 输出 NVFP4 路径 + N6144K5120 几何 |
| QKV key/value 行 [1024,5120] prefill | `ops::context_kv_materialize`（`draft.cpp:157-175`） | 硬要求 Q8_G32 [1024,5120]（`context_kv_materialize.cpp:37-52`）；融合 kernel 每 32 元素 1 字节码 + FP16 scale，7 条路线（`materialize.cu`、`launch.h:7-28`） | 需 NVFP4 codec + 7 路线重写 |
| QKV key/value（dflash v1 / MTP） | `ops::linear_pair`（`draft.cpp:203`、`mtp.cpp:87`） | Q8-only（`wrapper/linear_pair.cpp:42-52`） | **不在 DFlash2 路径上**，可不动 |
| gate_up [34816,5120] | `ops::linear_swiglu` | **已支持 NVFP4，且几何写死为 N34816K5120**（`wrapper/linear_swiglu.cpp:75-77,105,117-121`；`nvfp4_linear_swiglu_plan.cpp:56-67,148`） | A16 仅 T≤16（`plan.cpp:35`），T≥5 需 AllowA4 |
| output [5120,4096] | `linear_dynamic_grouped_conv_add` | 仅 Q8/Q5/Q4（`dynamic_grouped_conv.cpp:83-124,299-317`）；普通 NVFP4 `linear_add` 只认 [5120,6144]/[5120,17408]（`wrapper/linear_add.cpp:118-128,216-227`） | 需新 NVFP4 融合变体 + N5120K4096 几何 |
| down [5120,17408] | 同上 | 同上（普通 NVFP4 linear_add 支持该形状，但融合 conv-add 无 NVFP4） | 需新 NVFP4 融合变体 |
| kernel_projection [1280,5120] | `rmsnorm_dynamic_grouped_conv_prepare` | 仅 BF16/Q4（`dynamic_grouped_conv.cpp:43-55,57-75,236-250`） | 需新 NVFP4 prepare（partial+reduce） |
| feature_projection [5120,25600] | `ops::linear` | NVFP4 dispatch 是封闭 (n,k) 注册表，无 25600（`nvfp4_dispatch.cpp:9-21`；`nvfp4_geometry.h:44-73`） | 需新 shape + 几何 id |
| hidden_projection [256,5120] | `ops::linear` | 同上，无 [256,5120] | 2.5 MiB，收益 1.8 MiB，不值得 |
| codebook ×2 | `candidate_selector_path` | 仅 BF16 `Tensor`（`wrapper/candidate_selector.cpp:110-113`）；参数类型是裸 `Tensor`（`parameters.h:111-114`） | 需 codebook codec + 新 selector 路径 + Tensor→Weight + peer |

## 4. 三个硬阻塞

### 4.1 转换器：没有 float→NVFP4 量化器，也没有 draft 的 A4 标定来源

- `tools/convert/methods.py:296-301` 只注册 `cast_direct` / `grouped_absmax` / `fp8_row_maxabs` /
  `import_encoded`；唯一能产 NVFP4 的是 `import_encoded`（`:243-293`），它逐字节搬运预编码源，
  要求 `source.format == "nvfp4"`。`grouped_absmax` 只接受整数 QuantFormat（`:196-219`），
  `quantize_matrix` 对非 QuantFormat 直接抛错（`quantization/groupwise.py:83-85`）。
  `docs/maintainer/tensor-formats.md:92-93` 明写当前没有内置 float→NVFP4 量化器。
- draft 源是 compressed-tensors **动态激活** FP8：`DFlash2-FP8/manifest.json` 的
  `activation: float8, symmetric dynamic per-tensor`，只有 20 个 weight scale，**没有 input_global_scale**。
  而 A4 路由要求每个 (parameter,input) 的 `activation_input_divisor` Use（`load/prepare.cpp:44-54`），
  `import_encoded` 只在 `AllowA4` 且源里有 input scale 时才注入（`methods.py:269-280`）。
- 若只走 A16：`weight_input.cpp:108-110` 会为 NVFP4 A16 注入 divisor=1.0，但各 op 的 A16 路由
  对小 T 有限制（如 `nvfp4_linear_swiglu_plan.cpp:35` 仅 T≤16），draft 在 batch≥3 或 K>7 时
  T=(K+1)×B 会超界。要覆盖 1–8 并发必须 A4，也就必须新增一次 draft 激活标定。

### 4.2 Weight ABI：NVFP4 不允许子区域，融合 QKV 的 row-view 拓扑直接冲突

- `native_weight` 对 `QuantLayout::BlockScaleK16M128x4` 与 `RowScale` 的任何真子区域抛
  `"this native Weight input requires a complete FP8/NVFP4 parent"`（`src/core/weight_view.cpp:265-269`，
  `is_complete_weight` 要求 view.shape == parent.shape，`:175-185`）。
- `validate_nvfp4_weight` 还要求 `qdata==payload` 且 `scales==payload+scale_plane_offset`
  （`src/ops/linear/nvfp4/nvfp4_format.cpp:67-70`）；`weight_row_planes` 对 NVFP4 固定返回 parent 的
  scale base（`weight_view.cpp:219-223`），即 row_begin>0 的子区域连 plane 指针都不对。
- 生产路径中不存在非 payload base 的 NVFP4 Weight（`native_weight`、`tp_materialize.cpp:146-149`、
  `weight_splitter.cpp:140/187/400` 都在 base）。
- 而 draft 的 QKV 正是一个完整 [6144,5120] 对象，key/value/context_key/context_value 是行区间
  （`weight_view.cpp:193-199` 的 RowSplit 分支按 `row_begin*per_row_bytes` 偏移；`parameters.cpp:231-234`
  经 `prepare_linear_weight → native_weight` 绑定）。**当前只是一族 Q8 RowSplit 才能让这套共享成立。**
- 两条出路：
  1. 拆成三个完整 NVFP4 parent（query [4096,5120]、key/value [1024,5120]）。需要改官方配方的 share 结构，
     并给 `prepare_attn_input_proj_weights`（`ops/weight_input.cpp:165-179`，现在要求单连续 parent + Q8）
     加多 parent NVFP4 分支；prefill 的 context_kv_materialize 也才能拿到完整 parent。
  2. 扩展 ABI 让 `context_kv_materialize`/`linear_pair` 直接吃 `WeightRowPlanes`（已支持 swizzled NVFP4
     scales），但 GEMM 权重 staging 与 `validate_nvfp4_weight` 都假设 row_begin=0，属于共享 kernel 层改动。
- 附带约束：**一个对象只能有一种 dtype，而 QKV 同时被 decode（attn_input_proj）与 prefill
  （context_kv_materialize）消费** ⇒ 无法只把 decode 做 NVFP4、prefill 保持 Q8。

### 4.3 op 注册表与 kernel：封闭几何 + 手写融合 kernel

- NVFP4 的 (n,k) 是封闭集合：`resolve_nvfp4_geometry`（`nvfp4_geometry.h:44-73`）5 个 id、
  `nvfp4_dispatch.cpp:9-21` 10 个 shape；A4 TMA 描述符只对这 5 个 id 实例化（`nvfp4_w4a4_tma.cu`）。
  draft 需要的 N6144K5120、N5120K25600、N5120K4096、（N1024K5120）都不在其中。
- 需要新写的融合 kernel 族：`attn_input_proj` 3 输出（q4_q5 版是 `q4_q5_attn_input_*` 手写 MMA）、
  `context_kv_materialize`（`materialize.cu` 的 codes/scales staging 与 MMA B 片段解码，7 路线）、
  `dynamic_grouped_conv_add` 的 NVFP4 plan/materialized（q5/q4 版委托普通 GEMM + 共享
  `dynamic_conv_finish.cu`），`rmsnorm_dynamic_grouped_conv_prepare` 的 NVFP4 partial/reduce。
- `candidate_selector_path` 的 codebook 是裸 Tensor（`parameters.h:111-114`）且只有 BF16 变体
  （`wrapper/candidate_selector.cpp:110-113`）：要量化必须先有 codebook codec 并保持 top-k 域，
  再把 Tensor→Weight 管线与 peer 路径（`text.cpp:973+`、`draft.cpp:399-407`）接上。

### 4.4 TP-2 位置

draft 权重是 `Replicated shards=0x1`（只在 shard 0），codebook 是 `0x2`（只在 shard 1），
且这两条规则排在所有 shape 规则之前（`load/tp_split_spec.cpp:97-112`）⇒ draft 的融合 QKV
**永不按形状切分**。`weight_splitter` 本身支持 NVFP4 的 ColumnParallel/RowParallel 重布局
（`weight_splitter.cpp:112-190`），但 draft 用不到；codebook 量化则要覆盖 peer 上的 selector 路径。

## 5. 性能角度

- draft decode 的 T=(K+1)×B（默认 K=7、B=1 ⇒ 8）。NVFP4 的小 T 路由是 SIMT（`nvfp4_shapes.h`、
  `shapes/n34816_k5120.cu:31-39` 的 gemv/simt/exact），而现役 q4/q5 在小 T 已走 MMA
  （`linear_swiglu/q4/q4_linear_swiglu_plan.cpp:35-46` 的 `SmallTTiled = mma.small_t.tiled`）。
  ⇒ decode 换 NVFP4 **有退化风险**。
- NVFP4 的 TMA 张量核只在 T≥256（`select_a4` 分支）或 A4 MMA 的 T≥5 才启用；prefill/context append
  宽度 1..2048，能吃到一部分。但 DFlash2 的 prefill 代价本就只在 3–4% 量级，收益空间有限。
- A4 需要激活量化（额外 workspace + quantizer kernel），在 decode 每轮都要跑，进一步削弱小 T 的净收益。

## 6. 若要推进：改造清单（文件级）

1. **转换器**：`tools/convert/methods.py` 新增并注册 NVFP4 method（E2M1 codes + 每 16 K 的 E4M3FN
   scale + 每矩阵 FP32 divisor），与 `tools/artifact/codecs/nvfp4.py` 的 swizzle / `write_codes` 对齐；
   补 `tests/artifact/test_codecs.py`。**外加 draft 激活标定**（A4 必需）——仓库当前没有 draft 标定设施。
2. **配方**：`tools/convert/official_recipes.py:35-46` 的 dflash2 分支改为按需 NVFP4；draft 源是 FP8，
   新 method 必须支持 FP8→BF16→NVFP4。
3. **几何/形状**：`nvfp4_geometry.h`（+`resolve_nvfp4_geometry`）、`nvfp4_shapes.h`、
   `nvfp4_dispatch.cpp`、`shapes/*.cu`、`sources.cmake` 增加 N6144K5120 / N5120K25600 / N5120K4096。
4. **attn_input_proj 3 输出**：`wrapper/attn_input_proj.cpp`、`ops/weight_input.cpp`（
   `prepare_attn_input_proj_weights` 的 NVFP4/多 parent 分支）、新 kernel + plan。
5. **context_kv_materialize**：`materialize.cu` NVFP4 codec + `require_weight` 分支（7 路线）。
6. **dynamic grouped conv**：`wrapper/dynamic_grouped_conv.cpp`（`projection_planes`、
   `require_finish_projection_weight`、`kernel_projection_payload_bytes`、dispatch switch、容量 API）
   + `dynamic_grouped_conv/nvfp4/` + prepare 的 NVFP4 partial/reduce。
7. **candidate_selector**：codebook codec + 新 selector 路径 + `parameters.h`/`parameters.cpp` 的
   Tensor→Weight + `text.cpp` peer 路径。
8. **验证**（按 AGENTS.md 数值契约）：每个新 kernel 配独立 FP64 oracle（按存储 scale 独立解码）、
   路由边界 ±1、图重放；再补 §3.7/§8.2 的验收率 A/B、采样 A/B 与 DFlash2 三件套。

工作量粗估：r62–r66（4 项、已含 op oracle + 转换 + A/B）为一个基准单位；本清单约 3–4 个单位，
另加转换器/标定这一独立单位。

## 7. 建议

1. **不推进全 draft NVFP4**（显存动机不成立：对现役件净 −202 MiB，其中 174 来自 codebook）。
2. 若仍要释放显存，按性价比排序：
   - **codebook 量化**（−174 MiB）：独立杠杆，dtype 上 **Q4 比 NVFP4 更小**；需 codebook codec +
     top-k 域保持 + Tensor→Weight + peer。
   - **r65 融合 QKV→Q4**（−79.7 MiB）：RowSplit region 天然支持，但需 4-bit 融合材质化 kernel 一族 +
     3 输出 q4 变体 + 两套 oracle；接受率风险最高（K 投影误差在 2048 环内逐 token 累积）。
3. 只有当动机变成「draft 与主模型统一 W4A4 家族 / 吃 prefill TMA 张量核」时，才重新立项，
   并把激活标定与 decode 净收益测量作为前置门禁。
