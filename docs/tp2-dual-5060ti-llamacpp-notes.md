# llama.cpp 多卡张量并行(CUDA)实现剖析 —— C:/llama.cpp 只读分析

> 分析对象：C:/llama.cpp(只读，未做任何写入/构建)。
> 标注约定：【事实】= 我读到的代码/文档/git 原文，附 文件:行号；【推断】= 基于代码的推理，未运行验证。
> 行号均为该文件当前 HEAD 的内容。

---

## 0. 结论摘要(先看这段)

1. **这不是原版 llama.cpp。** 它是 zhuojun1024/llama.cpp 的 fork，当前在 ar3-opt 分支，HEAD 是一个自定义 commit，专门给 *3 卡无 P2P* 拓扑做 AllReduce 优化(README.md:20-56)。
2. **任务里假设的 ggml_cuda_op_mul_mat 行切分路径已经不存在了。** 上游在 2026-07-06 用 commit 74976e1ae('CUDA: remove -sm row, refactor cuBLAS', PR #24216)把 CUDA 的 row-split 实现删掉了；--split-mode row 在 CUDA 上现在会直接抛 'device %s does not support split buffers'(llama-model.cpp:1118-1119)，只有 SYCL 还留着 split buffer。
3. **当前的张量并行走的是完全不同的架构：** --split-mode tensor + ggml_backend_meta「虚拟设备」抽象(上游 PR #19378，commit d6f303004)。张量并行是**图级(graph-level)**的：切权重/切 KV 由 meta buffer 做，算子间插入 allreduce 由 meta backend 在图里切子图实现；ggml-cuda.cu 里只提供**通信原语**，不参与切分。
4. **跨卡通信不走 P2P、不走 cudaMemcpyPeerAsync、不走 NCCL(Windows 默认)**，而是 **pinned host staging + 内核内自旋(device-side spin)** 或 **copy-engine + event**(allreduce.cu:13-37)。fork 额外加了默认开启的 3-way 单 kernel allreduce(allreduce.cu:229-322, 1874-1983)。
5. 每层每 token 有 **2 次 allreduce barrier**(attn_output 行并行 + ffn_down 行并行)；层与层之间没有通信-计算重叠。fork 自己的诊断常量写着 '~81 subgraphs per decode step'(ggml-cuda.cu:1189-1196)。
6. KV cache 在 tensor 模式下**按 KV head 切分到各卡，不复制、不做跨卡 KV 拷贝**(llama-model.cpp:526-527 + llama-kv-cache.cpp:233-234, 1278-1284)。

---

## 1. 版本与拓扑(问题 1)

### 1.1 git 元数据

【事实】git log -1：

    92758c6475d48021fb32ab875a6dcec651e53df7
    Sat Sep 12 12:59:04 2026 +0800
    ggml-cuda: add 3-way AllReduce kernel (on by default) + timing instrumentation
    HEAD -> ar3-opt, fork/ar3-opt

- 分支：ar3-opt(本地还有 all / ar3-dual / ar3-ring / dflash2-tensor / local / master 等)。
- remote：fork https://github.com/zhuojun1024/llama.cpp 、 origin https://github.com/ggml-org/llama.cpp 。
- 上游基点：origin/master = 41fc7584f(2026-09-10 15:44:40 +0200, 'scripts : use sed instead of grep for version parsing (#28700)')；fork 的 merge 记录 824a41e11 'Merge branch master into all'、924cd0af7 'Merge branch all into ar3-opt'。
- git describe：b10825-91-g92758c647。

【事实】本地改动(git status)：只有 1 个文件被修改，**未提交**：

    ## ar3-opt
     M common/arg.cpp

内容(common/arg.cpp:883-893 附近)：mmproj 设备的默认选择在 LLAMA_SPLIT_MODE_TENSOR 下不再钉到 params.devices.front()，因为张量并行下 mmproj 要走 tensor_split：

    -    if (params.mmproj_use_gpu && params.mmproj_device == nullptr && !params.devices.empty()) {
    +    if (params.mmproj_use_gpu && params.mmproj_device == nullptr && !params.devices.empty()
    +        && params.split_mode != LLAMA_SPLIT_MODE_TENSOR) {

除此之外工作树干净(其余文件无修改)。

【事实】HEAD commit 92758c647 改了 4 个文件：README.md、ggml/src/ggml-cuda/allreduce.cu(+645)、ggml/src/ggml-cuda/ggml-cuda.cu(+257)、tools/server/server-context.cpp(+28)。

【事实】fork 自我描述的拓扑与目标(README.md:20-51)：目标是 '3-GPU inference on a mixed PCIe topology without P2P (two RTX 5060 Ti + one Tesla T10)'；给出 3 条 AllReduce 路线分支：ar3-opt(3-way kernel 小张量 + copy-engine 大张量)、ar3-dual(两条 2 卡 pipeline)、ar3-ring(ring)、all(无 3 卡 AR)。实测表(README.md:46-51)：two-pipeline ~37 tok/s decode / ~575 tok/s prefill，3-way kernel ~44.5 tok/s / ~665 tok/s(Qwen3.8-27B, draft-mtp)。
> 注意：用户任务说「双 GPU」，但这个检出实际是为 **3 卡** 调的；CUDA 内部 AR 原语本身只支持 **2 设备**，3 设备是两条 2 设备 pipeline 组合出来的。

### 1.2 支持哪几种 split mode，各走哪条代码路径

【事实】枚举(include/llama.h:198-203)：

| 值 | 名称 | 含义 |
|---|---|---|
| 0 | LLAMA_SPLIT_MODE_NONE | 单卡 |
| 1 | LLAMA_SPLIT_MODE_LAYER | 按层切 + KV 跟层 |
| 2 | LLAMA_SPLIT_MODE_ROW | 旧 row split(注释仍写 'use tensor parallelism if supported') |
| 3 | LLAMA_SPLIT_MODE_TENSOR | 新增，张量并行(meta device) |

CLI 解析：common/arg.cpp:2812-2823(none|layer|row|tensor)，帮助文本 common/arg.cpp:2808-2811。文档 docs/multi-gpu.md:22-27 明确把 row 标为 **Deprecated**、tensor 标为 **EXPERIMENTAL**。

**(a) layer(pipeline parallel)**
- 每个 GPU 拿一段连续层；KV cache 跟着层落在同一张卡。
- 相关代码：src/llama-context.cpp:428-433 决定是否启用 scheduler 的 pipeline parallel(split_mode() == LLAMA_SPLIT_MODE_LAYER && offload_kqv && n_gpu_layers > n_layer_all && !has_tensor_overrides())。
- **层间无集合通信**。

**(b) row(旧 row split)—— 在 CUDA 上已死**
- 路径：src/llama-model.cpp:1095-1120 make_gpu_buft_list()，当 split_mode == ROW 时向 backend reg 取 'ggml_backend_split_buffer_type' proc address；取不到就 throw std::runtime_error('device %s does not support split buffers')(llama-model.cpp:1118-1119)。
- 【事实】CUDA 后端注册表里已经没有这个符号：ggml/src/ggml-cuda/ggml-cuda.cu:5953-5971 的 ggml_backend_cuda_reg_get_proc_address() 只导出 ggml_backend_comm_init / comm_free / comm_allreduce_tensor / register_host_buffer / unregister_host_buffer / ggml_backend_get_features。
- 【事实】全仓库只剩 SYCL 还有 ggml_backend_sycl_split_buffer_type(ggml-sycl.cpp:1466、注册 7029-7030)。
- 【事实】删除它的 commit 是 74976e1ae 'CUDA: remove -sm row, refactor cuBLAS (#24216)'(2026-07-06)，且 git merge-base --is-ancestor 74976e1ae HEAD 为真。
- 【事实】ggml_cuda_op_mul_mat 这个函数在 ggml/src/ggml-cuda/ 全目录已经**不存在**：grep 只命中 ggml-cuda.cu:1601 的 typedef void (*ggml_cuda_op_mul_mat_t)(...) 以及 mmvf.cu:731 / mmvq.cu:1522 的单卡 kernel 入口。ggml_cuda_mul_mat(ggml-cuda.cu:2084)现在是纯单卡实现；ggml_cuda_mul_mat_id 里的 ggml_cuda_mul_mat(ctx, &src0_slice, ...)(ggml-cuda.cu:2315)是**单卡内 expert 切片**(配合 --n-cpu-moe)，不是多卡切分。
> 也就是说：任务里问的「ggml_cuda_op_mul_mat / ggml_cuda_mul_mat_batched 如何把权重行切到多卡」在当前检出的答案是——**这些函数已经没有了，CUDA 后端不再自己切多卡**。

**(c) tensor(当前唯一可用的张量并行)**
- 设备创建：src/llama.cpp:158-220 llama_prepare_model_devices()——当 split_mode == LLAMA_SPLIT_MODE_TENSOR 时，把所有非 CPU 设备包成**一个 Meta 设备**(ggml_backend_meta_device(devs, n, llama_meta_device_get_split_state, &ud)，llama.cpp:176-179 / 217-220)，模型只看到 1 个 device。
- 上下文约束：src/llama-context.cpp:3683-3695——TENSOR 强制 flash_attn = ENABLED(AUTO 会被改成 ON，显式 OFF 直接报错)；单设备使用会 warn。
- TENSOR 下不支持 backend sampling(llama-context.cpp:1225-1236)、--fit 不支持(common/fit.cpp:182-183)、row 的权重分配调整不支持(common/fit.cpp:476-477)。
- 架构白名单：src/llama-model.cpp:354-355 llm_arch_supports_sm_tensor(arch)；文档列了失败清单 docs/multi-gpu.md:89-93(MoE/SSM/RWKV 等多类)。
- 代码路径：**graph 层**由 ggml/src/ggml-backend-meta.cpp 主导，**权重布局**由 src/llama-model.cpp:371-839 的 split-state 回调决定，**通信**由 ggml/src/ggml-cuda/allreduce.cu + ggml-cuda.cu:965-1527 提供。

---

## 2. 张量并行的算子级实现(问题 2)

### 2.1 权重怎么切到多卡：不是算子切，是 buffer 切

【事实】每个张量的切分方式由回调 llama_meta_device_get_split_state() 给出(src/llama-model.cpp:371-839)。它用正则匹配张量名，返回一个 ggml_backend_meta_split_state{axis, ne[], nr[], n_segments}：

| 张量 | split 轴 | 证据 |
|---|---|---|
| blk.N.attn_q / attn_k / attn_v .weight | **axis 1**(输出维/head 维) | llama-model.cpp:511-513 |
| blk.N.attn_qkv.weight | axis 1 | llama-model.cpp:517-519 |
| blk.N.attn_output.weight | **axis 0**(输入维) | llama-model.cpp:529-531 |
| blk.N.ffn_up / ffn_gate(_exps) .weight | axis 1 | llama-model.cpp:561-563 |
| blk.N.ffn_down(_exps).weight | **axis 0** | llama-model.cpp:570-572 |
| cache_k_lN / cache_v_lN | **axis 0** | llama-model.cpp:526-527 |
| output.weight | axis 1(除非 output_mirrored / dsv4) | llama-model.cpp:581-586 |
| 其余(q/k bias、qk_norm、sinks…按需) | axis 0/1/MIRRORED | llama-model.cpp:514-575 |
| 兜底 | MIRRORED(所有卡全量复制) | llama-model.cpp:594 |

这正是 Megatron 式 **column-parallel(QKV/FFN-up，axis 1)+ row-parallel(attn_output/FFN-down，axis 0)**，row-parallel 的 matmul 输出被标记成 GGML_BACKEND_SPLIT_AXIS_PARTIAL(部分和，需要 allreduce)。

【事实】切分比例来自 tensor_split，并对齐量化块/head 粒度：llama-model.cpp:786-836(按 tensor_split_scan 求分界点，向下取整到 granule)，粒度由 get_split_granularity() 给(597-784)——例如 KV 权重/KV cache 用 granularity_kv = granularity_q / n_gqa(755-761)，保证每卡的 Q head 数正好是它的 KV head 数的 n_gqa 倍。还有 rotation 机制让不同层轮流向上/向下取整以更均衡(llama-model.cpp:453, 459, 462, 796, 815-818)。

【事实】物理切片发生在 meta buffer 层：ggml-backend-meta.cpp:1180-1250 为每个后端计算 t_ij(simple tensor，即该卡上的那块)的 ne/nb/view_offs；ggml_backend_meta_buffer_type_alloc_buffer 为每个后端各分配一块 buffer(1687-1701)。meta buffer 接口见 1659-1671。

### 2.2 “把部分和合并”：图级子图边界 + comm_allreduce，不是 ggml-cuda 里的 reduce

【事实】meta backend 的 graph_compute：
1. 先对整图逐节点求 split_state；凡是 axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL 的节点就是**子图边界**(ggml-backend-meta.cpp:2182-2222，判定在 2192)。
2. 循环里先让**所有**后端跑 cgraph_main[i](2434-2441)，再对每个后端该子图的**最后一个节点**调 comm_allreduce(2443-2454)：

       if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
           backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
       }
       if (!backend_allreduce_success) { allreduce_fallback(i); }   // 2456-2461

3. 若后端没提供 comm 实现，回退到 meta 的通用 butterfly：逐对 ggml_backend_tensor_copy_async + GGML_OP_ADD(2317-2431，n_reduce_steps = ceil(log2(n_devs)) 见 1810)。

【事实】comm_allreduce 是后端注册的 proc address，CUDA 端是 ggml_backend_cuda_comm_allreduce_tensor(ggml-cuda.cu:1521-1527)，由 comm_init 选出的函数指针 try_allreduce 分派：

| 优先级 | 实现 | 位置 |
|---|---|---|
| 1 | NCCL(ncclAllReduce，小张量 FP32 / 大张量 BF16) | ggml-cuda.cu:1009-1084, 1444-1474 |
| 2 | internal(自研 pinned-host AR) | ggml-cuda.cu:1290-1363, 1408-1442 |
| 3 | none → meta butterfly 兜底 | ggml-cuda.cu:1404-1406, 1439-1441 |

【事实】平台默认(Linux 走 NCCL，其它/Windows 走 internal)：

    // ggml-cuda.cu:1494-1501
    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
    #if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
    #else
        ggml_backend_cuda_comm_init_internal(ret);
    #endif
    }

NCCL 在虚拟设备下会被禁用(1446-1453)。GGML_CUDA_NCCL 是 CMake **默认 ON**(ggml/CMakeLists.txt:210)，但 Windows 上如果没找到 NCCL 包只打 warning(ggml/src/ggml-cuda/CMakeLists.txt:174-182)。

### 2.3 internal AllReduce 的机制(核心)：pinned host + 内核自旋 / copy-engine + event

【事实】ggml/src/ggml-cuda/allreduce.cu:13-37 的头部注释直接定义了两条策略：
- **chunked kernel path(小张量、decode)**：一个 kernel 同时把数据写到 pinned host 并做本地加；**跨 GPU 同步在 kernel 内部**(busy-wait host 内存 flag)。
- **copy-engine path(大张量、prefill)**：D2H + H2D cudaMemcpyAsync 分块 + 一个小的设备内 add kernel；**跨 GPU 同步在 kernel 外部**(CUDA event)。
- 明确写了 'targets setups without NVLink, where data is exchanged between the GPUs by staging it through pinned host memory over PCIe'(allreduce.cu:18-19)。

【事实】同步原语(allreduce.cu:40-66)：
- 每 (slot, rank) 在 pinned host 上有一个 int 到达令牌；写方 *(volatile int*)p = token，读方 *(const volatile int*)p 自旋等 token 相等(allreduce.cu:61-66)。
- 写方先 __threadfence_system() 再发 token(释放语义)；注释解释为什么不用 atomicAdd_system：'requires hostNativeAtomicSupported, which is unavailable on PCIe-attached consumer GPUs without NVLink'(allreduce.cu:52-58)。
- 每 64 B 一个到达 int，避免 false sharing(allreduce.cu:68-71)；8 个 block，每 block 独立令牌(allreduce.cu:73-76, 101-106)；spin 里 __nanosleep(100)(allreduce.cu:288-292)。
- 每个 token 是单调递增的 AR 调用序号，不回绕(31 位 int)，省掉每轮 reset(allreduce.cu:44-49)。ring 深度 GGML_CUDA_AR_POOL_SIZE = 2(allreduce.cu:401-407)。

【事实】pinned host 分配是 cudaHostAlloc(..., cudaHostAllocPortable | cudaHostAllocMapped) + cudaHostGetDevicePointer(allreduce.cu:441-474)，即 CPU 不碰、GPU 直接读写的映射内存。

【事实】2 设备 chunked 路径主循环(allreduce.cu:1265-1475)：
- 判类型/对齐/连续性，不支持就 return false 让上层兜底(1303-1334)；
- F32 且 nbytes >= bf16_threshold(默认 1，即恒开)时走 **F32→BF16 线上压缩往返**(1282-1289, 604-607)；
- 按 chunk 循环(每 chunk 上限 GGML_CUDA_AR_MAX_BYTES = 1 MB，allreduce.cu:411、1417)，每 chunk 每设备 launch 双向 kernel(1441-1462)，在**调用者的 compute stream** 上跑(allreduce.cu:1411-1416 注释明确说这是 barrier)；
- 结束记录 ev.ker(1467-1469)。

【事实】2 设备 copy-engine 路径(allreduce.cu:1116-1229)：stage 1 各卡 D2H 到自己的 pinned buffer、逐 chunk 记 event(1158-1167)；stage 2 各卡等 peer 的 chunk event、H2D 到自己 dev_tmp、记 host_large_read_done、再 cudaEventRecord(h2d) + compute stream cudaStreamWaitEvent(h2d)、跑 add kernel(1188-1217)。阈值 GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1 MB、单次上限 GGML_CUDA_AR_COPY_MAX_BYTES = 32 MB(419, 415)，超过 32 MB 由 outer chunker 切片(1231-1245)。

【事实】fork 的 3-way 单 kernel(allreduce.cu:229-322)：每个设备把自己的 shard cast 到 T_wire 写进自己的 pinned slot，然后**每 block 自旋等另外两个 peer 的 token**，最后读三个 slot 按**固定 device 顺序** ((w0+w1)+w2) 相加——注释说这样保证三卡结果 **bit-identical**(allreduce.cu:220-227, 301-321)。调用点 ggml_cuda_ar_allreduce3way(1727-1822)，默认开、GGML_CUDA_AR3WAY=0 关(1902-1938)；单次 tensor 必须放得进一个 rank slot(1 MB)否则 return false 回落到 2-pipeline(1751-1755)。

【事实】3 设备组合两条 2-device pipeline：(dev0,dev1) 与 (dev0,dev2) 共享 pivot dev0(ggml-cuda.cu:1408-1435)；序列是 AR(dev0,dev1) → AR(dev0,dev2) → host-staged broadcast dev0→dev1(allreduce.cu:1939-1975、广播 kernel 324-395)。大张量走专门的 3 设备 copy-engine(allreduce.cu:894-1115, 1888-1900)。广播失败时回退 ggml_backend_tensor_copy_async(1972-1975)。

### 2.4 P2P / cudaMemcpyPeerAsync / NCCL 的实际使用面

【事实】cudaDeviceEnablePeerAccess **只在 GGML_CUDA_P2P 环境变量存在时**才调用(ggml-cuda.cu:391-405)；NCCL 构建也会隐式开 peer access(ggml-cuda.cu:605-609, 611-640，VMM 分配要显式 cuMemSetAccess)。
【事实】cudaMemcpyPeerAsync 只出现在通用张量拷贝 cpy_tensor：buffer 版 ggml-cuda.cu:820-843(834 行)、backend async 版 2746-2801(2784 行)。两处都在 #ifdef GGML_CUDA_NO_PEER_COPY 下**直接 return false**(831-835 / 2781-2785)，并把 props.caps.events 置 false(5306-5311)、event_new 返回 nullptr(5819-5842)。GGML_CUDA_NO_PEER_COPY 由 CMake 选项 GGML_CUDA_NO_PEER_COPY 控制，**默认 OFF**(ggml/CMakeLists.txt:203；定义处 ggml/src/ggml-cuda/CMakeLists.txt:143-145)。
【事实】**AllReduce 路径完全不使用 peer copy**：internal AR 只做 host staging；NCCL 路径用 ncclAllReduce。也就是说在无 P2P 的机器上，只要不开 GGML_CUDA_P2P，机器上就不会有 D2D peer 流量(除非 meta butterfly 兜底里走 tensor_copy_async，而那条会走到 2784 的 cudaMemcpyPeerAsync)。

---

## 3. 跨卡通信次数与时机(问题 3)

### 3.1 次数

【事实】通信点 = 图中的 PARTIAL 边界(ggml-backend-meta.cpp:2188-2192)。PARTIAL 的产生规则在 handle_mul_mat(ggml-backend-meta.cpp:577-608)：
- src0 轴 1 split + src1 MIRRORED → 输出轴 0 split(列并行，**无通信**)；
- src0 轴 0 split + src1 轴 0 split → **PARTIAL**(行并行，**要 allreduce**)。

【事实】标准 decoder 层的两个行并行 matmul：attn_output.weight 轴 0(llama-model.cpp:529-531)和 ffn_down.weight 轴 0(570-572)。所以：
> **每个 transformer 层、每个 token = 2 次跨卡 allreduce**(一次在 attn_output 之后、一次在 ffn_down 之后)；n_layer 层约 2*n_layer 次/token。

【事实/佐证】fork 的子图计时诊断把 slot 数写成 **81**，注释原文 'one per meta subgraph, ~81 per decode step'(ggml-cuda.cu:1189-1196)，R = 81(1196)。81 = 2×40+1 的形式与“40 层、每层 2 个边界”完全吻合(该 fork 的模型是 Qwen3.8-27B)。
【推断】若模型是 MoE，PARTIAL 还来自 ffn_down_exps；若是线性注意力混合架构，ssm_out 等行并行层也各贡献一个边界。具体层数/模型会有差异。

【事实】通信量：每次 AR 是对**整个 hidden 尺寸**(row-parallel 的输出，形如 [n_embd, n_tokens])的求和；decode 时 n_tokens 小(注释举例 '80 KiB'，allreduce.cu:1751-1753)，prefill 时大。

### 3.2 时机与同步

| 场景 | 同步形式 | 证据 |
|---|---|---|
| 小张量(chunked kernel) | **in-kernel 自旋**(volatile + __threadfence_system + __nanosleep)，kernel 跑在 compute stream，是硬 barrier | allreduce.cu:61-66, 276-299, 1411-1416 |
| 大张量(copy-engine) | **stream 事件**(app/cpy/h2d/ker 事件池)，add kernel 在 compute stream，H2D 在专用非阻塞 stream | allreduce.cu:434-439, 562-567, 1137-1224 |
| ring 回绕 | **host 侧 cudaEventSynchronize**(POOL_SIZE=2，从第 3 次 AR 起每次都会做一次，等的是 *上一个* AR 的 ker 事件) | allreduce.cu:537-550 |
| 3-way kernel | in-kernel 自旋等两个 peer | allreduce.cu:280-299 |
| 设备整体同步 | ggml_backend_meta_synchronize 逐后端 synchronize；llama 层 llama_synchronize | ggml-backend-meta.cpp:1957-1962 |

【事实】没有 cudaStreamSynchronize 出现在 AR 路径里；host 侧唯一的阻塞点是 ring 回绕时的 cudaEventSynchronize(537-550)和计时节点的 cudaEventElapsedTime。

### 3.3 通信与计算的重叠

【事实】重叠机制**只存在于一次 AR 内部**，不存在跨层/跨子图的通信-计算重叠：
- copy-engine 路径让 D2H/H2D 在专用 AR stream 上由 copy engine 跑，add kernel 在 compute stream 跑，注释明确写 'This keeps the compute engine free while large transfers are in flight, which matters for prefill-sized tensors'(allreduce.cu:29-34)；AR stream 用 cudaStreamCreateWithFlags(cudaStreamNonBlocking)(617)。
- chunked 路径**故意**放在 compute stream 上(allreduce.cu:1411-1416)：注释说 'since AR is a barrier here, same-stream ordering subsumes any cross-stream event handshake... and skips the cross-stream scheduling overhead that was hurting the small-tensor (tg) latency'。
- meta backend 是**逐子图 lockstep**：先所有卡跑子图 i(异步 launch)，再 allreduce，再子图 i+1(ggml-backend-meta.cpp:2434-2461)。没有把子图 i+1 的计算与子图 i 的通信重叠的机制(它们在同一 per-device stream 上，天然串行)。
- 3-way kernel 必须等最慢的 peer(README.md:44)，所以 -ts 要均衡。

---

## 4. CUDA graph(问题 4)

### 4.1 编译与开关
- GGML_CUDA_GRAPHS 默认 ON(CMakeLists.txt:170)，开启后 add_compile_definitions(GGML_CUDA_USE_GRAPHS)(ggml/src/ggml-cuda/CMakeLists.txt:123-125)，common.cuh:1261-1263 把它映射成 USE_CUDA_GRAPH。
- 运行期可用 GGML_CUDA_DISABLE_GRAPHS 关掉(common.cuh:1292)。

### 4.2 捕获/复用位置
- 入口 ggml_backend_cuda_graph_compute()(ggml-cuda.cu:4681-4752)。
- 真正干活的是 ggml_cuda_graph_evaluate_and_capture()(ggml-cuda.cu:4446-4662)：cudaStreamBeginCapture(..., cudaStreamCaptureModeRelaxed)(4732) → 逐节点执行(or capture) → cudaStreamEndCapture(4635) → cudaGraphInstantiate(4650) → cudaGraphLaunch(4656)。
- 更新已有实例用 cudaGraphExecUpdate(2903)，失败则销毁重建(2910-2920)。

### 4.3 缓存 key 与复用判据
- **key = cgraph->nodes[0] 的指针**(ggml-cuda.cu:2852-2854)，存在 per-CUDA-context 的 std::unordered_map<const void*, ggml_cuda_graph>(common.cuh:1461-1489)。
- 命中后还要 cgraph->uid == graph->uid 才直接复用(2862-2867)；否则逐节点比较 node 结构 + 每个 src 的 data 指针 + ne/nb(2877-2895)，任何变化都要重捕。
- 所以「按什么 key 缓存」的准确答案是：**首节点地址为索引 + 全图逐节点形状/指针快照做等价性判定**，而**不是**参数化的 'envelope'/batch bucket。【推断】由于 llama 复用同一批 ggml_context/buffer，同一 ubatch 形状下首节点地址稳定 → 同一张图可长期复用；ubatch/批大小变了(例如 prefill chunk vs 单 token decode)会触发属性变化 → 重捕。

### 4.4 warmup 与“一次生成能复用多少次”
【事实】(ggml-cuda.cu:4690-4722)：
- 第一次调用：属性变化 → **直接执行、不捕获**(use_cuda_graph = false)；
- 第二次属性不变 → warmup_complete = true，这一次**捕获并 launch**(4701-4708)；
- 之后每次：use_cuda_graph = true，cuda_graph_update_required = (instance == nullptr) → 只 cudaGraphLaunch 复用(4716-4719)。
> **【推断】** 一轮请求若做 R 次 decode step，则约 warmup 1 次直接执行 + R−1 次捕获/复用(其中第 2 次是捕获+launch)。因为 decode 每 token 一次调用且形状不变，稳态下**每个 token 复用同一 instance**。
- 淘汰：每 5 s 扫一次，10 s 未用就删(common.cuh:1471-1481)。
- **注意 tensor 模式下是“每子图一张 CUDA graph”**：meta backend 把整图切成 ~81 个子图，每个子图在每个后端各调用一次 graph_compute → 每设备约 81 张图、每 decode step replay 81 次(证据 ggml-cuda.cu:1189-1196)。

### 4.5 replay 前需要的 host 侧 memcpy
【事实】输入张量(tokens/pos/kq_mask/k_idxs/…)在 replay 之前用 ggml_backend_tensor_set[_async] 写入(示例 src/llama-graph.cpp:73, 81, 98, 104, 113, 142, 172)。meta 设备会按 split_state 分发：MIRRORED → 每个后端各写一份；轴切分 → 用 set_tensor_2d_async 按块拼到各卡(ggml-backend-meta.cpp:1867-1910)。
【事实】CUDA 侧两条写入路径：
- buffer 同步版：cudaMemcpyAsync(..., cudaMemcpyHostToDevice, cudaStreamPerThread) **之后紧跟 cudaStreamSynchronize(cudaStreamPerThread)**(ggml-cuda.cu:784-790)——host 阻塞，源指针直接是调用方内存。
- backend 异步版：cudaMemcpyAsync(..., cuda_ctx->stream())(ggml-cuda.cu:2706-2713)，挂在 compute stream 上，因此与随后的 graph launch 天然有序，**不需要额外同步**。
【事实】**是否 pinned**：默认不是 pinned——cudaMemcpyAsync 的源是普通 host 内存(如 llama_batch/std::vector)。pinned 只在这些地方：
- ggml_cuda_host_malloc() 用 cudaMallocHost，可被 GGML_CUDA_NO_PINNED 关掉(ggml-cuda.cu:1545-1561)，并决定 device props 的 host_buffer cap(5306)；
- AR 的 staging/arrival 一定是 pinned+mapped(allreduce.cu:451-465)。
【事实】输出侧：meta 的 get_tensor_async 对轴切分张量按块从**各卡** D2H 再在 host 拼起来(ggml-backend-meta.cpp:1912-1955)——这也是 tensor 模式下 logits(vocab 轴 1 切分)的收集方式；MIRRORED 张量只从 device 0 读(1945-1950)。
【事实】KV cache 的写入是**图内 op**(cpy_k/cpy_v → GGML_OP_SET_ROWS 之类)，不是 host memcpy。
【事实】CUDA graph capture 里不会有这些 H2D：它们发生在 ggml_backend_graph_compute 之前(host 线程)，不参与捕获。

---

## 5. KV cache 多卡分布(问题 5)

【事实】KV 物理张量：ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)(src/llama-kv-cache.cpp:233-234)，即 [head_dim × n_head_kv, kv_size, n_stream]。
【事实】split-state 回调把 cache_k_lN / cache_v_lN 标成 **axis 0**(llama-model.cpp:526-527)，也就是**沿 n_embd_k_gqa 切 = 按 KV head 切**；粒度用 granularity_kv = granularity_q / n_gqa(llama-model.cpp:755-761)保证切点落在 head 边界。
【事实】读出时 get_k()/get_v() 返回 4D 视图 [head_dim, n_head_kv, n_kv, n_stream](llama-kv-cache.cpp:1278-1284 / 1301-1315)，随后 ggml_permute(q/k/v, 0, 2, 1, 3) 变成 head 在轴 2(src/llama-graph.cpp:2607-2611)；FA 输出形状是 {v->ne[0], q->ne[2], q->ne[1], q->ne[3]}(ggml/src/ggml.c:5525-5527)，所以 head 在轴 1。
【事实】handle_flash_attn_ext 断言 Q 是轴 2 split，K/V **要么轴 2 split、要么整体 MIRRORED**(ggml-backend-meta.cpp:773-791)。
> **结论(事实)：tensor/row 模式下 KV cache 不需要跨卡复制**——每张卡只持有自己那部分 KV head，本地做完自己 head 的 attention，partial 结果由 attn_output 的 allreduce 合并。**没有 KV 的跨卡 cudaMemcpy**。
【事实】例外(这些模型的 cache/state 被强制 MIRRORED，即每卡冗余复制全量 KV)：
- dsv4 的 cache_k/cache_v 和 dsv4_*_state_*(llama-model.cpp:474-478)；
- indexer cache(501-503)、PLE 表(505-508)；
- LFM2/LFM2MOE 的 cache_r/cache_s(546-549)。
【事实/注意】文档说 'SPLIT_MODE_TENSOR + KV cache 量化' 会在启动时报错(docs/multi-gpu.md:87, 121)，但我在本版本代码里**没有找到这条显式检查**(llama-context.cpp 中 TENSOR 相关检查只有 3683-3695 的 flash_attn 要求和 1225 的 backend sampling；llama-kv-cache.cpp 里也没有 TENSOR 字样)。【推断】该文档条目可能是过期的，或限制由别处(meta split-state 断言 / FA kernel 能力)间接体现——未验证。

---

## 6. 它“没做”的事 + Windows/WDDM/无 P2P 限制(问题 6)

### 6.1 明确没有做的(与“自研引擎可能做了”的对照)

| 项 | 现状 | 证据 |
|---|---|---|
| **通信与计算融合(in-kernel allreduce / 把 reduce 融进 GEMM epilogue)** | **没有**。AR 是独立 kernel，且每层 2 个 barrier。fork 的 'in-kernel' 只是指 *通信 kernel 内部的跨卡 spin*，不是与计算融合 | allreduce.cu:229-322；ggml-backend-meta.cpp:2434-2461 |
| **通信与下一层计算重叠(communication/computation overlap)** | **没有**。meta 逐子图 lockstep；AR 与 compute 同流(小张量)或只与同一次 AR 的拷贝重叠 | ggml-backend-meta.cpp:2434-2461；allreduce.cu:29-36, 1411-1416 |
| **按 envelope/桶参数化的图缓存** | **没有**。key 是首节点指针 + 逐节点属性快照；批大小变化触发 warmup reset + 重捕 | ggml-cuda.cu:2852-2894, 4712-4715 |
| **host checkpoint 环 / 激活环** | **没有**。AR ring 只有 2 个 slot，且仅用于通信 staging | allreduce.cu:401-407 |
| **跨卡 P2P/NVLink 直连 AR** | **没有**。AR 永远走 pinned host(cpy_tensor 才有 peer copy，且默认不启用 peer access) | allreduce.cu:18-19；ggml-cuda.cu:391-405, 2784 |
| **激活压缩(FP8/INT8)** | 只有 F32→BF16 往返(可关) | allreduce.cu:604-607, 1282-1289 |
| **量化 KV + 张量并行** | 未实现/未验证(文档声称报错，代码未找到检查) | docs/multi-gpu.md:87,121 |
| **张量并行下的 backend sampling** | 明确不支持，回退 CPU | llama-context.cpp:1225-1236 |
| **张量并行下 --fit 自动配内存** | 明确不支持 | common/fit.cpp:182-183, 476-477 |
| **CUDA 上的 row split** | 已删除 | commit 74976e1ae；llama-model.cpp:1118-1119 |
| **>3 卡的高效 AR** | internal AR 只支持 2 设备；3 设备靠两条 2-device pipeline；>3 设备回退 meta butterfly(逐对 tensor copy + ADD，最慢) | allreduce.cu:575-590；ggml-cuda.cu:1408-1441；ggml-backend-meta.cpp:2317-2431 |
| **每设备同一物理卡多虚拟设备** | 有处理(比较 physical device，同物理卡走 D2D 而非 peer copy) | ggml-cuda.cu:824-830, 2774-2780 |

### 6.2 Windows + WDDM + 无 P2P(SYS 拓扑)下的限制/回退路径

【事实】
1. **AR 后端选择**：非 Linux(含 Windows)默认用 **internal**(pinned host)而不是 NCCL(ggml-cuda.cu:1494-1501)；NCCL 未编译时打 warning 并回落 internal(1466-1471)；虚拟设备下禁用 NCCL(1446-1453)。
2. **所有跨卡数据都经过 pinned host、两趟 PCIe**(allreduce.cu:18-19, 441-474)——SYS/无 P2P 拓扑下的必然代价，也是 fork 做 3-way kernel 的动机。
3. **P2P 必须显式开** GGML_CUDA_P2P(ggml-cuda.cu:391-405)；文档警告某些主板/BIOS(IOMMU)下会崩溃或输出损坏(docs/multi-gpu.md:104-112, 127)。
4. **peer copy 的编译期开关** GGML_CUDA_NO_PEER_COPY(默认 OFF，ggml/CMakeLists.txt:203)：打开后 cpy_tensor 直接返回 false(ggml-cuda.cu:831-835, 2781-2785)、device props 的 events = false(5307-5311)、event_new 返回 nullptr(5826-5829)、feature 里报 NO_PEER_COPY(5918-5920)。作者加这个开关正是为了无 P2P 平台。
5. **内部 AR 的准入条件**：必须 cc >= Volta(要用 __nanosleep)且 n_devices == 2，否则 init 失败 return nullptr 并回落(allreduce.cu:575-590)。3 设备由两条共享 pivot 的 2 设备 pipeline 拼(ggml-cuda.cu:1408-1435)；再失败则回落 meta butterfly(1437-1441)——那是纯 tensor_copy_async + GGML_OP_ADD 的通用实现(ggml-backend-meta.cpp:2317-2431)，在无 P2P 时会变成 host 往返，性能最差。
6. **HIP/MUSA 直接不支持** internal AR：allreduce.cu:1985-2004 提供空实现(return nullptr/false)，注释说明 HIP 缺 cudaHostAllocPortable/Mapped、cudaHostGetDevicePointer、__nanosleep。
7. **Windows 相关构建修补**：fork 有 commit 6c98116ed 'ggml-cuda: fix winsock.h (v1) conflict with NCCL headers on Windows'(README.md:35；CMakeLists.txt:259 有注释)。
8. **WDDM**：全仓库 grep WDDM **零命中**——代码里**没有任何 WDDM 专门分支/回退**。【推断】WDDM 下 pinned memory 受限、cudaMemcpyAsync 页锁定 staging 更贵、GPU 提交走命令队列带来的额外延迟，本实现都没有针对性处理；唯一间接缓解是默认走 internal(避免 NCCL 在 WDDM 上的问题)。

### 6.3 环境变量/常量速查(便于对齐自研引擎)

| 名称 | 默认 | 作用 | 位置 |
|---|---|---|---|
| GGML_CUDA_ALLREDUCE | 平台默认(Linux=nccl，其它=internal) | 选 nccl/internal/none | ggml-cuda.cu:1494-1514 |
| GGML_CUDA_AR3WAY | 开 | =0 关 3-way kernel | allreduce.cu:1907-1908 |
| GGML_CUDA_AR_TIMING | 关 | 打 ar_timing/sg_timing/ar3way | ggml-cuda.cu:1097, 1337; allreduce.cu:1918 |
| GGML_CUDA_P2P | 关 | cudaDeviceEnablePeerAccess | ggml-cuda.cu:391 |
| GGML_CUDA_DISABLE_GRAPHS | 关 | 禁用 CUDA graph | common.cuh:1292 |
| GGML_CUDA_NO_PINNED | 关 | 禁 pinned host buffer | ggml-cuda.cu:1546, 5306 |
| GGML_CUDA_AR_COPY_THRESHOLD | 1 MB | chunked→copy-engine 阈值 | allreduce.cu:419, 595 |
| GGML_CUDA_AR_BF16_THRESHOLD | 1(恒开) | F32→BF16 线上压缩 | allreduce.cu:604-607 |
| GGML_CUDA_AR_COPY_CHUNK_BYTES | 0(启发式 nbytes/4) | copy chunk 大小 | allreduce.cu:598-603, 552-560 |
| LLAMA_SERVER_TIMING | 关 | server 计时 | README.md:56; server-context.cpp 改动 |
| GGML_CUDA_NO_PEER_COPY | CMake OFF | 禁 peer copy/event | ggml/CMakeLists.txt:203 |
| GGML_CUDA_NCCL | CMake ON | 编 NCCL | ggml/CMakeLists.txt:210 |
| GGML_CUDA_GRAPHS | CMake ON | 编 CUDA graph | CMakeLists.txt:170; ggml/CMakeLists.txt:209 |

---

## 附录：本文所有关键行号索引

- split mode 枚举：include/llama.h:198-203
- layer pipeline 判定：src/llama-context.cpp:428-433
- row 需要 split buffer：src/llama-model.cpp:1095-1120
- CUDA reg proc address(无 split_buffer)：ggml/src/ggml-cuda/ggml-cuda.cu:5953-5971
- TENSOR 建 Meta 设备：src/llama.cpp:158-220
- TENSOR 上下文约束：src/llama-context.cpp:3683-3695
- split-state 回调：src/llama-model.cpp:371-839(tensor_config 473-595、segments 597-669、granularity 671-784、扫描 786-836)
- meta buffer 切片/分配：ggml/src/ggml-backend-meta.cpp:1180-1250, 1659-1701
- 子图切分 + allreduce 调度：ggml/src/ggml-backend-meta.cpp:2182-2222, 2434-2461
- 通用 butterfly 兜底：ggml/src/ggml-backend-meta.cpp:2317-2431
- meta 输入/输出分发：ggml/src/ggml-backend-meta.cpp:1867-1955
- mul_mat split 规则：ggml/src/ggml-backend-meta.cpp:577-608
- FA/KV split 规则：ggml/src/ggml-backend-meta.cpp:773-791
- CUDA comm context / NCCL / internal 分派：ggml/src/ggml-cuda/ggml-cuda.cu:965-1084, 1290-1527
- 3 设备 pipeline 组合：ggml/src/ggml-cuda/ggml-cuda.cu:1408-1435
- internal AR 实现：ggml/src/ggml-cuda/allreduce.cu:13-37, 61-76, 108-322, 397-712, 894-1229, 1265-1475, 1631-1705, 1722-1983
- P2P/peer copy：ggml/src/ggml-cuda/ggml-cuda.cu:391-405, 605-648, 820-843, 2746-2801, 5306-5311, 5819-5842
- CUDA graph：ggml/src/ggml-cuda/ggml-cuda.cu:2818-2925, 4446-4662, 4681-4752；ggml/src/ggml-cuda/common.cuh:1261-1296, 1459-1489
- 子图诊断(81)：ggml/src/ggml-cuda/ggml-cuda.cu:1189-1196
- KV 张量与视图：src/llama-kv-cache.cpp:233-234, 1266-1316；src/llama-graph.cpp:2602-2611；ggml/src/ggml.c:5525-5527
- 文档：docs/multi-gpu.md:22-27, 37, 80-93, 104-112, 118-127；README.md:20-56
