# GGUF block formats

NInfer serves Qwen3.8-27B GGUF releases that give every tensor its own ggml quantization type,
such as ISTA-DASLab's [GSQ-RCO models](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF),
without converting their weights to another format. The converter recipe `qwen3_8_27b_gguf` copies
each quantized tensor's blocks into the artifact unchanged (see
[weight conversion](weight-conversion.md#a-mixed-precision-qwen38-27b-gguf)), and the runtime
multiplies them in place.

## Formats

All fifteen block types llama.cpp writes for dense models: `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`,
`Q5_K`, `Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`,
`IQ4_NL` and `IQ4_XS`, stored as the `gguf_*` formats of the `gguf_blocks_v1` layout
([tensor formats](maintainer/tensor-formats.md#35-gguf-block-formats),
[storage layouts](maintainer/storage-layouts.md#6-gguf_blocks_v1)). A projection may mix types: the
parts of a fused projection that share a type are multiplied together, the others one by one against
the same activation.

## Products

Every product quantizes its BF16 activation to ggml's `q8_1` numbers, one scale and one sum per 32
values, as llama.cpp does, and accumulates in FP32; the weights' represented values are exactly those
of `ggml-quants.c`.

- **Up to eight columns** (decode, speculative verification, small batches), the vector kernel
  (`src/ops/linear/gguf/ggml_bridge_vec.cuh`) decodes each 32-value slice of a row once into int8
  words and dots it with every column. A warp owns two rows; a `[gate; up]` pair of one type is one
  launch that writes `silu(gate) * up`.
- **More columns** (prompts) take llama.cpp's integer tensor-core kernel (`mul_mat_q`, vendored
  unmodified in `third_party/ggml-quants` under its MIT license) with its Ampere tile table.
  `IQ1_M` has no such kernel; its wide products dequantize to BF16 and run cuBLAS.
- **Token table** rows are dequantized exactly.

The scales of `IQ2_XXS`, `IQ2_XS`, `IQ2_S` and `IQ3_XXS` are applied in FP32 where llama.cpp's
vector kernels round an integer rescale, so a decode step's logits can differ from llama.cpp's in the
last bits; prompts use llama.cpp's own arithmetic.

The GGUF head of the reduced proposal table materializes BF16 logits per chunk and ranks them through
the same grouped K-split reduction the fused producers use.

## Serving

A converted artifact starts like any other; MTP, DFlash2 and Vision work as with the official
artifact:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_gsq_rco_iq3_s.ninfer \
  --model-id qwen3.8-27b --kv-dtype rk8v4 --spec mtp --draft-tokens 3
```

The converter recipe, the artifact and the products above are qualified by
[`tests/ops/linear/test_gguf.cpp`](../tests/ops/linear/test_gguf.cpp), which compares every block
type against the FP64 product of its exactly dequantized weights.
