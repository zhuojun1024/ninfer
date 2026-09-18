# Dual RTX 5060 Ti TP-2 tuning (Qwen3.8-27B NVFP4)

This note records the tensor-parallel-2 (TP-2) adaptation of this engine for two RTX 5060 Ti
(16 GiB) cards running `Qwen3.8-27B` NVFP4. Numbers were measured on this machine with
`ninfer-serve` on `--devices 0,1`, greedy sampling, one request at a time. The routines live in
`tools/tp_bootstrap/`.

## Recommended configuration

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 131072 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088
```

`--kv-dtype fp8 --max-context 131072` doubles the usable context (65,536 -> 131,072 tokens) at
essentially the same device memory and throughput as bf16 KV at 65,536 (see below).

## Benchmarks

Prefill is measured with a 279-token prompt and an 8-token output (615 tok/s at 65,536 bf16); the
first request after startup also pays CUDA-graph capture (14.4 s observed on the fp8/131,072 config),
so the steady-state number is the meaningful one. Decode is a 70-token prompt with a 256-token output
reported end-to-end (prefill included), so it understates sustained decode. Sustained decode, measured
from committed-token traces at temperature 0: plain 31.5 tok/s, MTP K=1 42.0, K=2 50.5-54.0,
K=3 47.9-48.0 tok/s.

| Configuration | KV capacity | Memory per GPU | Prefill | Decode (256 tok) |
|---|---|---|---|---|
| plain, bf16 KV | 65,536 | 13,888 MiB | 0.453 s (279 tok, 615 tok/s) | 8.153 s (31.4 tok/s) |
| MTP K=3, bf16 KV | 65,536 | 14,582 / 14,322 MiB | 0.526 s | 6.532 s (39.2 tok/s) |
| plain, fp8 KV | 65,536 | - | - | 8.091 s |
| plain, nvfp4 KV | 65,536 | - | - | 8.093 s |
| plain, int8 KV | 65,536 | - | - | 8.140 s |
| plain, bf16 KV | 131,072 | out of memory | - | - |
| **plain, fp8 KV** | **131,072** | **13,904 MiB** | - | **8.102 s** |
| plain, int8 KV | 131,072 | 13,952 MiB | - | 8.088 s |
| plain, fp8 KV | 262,144 | out of memory | - | - |

262,144 and higher contexts need more KV memory than two 16 GiB cards offer even at fp8.

## KV cache quantization

`--kv-dtype` supports `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`. All of them work on the
TP-2 route; the quantized variants needed one fix (below) because the quantized prompt-attention
kernels opt in to more than 48 KiB of dynamic shared memory.

The 131,072-token capacity is usable end to end: a single **118,869-token** prompt returned HTTP 200
in 96.8 s on the fp8 configuration (and a 19,869-token prompt in 19.5 s), against a 65,536-token
ceiling for bf16 KV on the same cards.

fp8 KV does not measurably change output quality. Four 1,400-token budget samples of the same essay
prompt produced content lengths 1365 / 859 / 1382 / 1602 (median ~1374) with no repeated-token runs,
against 1233 / 763 / 1469 / 1456 / 1430 / 1621 (median 1443) for the bf16 route at 65,536 tokens.

## Multi-token prediction (MTP)

`--spec mtp --draft-tokens K` (K in 1..5) drafts K tokens with the MTP layer and verifies them in a
single window forward, which is ~25% faster end-to-end and ~50% faster in sustained decode at K=2-K=3.

MTP is **experimental on this route**: it is not token-for-token identical to the plain route. The
verify window runs in `Phase::Prefill` while the decode path runs in `Phase::Verify`; the two use
different kernels, so their per-column logits differ (10-20% of columns flip argmax on this NVFP4
model), and the KV rows written for accepted tokens carry those differences permanently, so the two
routes drift apart after ~50-80 tokens. The decode-equivalent verify exists in the engine
(`TextPhase::Verify` supports `width > 1`), but the TP-2 shard path for it is incomplete. With the
sequence bindings, the per-device valid-column and backend-row tables, and the shard's decomposed
GDN recurrent path (sequential in both phases, so numerically phase-independent) all in place, the
launch still faults in RoPE (`rope.cu:189`, illegal address): the shard path for a multi-column
verify window needs its shape and workspace plumbing completed. Until that is done, MTP stays
experimental and the plain route is the recommended configuration when output consistency matters.

## Fixes applied on this branch

- `copy_row_block` and the ids plumbing: a `void*` byte-offset bug corrupted window ids in the MTP
  warmup (`cudaErrorIllegalAddress`).
- MTP rounds snapshot the pre-verify GDN state and replay the fold at the committed width, so the
  verify no longer double-advances the recurrent state.
- Quantized prompt attention (`prompt_fp8.cu`, `prompt_nvfp4_non_rdc.cu`, `prompt_k8v4.cu`) now
  perform the `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` opt-in **per device**; a
  function-local static configured only the first shard's device, so the second device rejected every
  >48 KiB launch with `cudaErrorInvalidValue` and no quantized KV dtype could start on TP-2.

## Work log

[TP-2 adaptation work log](tp2-dual-5060ti-worklog.md) is the chronological round-by-round record:
design decisions, failed attempts, and the evidence behind the measurements above. It is a
historical log, not an active reference; the current state and the open item are as described in
this note.
