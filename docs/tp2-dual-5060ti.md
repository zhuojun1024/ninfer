# Dual RTX 5060 Ti TP-2 tuning (Qwen3.8-27B NVFP4)

This note records the tensor-parallel-2 (TP-2) adaptation of this engine for two RTX 5060 Ti
(16 GiB) cards running `Qwen3.8-27B` NVFP4. Numbers were measured on this machine with
`ninfer-serve` on `--devices 0,1`, greedy sampling, one request at a time. The routines live in
`tools/tp_bootstrap/`.

## Recommended configuration

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 131072 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2
```

`--kv-dtype fp8 --max-context 131072` doubles the usable context (65,536 -> 131,072 tokens) at
essentially the same device memory and throughput as bf16 KV at 65,536 (see below).
`--spec mtp --draft-tokens 2` adds ~45% sustained decode (46-49 tok/s against 31.6 plain) and is
verified non-degenerate (see the MTP section); drop those two flags for the plain route.

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
TP-2 route, but the quantized variants needed two fixes (below): the quantized prompt-attention
kernels, and the quantized small-T decode kernels used by a multi-column (MTP) window, both opt in to
more than 48 KiB of dynamic shared memory, and in both cases the opt-in was a function-local
`static` that configured only the first shard's device.

The 131,072-token capacity is usable end to end: a single **118,869-token** prompt returned HTTP 200
in 96.8 s on the fp8 configuration (and a 19,869-token prompt in 19.5 s), against a 65,536-token
ceiling for bf16 KV on the same cards.

fp8 KV does not measurably change output quality. Four 1,400-token budget samples of the same essay
prompt produced content lengths 1365 / 859 / 1382 / 1602 (median ~1374) with no repeated-token runs,
against 1233 / 763 / 1469 / 1456 / 1430 / 1621 (median 1443) for the bf16 route at 65,536 tokens.

## Multi-token prediction (MTP)

`--spec mtp --draft-tokens K` (K in 1..5) drafts K tokens with the MTP layer and verifies them in a
single window forward, which is ~25% faster end-to-end and ~50% faster in sustained decode at K=2-K=3.

MTP is not token-for-token identical to the plain route, and no engine gives that. A verify window
is a different execution shape from a single-token decode -- here the chunked attention/convolution
route, in llama.cpp the chunked GDN kernel -- so its per-column logits differ, and the KV rows written
for accepted tokens carry those differences permanently. llama.cpp's Qwen3.5 MTP has the same property
and its documentation says exact output matching requires greedy sampling. The relevant criterion is
therefore that MTP is not *worse*, and it now qualifies: six samples of the same essay prompt at
temperature 0.7 / top-k 20 / top-p 0.80 produced content lengths 1519 / 1318 / 1294 / 1195 / 1683 / 0
(median 1306, no repeated-token runs) on the MTP route against 1165 / 1452 / 1546 / 672 / 224 / 1237
(median 1201) on the plain route; the single zero is a reasoning-budget artefact that the plain route
shows too. Sustained decode is 46-49 tok/s at K=2 against 31.6 tok/s plain, and the combination
with the recommended KV configuration is clean as well: three samples on fp8 KV at 131,072 tokens
produced content 1530 / 1481 / 1708 with no repeated-token runs.

This paragraph replaced a defect, not a limitation. A head-split shard was not recording its linear
attention transitions: the two record-producing branches in `gdn_mix` were gated on
`shard_config_ == nullptr`, because the fused width>1 record ops are registered for the full
16384-row fused parent only. The round nevertheless restored the pre-round state and ran
`gdn_replay_fold` over those record planes, which had never been written -- so every MTP round
replayed an empty transition log into the 48 linear-attention layers, which stopped advancing (and had
their convolution history rebuilt from the same empty log), while the 16 full-attention layers kept
working. That is exactly the observed signature: locally coherent output for a few tokens, then
collapse into `0` repetition. The shard now records on its decomposed route: the convolution record
is the raw pre-convolution window that route already materializes as `[q|k|v]`, and the recurrence
uses `ops::gated_delta_net_replay_record`, which is registered for the shard geometry (8 key heads,
24 value heads) and whose live outputs are defined to be bit-identical to the normalized
`gated_delta_net` it replaces, so window logits and acceptance are unchanged.

An earlier diagnosis of this item was wrong and is worth recording: the verify window's phase was
blamed for the drift. `Phase` has no semantic effect on a head-split shard -- `attn_mix` ignores its
phase argument entirely, and both `Phase::Verify` gates in `gdn_mix` are excluded for shards -- so
switching the window to `Phase::Verify` could not have changed a single number. The reported fault
site was likewise a red herring: `src/ops/launcher/rope.cu:189` is a `CUDA_CHECK(cudaGetLastError())`
checkpoint, and with two devices interleaved an asynchronous illegal address surfaces at whichever
checkpoint runs first, not necessarily at the faulting kernel.

## Fixes applied on this branch

- `copy_row_block` and the ids plumbing: a `void*` byte-offset bug corrupted window ids in the MTP
  warmup (`cudaErrorIllegalAddress`).
- MTP rounds snapshot the pre-verify GDN state and replay the fold at the committed width, so the
  verify no longer double-advances the recurrent state.
- Head-split shards now produce the replay records that fold consumes (convolution record from the
  decomposed route's pre-convolution window; recurrence via `gated_delta_net_replay_record` at the
  shard geometry). Without them the fold replayed an empty log every round and MTP degenerated.
- Quantized prompt attention (`prompt_fp8.cu`, `prompt_nvfp4_non_rdc.cu`, `prompt_k8v4.cu`) now
  perform the `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` opt-in **per device**; a
  function-local static configured only the first shard's device, so the second device rejected every
  >48 KiB launch with `cudaErrorInvalidValue` and no quantized KV dtype could start on TP-2.
- The same defect existed in the small-T decode kernels (`small_t_fp8.cu`, `small_t_k8v4.cu`,
  `small_t.cu`), where it stayed latent because a single-token decode uses a 32 KiB window that fits
  the 48 KiB default. The MTP window (`T = K+1`) switches to a 64 KiB tile that does not, so
  `--spec mtp` failed on every first quantized-KV request with `small_t_fp8.cu:56
  cudaErrorInvalidValue`. All four quantized/bf16 small-T routes now share one per-device opt-in
  authority (`src/ops/common/cuda_smem.h`).

## Work log

[TP-2 adaptation work log](tp2-dual-5060ti-worklog.md) is the chronological round-by-round record:
design decisions, failed attempts, and the evidence behind the measurements above. It is a
historical log, not an active reference; the current state and the open item are as described in
this note.
