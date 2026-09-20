# Dual RTX 5060 Ti TP-2 tuning (Qwen3.8-27B NVFP4)

This note records the tensor-parallel-2 (TP-2) adaptation of this engine for two RTX 5060 Ti
(16 GiB) cards running `Qwen3.8-27B` NVFP4. Numbers were measured on this machine with
`ninfer-serve` on `--devices 0,1`, greedy sampling, one request at a time. The routines live in
`tools/tp_bootstrap/`.

## Recommended configuration

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft
```

The full `max_position_embeddings` of the artifact (262,144 tokens) now fits: 15,614 MiB on shard 0
and 15,094 MiB on shard 1, i.e. 697 / 1,217 MiB still free per card (`nvidia-smi`). Splitting the
vocabulary- and hidden-parallel shard weights is what closed the gap; with the reduced state and
workspace pools alone, fp8 KV topped out near 229,376 tokens. `--kv-dtype fp8` costs 16.125 KiB per
token per shard. `--spec mtp --draft-tokens 2` adds ~90% sustained decode over the plain route and is
verified non-degenerate (see the MTP section); `--lm-head-draft` selects the artifact's reduced
proposal head for the draft side; drop those two flags for the plain route. Every one of those
choices is numerically transparent: the greedy output at 262,144 is byte-identical to the pre-split
build at 131,072 (see Verification).

## Benchmarks

| Configuration | KV capacity | Memory per GPU | Prefill | Decode |
|---|---|---|---|---|
| plain, bf16 KV | 65,536 | 13,888 MiB | 0.453 s (279 tok, 615 tok/s) | 8.153 s (31.4 tok/s) |
| MTP K=3, bf16 KV | 65,536 | 14,582 / 14,322 MiB | 0.526 s | 6.532 s (39.2 tok/s) |
| plain, fp8 KV | 65,536 | - | - | 8.091 s |
| plain, nvfp4 KV | 65,536 | - | - | 8.093 s |
| plain, int8 KV | 65,536 | - | - | 8.140 s |
| plain, bf16 KV | 131,072 | out of memory | - | - |
| **plain, fp8 KV** | **131,072** | **13,904 MiB** | - | **8.102 s** |
| plain, int8 KV | 131,072 | 13,952 MiB | - | 8.088 s |
| MTP K=2 fp8, `--lm-head-draft` | 131,072 | 14,940 / 14,678 MiB | - | 46-49 tok/s |
| plain, fp8 KV (pre-split build) | 262,144 | out of memory | - | - |
| **MTP K=2 fp8, `--lm-head-draft`, shard split** | **262,144** | **15,614 / 15,094 MiB** | **1,588 tok/s** | **59.0 tok/s** |

The 65,536 and 131,072 rows use the historical protocol (prefill: 279-token prompt, 8-token output;
decode: 256-token output reported end-to-end, so it understates sustained decode). The 262,144 rows
use `temperature 0` per request and the response's `usage` token counts, one request at a time:

- **Prefill**: a 16,057-token prompt produced its first token in 10.11 s (1,588 tok/s); a 65k-token
  prompt returned a complete answer in 48.1 s end to end (~1.35k tok/s of prompt), against 96.8 s for
  a 118,869-token prompt at the previous 131,072 ceiling.
- **Decode**: a 512-token answer took 8.68 s (59.0 tok/s over 223 MTP rounds, 2.3 committed tokens per
  round, 64.6% per-draft acceptance); a 64-token answer measured 63.6 tok/s. The previous ceiling's
  sustained decode was 46-49 tok/s, so doubling the context also moved throughput ~20% forward: the
  split halves each shard's proposal-head read and lets both shards draft in parallel.
- **Prefix reuse**: repeating an identical 16,057-token prompt returned in 0.30 s (0.33 s for the
  65k-token one) against 10.11 s / 48.1 s fresh, so one rewind snapshot behind the prefill end covers
  the chat-turn pattern (see Resident memory).
- **Concurrency**: two simultaneous streaming requests both completed (24 and 27 chunks, 1.99 s).

The startup after CUDA-graph capture is a one-off 14.4 s on this configuration; it is outside the
numbers above.

## Cross-engine comparison: llama.cpp `ar3-opt`

Its own tensor-parallel implementation — graph-level sharding, the allreduce transports, and why
neither `-sm row` nor NCCL applies on this host — is analysed source by source in
[llama.cpp TP notes](tp2-dual-5060ti-llamacpp-notes.md).

The reference engine on this machine is a tuned llama.cpp build (`C:\llama_ar3-opt`, source branch
`ar3-opt` at `C:\llama.cpp`), so both engines were measured back to back on the same two cards with
the same protocol and exactly one engine resident at a time: one non-streaming chat completion at a
time, `temperature 0`, 256 generated tokens, `--reasoning-effort medium`, MTP speculative decoding
on both sides, identical prompt text, and every number taken from the server's own `timings` block.
Prefill rows send a fresh nonce-prefixed filler prompt with `max_tokens 1`, so prefix caching cannot
fake them. ninfer runs `--devices 0,1 --max-context 131072 --kv-dtype fp8 --spec mtp
--draft-tokens 2 --lm-head-draft`; llama.cpp runs the owner's recipe via
`tools/win_port/serve_llama.ps1` (`-sm tensor -ts 1,1 -c 262144 --spec-type draft-mtp
--spec-draft-n-max 3 -fa on -ctk q8_0 -ctv q5_0`). Script `tools/win_port/bench_serve.ps1`,
raw results `build-win/bench-ninfer.json` and `build-win/bench-llama.json`.

| Workload | ninfer | llama.cpp | Ratio |
|---|---|---|---|
| prefill, 1.9k-token prompt | 1,401 tok/s | 764 tok/s | 1.83x |
| prefill, 7.2k-token prompt | 1,667 tok/s | 1,040 tok/s | 1.60x |
| prefill, 28.5k-token prompt | 1,552 tok/s | 1,016 tok/s | 1.53x |
| decode, 256 greedy tokens, 3 reps | 56.7 / 56.7 / 56.7 tok/s | 51.7 / 52.6 / 52.7 tok/s | 1.08x |

Round breakdown for the identical decode prompt:

| | ninfer K=2 | ninfer K=3 | llama.cpp n_max=3 |
|---|---|---|---|
| round time | 38.2 ms | 40.7 ms | 49.4 ms |
| committed tokens per round | 2.18 | 2.27 | 2.60 |
| per-draft acceptance | 58.8% | 42.4% | 53.8% |

**Decode does not separate because batch-1 decode is weight streaming, not compute.** Each engine
reads every weight once per forward, so `tok/s = committed tokens per round / round time`, and the
floor of the round time is `weight bytes per shard / 448 GB/s`. ninfer streams 10.15 GB per shard per
forward; llama.cpp streams 8.56 GB (`llama-gguf` reports 15.94 GiB for its GGUF: 10.38 GiB NVFP4 MLP,
3.09 GiB q5_K attention/GDN, 1.55 GiB q6_K embedding, 1.29 GiB q8_0 head). The two engines land at
38.2 ms and 49.4 ms per round, 59% and 39% of their respective floors, once the draft chain and the
host-side steps are counted. Two 5060 Ti together have 896 GB/s, half of one RTX 5090's 1,792 GB/s,
which is why this artifact decodes at 71.2 tok/s on a 5090 and at ~57 tok/s here: TP-2 is a capacity
decision (the 27B artifact does not fit on a single 16 GiB card), not a speed one.

Engine work is where ninfer is ahead - a 23% shorter round than llama.cpp and 1.5-1.8x the prefill
throughput - and four things consume that on decode:

- **Draft depth.** llama.cpp commits 2.60 tokens per round against 2.18 here. Raising
  `--draft-tokens` to 3 does not pay on this artifact: the round grows to 40.7 ms while acceptance
  falls to 42.4%, so decode drops to 55.7 tok/s. Each extra draft is host-serialized
  (`mtp_propose_window` synchronizes per draft) and adds only ~0.1 committed tokens.
- **Weight bytes.** Roughly a quarter of the per-token traffic is stored at one byte per element:
  attention/GDN projections, embedding, output head and the last eight layers' FFN are FP8 where the
  reference GGUF spends 0.56-0.69 B/element. Repacking them as NVFP4 would remove ~2.6 GB per shard
  per token, at a quality cost.
- **TP-2 structural losses.** This route runs with `use_cuda_graph = false`
  (`src/runtime/engine/model_instance.cpp:99`), so a verify forward launches ~1,009 kernels and pays
  ~2.9 ms of launch gaps; splitting heads doubles the collectives to 128 allreduces per token
  (~2.6 MB round trip, 9-18 us each); the fused `linear_swiglu`/`attn_input_proj` kernels are
  registered for full-model rows only and are bypassed on shard-local rows. llama.cpp captures the
  whole round, draft chain included, in CUDA graphs (95-96 replays per 256-token answer).
- **Draft chain on the host.** Two drafts cost 7-10 ms of the 38.2 ms round on top of the ~30 ms
  verify forward, so the MTP chain, not the kernels, is the largest single overhead.

Caveat: llama.cpp ran with the owner's exact command line and therefore its default batch/ubatch; a
larger `-ub` was not tried and could move its prefill numbers.

## KV cache quantization

`--kv-dtype` supports `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`. All of them work on the
TP-2 route, but the quantized variants needed two fixes (below): the quantized prompt-attention
kernels, and the quantized small-T decode kernels used by a multi-column (MTP) window, both opt in to
more than 48 KiB of dynamic shared memory, and in both cases the opt-in was a function-local
`static` that configured only the first shard's device.

The capacity is usable end to end: a **65k-token** prompt returned HTTP 200 after 48.1 s on the
shipped 262,144-token fp8 configuration, and a **118,869-token** prompt took 96.8 s at the earlier
131,072-token ceiling, against a 65,536-token ceiling for bf16 KV on the same cards.

fp8 KV does not measurably change output quality. Four 1,400-token budget samples of the same essay
prompt produced content lengths 1365 / 859 / 1382 / 1602 (median ~1374) with no repeated-token runs,
against 1233 / 763 / 1469 / 1456 / 1430 / 1621 (median 1443) for the bf16 route at 65,536 tokens.
The fp8 KV route is also *exactly* reproducible across builds: every golden prompt below matches byte
for byte at 131,072 and at 262,144, across the shard splits and the reduced arena sizes.

The MTP layer's own cache follows `--kv-dtype` as well. That cache is draft-only -- every draft is
verified by the target -- so quantizing it is a tempting 228 MiB at 262,144 tokens, and it was measured
rather than assumed. With the MTP cache at nvfp4 and the target KV still fp8, the same cumulative
acceptance counter read 392/510 against 391/510 (77.2% against 76.9%) over the golden workload, but
three of the five golden prompts produced different greedy text. The mechanism is not the draft cache's
precision leaking into an emitted token: the verify-and-fold path is a different execution shape from
single-token decode, so a changed draft *pattern* shifts the trajectory on which near-ties are resolved.
The shipped configuration therefore keeps the MTP cache at the target dtype and pays the 228 MiB, which
is the price of the byte-identical guarantee below.

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


### Verify CUDA graph

The MTP verify window is captured as one CUDA graph per device (src/core/decode_graph.h, capture_group
begins capture on both device streams and records each stream into its own definition), so a round's
~1,000 kernels are replayed from ~10 host calls instead of being launched one by one. The window's host
inputs (token ids, positions) are written into a portable pinned buffer that the graph's leading memcpy
nodes re-read on every replay, and the attention envelope is baked per capture: the eight buckets come
from mtp_graph_profiles(capacity, draft_tokens), so a round picks the bucket whose visible range covers
it. The captured body allocates from the round's arena watermark, which DeviceArena::rewind restores
before each round; a bucket captured at a lower watermark (the startup warm-up request) is re-captured
at the first real request's.

NINFER_TP2_VERIFY_GRAPH=0 runs the same window eagerly. Both routes call the same forward_tp2_window
with the same bucket envelope, so the switch changes only how the kernels reach the GPU:

| Metric (bench_serve.ps1, fp8 KV, context 131072, K=2) | eager | graph | delta |
|---|---|---|---|
| round time (NINFER_TP2_TIMING=1) | 39.6 ms | 35.3 ms | -4.3 ms |
| of which verify | 35.0 ms | 30.6 ms | -4.4 ms |
| 2048-token-context decode (64 tokens both) | 65.6 tok/s | 76.5 tok/s | +16.6% |
| prefill, 512-token prompt | 1419 tok/s | 1419 tok/s | flat |

Decode throughput also depends on the draft acceptance rate, which the round's own text decides, so the
round time and the equal-output 2048-token run are the trustworthy comparisons. The whole 4.4 ms is
launch and scheduling overhead: the verify's kernels, parameters and operands are identical in both
routes, and five prompts x up to 256 greedy tokens are byte-identical between them.
The graph's replay is also insensitive to added instrumentation, which is what made the round-boundary
synchronization defect below visible in the first place.

## Proposal head

`--lm-head-draft` switches the draft head from the weight-tied full output head (248,320 rows,
Q8_G32_FP16) to the artifact's indexed proposal head (131,072-row frequency shortlist, Q4_G64_FP16,
plus a row-to-token-id map). Target verification still uses the full head, so this changes only what
the draft proposes, never which token is emitted. A/B at the recommended fp8/131,072 MTP K=2
configuration, two six-sample runs, one request at a time:

| Workload (1,400-token budget) | Route | Decode tok/s, median (range) | Draft acceptance | Content median |
|---|---|---|---|---|
| Chinese essay | full | 45.18 (43.20-52.25) | 54.3% (2780/5118) | 1294 |
| Chinese essay | `--lm-head-draft` | **50.10** (47.53-55.35) | 53.5% (2749/5136) | 1336 |
| Python coding | full | 48.59 (45.62-51.07) | 58.6% (4532/7730) | budget-limited |
| Python coding | `--lm-head-draft` | **49.58** (48.85-54.87) | 53.1% (4326/8142) | budget-limited |

Acceptance is derived from the `NINFER_TP2_TIMING=1` decode line
(`[tp2-time] decode rounds=R committed=C`): accepted drafts are C - R out of R*K, where K is
`--draft-tokens`. That is the only reliable source on this route: the response's `timings.draft_n`
stays zero. Both routes were
non-degenerate on the essay prompt (no repeated-token runs); the coding prompt exhausted the token
budget in reasoning on every sample, so its content is not comparable.

The shortlist and the indexed head's Q4 quantization both move its argmax away from the full head's,
so acceptance falls -- 0.8 points on prose and 5.5 on code. The cheaper head read still wins on both
workloads, but the margin shrinks as acceptance drops, so this is a throughput trade rather than a
free win.

The head is 131,072 x 5,120 Q4_G64_FP16 (341 MiB) and only shard 0 ever samples from it, so the pair
now holds one vocabulary half each (65,536 rows, 171 MiB) and reassembles the draft logits before the
argmax. That needed a Q4 route for the half shape: the Q4 A16 dispatch is an enumerated (n, k) table
and no admitted shape had an admitted half, so `select_q4_n65536_k5120` was added next to the
131,072-row entry. The merge is exact -- disjoint row blocks, one in-place all-reduce -- and
`ninfer_linear_tp2_split_grouped_head_test` runs both shard halves against the full-weight Op at
exactly this shape (T=1 and T=2) and compares them bit for bit.

## Device memory budget

Measured by starting the engine under controlled configurations and reading
`nvidia-smi --query-gpu=memory.used` after load with no requests. CUDA devices 0 and 1 are nvidia-smi
indices 0 and 2; nvidia-smi index 1 is an unrelated idle Tesla T10. At the shipped configuration
(262,144 tokens, fp8 KV, MTP K=2, `--lm-head-draft`) the per-card totals are 15,614 MiB (shard 0,
which also owns the MTP layer) and 15,094 MiB (shard 1): 697 and 1,217 MiB still free of 16,311 MiB.

The engine prints one ledger line per shard at startup -- the blocks it allocates, in MiB, at the
requested ceiling:

```
[mem] shard 0 capacity 262144 | weights+ctx 11494.6 | kv 4644.2 | state 293.6 | record 2.6 | round 0.0 | workspace 192.0 | free 0.0 of 16310.6 MiB
[mem] shard 1 capacity 262144 | weights+ctx 11494.6 | kv 4128.0 | state 293.6 | record 2.6 | round 0.0 | workspace 192.0 | free 196.0 of 16310.6 MiB
```

| Component | shard 0 | shard 1 |
|---|---:|---:|
| Weights + CUDA context (ledger `weights+ctx`) | 11,494.6 | 11,494.6 |
| Text KV cache, fp8, 262,144 tokens | 4,128.0 | 4,128.0 |
| MTP layer KV cache, fp8, 262,144 tokens | 516.2 | 0.0 |
| Linear-attention state arena (2 live + 1 round scratch + 2 reuse snapshots) | 293.6 | 293.6 |
| Program workspace arena | 192.0 | 192.0 |
| MTP replay records and round anchor | 2.6 | 2.6 |
| **Sum of the engine's blocks** | **16,627** | **16,111** |

The per-token costs are the ones to plan with: text KV is 16 full-attention layers x 2 local KV heads
(of four, head-parallel) x head_dim 256 x K+V x 1 byte = 16.125 KiB per token per shard, 4,128 MiB at
262,144 tokens; MTP KV is 2 KiB per token on shard 0 only, because that layer is replicated rather
than head-split. The ledger's `free` column is `cudaMemGetInfo`, which this WSL2 driver under-reports
(it reads 0.0 while nvidia-smi shows 697 MiB free and the engine starts and serves reliably); the sum
of the rows therefore exceeds the reported device total of 16,310.6 MiB by ~1 GiB. Treat the row
sizes and the nvidia-smi totals as the budget, not the `free` column.

Three things had to give to reach 262,144 tokens, all of them memory-only and all of them verified
numerically transparent:

- **Vocabulary and hidden parallelism of the shard weights.** `text/output_head` (248,320 x 5,120,
  FP8 row-scale), the reduced `proposal/head` (131,072 x 5,120, Q4_G64_FP16) and
  `text/token_embedding` (248,320 x 5,120, FP8 row-scale) used to be replicated on both shards,
  4.8 GiB of duplicated tables on the pair. The two heads are now split by vocabulary row and the
  embedding by hidden column, so each shard holds half of each table and the pair assembles the
  result before sampling or drafting: ~1.2 GiB per card for the head and embedding, 170 MiB for the
  proposal head.
- **Prefix-reuse snapshots 3 -> 2.** The state arena held five 73.4 MiB planes: two live buffers, the
  MTP round scratch, and three prefix-reuse snapshots at rewind depths 0 / near / far. The chat-turn
  pattern the snapshots exist for needs the near rewind only, and the measurement above confirms it
  (identical 16k and 65k prompts re-enter in 0.30-0.33 s).
- **Workspace 384 -> 192 MiB.** The arena's per-layer prefill peak is ~140 MiB at the maximum prefill
  chunk of 1,024 tokens, so 192 MiB still leaves headroom; a larger request is a reported arena
  overflow, not corruption.

One block stays deliberately replicated: the MTP layer's own weights (430 MiB), because the draft path
runs on shard 0 alone and handing it a collective per draft step would cost more than the memory. The
KV cache stays fp8: nvfp4 or k8v4 KV (`--kv-dtype`) would free further memory at a value-precision
cost, and int8 is 48 MiB *larger* than fp8 at the same ceiling because of its 64-element scale groups.
The context cache is disabled on this route -- the core owns the prefix-reuse snapshots instead.

## Verification

| Check | Command | Result |
|---|---|---|
| Greedy identity of the whole stack | `r52_ab.sh final` then `r52_cmp3.sh` | 5/5 prompts byte-identical to the pre-split build at 131,072 (`ab-embed`, itself identical to the pre-optimization `ab-control`) |
| Route loading | `ninfer_qwen3_5_tp2_load_test` (plain, `--spec mtp`, `--spec mtp --lm-head-draft`) | expected shard shapes: head 124,160 / 248,320 rows, embedding `[248320,2560]`, proposal head `[65536,5120]` |
| TP-2 execution | `ninfer_qwen3_5_tp2_forward_test` | 11 probes shard-consistent, 32-step decode, prefill chunk-split invariant |
| Vocabulary-parallel head | `ninfer_linear_tp2_split_fp8_head_test` | exact at `[248320,5120]` and `[16384,5120]` |
| Grouped proposal head | `ninfer_linear_tp2_split_grouped_head_test` | exact at `[131072,5120]`, T=1 and T=2 |
| Half-width FP8 embedding | `ninfer_embedding_test` | full sweep at `[248320,2560]` |
| NVFP4 split Linear | `ninfer_linear_tp2_split_nvfp4_test` | passes |
| Collective staging | `ninfer_tp_device_pair_test` | passes |

"Exact" is bit for bit: the split Op output equals the same Op run with the full weight on the same
device, which is the property the merge relies on. `temperature 0, top_k 1` is what makes the greedy
comparison meaningful; the served configuration samples (`0.7 / 20 / 0.80`).

## Fixes applied on this branch

- Vocabulary-parallel and hidden-parallel shard weights: `text/output_head` and the reduced
  `proposal/head` split by vocabulary row, `text/token_embedding` by hidden column, with the pair
  merging the reduced block before sampling or drafting. That required shard-local `ColumnParallel`
  view offsets in `tp_shard_views.cpp`, FP8 slice descriptors that follow the new row count
  (`scale_ne[0]`, `scale_nb[1..3]`, `group`/`group_size` in `weight_splitter.cpp`), an offset
  correction for the peer half during the merge, a 16-byte-padded ids broadcast (the collective
  requires a multiple of 16 bytes), and a grouped-format (Q4/Q5/Q6/Q8) row slice plus `shard_geometry`
  knowledge of the `RowSplit` planes.
- A Q4 A16 route for the half proposal head (`select_q4_n65536_k5120`), because the Q4 dispatch is an
  enumerated shape table and no admitted shape had an admitted half.
- A `[248320, 2560]` FP8 embedding-gather domain: the gather kernel was templated on the hidden width
  and the wrapper now queries `embed_gather_fp8_supports_width` instead of hard-coding 5,120.
- The startup `[mem]` ledger line per shard, next to the existing `capacity | KV` line.



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
- The in-kernel allreduce's arrival tokens (the monotonic per-call counter that keeps a captured
  sequence's handshake honest) now live in **mapped pinned host memory** instead of on a device heap.
  The caller chooses which shard drives a pair, so an allreduce's stream is not necessarily device A's
  -- the TP-2 forward test drives the same pair from either shard -- and a device allocation was then
  dereferenced from the other device's context (`cudaErrorIllegalAddress` inside the first layer).
  The kernel reads the token once per block and publishes it through shared memory, because a
  per-thread read of a system-scope location costs ~4 us per allreduce, ~0.5 ms per round.
- TP-2 decode rounds now synchronize **both** devices at the round boundary, not just shard A. The
  host folds the verified columns, restores the pre-round GDN state and rebuilds the next round's
  window right after that sync, while shard B's verify tail could still be running; the pair's
  in-kernel allreduce is the only thing that orders the two devices' kernels against each other, so
  the peer's tail was unordered against the host's next round. It was latent in the eager path (where
  ~1,000 launches per round kept the host behind the GPU) and only became deterministic once the
  verify turned into a graph replay. The added sync costs 0.01 ms because shard B trails by less than
  one kernel, and eager and graph now agree byte for byte on every prompt.
- Every token selection on the TP-2 route (`ops::sample` for the prefill token and for plain decode, the
  MTP target `ops::argmax`, and `speculative_accept_greedy_drafts`) is bounded by the public tokenizer
  vocabulary instead of the packed embedding row count. A padding row could be selected, corrected or
  licensed, and the resulting id then failed tokenizer decoding and killed the whole request; the
  single-GPU route and the draft selector already passed `public_token_count`.
- Presence and frequency penalties reach the sampler on the TP-2 route. `make_sampling_config` left
  `token_counts` null and the penalty only reads that array, so across rounds both penalties were inert
  and only the MTP verify window's round-local overlay still applied. The request now owns an
  `I32[public_token_count]` count array, reset with the request, that `ops::sample` and the speculative
  accept kernel increment as they produce tokens.

## Work log

[TP-2 adaptation work log](tp2-dual-5060ti-worklog.md) is the chronological round-by-round record:
design decisions, failed attempts, and the evidence behind the measurements above. It is a
historical log, not an active reference; the current state and the open item are as described in
this note.
