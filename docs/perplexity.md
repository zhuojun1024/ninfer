# Perplexity evaluation

`ninfer-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API.
Only Text weights and resources are loaded; Vision and speculative components are not required.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype fp8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `./build/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`.

An artifact that does not fit one card is scored with `--devices A,B`, which runs the reference
tensor-parallel route: both devices hold half the weights, so a model that needs more memory than
any single device has can still be evaluated. `--device N` and `--devices A,B` are mutually
exclusive, and a two-device run writes its reports under an extra `tp2-A-B` path component so its
numbers never sit next to the single-device ones for the same artifact and KV format. The scoring
normalization is the same either way: one request at a time, one prefill chunk of 1,024 tokens, no
CPU KV ring and no speculative decoding.

The two routes are not bit-identical. The tensor-parallel head splits the vocabulary across the pair
and sums two half-K partials for every logit, so the same artifact, corpus and KV format scored on
one device and on two differ by roughly 1.5e-3 in `mean_nll` (measured `bf16` KV, `--quick`,
`--context 4096`: 1.508191 on one device against 1.506711 on two, perplexity 4.518551 against
4.511865). That is the same order as a KV-format effect, so compare KV formats within one route and
treat a cross-route difference of that size as the route rather than the format.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_vision.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --devices 0,1 \
  --kv-dtype fp8
```

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Per-token NLL

`mean_nll` averages over the whole run, so it cannot show how the cost of a representation varies
with how much history each target actually saw. `--dump-token-nll` writes the unaggregated scores
next to the report as one `token_nll.<stream-id>.tsv` file per stream, with a `# target_index\tnll`
header followed by one row per scored token:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_vision.ninfer \
  --text long_document.txt \
  --context 262144 --stride 131072 \
  --devices 0,1 --kv-dtype fp8 \
  --dump-token-nll --output profiles/perplexity/depth-run
```

`target_index` is the index of the scored token in the stream, and because the token at index `t` is
predicted from `[b,t)` within its window, its history depth is `t - b`. With `--context` at least the
stream length there is a single window, so `b` is `0` and `target_index` is the history depth
directly. Comparing two runs that differ only in KV format then gives a paired `delta NLL(depth)`
per token, which cancels the text and leaves the representation's contribution.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format. Reports are grouped by artifact prefill signature and KV format; because `--context`
and `--stride` are not part of the path, read `execution.context_tokens` and
`execution.stride_tokens` from `report.json` before comparing two runs that share a directory level.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The schema-v2 report identifies the artifact's architecture, public name, actual weight formats
and prefill signature alongside the workload and numerical results.
