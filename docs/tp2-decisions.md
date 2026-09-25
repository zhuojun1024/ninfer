# TP-2 (2× RTX 5060 Ti) settled decisions

Quick reference of settled conclusions from the 2× RTX 5060 Ti TP-2 campaign
(Qwen3.8-27B NVFP4). Check this file before re-investigating any topic: entries
record what was tried, what the evidence showed, and where the full record lives.

Historical reference only: current state and remaining work live in `PLAN.md`
(repository root); product behavior and measurements live in
[tp2-dual-5060ti.md](tp2-dual-5060ti.md). Worklog line numbers refer to
[tp2-dual-5060ti-worklog.md](tp2-dual-5060ti-worklog.md) as of 2026-09-25
(6,499 lines).

## Architecture and ownership

- **Fixed TP=2, dense 27B only.** No MoE split, no PP/DP; `--devices <name1>,<name2>`
  selects by GPU name, not index. (worklog §0–§2)
- **Load-time weight splitting; converter untouched.** Materialization reads the
  complete tensor, splits, and uploads per shard; reuses the official 23.7 GB
  artifact. (worklog §2)
- **DFlash2 and MTP are mutually exclusive backends.** Choosing DFlash2 frees
  shard 0's MTP weights (430 MiB) + KV (516 MiB). (worklog L3757–3760)
- **Whole-copy draft is infeasible.** At full context (262,144 / fp8 / MTP K=2)
  free is only 697/1,217 MiB vs a 2.07 GiB/card draft; after splitting ~1.04
  GiB/card. The proposal condition is a 5-block residual concatenation
  projection ⇒ cross-card handoff is required (residuals bit-identical across
  cards; only shard 0 captured). The candidate selector reads the full-vocab
  codebook directly (~254 MiB) and must stay whole. Do not re-investigate.
  (worklog L3757–3762; PLAN §3.4)
- **`--spec dflash` (v1) is rejected at construction time** — product decision;
  DFlash2 and MTP are the only supported draft backends. (worklog L3717)
- **The `--chat-template` whitelist is intentional.** The frontend is not a
  generic Jinja interpreter: each semantic has a hand-written C++ renderer, and
  only the two artifact template semantics are accepted (thinking-toggle
  `e84f32a2…`, reasoning-effort `c3cf9e34…`). A third template in a model dir
  is rejected because its rendering semantics (tool-call XML/JSON, effort
  aliases, …) are unimplemented; adding one is a separate feature. (worklog
  L2554–2565)
- **Vision: single-image cap 8,192 tokens**, static shard split (MTP→shard 0,
  vision→shard 1); images above the cap are rejected at request time with a
  pointer to the `[mem] vision` ledger. (worklog L2456)

## Rejected optimizations (measured)

- **Prefill AR∥MMA sub-block pipeline: ~5% slower, rolled back.** In-kernel AR
  spin time is already masked by the peer's compute; the sub-block split also
  re-streams weights per layer. If a truly compute-bound prefill appears later,
  reuse the 64-alignment constraint and event topology. (worklog L1425–1468)
- **Multi-block AR: no difference** (TTFT 859/877/879/879/830/861 ms for
  1/2/4/8/16 blocks) — not a single-SM issue. (worklog L653–655)
- **Copy-engine rendezvous (small payloads): slower (951 vs 857 ms) and shard
  logits disagree** — rendezvous/double-buffer semantics do not hold. (worklog
  L655–656)
- **Large-payload event path (Phase 3): rejected.** Copy-engine floor 3.32 ms vs
  sliced 2.70 ms; the event fence alone is +23%. (worklog L5059, L5114)
- **Copy-engine large-payload arm (design stage): not implemented.** The real
  lever is the 128 calls/round count, not the transfer mechanism; the prefill
  control arm was left untouched and proved it. (worklog L3503)
- **GQA-aware prompt kernel (L1): falsified at the measurement gate.** ncu on
  the production prompt kernel: DRAM 1.09% (KV traffic absorbed by L2), tensor
  pipeline 63.5% busiest, occupancy locked at 33% (100 regs + ~100 KB smem per
  thread). Warp doubling: −12% (register spilling). Bc 64→128: cannot launch.
  The 6× KV-request redundancy is not a DRAM cost. (worklog L5290, L5344–5365)
- **Reducing AR call count requires giving up weight splitting** ⇒ always a loss
  in decode. The only correct direction is overlap. (worklog L2757–2758)
- **Prefill allreduce is at the hardware floor.** Card 2 sits in a chipset
  Gen4 x4 slot (~7 GB/s) ⇒ 2.67k tok/s ceiling; the measured marginal slope
  (0.368 ms/token) sits exactly on it, independent of chunk width. The largest
  single lever is hardware: move card 2 to a CPU-direct Gen5 x8 slot. (worklog
  L642–659)
- **Overclock/power wall and P2P are infeasible** (already at full clock,
  93/85 W; the consumer driver masks P2P, `canAccessPeer=0`, SYS topology
  across root complexes). (worklog L2543, L5282)
- **Device-plane divergence anchor (+73.4 MiB/card): abandoned.** Saves only
  ~9.9 ms per session; in real chat the reachability guard usually holds and
  the device path is taken. (worklog L1976–1984)

## Withdrawn approaches (falsified by real traffic)

- **"Continuation boundary" (reuse to the end of the last generation):
  withdrawn.** DSH re-renders the assistant turn instead of echoing the
  generated token stream; the continuation slot was used 0/36 in real traffic.
  (worklog L1495–1510)
- **Template `|trim` → `rstrip('\n')`: rolled back.** Even with a maintained
  override template, verbatim replay of `reasoning_content` still stops at the
  previous prompt length. (worklog L5405–5406)
- **Text-protocol replay cannot guarantee token-exact reproduction.** The first
  generated token is not in the published reasoning text (`max_tokens=1` ⇒
  empty `reasoning_content`); with the DFlash2 cost gate (decline draft ≈ 3×
  decode vs saved prefill), recomputing is the rational choice. The only lever
  for token-exact reuse is a continuation protocol carrying token ids
  (unscheduled). (worklog L5408–5413, L5457)
- **"Unaligned reuse must decline the masked draft": premise overturned.** Same
  round: declining 22.5 vs keeping 87.1 tok/s with near-identical acceptance ⇒
  the gate, `draft_context_declined`, `reuse_grid` and session-mean pricing
  were removed; a single scan with the deepest boundary wins. (worklog
  L4251–4279 background; PLAN §2)
- **DFlash2 TP-2 option (b) (2026-09-22): implemented, then rolled back.** It
  was not the cause of the solo-probe instability; the route went back to
  construction-time rejection until the clamping-round root cause was fixed
  (B5). (worklog L3974–3978)

## Measurement and methodology rules

- **Acceptance-rate verdicts need a 30-rep pool.** The 15-rep inter-arm noise
  floor is ~±2 pp; two "−2 pp cost" verdicts from 15-rep pools were later shown
  to be noise. (worklog L6200–6206; PLAN §2)
- **Greedy cross-arm comparison is invalid.** Draft proposal length changes the
  verify batch shape ⇒ target numerics change ⇒ text drifts (6/7 probe prompts
  differ). Greedy probes are deterministic regression checks only; pooled
  sampling is the acceptance-rate statistic. (worklog L5788–5793)
- **`CUDA_LAUNCH_BLOCKING=1` / compute-sanitizer are incompatible with the
  TP-2 in-kernel AR** — the A-side launch becomes blocking, the B-side kernel
  waits for it ⇒ the first collective always gives up (self-deadlock).
  Kernel-level attribution requires source-level probe bisection. (worklog
  L6431–6433)
- **Boundary sensitivity:** the last chunk width is not the variable; the
  boundary position is; 120-step decode shows zero flips. (worklog L3587)
- **B7 (verify graph) settling needs ≥24 reps/arm.** Current A/B: 6 reps/arm,
  aggregate +5.2%, per-request t=0.99 (not significant). (PLAN §3.4)
- **The Q4 selector uses coarse T buckets, not per-T capacity buckets** —
  per-T instantiation is pure code bloat under the mask path. (worklog
  L5709–5712)
- **Draft-quantization overrides are cumulative by design** — each arm must be
  the cumulative artifact, otherwise it falls back to the official Q8 recipe
  and the test is not of the product path. (worklog L5719–5721)

## Root causes (settled)

- **MTP consistency (①): shards never wrote replay records** (fold consumed an
  empty log). Fixed; criteria were no-regression + same quality tier + speedup.
  (worklog L1255)
- **Spin hang (2026-09-23): token counter cross-card false sharing.** Fixed;
  `DevicePair::start_ar_watchdog` retained as a diagnostic. (worklog L4648)
- **Clamping rounds: `forward_tp2_window` never bound
  `active_valid_columns_` / `valid_columns`** ⇒ paged-KV same-slot tearing.
  Fixed. (worklog L4137, L4816)
- **DFlash2 22 tok/s: the decline guard misfired on the template prefix**
  (~37–57 tokens always hit the unaligned fallback ⇒ `extent=0` every round).
  Fixed with a 16×-budget cost gate. (worklog L4251–4279)
- **TP-2 decode double preview:** `preview_model` already handles termination;
  the decode loop was publishing a second preview. (worklog L395–396)
- **Early GDN test failures: test bug** (GDN state pool not reset between the
  card-local double forwards). (worklog L335)
- **Bounded-spin give-up inside a captured window:** the round keeps running
  with locally-only partial sums; the window's kernels consume the inconsistent
  data as addresses ⇒ illegal address; no host-side check can make it in time.
  Only eager-prefill give-ups become clean 503s. Mitigations in place:
  abandon-output (1b), watchdog (default on), timeout default 2000→10,000 ms
  (a second-level stall is a host/driver event, not a slow kernel). (worklog
  L6436–6441, L6472–6481; PLAN §3.6)
- **Two low-frequency failures (r67 IllegalAddress req#14; r69 HTTP 503
  req#14) are the same event class.** Natural rate ≲1/1,500 requests; long runs
  (1,500 requests, watchdog on) have not caught the trigger. (PLAN §3.3, §3.6)
- **BF16-add allreduce is bit-safe only for BF16 payloads.** I32/FP32 ids can
  carry signaling-NaN bit patterns; use `DevicePair::sendrecv` for byte-exact
  cross-card exchange. DFlash2's equivalent paths are fixed; the MTP route at
  `execution/text.cpp:486` is not (verification cost high). (PLAN §2, §3.3)
- **Official NVFP4 artifact's lower acceptance rate vs synthetic/self-
  converted:** the difference is in the text weights, not the components.
  (PLAN §2)
- **Solo-probe determinism: ~6% flip rate** refutes "byte-identical across
  runs" for the dflash2 solo probe. (worklog L4097)
- **The sessions plain-route failure is pre-existing** (the r57 baseline
  artifact reproduces it token-exactly; dflash2/mtp routes pass) ⇒ blocks the
  three-suite all-green; the sessions gate runs on the dflash2 route. (worklog
  L5811–5815; PLAN §3.3)

## Tool and environment boundaries

- **WSL2 CUDA is unusable for execution** (PTX JIT segfault in
  `cudaGetDeviceCount()`; GPU held by Windows services) ⇒ the WSL tree is
  compile-verification only; all tests run on the Windows side
  (`tools/win_port/test.ps1`). (PLAN §3.5)
- **Three GPUs visible:** nvidia-smi order 0/2 = 5060 Ti, 1 = Tesla T10;
  `--devices 0,1` picks the two 5060 Tis by CUDA ordinal (offset from
  nvidia-smi order). (PLAN §1)
- **Windows PATH must include the FFmpeg and libcurl `bin` dirs** or
  solo/sessions die at load with `0xC0000135` (append does not use the media
  path). (PLAN §1)
- **Windows libcurl link:** `find_library` picks the static `libcurl.a`
  (MSVC cannot link it); prefer `find_file(... libcurl.dll.a)`. (worklog
  L2680)
- **Process discipline:** one model process machine-wide (WSL 8088 vs Windows
  8099 mutually exclusive); stop the service before relinking the exe
  (LNK1104); `build-win/apps/ninfer-serve.exe` is not auto-synced to
  `C:\ninfer\`. (PLAN §1)
- **Watchdog cost model:** `note_ar_call` was already unconditional; the
  thread polls host counters every 25 ms, calls no CUDA API, touches no
  stream; large dumps are rate-limited to one per stall. A/B: no measurable
  cost. (worklog L6485–6492)
- **The frontend test fixture rejection is a pre-existing failure** (fixture
  template rejected by the chat-template whitelist; expected to turn green
  after the cherry-pick). (worklog L4321)

## Deferred (settled as "not now")

- **Full draft NVFP4: not advancing** (source-level audit;
  [tp2-dflash2-draft-nvfp4.md](tp2-dflash2-draft-nvfp4.md)). Only ~202.3 MiB
  left vs r66 q4all, of which 174.3 MiB is the non-GEMM codebook; NVFP4
  0.5625 B/elem > Q4 0.53125 B/elem, so re-quantizing already-Q4 tensors would
  grow them; three hard blockers (no float→NVFP4 quantizer in the converter and
  no A4 activation calibration for the dynamic-FP8 draft source; NVFP4 native
  Weight only takes full parents; consumer op NVFP4 geometries are closed
  sets). Reopen only with a W4A4-family / prefill-TMA motivation. (worklog
  L6053–6060)
- **Ring K/V quantization (CyclicKVCache): deferred.** Locked in three places
  (`CyclicKVCache` layout, SWA validation, `context_kv_materialize`
  `validate_cache`); ~55 MiB, worst cost/benefit. (worklog L5534)
- **L2 (AR∥MMA overlap, ring staging + event sync): unscheduled.** Shallow
  ceiling ~1.5×; large refactor with cross-card rendezvous deadlock risk
  (Round 35c evaluated). (worklog L5291)
- **Gate revision (sampling-led acceptance verdict): separate from the
  promotion.** The literal greedy gate `|Δacc| ≤ 1.0pp` failed at K=7
  (−1.60 pp, proven cross-arm text drift); the promotion is done (r69,
  user-confirmed); 30-rep pooled sampling is the de facto method. Formalize in
  one place if/when. (worklog L5826; PLAN §3.4)
- **Genuinely open items** (tracked in PLAN §3, not settled): bf16 KV + MTP
  first-prefill crash; MTP I32 sNaN risk; MTP3 ≥70 tok/s; perplexity on TP-2;
  trigger source of the low-frequency transport failures; B7 settling
  (≥24 reps/arm); sessions plain-route failure.
