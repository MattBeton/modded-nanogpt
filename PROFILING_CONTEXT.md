# Polar Express profiling — context for sanity check

Working doc, 2026-09-19. Branch `profiling` on `github.com/MattBeton/modded-nanogpt`
(fork of `KellerJordan/modded-nanogpt`). Untracked file; not part of the branch.

**The question I want challenged:** is optimizing Polar Express (the orthogonalization inside
the NorMuon optimizer) a viable route to a speedrun record, or are we polishing something too
small to matter? I have a bias to declare: I did the measurement, I like the finding, and I
want someone to attack the arithmetic in §6.

---

## 1. What we set out to do

Profile the modded-nanogpt training step to find where time goes, with a prior interest in the
Polar Express (PE) step of the optimizer being slower than it should be. The original idea was
fused kernels for PE.

## 2. Setup and methodology

- **Hardware: 1x H100 80GB HBM3** (RunPod, driver 580.126.09). We did *not* have 8 GPUs.
- torch 2.10.0+cu128, CUDA 12.8, Python 3.12, custom Docker image.
- Two instruments, both added in this branch under `profile/`:
  - `bench_polar_express.py` — extracts `polar_express` from `train_gpt.py` via `ast` (so it
    can't drift from the real code) and times it on the **per-rank chunk shapes at
    `world_size=8`**. This is the single-GPU measurement that *does* reflect the 8-GPU config.
  - `analyze_trace.py` — consumes a `torch.profiler` trace from an instrumented training run
    (`PROFILE_STEPS="20:26"` env var), reports per-step GPU busy/idle, phase breakdown,
    optimizer breakdown, PE launch-gap table, inter-step bubble.

### Caveats that materially affect the conclusions

1. **`world_size=1` changes the optimizer.** It sets `grad_accum_steps=8` and forces
   `comms="none"`, so the *training profile* runs 8 micro-batches on one GPU and the optimizer
   sees **full banks, not per-rank chunks**. The PE microbenchmark is unaffected (it is run on
   chunk shapes directly); the training run's `opt` share is not representative.
2. **I profiled steps 20–26, which is the cheapest regime of the entire run.**
   `TRAINING_STAGES` ramps batch size 8→16→24 and seq len 896→2048 at 1/3 of training, plus a
   sliding-window schedule. PE's cost is **constant** (it depends only on weight shapes), while
   fwd/bwd grows several-fold. So PE's *share* at steady state is substantially smaller than
   what the step-20 profile shows. This cuts against the whole thesis and is the caveat I most
   want checked.
3. **No NCCL at all** in these traces, so nothing here says anything about comms overlap.
4. **`ncu` hardware counters are unavailable** (`ERR_NVGPUCTRPERM` — restricted by the host
   driver, not fixable inside the container). So I have *no* occupancy / tensor-pipe / L2 data.
   Every claim below about *why* a kernel is slow is inference from timing and shape, not
   from counters.

## 3. Measurement: PE is launch-latency-bound on the small banks

`bench_polar_express.py`, per-rank chunk shapes, 50 iters, no profiler attached:

| bank | shape | ms/call | GFLOP/call | eff. TFLOP/s | % of ~990 peak | kernels/call |
|---|---|---|---|---|---|---|
| qk_bank | (8, 256, 768) | 0.253 | 9.4 | 37 | 3.7% | 18 |
| vo_bank | (3, 768, 768) | 0.249 | 40.8 | 164 | 16.5% | 18 |
| mlp_bank | (3, 3072, 768) | 0.399 | 122.3 | 306 | 31.0% | 23 |
| **total** | | **0.901** | **172.5** | **191** | **19.1%** | |

From a `torch.profiler` trace of the same loop (profiler inflates gaps; kernel times are sound):

| bank | GPU busy | gap (unprofiled, derived) | median gap/kernel (profiled) |
|---|---|---|---|
| qk_bank | 107 us | ~150 us | 12.2 us |
| vo_bank | 138 us | ~120 us | 10.5 us |
| mlp_bank | 341 us | ~59 us | 1.2 us |

**The load-bearing observation:** qk_bank and vo_bank cost the same wall time despite qk doing
~4x fewer FLOPs. Both run 18 kernels of 6–9 us separated by ~10–12 us gaps. mlp_bank is the
only one actually compute-bound.

Per-kernel structure (5 PE iterations x 3 kernels + ~3 setup):
`XXT_kernel` / `XTX_kernel` → `ba_plus_cAA_kernel` → `baddbmm` (or split `bmm`+`add_` when
`shape[-2] > 1024`). The first two are already custom Triton kernels in `triton_kernels.py`.

**Kernel-busy-only, i.e. the floor if every launch gap vanished:**

| bank | busy-only ms | % of peak |
|---|---|---|
| qk_bank | 0.107 | 8.9% |
| vo_bank | 0.138 | 29.8% |
| mlp_bank | 0.341 | 36.2% |
| total | 0.586 | 29.4% |

So even with *zero* launch overhead, qk_bank still runs at ~9% of peak. That residual is what
a fused kernel would attack; CUDA graphs cannot touch it.

## 4. Measurement: CUDA graphs, and why the automatic route fails

- `torch.compile(mode="reduce-overhead")` **refuses to apply cudagraphs**:
  *"skipping cudagraphs due to mutated inputs (2 instances)"*, pointing at the in-place Nesterov
  `lerp_` on `grad_chunk` / `momentum_buffer` (train_gpt.py:211). Net result 0.983 ms — **9%
  slower** than baseline. It is also banned by speedrun rule 3 (no extra `torch.compile` flags).
- **Hand-captured** `torch.cuda.CUDAGraph()` works. In training the mutated buffers are exactly
  the persistent ones (momentum is optimizer state; the grad chunk can live in the all-gather's
  destination buffer). `momentum_t` must move to GPU so the schedule can update between replays.

| bank | baseline | graph | graph + d2d copy | speedup |
|---|---|---|---|---|
| qk_bank | 0.253 | 0.121 | 0.126 | 2.0x |
| vo_bank | 0.249 | 0.153 | 0.164 | 1.5x |
| mlp_bank | 0.399 | 0.355 | 0.371 | 1.08x |
| **total** | **0.901** | **0.630** | **0.662** | **1.36x** |

## 5. Measurement: the training step as a whole

Steps 20–26, 1 GPU, 8 micro-batches/step:

- **~235 ms/step, 224 ms GPU busy, 10 ms idle (4.3%)**, 6046 kernels/step, peak mem 39.4 GiB.
- The GPU is **96% busy**. There is no bubble to reclaim; wins must come from making kernels
  faster, not from filling gaps.
- Kernel categories per step: gemm 86.0 ms, inductor 63.9 ms, triton_custom 33.7 ms,
  attention 18.9 ms, aten_other 18.9 ms.
- Biggest single kernels: `linear_relu_square_kernel` 23.9 ms/step (10.7%);
  `nvjet_*_NTT` 18.4 ms; `nvjet_*_NNT` 17.9 ms; RMS-norm inductor kernels ~22.2 ms combined
  (~10%).
- `opt` is 4.47 ms = 2.0% of step GPU busy — **but unsharded**, so it overstates the per-rank
  8-GPU cost.

### Two bugs in my own instrumentation, found by running it on real data (both fixed)

1. `bench_polar_express.py` matched only `cuda_runtime` launches; torch 2.10 records
   triton/cublas launches as `cuda_driver` (`cuLaunchKernel`), so the gap table was empty.
2. `analyze_trace.py` attributed ~0 to `bwd` because the compiled backward launches from the
   autograd engine's thread, which carries no phase annotations — 10% of step GPU was landing
   in "other". Fixed by falling back to the main thread's enclosing phase by wall-clock.

### Finding about the codebase worth knowing

The model's compiled region is a **joint forward+backward graph**. The main-thread graph
(`f4rcuygz`, 8 calls/step, 195 ms/step GPU) is 20.6% backward-named kernels, while the compiled
backward on the autograd thread (`fu7h6lhi`) is only 14.4 ms/step. **CPU-side `fwd`/`bwd`
markers cannot separate forward from backward work in this repo.** Kernel-name split is
76.7% forward/other, 23.3% backward.

## 6. The arithmetic — THIS IS WHAT I WANT CHECKED

Run length for this config: **1285 steps** (from the run's own output).

Record scale (from PR research, see §7): the current record is **~74 s**, and open PR #360
claims **39.9 s**.

PE costs (per rank, per step, x1285 steps):

| scenario | ms/step | total over run | % of a 74 s record | % of a 40 s record |
|---|---|---|---|---|
| PE today | 0.901 | **1.16 s** | 1.6% | 2.9% |
| PE with hand-captured graphs | 0.662 | 0.85 s | — | — |
| → graph saving | 0.239 | **0.31 s** | 0.4% | 0.8% |
| PE kernel-busy floor (graphs, zero gaps) | 0.586 | 0.75 s | — | — |
| PE made *entirely free* (impossible) | 0 | **1.16 s saved** | 1.6% | 2.9% |

**The ceiling on this entire line of work is 1.16 s.** That is the whole of PE. A realistic
fused kernel that got PE to ~0.3 ms/step would save ~0.77 s vs today, or **~0.46 s on top of a
CUDA-graphed baseline**.

Additional lever found: **the three banks are independent but serialized.** `_normuon_update`
is called per-param in a plain Python loop (train_gpt.py:893), so qk → vo → mlp run back to
back. qk and vo occupy a small fraction of the SMs. Serial sum 0.901 ms; floor if perfectly
overlapped is `max` = 0.40 ms → **~0.5 ms/step ≈ 0.64 s**, needing only separate CUDA streams,
no new kernel. This has the best prize-to-effort ratio of anything found.

## 7. Rules research (separate agent swept all PRs/issues/discussions)

- **PR #80** (`mode="max-autotune-no-cudagraphs"`) — **rejected**. KellerJordan: *"sorry, extra
  torch.compile options are banned. they cause the run to take way too long to compile. it's
  supposed to be part of rule 3, but I see it isn't totally clear now. I'll make it more clear"*
  He then edited rule 3 to add "or `torch.compile`" one minute later. Rule 3's current wording
  exists because of this PR.
- **PR #102** (record #24) originally shipped `mode="reduce-overhead"` +
  `cudagraph_mark_step_begin()`, then silently reverted it before merge (commit "Upstream
  style"). No one ever commented on it.
- **PR #360** (open, @devenpzak, claims 39.9 s record) uses **hand-captured
  `torch.cuda.CUDAGraph()`** extensively — fwd/bwd runners and optimizer tail update — for
  **+2.99 s and +0.92 s** of its 34 s gain, with **valΔ +0.0**. Every `torch.compile` call
  stays flags-free. KellerJordan: *"Personally I would consider all of the changes described
  here to be 100% legitimate."* ClassicLarry (2026-09-18): *"Ok this one is up next. I'll try
  to merge it in with the latest main."*
- **Nobody has ever articulated the manual-capture vs `reduce-overhead` distinction in words.**
  It has only been settled implicitly by what got accepted.
- Relevant precedent: rule 2 waives statistical validation for pure systems wins; the
  discretionary list says *"A 200 line kernel to drop 300ms is considered worthwhile"*; custom
  Triton kernels are the most common category of recent records.

**Implication that hurts us:** PR #360 already captures the optimizer update in CUDA graphs. If
it merges, the 0.31 s launch-gap prize in §6 is **already taken**, and the only remaining PE
prize is the kernel-busy time (0.75 s) minus whatever a fused kernel can't recover.

### 7b. What PR #360 does to Polar Express itself (read from its source)

Fetched from `devenpzak/modded-nanogpt@anvil2-record`.

**ANVIL is not a separate algorithm — it is Polar Express.** The section is headed
`# Polar Express` (their train_gpt.py:210) and `AnvilAndAdam`'s docstring describes
*"A six-map Polar-Express cascade on ANVIL_MAPS"*, citing arxiv 2505.16932, the same paper.
Changes vs today's `polar_express`:

- **6 quintic maps instead of 5**, coefficients "re-derived rather than taken from the Polar
  Express reference" → the cascade gets ~20% *more* expensive, not less.
- **Twin-rail velocity** (fast rail beta 0.85, zero-init slow rail 0.98, blend 0.4385, engaging
  at step 514) replacing the single momentum buffer.
- **Cheaper normalization**: computes the Gram once up front, uses its trace
  (`d = sqrt(tr)*1.05`) to scale both X and A, then hoists that first Gram out of the loop via
  `if k > 0: XXT(X, out=A)`. **Saves one kernel per call** — an optimization we had not spotted.

**`fuse_tiny_kernels.py` does not touch the cascade.** Its docstring: tiny-kernel fusions for
"AnvilAndAdam's optimizer TAIL -- the step plus the fp8 requantize, everything after the main
compiled graph... Three launch-overhead removals, none of which changes any arithmetic." It is
a fused replicated-Adam elementwise kernel, not cascade work.

**The cascade inner loop remains completely unfused**, identical three-kernel structure
(`XXT`/`XTX` → `ba_plus_cAA` → `baddbmm`-or-split), still only `@torch.compile(dynamic=False,
fullgraph=True)`. They remove the launch gaps by capturing the whole bank-update body in a CUDA
graph ("one CUDA graph per ANVIL bank update body"), **not** by fusing.

**Net effect on the thesis:** the graph half is taken; the fusion half is not, and is now
*worth slightly more* (6 maps instead of 5). But any fused kernel must beat the graphed
baseline (0.662 ms/step), not the naive 0.901 ms/step.

## 7c. CORRECTION — the thesis is dead, and why (2026-09-19, after review)

Two errors in §6, both found in review. They kill the fused-kernel idea.

**Error 1: §6 assumes PE time is on the critical path. At world_size=8 it largely is not.**
`step_optimizers` (train_gpt.py:751-820) is deliberately two-phase:

    Phase 1: for label in scatter_order:  launch reduce_scatter (async)
    Phase 2: for label in work_order:     future.wait() -> _normuon_update (PE) -> _launch_gather

So PE math on one bank runs while NCCL for other banks is in flight. This is the documented
purpose of `scatter_order` / `work_order` ("Reductions are launched in scatter_order, while
update math and final gathers are executed in work_order").

Sizing the comms: Muon params = mlp 12x2x768x3072 (56.6M) + vo 24x768^2 (14.2M) + qk ~12.6M
= ~83M params = ~166 MB bf16. Ring reduce_scatter moves 7/8 of that per rank; at ~450 GB/s
effective that is ~0.3 ms, and the all_gather another ~0.3 ms. **NCCL on the Muon banks is
~0.65 ms against 0.901 ms of PE compute — the same order of magnitude.**

Consequence: the optimizer tail costs roughly `max(PE, NCCL)`. PE exceeds NCCL by only
~0.25 ms, so **the total bankable saving from all PE work is ~0.25 ms/step = ~0.32 s**, not
the 1.16 s in §6. Speeding PE below ~0.65 ms banks nothing — it just exposes NCCL.
§6's ceiling was ~4x too high. PR #360's optimizer graphs already claim most of the 0.32 s.

**Error 2: §6 invites adding the graph prize (0.31 s) to the stream prize (0.64 s).** They
attack the same idle time. Composed correctly: post-graph per bank is qk 0.121 / vo 0.153 /
mlp 0.355; overlapped you pay `max` = 0.355 ms, i.e. 0.546 ms/step = 0.70 s for both together
(and that is before the NCCL floor above, which cuts it further).

**Why fusion specifically dies:** once the banks are overlapped, mlp_bank *is* the cost — qk
and vo hide inside it. So a fused kernel only pays on mlp_bank, the one bank that was never
launch-bound (1.2 us gaps, 31% of peak). Taking mlp from 31% to an optimistic 60% is
0.341 -> ~0.18 ms, ~0.2 s on top of graphs+streams, for a cooperative-groups grid-synced
kernel — the hardest thing in the repo to write correctly.

*One caveat on that composition, which does not rescue it:* "31% of peak" is FLOP efficiency,
not SM occupancy. If mlp_bank occupies all SMs, qk/vo contend rather than hide free — which
makes the stream prize smaller while making fusion slightly less pointless. Unresolvable
without occupancy data (see caveat 4, ncu blocked). Second-order either way.

**Verdict: do not pursue the fused Polar Express cascade as a record attempt.**

**What survives:** the retargeting, and it survives *this same argument*. The overlap objection
bites in the optimizer tail because NCCL there is the same order as PE. In fwd/bwd, compute is
~27 ms/rank against ~1 ms of gradient comms — compute dominates ~25x, so kernel time lands on
the critical path. `linear_relu_square_kernel` (10.7% of step) and the RMS-norm inductor
kernels (~10%) are each 3-10x the original PE prize and ~30x the bankable one, with no comms
hiding them.

## 7d. MEASURED: PR #360's cascade does NOT close the gaps (2026-09-19)

§7c called the thesis dead partly on inference about #360's code. Measured instead, on the pod,
using their `train_gpt.py` + their `triton_kernels.py` (XXT_kernel / XTX_kernel /
ba_plus_cAA_kernel verified byte-identical to master's), on their own per-rank chunk shapes
(qk (6,256,768), vo (2,768,768), mlp (3,2816,768) -- all smaller than master's, since qk padded
64->48 and vo 24->16).

| | master polar_express | PR#360 anvil_cascade |
|---|---|---|
| maps / kernels per qk call | 5 / 18 | 6 / **25** |
| baseline total | 0.901 ms | **1.211 ms (+34%)** |
| qk eff. TFLOP/s | 37.1 | **25.3** (2.6% of peak) |
| vo eff. TFLOP/s | 164 | 98 |
| mlp eff. TFLOP/s | 306 | 250 |
| qk gap% (profiled) | 75.9% | **77.6%** |
| vo gap% | 68.0% | 71.1% |
| CUDA-graph total | 0.662 ms | **0.857 ms** |

**The PR makes the cascade more launch-bound, not less** -- one extra map and smaller banks
(less parallelism per kernel). It recovers this at a higher level with graph capture of the whole
bank-update body (29% here vs 27% on master), not by touching the cascade.

**After their graphs the launch overhead is spent.** Per-bank kernel-busy floor is
133/164/474 us = 0.771 ms against a graphed 0.857 ms, i.e. ~0.09 ms/step of residual overhead.
What remains is 0.77 ms of genuine kernel time at 2.6% / 10% / ~30% of peak -- the only thing a
fused kernel can still attack.

**Revised arithmetic on the new baseline.** Their Muon banks are ~141 MB bf16, so NCCL ~0.55 ms
against a graphed cascade of 0.857 ms: bankable ~0.3 ms/step ~= **0.39 s**, now against a
**39.9 s** record instead of 74 s -- so ~**1.0%**, versus 0.4% on master. The cascade got more
expensive while the record got shorter; both move the ratio the same way.

**Verdict update:** §7c's "dead" was right for master and too strong for the new baseline. The
structural objection is unchanged and still decisive: that 0.3 ms is bankable only if the cascade
sits on the critical path rather than hidden under NCCL, which still needs an 8-GPU trace.
Also note their docstring independently confirms the capture hazard we found: the 0-D scalars
"must be the persistent 0-D CUDA mirrors ... a weight tensor here is a blocking H2D inside the
graph (+1.4 ms/step)".

## 8. What I want the sanity-checker to attack

1. **Is the §6 arithmetic right?** Especially: is 1285 steps x 0.901 ms the correct way to
   total PE cost, given PE runs once per bank per rank per step?
2. **How much does caveat 2 (§2) hurt?** I profiled the cheapest regime. Does PE's constant cost
   against a growing step make this materially worse than stated, or is the total-seconds
   framing (which is schedule-invariant) the right one?
3. **Is a ~0.5 s win worth pursuing** when recent records move by ~0.5–3 s, given the work is a
   cooperative-groups fused kernel — one of the hardest things to write correctly?
4. **Is the bank-overlap idea (§6) real**, or will mlp_bank already saturate the SMs such that
   overlapping qk/vo buys nothing? I could not check occupancy (no ncu).
5. **Should we instead target the big items?** `linear_relu_square_kernel` is 10.7% of step and
   RMS-norm inductor kernels ~10% — each roughly 3-10x the entire PE prize. Is there a reason
   to think those are already optimal that I'm missing?
6. Is there a flaw in using effective TFLOP/s vs a ~990 TFLOP/s bf16 peak as the headroom
   measure, given these are fp32 inputs cast to bf16 with small batch dimensions?
7. Given §7b — the graph prize is taken but the cascade is still unfused — is "fuse the cascade,
   beat a CUDA-graphed baseline by ~0.4 s" a thesis worth months, or is it the sunk-cost version
   of a finding that was interesting but is now mostly claimed?
