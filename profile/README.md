# Profiling the speedrun step

Tooling to answer, with numbers rather than guesses:

1. **What fraction of the step is the optimizer?** (and fwd / bwd / fp8 requant / NCCL)
2. **L2 hit rate and DRAM pressure on the Polar Express loop** (`ncu`)
3. **Launch gaps vs kernel time in the Polar Express region** — are we launch-bound?
4. **Tensor-core pipe utilisation inside the PE GEMMs** (`ncu`)
5. **The inter-step bubble**: from the last backward kernel to the first forward kernel of the next
   step, split into compute-busy / NCCL-busy / both / idle.

Items 1, 3, 5 come from a `torch.profiler` trace of the real 8-GPU run; 2 and 4 are hardware counters
and come from `ncu` on a single-GPU microbenchmark of the exact per-rank shapes.

## 1. Full-run trace (8×H100)

```bash
PROFILE_STEPS="20:26" ./run.sh                  # stage 1 (bsz 8×2048×8, seq 896)
PROFILE_STEPS="20:26,900:906" ./run.sh          # + stage 3 (bsz 24×2048×8, seq 2048)
```

* `PROFILE_STEPS` is a comma-separated list of `[start:end)` step windows. One extra warmup step is
  recorded-and-discarded before each window (CUPTI init), and the run **exits after the last window**,
  so the second command above takes ~1 min of training plus compile time.
* Every rank writes `profiles/<run_id>/rank<r>_steps<start>-<end>.json` (chrome trace, open in
  https://ui.perfetto.dev for the timeline).
* Avoid windows containing a validation step (multiples of `val_loss_every`, default 250) or a stage
  transition (~423, ~847 with 1270 scheduled steps) — those steps contain extra work / recompiles.
* Adam params update only on odd steps (`do_adam`), so consecutive steps differ; a 6-step window
  shows both kinds. Compare rows, not just the median.
* CUPTI adds overhead to every launch, so absolute times are inflated (esp. the many-small-kernel
  optimizer); use the trace for structure and ratios, then confirm any headline number with
  `torch.cuda.Event` timing.

Analyze:

```bash
python profile/analyze_trace.py profiles/<run_id>/                 # rank0 report + cross-rank table
python profile/analyze_trace.py profiles/<run_id>/rank0_steps20-26.json --pe-detail   # every PE kernel
```

Kernels are attributed to the `rf("...")` markers in `train_gpt.py` (`step` → `fwd`/`bwd`/`opt`/
`fp8_quant` → `opt/{scatter,wait,update,gather}/<label>` → `pe`/`normuon_vr`/`cautious_update`) via the
CUDA runtime correlation id of the launching call, so NCCL kernels land under `opt/scatter/*` and
`opt/gather/*` even though they execute later on the NCCL stream.

## 2. Polar Express microbenchmark + ncu (single H100)

```bash
python profile/bench_polar_express.py                        # ms per call per bank shape, CUDA-event timed
python profile/bench_polar_express.py --trace pe.json        # launch-gap summary without torchrun/CUPTI-in-NCCL noise
profile/ncu_polar_express.sh                                 # per-kernel L2 hit %, DRAM %, tensor-pipe %, occupancy
```

`bench_polar_express.py` extracts `polar_express` from `train_gpt.py` with `ast` (no copied code) and runs
it on the per-rank chunk shapes at `world_size=8`: `qk_bank (8,256,768)`, `vo_bank (3,768,768)`,
`mlp_bank (3,3072,768)`. `ncu` needs counter access (root or `NVreg_RestrictProfilingToAdminUsers=0`);
its timings are not representative (kernel replay), only its counters are.

## Reading the results

* `opt` share of `gpu_busy` in the phase table → item 1. Multiply by ~1390 steps for the total prize.
* PE table: `gap%` high (say >30%) with ~15 kernels per call → launch/serialisation-bound, a persistent
  or fused kernel wins on launch overhead alone. `gap%` low → the kernels themselves are the cost; look at
  the ncu table (tensor% low → shape/occupancy problem; l2hit% already high → a megakernel buys little
  from cache locality).
* Bubble table: `nccl` + `both` large vs `compute` → the tail is communication-bound and the lever is
  scheduling (split banks so early layers gather first), not kernels. `idle` large → CPU launch-bound
  or stream sync stalls; check the `cpu_wall` column for `opt` against its `gpu_sum`.
