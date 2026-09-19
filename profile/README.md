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

## Measured results (1x H100 80GB, torch 2.10.0+cu128, 2026-09-19)

Run on a single H100 (RunPod). Caveat: `world_size=1` sets `grad_accum_steps=8` and forces the
optimizer to `comms="none"`, so the training profile below runs 8 micro-batches on one GPU and
the optimizer sees *full* banks, not per-rank chunks. The Polar Express microbenchmark is the one
that reflects the 8-GPU config, because it is run on the per-rank chunk shapes directly.

### Polar Express is launch-latency-bound on the small banks

`bench_polar_express.py`, per-rank chunk shapes, 50 iters:

| bank | shape | ms/call | eff. TFLOP/s | kernels/call | GPU busy | gap |
|---|---|---|---|---|---|---|
| qk_bank | (8, 256, 768) | 0.253 | 37 | 18 | 107 us | ~150 us |
| vo_bank | (3, 768, 768) | 0.249 | 164 | 18 | 138 us | ~120 us |
| mlp_bank | (3, 3072, 768) | 0.399 | 306 | 23 | 341 us | ~59 us |
| total | | 0.901 | | | | |

qk_bank and vo_bank cost the same wall time despite qk doing ~4x fewer FLOPs: both run 18 kernels of
6-9 us separated by ~10-12 us gaps (median gap 12.2 us / 10.5 us under the profiler). mlp_bank's
kernels are 20 us with 1.2 us gaps, i.e. it is the only one that is actually compute-bound.

### CUDA graphs recover ~27% of PE, but only if captured by hand

`torch.compile(mode="reduce-overhead")` **refuses** to apply cudagraphs here -- *"skipping cudagraphs
due to mutated inputs (2 instances)"*, from the in-place Nesterov `lerp_` on `grad_chunk` /
`momentum_buffer` -- and ends up 9% slower (0.983 ms). It is also banned by speedrun rule 3
(no extra `torch.compile` flags).

Hand-captured `torch.cuda.CUDAGraph()` works, because in training the mutated buffers are exactly the
persistent ones (momentum is optimizer state; the grad chunk can live in the all-gather's destination).
`momentum_t` must move to the GPU so the schedule can be updated between replays without re-capturing.

| bank | baseline | graph | graph + d2d copy | speedup |
|---|---|---|---|---|
| qk_bank | 0.253 | 0.121 | 0.126 | 2.0x |
| vo_bank | 0.249 | 0.153 | 0.164 | 1.5x |
| mlp_bank | 0.399 | 0.355 | 0.371 | 1.08x |
| total | 0.901 | 0.630 | 0.662 | 1.36x |

~0.24 ms/step/rank, so ~0.3 s over the 1285-step run. Systems-only, so rule 2's statistical
validation requirement is waived.

### Training step structure (1 GPU, steps 20-26)

~235 ms/step, 224 ms GPU busy, 10 ms idle (4.3%), 6046 kernels/step, peak memory 39.4 GiB.

The model's compiled region is a **joint forward+backward graph**: the main-thread graph
(`f4rcuygz`, 8 calls/step, 195 ms/step GPU) is 20.6% backward-named kernels, while the compiled
backward on the autograd thread (`fu7h6lhi`) is only 14.4 ms/step. So the CPU-side `fwd`/`bwd`
markers do **not** separate forward from backward work -- read the kernel-name split instead
(76.7% forward/other, 23.3% backward).

`opt` is 4.47 ms = 2.0% of step GPU busy (unsharded, so this overstates the per-rank 8-GPU cost),
of which PE is 3.70 ms.

### Not measured

`ncu_polar_express.sh` fails on RunPod with `ERR_NVGPUCTRPERM`: GPU performance counters are
restricted by the host driver and cannot be enabled from inside the container. Needs a host with
`NVreg_RestrictProfilingToAdminUsers=0`.
