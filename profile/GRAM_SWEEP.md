# ANVIL Gram tile experiment

This experiment changes launch configurations of the existing symmetric Triton
kernels, without editing the production optimizer or its coefficients.

## Reproduce

On an H100 with PyTorch and Triton installed:

```bash
python profile/sweep_gram_tiles.py --source /path/to/anvil --output /path/to/results
python profile/sweep_gram_tiles.py --source /path/to/anvil --output /path/to/results --phase cascade
```

Use a new output directory for a new sweep. Source files are copied into
`results/source` on the first invocation; later phases read those frozen files.
The manifest records source hashes and the GPU/software environment.

## Scope

- Per-rank shapes for the eight-GPU ANVIL configuration: QK `(6,256,768)`,
  VO `(2,768,768)`, and MLP `(3,2816,768)`.
- Both the Gram (`XXT` or `XTX`) and `b*A+c*A*A` kernels.
- Square output tiles of 32, 64, 128, 256; reduction tiles of 32, 64, 128;
  4 or 8 warps; 2, 3, or 4 pipeline stages: 72 configurations per operation,
  432 attempted configurations overall. Resource-limit failures are recorded.
- Square tiles preserve the alignment assumptions of the existing triangular
  skip/mirror logic. Rectangular tiling needs a separate correctness review.
- The original configuration is `(128,64,8,4)` in the order above.

## Measurement

Compilation, warmup, graph capture, input construction, and correctness checks
are outside timed regions. Standalone graphs contain 64 repeated launches;
CUDA events measure five replays per sample. These intentionally reuse buffers.
Every candidate is bracketed by baseline measurements. Finalists receive twelve
shuffled paired remeasurements with alternating A/B order.

Synthetic gradients and momentum are run through the source cascade to collect
its actual intermediate operands. Each candidate is compared to the source
kernel on early, middle, and final-iteration operands. The loose relative-L2
screen only excludes broken candidates; it is not a claim of training equivalence.
Bitwise agreement is tracked separately.

The cascade phase compiles the original function body, changing only its Gram
kernel wrappers. It compares baseline, fastest screened, and fastest sampled-bitwise
configurations under one-cascade-per-graph replay. It tests three synthetic
initialization/blend scenarios with eight matched sequential updates each.
Timing restores state before each sample, then permits state evolution during
replays equally for both variants. Capture and restoration costs are excluded.

## Limits

This is a single-GPU experiment. It measures local kernel and cascade latency,
not end-to-end eight-GPU training savings. It does not include NCCL overlap,
model forward/backward cache effects, or the equalization/weight-update tail.
Inputs are synthetic rather than captured training tensors. Sampled bitwise
agreement is strong evidence for these cases, not an exhaustive proof.

## Measured results — 2026-09-19

Host: one idle RunPod H100 80GB HBM3, driver 580.126.09, PyTorch 2.10.0+cu128,
CUDA runtime 12.8, Triton 3.6.0. This is the profiling environment, not a claim
that all software versions match the eventual production record environment.
The ANVIL files were frozen from `/workspace/pr360run`; their full SHA256 hashes
are recorded in the manifests. Remote experiment directory:
`/workspace/gram-tile-sweep`.

Of 432 attempted configurations, 372 ran successfully and 60 exceeded shared
memory limits. Every successful configuration matched the reference bitwise on
the three sampled operands. No numerical screening failures occurred.

### Standalone kernel finalists

Times are median graph-replayed microseconds, with twelve paired finalist rounds.
Configuration is `(output tile, reduction tile, warps, stages)`; output tiles are square.

| Operation | Original µs | Best µs | Speedup | Selected configuration |
|---|---:|---:|---:|---|
| QK Gram | 7.337 | 4.158 | 1.765× | `(64,128,4,3)` |
| QK polynomial | 5.973 | 3.495 | 1.709× | `(64,64,4,4)` |
| VO Gram | 7.421 | 6.557 | 1.132× | `(64,128,4,3)` |
| VO polynomial | 8.424 | 6.838 | 1.232× | `(64,128,4,3)` |
| MLP Gram | 16.535 | 16.516 | 1.001× | `(128,64,4,4)` |
| MLP polynomial | 8.705 | 7.917 | 1.100× | `(64,128,4,3)` |

The MLP Gram difference is noise: paired savings ranged from -0.030 to +0.024 µs.
There is no evidence to replace its production configuration based on this result.
For comparison, QK Gram paired savings were +3.154 to +3.306 µs.

### Full compiled cascade

The second integration run (24 alternating-order rounds per bank) measured:

| Bank | Original µs | Tuned µs | Reduction |
|---|---:|---:|---:|
| QK | 156.960 | 121.421 | 22.6% |
| VO | 185.533 | 170.282 | 8.2% |
| MLP | 479.882 | 476.803 | 0.6% |

The first integration run independently gave 155.978→120.547, 186.198→170.634,
and 480.794→477.533 µs. The main gain is consistently in QK and VO.

Across three initial-state/blend scenarios and eight sequential matched updates
per bank (72 cascade steps per candidate), all output and momentum-state
comparisons were bitwise identical to the baseline. These are synthetic inputs,
not training validation.

### Sequential three-bank graph replay

Replaying the three separate bank graphs consecutively, in QK→VO→MLP order,
gave **829.558→780.077 µs**, a **5.96% latency reduction**. Across 36 paired
rounds, median saving was 49.459 µs and individual paired savings ranged from
47.238 to 51.046 µs.

The separately captured `exact` variant selected the same configurations as
`tuned` because all winners passed the sampled bitwise checks. It measured
778.179 µs (6.19% reduction). This is a second capture/allocation of the same
choices, not a different optimization; the ~2 µs variation is small relative
to the ~50 µs gain. The sequential test retains separate bank graphs and does
not fuse the banks into one graph.

This is **about 6% of local cascade latency**, not 6% of training time. For an
illustrative 1,285 optimizer steps, 50 µs per step is about 64 ms before accounting
for overlap. The actual PR run's invocation count and exposed critical path must
be measured before converting this into a record improvement.

### Interpretation

- The tiny QK kernels have meaningful launch-configuration headroom even under
  graph replay. Changing the mathematical optimizer is unnecessary for this win.
- VO benefits modestly; MLP's Gram is already close to the best configuration
  in this search space. MLP should not inherit the QK configuration blindly.
- The results support integration testing of shape-specific configurations, but
  do not establish convergence or eight-GPU timing.
- No production kernel defaults or optimizer source files were modified.

Raw data and execution artifacts are in [results/gram_tiles_2026-09-19](results/gram_tiles_2026-09-19):
`sweep.jsonl`, `summary.json`, `cascade_first.json`, `cascade.json`, `sequential.json`,
software/source manifests, logs, and the exact remotely executed harness.

## Follow-up single-GPU experiments

The first sweep tied the two output tile dimensions together. A follow-up tested
rectangular `(BLOCK_SIZE_M, BLOCK_SIZE_N)` choices. The best standalone cases were:

| Bank | Operation | Configuration | Original µs | Rectangular µs |
|---|---|---|---:|---:|
| QK | Gram | `(64,32,64,4,3)` | 7.58 | 5.00 |
| QK | polynomial | `(64,32,64,4,3)` | 5.88 | 3.62 |
| VO | Gram | `(64,128,128,4,3)` | 7.40 | 5.72 |
| VO | polynomial | `(64,128,128,4,3)` | 8.39 | 6.33 |
| MLP | Gram | `(64,128,128,4,3)` | 16.60 | 14.74 |
| MLP | polynomial | `(64,128,128,4,3)` | 8.59 | 7.26 |

Those standalone results do not directly choose the production defaults: the
earlier square QK candidate was faster in the full cascade. Testing the selected
rectangular choices in the complete graph-replayed cascade gave QK 157.05→123.70 µs,
VO 188.04→162.86 µs, and MLP 481.10→463.17 µs. All eight matched multi-step
output/state comparisons were bitwise identical. The square tuned cascade remains
better for QK (about 121.4 µs), while rectangular tiles are better for VO and MLP;
a hybrid should be tested before choosing defaults.

The separate-bank graph overlap test used the square tuned candidates. Serial
QK→VO→MLP replay was 781.11 µs; replaying the three independent graphs on separate
CUDA streams was 691.57 µs, about 11.5% lower. This is a local compute-overlap
measurement only: it has no NCCL and therefore cannot establish whether collectives
would consume the same resources or hide this work in the eight-GPU run.

Finally, copying a fresh gradient into the graph's persistent gradient input before
replay added 5.7% for QK, 3.9% for VO, and 4.5% for MLP. Production ANVIL binds
the reduce-scatter destination directly as the captured input, so this is an upper
bound for a separate-input implementation rather than an expected production cost.

The rectangular first pass originally reported times 64× too high because its graph
contained 64 calls and the timing denominator omitted that factor; it was corrected
and rerun. Its JSON and log are retained so the correction is auditable.
