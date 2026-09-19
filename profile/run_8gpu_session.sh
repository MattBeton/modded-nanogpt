#!/usr/bin/env bash
# One 8xH100 session. Goal, in priority order:
#   1. Is the optimizer tail on the critical path, or hidden under NCCL?  <- decides everything
#   2. A real world_size=8 step breakdown (GEMM / attention / comms) with sharded optimizer.
#   3. Does the swept tile config shrink the tail by ~50 us, and is that exposed?
#   4. ncu counters on the cascade kernels (blocked on RunPod so far: ERR_NVGPUCTRPERM).
# Wall-clock A/B is phase 5 and is NOT the evidence: the effect is ~0.16% of the run, below
# run-to-run noise. Read the traces, not the stopwatch.
#
#   profile/run_8gpu_session.sh all                 # everything, in order
#   profile/run_8gpu_session.sh setup arm-base      # individual phases (resumable)
#   DRY_RUN=1 profile/run_8gpu_session.sh all       # 1 GPU, validates plumbing, no 8-GPU work
#
# Phases, in order: setup verify bench arm-base arm-tuned ncu
# Env: TREE (checkout to run, default .), OUT (results dir), NGPU, WINDOW (PROFILE_STEPS),
#      TIMED_RUNS (per arm, default 3), SHARDS.
set -uo pipefail

TREE="${TREE:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-/workspace/session8}"
NGPU="${NGPU:-8}"
WINDOW="${WINDOW:-1100:1106}"
TIMED_RUNS="${TIMED_RUNS:-3}"
SHARDS="${SHARDS:-6}"
DRY_RUN="${DRY_RUN:-0}"
PROF="$TREE/profile"

# Inductor/Triton caches persist across processes, so only the FIRST run of each arm compiles.
# Pin them somewhere stable rather than a per-boot tmpdir.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

if [ "$DRY_RUN" = "1" ]; then NGPU=1; OUT="${OUT}_dry"; WINDOW="${WINDOW_DRY:-3:6}"; TIMED_RUNS=0; fi
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/session.log"; }
have_phase() { [[ " $PHASES " == *" $1 "* || " $PHASES " == *" all "* ]]; }
PHASES="${*:-all}"

log "TREE=$TREE OUT=$OUT NGPU=$NGPU WINDOW=$WINDOW DRY_RUN=$DRY_RUN phases='$PHASES'"

# ---------------------------------------------------------------- setup
if have_phase setup; then
  log "=== setup ==="
  n=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  log "visible GPUs: $n (want $NGPU)"
  [ "$n" -lt "$NGPU" ] && { log "FATAL: fewer GPUs than NGPU"; exit 1; }
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "$OUT/session.log"
  ( cd "$TREE" && python data/cached_fineweb10B.py "$SHARDS" ) >>"$OUT/session.log" 2>&1
  log "data: $(ls "$TREE"/data/fineweb10B/*.bin 2>/dev/null | wc -l) shards"
  python "$PROF/gram_tiles.py" --file "$TREE/triton_kernels.py" --check | tee -a "$OUT/session.log"
fi

# ---------------------------------------------------------------- correctness of the tile change
if have_phase verify; then
  log "=== verify: tile change must be arithmetic-neutral ==="
  python "$PROF/gram_tiles.py" --file "$TREE/triton_kernels.py" --revert >/dev/null 2>&1
  ( cd "$TREE" && python profile/verify_gram_tiles.py --src "$TREE/train_gpt.py" --out "$OUT/tiles_before.pt" ) 2>&1 | tee -a "$OUT/session.log"
  python "$PROF/gram_tiles.py" --file "$TREE/triton_kernels.py" | tee -a "$OUT/session.log"
  ( cd "$TREE" && python profile/verify_gram_tiles.py --src "$TREE/train_gpt.py" --out "$OUT/tiles_after.pt" ) 2>&1 | tee -a "$OUT/session.log"
  ( cd "$TREE" && python profile/verify_gram_tiles.py --compare "$OUT/tiles_before.pt" "$OUT/tiles_after.pt" ) 2>&1 | tee -a "$OUT/verify.txt" | tee -a "$OUT/session.log"
  python "$PROF/gram_tiles.py" --file "$TREE/triton_kernels.py" --revert >/dev/null
  grep -q "BITWISE IDENTICAL" "$OUT/verify.txt" || log "WARNING: tile change is NOT bitwise-neutral -- see $OUT/verify.txt"
fi

# ---------------------------------------------------------------- cascade microbenchmark, both arms
if have_phase bench; then
  log "=== bench: cascade microbenchmark, baseline vs tuned tiles ==="
  for arm in baseline tuned; do
    [ "$arm" = tuned ] && python "$PROF/gram_tiles.py" --file "$TREE/triton_kernels.py" >/dev/null \
                       || python "$PROF/gram_tiles.py" --file "$TREE/triton_kernels.py" --revert >/dev/null 2>&1
    ( cd "$TREE" && python profile/bench_polar_express.py --src "$TREE/train_gpt.py" --iters 50 ) \
      2>&1 | grep -viE "warn|check\(" | tee "$OUT/bench_$arm.txt"
    ( cd "$TREE" && python profile/bench_polar_express.py --src "$TREE/train_gpt.py" --iters 50 --cudagraph ) \
      2>&1 | grep -viE "warn|check\(" | tee "$OUT/bench_${arm}_graph.txt"
  done
  python "$PROF/gram_tiles.py" --file "$TREE/triton_kernels.py" --revert >/dev/null 2>&1
fi

# ---------------------------------------------------------------- the traces (THE deliverable)
run_trace() {  # $1 = arm name
  local arm=$1 dir="$OUT/trace_$arm"
  mkdir -p "$dir"
  log "--- trace [$arm]: nproc=$NGPU PROFILE_STEPS=$WINDOW ---"
  ( cd "$TREE" && PROFILE_STEPS="$WINDOW" PROFILE_DIR="$dir" \
      torchrun --standalone --nproc_per_node="$NGPU" train_gpt.py ) >"$OUT/train_$arm.log" 2>&1
  local rc=$?
  log "    exit=$rc  traces: $(find "$dir" -name '*.json' | wc -l)"
  [ $rc -ne 0 ] && tail -25 "$OUT/train_$arm.log" | tee -a "$OUT/session.log"
  ( cd "$TREE" && python profile/analyze_trace.py "$dir"/*/ ) >"$OUT/analysis_$arm.txt" 2>&1
  log "    analysis -> $OUT/analysis_$arm.txt"
}

timed_runs() {  # $1 = arm name; assumes tiles for that arm are already applied AND compiled
  [ "${TIMED_RUNS:-0}" -eq 0 ] && return 0
  for i in $(seq 1 "$TIMED_RUNS"); do
    ( cd "$TREE" && torchrun --standalone --nproc_per_node="$NGPU" train_gpt.py ) >"$OUT/timed_$1_$i.log" 2>&1
    local t v
    t=$(grep -oE "train_time:[0-9]+ms" "$OUT/timed_$1_$i.log" | tail -1 | grep -oE "[0-9]+")
    v=$(grep -oE "val_loss:[0-9.]+" "$OUT/timed_$1_$i.log" | tail -1)
    log "    $1 timed run $i: train_time=${t:-FAILED}ms ${v:-}"
    echo "$1 $i ${t:-NA} ${v:-NA}" >>"$OUT/timed_summary.txt"
  done
}

# Each arm is done contiguously: trace first (pays the compile), then the timed runs reuse the
# warm cache. Flipping tiles between phases would force a recompile every time.
if have_phase arm-base; then
  log "=== ARM: baseline (cold compile expected ~7-10 min, then warm) ==="
  python "$PROF/gram_tiles.py" --file "$TREE/triton_kernels.py" --revert >/dev/null 2>&1
  run_trace baseline
  timed_runs baseline
fi

if have_phase arm-tuned; then
  log "=== ARM: tuned tiles (only the 2 Triton kernels + cascade graph recompile) ==="
  python "$PROF/gram_tiles.py" --file "$TREE/triton_kernels.py" | tee -a "$OUT/session.log"
  run_trace tuned
  timed_runs tuned
  python "$PROF/gram_tiles.py" --file "$TREE/triton_kernels.py" --revert >/dev/null
fi

# ---------------------------------------------------------------- ncu (allowed to fail)
if have_phase ncu; then
  log "=== ncu: cascade kernel counters (may hit ERR_NVGPUCTRPERM) ==="
  ( cd "$TREE" && bash profile/ncu_polar_express.sh "$OUT/ncu_cascade" ) >"$OUT/ncu.log" 2>&1
  if grep -q "ERR_NVGPUCTRPERM" "$OUT/ncu.log" "$OUT/ncu_cascade.csv" 2>/dev/null; then
    log "    counters unavailable on this host (needs NVreg_RestrictProfilingToAdminUsers=0)"
  else
    log "    ncu OK -> $OUT/ncu_cascade.csv"; tail -20 "$OUT/ncu.log" | tee -a "$OUT/session.log"
  fi
fi

# ---------------------------------------------------------------- report
log "=== done. key files in $OUT ==="
for f in verify.txt analysis_baseline.txt analysis_tuned.txt timed_summary.txt; do
  [ -f "$OUT/$f" ] && log "    $f"
done
cat <<'NOTE' | tee -a "$OUT/session.log"

READ THE ANALYSIS IN THIS ORDER:
  1. analysis_baseline.txt, "per step" table: is `nccl` large and `gpu_idle` small?
  2. The "opt" row of the phase breakdown: gpu_sum vs cpu_wall. If cpu_wall >> gpu_sum the
     tail is waiting on collectives, i.e. the cascade is HIDDEN and none of the tile/fusion
     work is bankable. If they track each other, the tail is EXPOSED and it is.
  3. "inter-step bubble" table: compute vs nccl vs both vs idle.
  4. analysis_tuned.txt: the same "opt" row should be ~50us smaller per step. If the tail is
     hidden, expect step time NOT to move -- that is the decisive negative result.
NOTE
