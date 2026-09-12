#!/usr/bin/env bash
# Hardware counters for every kernel in one Polar Express call per bank shape (single GPU, no torchrun).
# Answers: L2 hit rate / DRAM pressure on the PE loop, tensor-pipe utilisation inside the GEMMs, occupancy.
#
#   profile/ncu_polar_express.sh [out_prefix]
#
# Needs Nsight Compute (`ncu`) on PATH and counter access (root, or NVreg_RestrictProfilingToAdminUsers=0).
# ncu serialises and replays kernels, so its timings are NOT representative — use bench_polar_express.py for time.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-profiles/ncu_polar_express}"
mkdir -p "$(dirname "$OUT")"

METRICS=(
  gpu__time_duration.sum
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active   # tensor core pipe busy (bf16 mma/wgmma)
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  lts__throughput.avg.pct_of_peak_sustained_elapsed                         # L2 bandwidth
  lts__t_sector_hit_rate.pct                                                # L2 hit rate
  sm__warps_active.avg.pct_of_peak_sustained_active                         # achieved occupancy
  launch__grid_size
  launch__block_size
)
METRIC_LIST=$(IFS=,; echo "${METRICS[*]}")

ncu --profile-from-start off \
    --target-processes all \
    --metrics "$METRIC_LIST" \
    --csv --page raw \
    python profile/bench_polar_express.py --ncu > "$OUT.csv"
echo "raw csv: $OUT.csv"

# Compact table: one row per kernel launch in order (kernel name, duration, and the % metrics).
python3 - "$OUT.csv" <<'EOF'
import csv, sys, re
rows = list(csv.reader(open(sys.argv[1])))
# ncu prints a preamble; the header row is the one starting with "ID"
start = next(i for i, r in enumerate(rows) if r and r[0] == "ID")
header, units, data = rows[start], rows[start + 1], rows[start + 2:]
col = {h: i for i, h in enumerate(header)}
def get(r, key):
    return r[col[key]] if key in col else ""
short = lambda n: re.sub(r"\(.*$", "", re.sub(r"^void ", "", n)).split("::")[-1][:44]
print(f"\n{'#':>3} {'kernel':<44} {'us':>8} {'sm%':>6} {'tensor%':>8} {'dram%':>6} {'l2bw%':>6} {'l2hit%':>7} {'occ%':>6} {'grid':>7} {'block':>6}")
for i, r in enumerate(data):
    dur = float(get(r, "gpu__time_duration.sum").replace(",", "") or 0)
    unit = units[col["gpu__time_duration.sum"]] if "gpu__time_duration.sum" in col else "ns"
    us = dur / 1e3 if unit == "nsecond" else dur if unit == "usecond" else dur * 1e3
    f = lambda k: (get(r, k).replace(",", "") or "-")[:7]
    print(f"{i:>3} {short(get(r, 'Kernel Name')):<44} {us:8.1f} {f('sm__throughput.avg.pct_of_peak_sustained_elapsed'):>6} "
          f"{f('sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active'):>8} "
          f"{f('dram__throughput.avg.pct_of_peak_sustained_elapsed'):>6} {f('lts__throughput.avg.pct_of_peak_sustained_elapsed'):>6} "
          f"{f('lts__t_sector_hit_rate.pct'):>7} {f('sm__warps_active.avg.pct_of_peak_sustained_active'):>6} "
          f"{f('launch__grid_size'):>7} {f('launch__block_size'):>6}")
EOF
