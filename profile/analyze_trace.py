"""
Analyze torch.profiler chrome traces produced by `PROFILE_STEPS=... ./run.sh` (see profile/README.md).

Pure Python (no torch needed). Attributes every GPU kernel to the CPU-side `rf(...)` phase marker that
launched it (via the CUDA runtime correlation id) and reports, per training step:
  - CPU wall time vs GPU span / busy / idle
  - GPU time by phase (fwd / bwd / opt / fp8_quant) and by kernel category (gemm / attention / nccl / ...)
  - optimizer breakdown per parameter label (scatter / update / gather) and, for NorMuon params,
    the Polar Express region: kernel time vs launch gaps (launch-bound or not)
  - the inter-step bubble: last backward kernel -> first forward kernel of the next step, split into
    compute-busy / nccl-busy / both / idle

Usage:
  python profile/analyze_trace.py profiles/<run_id>/rank0_steps20-26.json
  python profile/analyze_trace.py profiles/<run_id>/            # all ranks: full report for rank0 + cross-rank table
  python profile/analyze_trace.py trace.json --pe-detail        # print every Polar Express kernel for the median step
"""
import argparse
import bisect
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PHASES = ["dataload", "fwd", "bwd", "opt", "fp8_quant"]
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


# ----------------------------------------------------------------------------
# Kernel categorisation

def _custom_triton_kernel_names() -> list[str]:
    names = []
    for fname in ["triton_kernels.py", "dc_triton_kernels.py"]:
        path = Path(__file__).resolve().parent.parent / fname
        if path.exists():
            names += re.findall(r"@triton\.jit\s*\ndef\s+(\w+)", path.read_text())
    return names


CUSTOM_TRITON = _custom_triton_kernel_names()


def categorize(ev) -> str:
    if ev["cat"] in ("gpu_memcpy", "gpu_memset"):
        return "memcpy"
    name = ev["name"]
    if "nccl" in name.lower():
        return "nccl"
    if any(name.startswith(k) for k in CUSTOM_TRITON):
        return "triton_custom"
    if name.startswith("triton_tem") or name.startswith("triton_mm"):
        return "gemm"
    if name.startswith("triton_"):
        return "inductor"
    if re.search(r"flash|fmha|attn|attention", name, re.I):
        return "attention"
    if re.search(r"gemm|cutlass|nvjet|xmma|cublas|Cijk|gemv", name, re.I):
        return "gemm"
    return "aten_other"


def short_name(name: str, width: int = 64) -> str:
    n = name
    if n.startswith("void "):
        n = n[5:]
    n = re.sub(r"\(.*$", "", n)          # drop argument list
    n = re.sub(r"<.*$", "", n)           # drop template args
    n = n.split("::")[-1] if "::" in n else n
    return n[:width]


# ----------------------------------------------------------------------------
# Interval helpers (all times in microseconds)

def union_len(intervals) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s)


def clip(intervals, lo, hi):
    return [(max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi]


def intersect_len(a, b) -> float:
    """Length of the intersection of two interval sets."""
    a, b = sorted(a), sorted(b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


# ----------------------------------------------------------------------------
# Trace loading & attribution

class Trace:
    def __init__(self, path: Path):
        self.path = path
        with open(path) as f:
            data = json.load(f)
        events = [e for e in data["traceEvents"] if e.get("ph") == "X" and "dur" in e]

        self.gpu = [e for e in events if e.get("cat") in GPU_CATS]
        runtime = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")]
        annotations = [e for e in events if e.get("cat") == "user_annotation"
                       and not e["name"].startswith("ProfilerStep")]
        self.annotations = annotations

        # correlation id -> (tid, launch timestamp)
        launch_by_corr = {}
        for e in runtime:
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                launch_by_corr[corr] = (e["tid"], e["ts"])

        # Per-thread annotation stack lookup: sweep annotations and launches in time order.
        ann_by_tid = defaultdict(list)
        for a in annotations:
            ann_by_tid[a["tid"]].append(a)
        launches_by_tid = defaultdict(list)
        for corr, (tid, ts) in launch_by_corr.items():
            launches_by_tid[tid].append((ts, corr))

        self.stack_by_corr = {}
        for tid, anns in ann_by_tid.items():
            anns.sort(key=lambda a: (a["ts"], -a["dur"]))
            launches = sorted(launches_by_tid.get(tid, []))
            stack = []
            ai = 0
            for ts, corr in launches:
                while ai < len(anns) and anns[ai]["ts"] <= ts:
                    a = anns[ai]
                    while stack and stack[-1]["ts"] + stack[-1]["dur"] <= a["ts"]:
                        stack.pop()
                    stack.append(a)
                    ai += 1
                while stack and stack[-1]["ts"] + stack[-1]["dur"] <= ts:
                    stack.pop()
                self.stack_by_corr[corr] = list(stack)

        # Training-step annotations, indexed in time order
        self.step_anns = sorted([a for a in annotations if a["name"] == "step"], key=lambda a: a["ts"])
        step_id = {id(a): i for i, a in enumerate(self.step_anns)}

        # Annotate each GPU event with (step, phase, path, category, stream)
        for e in self.gpu:
            corr = e.get("args", {}).get("correlation")
            stack = self.stack_by_corr.get(corr, [])
            names = [a["name"] for a in stack]
            e["_step"] = next((step_id[id(a)] for a in stack if a["name"] == "step"), None)
            e["_phase"] = next((n for n in names if n in PHASES), "other")
            e["_path"] = names
            e["_cat"] = categorize(e)
            e["_stream"] = e.get("args", {}).get("stream", -1)
            e["_end"] = e["ts"] + e["dur"]
        self.gpu.sort(key=lambda e: e["ts"])

        # Fallback for kernels with no correlated launch (rare): assign by GPU-time containment.
        spans = []
        for i in range(len(self.step_anns)):
            ks = [e for e in self.gpu if e["_step"] == i]
            if ks:
                spans.append((min(k["ts"] for k in ks), max(k["_end"] for k in ks), i))
        starts = [s for s, _, _ in spans]
        for e in self.gpu:
            if e["_step"] is None and spans:
                j = bisect.bisect_right(starts, e["ts"]) - 1
                if j >= 0 and e["ts"] < spans[j][1]:
                    e["_step"] = spans[j][2]

        self.nccl_streams = {e["_stream"] for e in self.gpu if e["_cat"] == "nccl"}
        self.by_step = defaultdict(list)
        for e in self.gpu:
            if e["_step"] is not None:
                self.by_step[e["_step"]].append(e)

    # ------------------------------------------------------------------
    def step_summary(self, i: int) -> dict:
        ks = self.by_step[i]
        ann = self.step_anns[i]
        iv = [(k["ts"], k["_end"]) for k in ks]
        compute = [(k["ts"], k["_end"]) for k in ks if k["_cat"] != "nccl"]
        nccl = [(k["ts"], k["_end"]) for k in ks if k["_cat"] == "nccl"]
        span = (max(e for _, e in iv) - min(s for s, _ in iv)) if iv else 0.0
        busy = union_len(iv)
        out = dict(
            cpu_ms=ann["dur"] / 1e3, gpu_span_ms=span / 1e3, gpu_busy_ms=busy / 1e3,
            gpu_idle_ms=(span - busy) / 1e3, compute_busy_ms=union_len(compute) / 1e3,
            nccl_busy_ms=union_len(nccl) / 1e3, n_kernels=len(ks),
        )
        # per phase: kernel-time sum (all streams) and CPU wall of the marker
        phase_cpu = {}
        for a in self._children(ann):
            if a["name"] in PHASES:
                phase_cpu[a["name"]] = phase_cpu.get(a["name"], 0.0) + a["dur"]
        out["phases"] = {}
        for ph in PHASES + ["other"]:
            pk = [k for k in ks if k["_phase"] == ph]
            out["phases"][ph] = dict(
                gpu_sum_ms=sum(k["dur"] for k in pk) / 1e3,
                gpu_compute_ms=sum(k["dur"] for k in pk if k["_cat"] != "nccl") / 1e3,
                gpu_nccl_ms=sum(k["dur"] for k in pk if k["_cat"] == "nccl") / 1e3,
                cpu_ms=phase_cpu.get(ph, 0.0) / 1e3, n=len(pk),
            )
        out["cats"] = defaultdict(float)
        for k in ks:
            out["cats"][k["_cat"]] += k["dur"] / 1e3
        return out

    def _children(self, ann):
        s, e = ann["ts"], ann["ts"] + ann["dur"]
        return [a for a in self._all_annotations() if a["tid"] == ann["tid"] and s <= a["ts"] and a["ts"] + a["dur"] <= e and a is not ann]

    def _all_annotations(self):
        return self.annotations

    def opt_breakdown(self, i: int) -> dict:
        """GPU kernel time inside `opt`, keyed by 'scatter/<label>', 'update/<label>', 'gather/<label>', 'finalize'."""
        out = defaultdict(lambda: dict(gpu_ms=0.0, n=0, sub=defaultdict(float)))
        for k in self.by_step[i]:
            if k["_phase"] != "opt":
                continue
            key = next((n[len("opt/"):] for n in k["_path"] if n.startswith("opt/")), "(opt, unmarked)")
            out[key]["gpu_ms"] += k["dur"] / 1e3
            out[key]["n"] += 1
            sub = next((n for n in k["_path"] if n in ("pe", "normuon_vr", "cautious_update")), "other")
            out[key]["sub"][sub] += k["dur"] / 1e3
        return out

    def pe_regions(self, i: int) -> dict:
        """Polar Express kernels per label for step i, in launch order, with same-stream gaps."""
        regions = defaultdict(list)
        for k in self.by_step[i]:
            if "pe" in k["_path"]:
                label = next((n.split("/")[-1] for n in k["_path"] if n.startswith("opt/update/")), "?")
                regions[label].append(k)
        out = {}
        for label, ks in regions.items():
            ks.sort(key=lambda k: k["ts"])
            rows, prev_end = [], None
            for k in ks:
                gap = (k["ts"] - prev_end) if prev_end is not None else 0.0
                rows.append(dict(name=short_name(k["name"]), dur_us=k["dur"], gap_us=max(gap, 0.0), cat=k["_cat"]))
                prev_end = max(prev_end or 0, k["_end"])
            ann = next((a for a in self._all_annotations() if a["name"] == "pe"
                        and a["ts"] <= self._launch_ts(ks[0]) <= a["ts"] + a["dur"]), None)
            out[label] = dict(
                rows=rows, kernel_us=sum(r["dur_us"] for r in rows), gap_us=sum(r["gap_us"] for r in rows),
                span_us=ks[-1]["_end"] - ks[0]["ts"], cpu_us=ann["dur"] if ann else float("nan"),
            )
        return out

    def _launch_ts(self, k):
        corr = k.get("args", {}).get("correlation")
        stack = self.stack_by_corr.get(corr)
        return stack[-1]["ts"] if stack else k["ts"]

    def bubble(self, i: int) -> dict | None:
        """From the last backward kernel of step i to the first forward kernel of step i+1."""
        if i + 1 not in self.by_step:
            return None
        bwd = [k for k in self.by_step[i] if k["_phase"] == "bwd" and k["_cat"] != "nccl"]
        nxt = [k for k in self.by_step[i + 1] if k["_phase"] == "fwd" and k["_cat"] != "nccl"]
        if not bwd or not nxt:
            return None
        lo = max(k["_end"] for k in bwd)
        hi = min(k["ts"] for k in nxt)
        window = [k for k in self.gpu if k["_end"] > lo and k["ts"] < hi]
        compute = clip([(k["ts"], k["_end"]) for k in window if k["_cat"] != "nccl"], lo, hi)
        nccl = clip([(k["ts"], k["_end"]) for k in window if k["_cat"] == "nccl"], lo, hi)
        c, n, both = union_len(compute), union_len(nccl), intersect_len(compute, nccl)
        total = hi - lo
        return dict(total_ms=total / 1e3, compute_only_ms=(c - both) / 1e3, nccl_only_ms=(n - both) / 1e3,
                    both_ms=both / 1e3, idle_ms=(total - c - n + both) / 1e3,
                    n_compute_kernels=sum(1 for k in window if k["_cat"] != "nccl"),
                    n_nccl_kernels=sum(1 for k in window if k["_cat"] == "nccl"))

    def top_kernels(self, n: int = 25):
        agg = defaultdict(lambda: [0.0, 0, ""])
        for k in self.gpu:
            if k["_step"] is None:
                continue
            a = agg[k["name"]]
            a[0] += k["dur"]
            a[1] += 1
            a[2] = k["_cat"]
        rows = sorted(agg.items(), key=lambda kv: -kv[1][0])[:n]
        nsteps = max(1, len(self.by_step))
        return [(short_name(name), tot / 1e3 / nsteps, cnt / nsteps, tot / cnt, cat) for name, (tot, cnt, cat) in rows]


# ----------------------------------------------------------------------------
# Reporting

def fmt_ms(x):
    return f"{x:7.2f}"


def report(tr: Trace, pe_detail: bool, top_n: int):
    steps = sorted(tr.by_step)
    print(f"== {tr.path.name}: {len(tr.step_anns)} step markers, {len(steps)} with GPU work, "
          f"{len(tr.gpu)} GPU events, nccl streams={sorted(tr.nccl_streams)}")
    if not steps:
        print("no attributed GPU work found — was the trace recorded inside the training loop?")
        return
    summaries = {i: tr.step_summary(i) for i in steps}

    print("\n-- per step (ms) --")
    print(f"{'step':>4} {'cpu':>8} {'gpu_span':>9} {'gpu_busy':>9} {'gpu_idle':>9} {'compute':>8} {'nccl':>7} {'kernels':>8}")
    for i in steps:
        s = summaries[i]
        print(f"{i:>4} {s['cpu_ms']:8.2f} {s['gpu_span_ms']:9.2f} {s['gpu_busy_ms']:9.2f} {s['gpu_idle_ms']:9.2f} "
              f"{s['compute_busy_ms']:8.2f} {s['nccl_busy_ms']:7.2f} {s['n_kernels']:8d}")

    # Use the median-cpu-time step for the detailed sections (the first recorded step often carries warmup noise)
    med = sorted(steps, key=lambda i: summaries[i]["cpu_ms"])[len(steps) // 2]
    s = summaries[med]
    print(f"\n-- phase breakdown, step {med} (median by cpu time), ms --")
    print(f"{'phase':<10} {'gpu_sum':>8} {'compute':>8} {'nccl':>7} {'cpu_wall':>9} {'kernels':>8} {'%step_gpu':>10}")
    for ph, d in s["phases"].items():
        pct = 100 * d["gpu_compute_ms"] / max(s["compute_busy_ms"], 1e-9)
        print(f"{ph:<10} {d['gpu_sum_ms']:8.2f} {d['gpu_compute_ms']:8.2f} {d['gpu_nccl_ms']:7.2f} {d['cpu_ms']:9.2f} {d['n']:8d} {pct:9.1f}%")

    print(f"\n-- kernel category breakdown, step {med} (sum of kernel durations, ms) --")
    for cat, ms in sorted(s["cats"].items(), key=lambda kv: -kv[1]):
        print(f"{cat:<14} {ms:8.2f}")

    print(f"\n-- optimizer breakdown, step {med} (GPU kernel time inside opt, ms) --")
    ob = tr.opt_breakdown(med)
    print(f"{'region':<28} {'gpu_ms':>8} {'n':>5}   sub-regions")
    for key, d in sorted(ob.items(), key=lambda kv: -kv[1]["gpu_ms"]):
        sub = "  ".join(f"{k}={v:.2f}" for k, v in sorted(d["sub"].items(), key=lambda kv: -kv[1]) if k != "other")
        print(f"{key:<28} {d['gpu_ms']:8.2f} {d['n']:5d}   {sub}")
    opt_total = sum(d["gpu_ms"] for d in ob.values())
    print(f"{'total':<28} {opt_total:8.2f}   ({100 * opt_total / max(s['gpu_busy_ms'], 1e-9):.1f}% of step GPU busy, "
          f"opt cpu wall {s['phases']['opt']['cpu_ms']:.2f} ms)")

    print(f"\n-- Polar Express regions, step {med} --")
    print(f"{'label':<12} {'kernels':>7} {'kernel_us':>10} {'gap_us':>8} {'span_us':>8} {'cpu_us':>8} {'gap%':>6}")
    pe = tr.pe_regions(med)
    for label, d in pe.items():
        gap_pct = 100 * d["gap_us"] / max(d["span_us"], 1e-9)
        print(f"{label:<12} {len(d['rows']):7d} {d['kernel_us']:10.1f} {d['gap_us']:8.1f} {d['span_us']:8.1f} {d['cpu_us']:8.1f} {gap_pct:5.1f}%")
    if pe_detail:
        for label, d in pe.items():
            print(f"\n   [{label}]  {'kernel':<50} {'dur_us':>8} {'gap_us':>8}  cat")
            for r in d["rows"]:
                print(f"   {'':<10}  {r['name']:<50} {r['dur_us']:8.1f} {r['gap_us']:8.1f}  {r['cat']}")

    print("\n-- inter-step bubble: last bwd kernel of step i -> first fwd kernel of step i+1 (ms) --")
    print(f"{'step':>4} {'total':>7} {'compute':>8} {'nccl':>7} {'both':>7} {'idle':>7} {'#compute':>9} {'#nccl':>6}")
    bubbles = []
    for i in steps:
        b = tr.bubble(i)
        if b:
            bubbles.append(b["total_ms"])
            print(f"{i:>4} {b['total_ms']:7.2f} {b['compute_only_ms']:8.2f} {b['nccl_only_ms']:7.2f} {b['both_ms']:7.2f} "
                  f"{b['idle_ms']:7.2f} {b['n_compute_kernels']:9d} {b['n_nccl_kernels']:6d}")
    if bubbles:
        print(f"median bubble {statistics.median(bubbles):.2f} ms = "
              f"{100 * statistics.median(bubbles) / statistics.median(summaries[i]['cpu_ms'] for i in steps):.1f}% of step")

    print(f"\n-- top {top_n} kernels (per-step averages) --")
    print(f"{'kernel':<64} {'ms/step':>8} {'n/step':>7} {'mean_us':>8}  cat")
    for name, ms, cnt, mean_us, cat in tr.top_kernels(top_n):
        print(f"{name:<64} {ms:8.3f} {cnt:7.1f} {mean_us:8.1f}  {cat}")


def cross_rank(traces: list[Trace]):
    print("\n== cross-rank summary (median step, ms) ==")
    print(f"{'file':<28} {'cpu':>8} {'gpu_busy':>9} {'idle':>7} {'nccl':>7} {'opt_gpu':>8} {'bubble':>7}")
    for tr in traces:
        steps = sorted(tr.by_step)
        if not steps:
            continue
        sums = {i: tr.step_summary(i) for i in steps}
        med = sorted(steps, key=lambda i: sums[i]["cpu_ms"])[len(steps) // 2]
        s = sums[med]
        opt = sum(d["gpu_ms"] for d in tr.opt_breakdown(med).values())
        bubbles = [b["total_ms"] for b in (tr.bubble(i) for i in steps) if b]
        bub = statistics.median(bubbles) if bubbles else float("nan")
        print(f"{tr.path.name:<28} {s['cpu_ms']:8.2f} {s['gpu_busy_ms']:9.2f} {s['gpu_idle_ms']:7.2f} {s['nccl_busy_ms']:7.2f} {opt:8.2f} {bub:7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="trace .json file(s) or a directory of them")
    ap.add_argument("--pe-detail", action="store_true", help="list every Polar Express kernel for the median step")
    ap.add_argument("--top", type=int, default=25, help="number of top kernels to list")
    ap.add_argument("--all-ranks", action="store_true", help="print the full report for every file, not just the first")
    args = ap.parse_args()

    files = []
    for p in map(Path, args.paths):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    if not files:
        sys.exit("no trace files found")

    traces = [Trace(f) for f in files]
    for tr in (traces if args.all_ranks else traces[:1]):
        report(tr, args.pe_detail, args.top)
        print()
    if len(traces) > 1:
        cross_rank(traces)


if __name__ == "__main__":
    main()
