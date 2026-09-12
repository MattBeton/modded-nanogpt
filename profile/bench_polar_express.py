"""
Single-GPU microbenchmark of the Polar Express step on the exact per-rank chunk shapes used in
train_gpt.py (world_size=8), for timing and for hardware-counter profiling with ncu.

The `polar_express` function and its coefficients are extracted from train_gpt.py with `ast`, so this
always benchmarks the code that actually runs in training (no copy to drift).

  python profile/bench_polar_express.py                 # CUDA-event timing per shape
  python profile/bench_polar_express.py --trace out.json  # torch.profiler trace of the PE loop (launch gaps, no torchrun needed)
  profile/ncu_polar_express.sh                          # L2 hit rate / DRAM / tensor-pipe utilisation per kernel
"""
import argparse
import ast
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from triton_kernels import XXT, XTX, ba_plus_cAA  # noqa: E402

# Per-rank NorMuon chunk shapes at world_size=8 (see GPT.init_attn / init_mlp and NorMuonAndAdam._init_state)
SHAPES = {
    "qk_bank":  (8, 256, 768),    # (64, 256, 768) / 8   -> wide path, baddbmm
    "vo_bank":  (3, 768, 768),    # (24, 768, 768) / 8   -> wide path (square), baddbmm
    "mlp_bank": (3, 3072, 768),   # (24, 3072, 768) / 8  -> tall path, split bmm + add_
}


def load_polar_express():
    tree = ast.parse((REPO / "train_gpt.py").read_text())
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "polar_express_coeffs" for t in node.targets):
            wanted.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "polar_express":
            wanted.append(node)
    assert len(wanted) == 2, "could not find polar_express / polar_express_coeffs in train_gpt.py"
    ns = dict(torch=torch, XXT=XXT, XTX=XTX, ba_plus_cAA=ba_plus_cAA)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "train_gpt.py:polar_express", "exec"), ns)
    return ns["polar_express"]


def make_inputs(shape, device):
    grad = torch.randn(shape, dtype=torch.float32, device=device)
    mom = torch.randn(shape, dtype=torch.float32, device=device)
    momentum_t = torch.tensor(0.95)  # 0-D CPU tensor, as in training
    return grad, mom, momentum_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--trace", type=str, default=None, help="write a torch.profiler chrome trace of the timed loop")
    ap.add_argument("--ncu", action="store_true", help="cudaProfilerStart/Stop around one call per shape (for ncu --profile-from-start off)")
    ap.add_argument("--shapes", type=str, default=",".join(SHAPES), help="comma-separated subset of " + ",".join(SHAPES))
    args = ap.parse_args()

    device = torch.device("cuda")
    polar_express = load_polar_express()
    shapes = {k: SHAPES[k] for k in args.shapes.split(",")}

    inputs = {k: make_inputs(s, device) for k, s in shapes.items()}

    def run(label):
        grad, mom, momentum_t = inputs[label]
        split = shapes[label][-2] > 1024  # matches `is_large_matrix` in _normuon_update
        return polar_express(grad.clone(), mom, momentum_t, split_baddbmm=split)

    # compile + warmup (each shape / split_baddbmm combination is its own graph)
    t0 = time.perf_counter()
    for _ in range(args.warmup):
        for label in shapes:
            run(label)
    torch.cuda.synchronize()
    print(f"compiled + warmed up in {time.perf_counter() - t0:.1f}s")

    if args.ncu:
        for label in shapes:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            run(label)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"profiled one polar_express call for {label} {shapes[label]}")
        return

    prof = None
    if args.trace:
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof.__enter__()

    total = 0.0
    for label in shapes:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.profiler.record_function(f"pe/{label}"):
            start.record()
            for _ in range(args.iters):
                run(label)
            end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / args.iters
        total += ms
        m, n = shapes[label][-2:]
        b = shapes[label][0]
        # 5 iters x (small gram matmul + gram^2 + big matmul): rough FLOP count for context
        k = min(m, n)
        flops = 5 * b * (2 * m * n * k + 2 * k * k * k + 2 * m * n * k)
        print(f"{label:<9} {str(shapes[label]):<16} {ms:7.3f} ms/call   ~{flops / ms / 1e9:6.1f} TFLOP/s effective")
    print(f"{'total':<9} {'':<16} {total:7.3f} ms/step (one call per shape per step)")

    if prof is not None:
        prof.__exit__(None, None, None)
        prof.export_chrome_trace(args.trace)
        print(f"wrote {args.trace}")
        _summarize_gaps(args.trace, shapes, args.iters)


def _summarize_gaps(path, shapes, iters):
    """Kernel-time vs gap-time inside each pe/<label> region, per call."""
    import json
    data = json.load(open(path))["traceEvents"]
    ev = [e for e in data if e.get("ph") == "X"]
    ann = {e["name"]: e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("pe/")}
    launches = {e["args"]["correlation"]: e["ts"] for e in ev if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
    kern = [e for e in ev if e.get("cat") == "kernel"]
    print(f"\n{'label':<9} {'kernels/call':>12} {'kernel_us':>10} {'gap_us':>8} {'span_us':>8} {'gap%':>6}")
    for label in shapes:
        a = ann.get(f"pe/{label}")
        if not a:
            continue
        lo, hi = a["ts"], a["ts"] + a["dur"]
        ks = sorted((k for k in kern if lo <= launches.get(k["args"].get("correlation"), -1) <= hi), key=lambda k: k["ts"])
        if not ks:
            continue
        kernel = sum(k["dur"] for k in ks)
        span = ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]
        gap = span - kernel
        print(f"{label:<9} {len(ks) / iters:12.1f} {kernel / iters:10.1f} {gap / iters:8.1f} {span / iters:8.1f} {100 * gap / span:5.1f}%")


if __name__ == "__main__":
    main()
