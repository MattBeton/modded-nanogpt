"""
Single-GPU microbenchmark of the Polar Express / ANVIL cascade on the exact per-rank chunk shapes
used in training (world_size=8), for timing and for hardware-counter profiling with ncu.

The cascade function and its coefficients are extracted from a train_gpt.py with `ast`, so this
always benchmarks the code that actually runs in training (no copy to drift). Two variants are
recognised automatically:

  polar_express(grad, momentum_buffer, momentum_t, split_baddbmm)                  -- current master
  anvil_cascade(grad, velocity, momentum_t, split_baddbmm, bimax_bf_t, bimax_w_t)  -- PR #360

  python profile/bench_polar_express.py                   # CUDA-event timing per shape
  python profile/bench_polar_express.py --src other/train_gpt.py   # benchmark another tree
  python profile/bench_polar_express.py --trace out.json  # torch.profiler trace of the loop
  python profile/bench_polar_express.py --cudagraph       # hand-captured CUDA graph replay
  profile/ncu_polar_express.sh                            # L2 / DRAM / tensor-pipe per kernel
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

# Per-rank chunk shapes at world_size=8, i.e. <full bank> / 8.
#   master: qk (64,256,768), vo (24,768,768), mlp reshape (24,3072,768)
#   PR#360: qk (48,256,768), vo (16,768,768), mlp reshape (24,2816,768)
SHAPES_BY_VARIANT = {
    "polar_express": {
        "qk_bank":  (8, 256, 768),
        "vo_bank":  (3, 768, 768),
        "mlp_bank": (3, 3072, 768),
    },
    "anvil_cascade": {
        "qk_bank":  (6, 256, 768),
        "vo_bank":  (2, 768, 768),
        "mlp_bank": (3, 2816, 768),
    },
}

CASCADES = ("polar_express", "anvil_cascade")


def load_cascade(src: Path, mode=None):
    """Extract the cascade function from `src` plus the module-level constants it reads.

    mode=None keeps the source's own @torch.compile decorator; a mode string (e.g.
    "reduce-overhead", which wraps the graph in CUDA graphs) strips it and recompiles instead.
    Returns (fn, variant_name).
    """
    tree = ast.parse(src.read_text())
    fn_node = next((n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name in CASCADES), None)
    assert fn_node is not None, f"no {' or '.join(CASCADES)} found in {src}"

    # Module-level constants the body references (coefficient tables, rail betas, ...).
    used = {n.id for n in ast.walk(fn_node) if isinstance(n, ast.Name)}
    def bound_names(assign):  # handles tuple targets: a, b, c = 1, 2, 3
        return {n.id for t in assign.targets for n in ast.walk(t) if isinstance(n, ast.Name)}

    consts = [n for n in tree.body
              if isinstance(n, ast.Assign) and (bound_names(n) & used)]

    if mode is not None:
        fn_node = ast.parse(ast.unparse(fn_node)).body[0]  # detach from the original tree
        fn_node.decorator_list = []

    ns = dict(torch=torch, XXT=XXT, XTX=XTX, ba_plus_cAA=ba_plus_cAA)
    module = ast.Module(body=[*consts, fn_node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), f"{src}:{fn_node.name}", "exec"), ns)

    fn = ns[fn_node.name]
    if mode is not None:
        fn = torch.compile(fn, dynamic=False, fullgraph=True, mode=mode)
    return fn, fn_node.name


def make_inputs(shape, device, variant, on_gpu_scalars=False):
    """State tensors for one bank. Scalars are 0-D tensors so the per-step refill never
    recompiles; they must be on the GPU for capture (a CPU scalar is a blocking H2D in-graph)."""
    def scalar(v):
        return torch.tensor(v, device=device) if on_gpu_scalars else torch.tensor(v)

    grad = torch.randn(shape, dtype=torch.float32, device=device)
    if variant == "anvil_cascade":
        state = torch.randn((2, *shape), dtype=torch.float32, device=device)  # twin-rail velocity
        scalars = dict(momentum_t=scalar(0.95), bimax_bf_t=scalar(0.85), bimax_w_t=scalar(0.4385))
    else:
        state = torch.randn(shape, dtype=torch.float32, device=device)  # momentum buffer
        scalars = dict(momentum_t=scalar(0.95))
    return grad, state, scalars


def call_cascade(fn, variant, grad, state, scalars, split):
    if variant == "anvil_cascade":
        return fn(grad, state, scalars["momentum_t"], split_baddbmm=split,
                  bimax_bf_t=scalars["bimax_bf_t"], bimax_w_t=scalars["bimax_w_t"])
    return fn(grad, state, scalars["momentum_t"], split_baddbmm=split)


def _tflops(shape, ms, maps=5):
    """Rough FLOP count for context: `maps` iterations x (gram + gram^2 + big matmul)."""
    b, m, n = shape[0], shape[-2], shape[-1]
    k = min(m, n)
    return maps * b * (2 * m * n * k + 2 * k ** 3 + 2 * m * n * k) / ms / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(REPO / "train_gpt.py"),
                    help="train_gpt.py to extract the cascade from")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--trace", type=str, default=None, help="write a torch.profiler chrome trace of the timed loop")
    ap.add_argument("--ncu", action="store_true", help="cudaProfilerStart/Stop around one call per shape")
    ap.add_argument("--cudagraph", action="store_true", help="capture one call per shape in a CUDA graph and time the replay")
    ap.add_argument("--mode", type=str, default=None, help="torch.compile mode instead of the source's decorator, e.g. reduce-overhead")
    ap.add_argument("--shapes", type=str, default=None, help="comma-separated subset of the bank names")
    args = ap.parse_args()

    device = torch.device("cuda")
    cascade, variant = load_cascade(Path(args.src), args.mode)
    all_shapes = SHAPES_BY_VARIANT[variant]
    keys = args.shapes.split(",") if args.shapes else list(all_shapes)
    shapes = {k: all_shapes[k] for k in keys}
    n_maps = 6 if variant == "anvil_cascade" else 5
    print(f"variant={variant} ({n_maps} maps)  src={args.src}"
          + (f"  torch.compile(mode={args.mode!r})" if args.mode else ""))

    inputs = {k: make_inputs(s, device, variant, on_gpu_scalars=args.cudagraph)
              for k, s in shapes.items()}

    def run(label):
        grad, state, scalars = inputs[label]
        split = shapes[label][-2] > 1024  # matches `is_large_matrix` at the call site
        return call_cascade(cascade, variant, grad.clone(), state, scalars, split)

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
            print(f"profiled one {variant} call for {label} {shapes[label]}")
        return

    if args.cudagraph:
        _bench_cudagraph(cascade, variant, shapes, inputs, args.iters)
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
        print(f"{label:<9} {str(shapes[label]):<16} {ms:7.3f} ms/call   ~{_tflops(shapes[label], ms, n_maps):6.1f} TFLOP/s effective")
    print(f"{'total':<9} {'':<16} {total:7.3f} ms/step (one call per shape per step)")

    if prof is not None:
        prof.__exit__(None, None, None)
        prof.export_chrome_trace(args.trace)
        print(f"wrote {args.trace}")
        _summarize_gaps(args.trace, shapes, args.iters)


def _bench_cudagraph(cascade, variant, shapes, inputs, iters):
    """Replay-time of one cascade call captured in a CUDA graph.

    torch.compile(mode="reduce-overhead") refuses to apply cudagraphs here because the cascade
    mutates its inputs (the in-place Nesterov lerp_ on grad_chunk / momentum / velocity rails).
    Capturing by hand is fine for the training use: the momentum/velocity buffers are persistent
    optimizer state, and the grad chunk can live in the all-gather's destination buffer.
    """
    print(f"{'label':<9} {'':<16} {'graph':>10} {'graph+copy':>12}")
    total_g = total_gc = 0.0
    for label, shape in shapes.items():
        grad, state, scalars = inputs[label]
        split = shape[-2] > 1024
        static_grad, src = grad.clone(), grad.clone()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                call_cascade(cascade, variant, static_grad, state, scalars, split)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call_cascade(cascade, variant, static_grad, state, scalars, split)

        def timeit(fn):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(iters):
                fn()
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b) / iters

        ms_g = timeit(g.replay)
        ms_gc = timeit(lambda: (static_grad.copy_(src), g.replay()))
        total_g += ms_g
        total_gc += ms_gc
        print(f"{label:<9} {str(shape):<16} {ms_g:7.3f} ms {ms_gc:9.3f} ms")
    print(f"{'total':<9} {'':<16} {total_g:7.3f} ms {total_gc:9.3f} ms")


def _summarize_gaps(path, shapes, iters):
    """Kernel-time vs gap-time inside each pe/<label> region, per call."""
    import json
    data = json.load(open(path))["traceEvents"]
    ev = [e for e in data if e.get("ph") == "X"]
    ann = {e["name"]: e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("pe/")}
    # torch 2.10 records triton/cublas launches under cuda_driver (cuLaunchKernel), not cuda_runtime
    launches = {e["args"]["correlation"]: e["ts"] for e in ev
                if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
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
