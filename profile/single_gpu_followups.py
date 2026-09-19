"""Follow-up single-H100 tests for the ANVIL Gram kernels.

Run inside the profiling image with --source pointing at the frozen ANVIL tree.
This deliberately reuses the sweep harness' source extraction and graph timing.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

import sweep_gram_tiles as sw


def rect_launcher(kernels, op, cfg):
    bm, bn, bk, warps, stages = cfg
    name = {"xxt": "XXT_kernel", "xtx": "XTX_kernel", "poly": "ba_plus_cAA_kernel"}[op]
    kernel = kernels[name]

    def run(x, out, alpha=1.0, beta=0.0):
        m, k = x.shape[-2:]
        d = k if op == "xtx" else m
        grid = (x.shape[0] * torch.ceil(torch.tensor(d / bm)).int().item() *
                torch.ceil(torch.tensor(d / bn)).int().item(),)
        extra = {"alpha": alpha, "beta": beta} if op == "poly" else {"K": k}
        kernel[grid](A_ptr=x, C_ptr=out, M=m,
                     a_stride_b=x.stride(0), a_stride_r=x.stride(1), a_stride_c=x.stride(2),
                     c_stride_b=out.stride(0), c_stride_r=out.stride(1), c_stride_c=out.stride(2),
                     BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                     GROUP_SIZE_M=8, LOWER_UPPER=1, num_warps=warps, num_stages=stages,
                     **extra)
        return out
    return run


def measure_graph(graph, repeats=5):
    vals = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(8):
            graph.replay()
        b.record()
        b.synchronize()
        # sw.capture_graph records 64 calls per replay; each timing loop replays it 8 times.
        vals.append(a.elapsed_time(b) * 1000 / (8 * 64))
    return vals


def measure_one_graph(graph, repeats=5):
    vals = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(8):
            graph.replay()
        b.record(); b.synchronize()
        vals.append(a.elapsed_time(b) * 1000 / 8)
    return vals


def graph_call(fn):
    static = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            static["result"] = fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static["result"] = fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def rectangular(args, kernels):
    shapes = sw.SHAPES
    pairs = [(32, 64), (64, 32), (64, 128), (128, 64), (128, 256), (256, 128)]
    results = []
    for bank, shape in shapes.items():
        rec = sw.operands(args.source / "train_gpt.py", kernels, shape)
        for kind in ("gram", "poly"):
            op = "poly" if kind == "poly" else ("xtx" if shape[-2] > shape[-1] else "xxt")
            samples = [rec[kind][i] for i in (0, 2, 5)]
            x, expected, kw = samples[1]
            out = torch.empty_like(expected)
            base = sw.launcher(kernels, op, sw.BASE)
            base(x, out, **kw)
            bg = sw.capture_graph(lambda: base(x, out, **kw))
            for bm, bn in pairs:
                for bk, warps, stages in ((64, 4, 3), (128, 4, 3)):
                    cfg = (bm, bn, bk, warps, stages)
                    run = rect_launcher(kernels, op, cfg)
                    checks = []
                    try:
                        for sx, sy, skw in samples:
                            run(sx, out, **skw)
                            checks.append(sw.metrics(out, sy))
                        if not all(c["bitwise"] for c in checks):
                            raise ValueError("not bitwise identical")
                        graph = sw.capture_graph(lambda: run(x, out, **kw))
                        b = statistics.median(measure_graph(bg))
                        t = statistics.median(measure_graph(graph))
                        row = dict(bank=bank, kind=kind, config=cfg, baseline_us=b,
                                   candidate_us=t, speedup=b / t, checks=checks)
                        results.append(row)
                        print("RECT", bank, kind, cfg, round(b, 3), round(t, 3), round(b / t, 3), flush=True)
                    except Exception as exc:
                        results.append(dict(bank=bank, kind=kind, config=cfg, error=str(exc)[:300]))
                        print("RECT-ERR", bank, kind, cfg, str(exc)[:160], flush=True)
    args.output.joinpath("rectangular.json").write_text(json.dumps(results, indent=2))


def streams(args, kernels):
    summary = json.loads(args.output.joinpath("summary.json").read_text())
    contexts = {}
    for bank, shape in sw.SHAPES.items():
        grad, state, scalars = sw.inputs(shape)
        cfgs = []
        for kind in ("gram", "poly"):
            cfgs.append((summary[f"{bank}/{kind}"]["best_exact"] or summary[f"{bank}/{kind}"]["best"])["config"])
        tuned = sw.wrappers(kernels, *cfgs)
        fn = sw.load_cascade(args.source / "train_gpt.py", tuned, f"followup_{bank}", compiled=True)
        holder = {}
        def call():
            holder["value"] = sw.invoke(fn, grad, state, scalars)
        graph = graph_call(call)
        contexts[bank] = dict(graph=graph, grad=grad, state=state, scalars=scalars)
    torch.cuda.synchronize()

    def serial():
        for ctx in contexts.values():
            ctx["graph"].replay()

    streams = [torch.cuda.Stream() for _ in contexts]
    def parallel():
        events = []
        for stream, ctx in zip(streams, contexts.values()):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                ctx["graph"].replay()
                event = torch.cuda.Event()
                event.record(stream)
                events.append(event)
        for event in events:
            torch.cuda.current_stream().wait_event(event)

    # Warm and measure each form in alternating order.
    serial_vals, parallel_vals = [], []
    for i in range(36):
        fn_a, fn_b = (serial, parallel) if i % 2 == 0 else (parallel, serial)
        for fn, dst in ((fn_a, serial_vals if fn_a is serial else parallel_vals),
                        (fn_b, serial_vals if fn_b is serial else parallel_vals)):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(8):
                fn()
            b.record(); b.synchronize()
            dst.append(a.elapsed_time(b) * 1000 / 8)
    result = dict(serial_us=statistics.median(serial_vals), parallel_us=statistics.median(parallel_vals),
                  serial_samples_us=serial_vals, parallel_samples_us=parallel_vals)
    args.output.joinpath("streams.json").write_text(json.dumps(result, indent=2))
    print("STREAMS", json.dumps({k: round(v, 3) for k, v in result.items() if k.endswith("_us")}), flush=True)


def graph_copy(args, kernels):
    summary = json.loads(args.output.joinpath("summary.json").read_text())
    result = {}
    for bank, shape in sw.SHAPES.items():
        grad, state, scalars = sw.inputs(shape)
        cfgs = [(summary[f"{bank}/{kind}"]["best_exact"] or summary[f"{bank}/{kind}"]["best"])["config"]
                for kind in ("gram", "poly")]
        tuned = sw.wrappers(kernels, *cfgs)
        fn = sw.load_cascade(args.source / "train_gpt.py", tuned, f"copy_{bank}", compiled=True)
        src = grad.clone()
        graph = graph_call(lambda: sw.invoke(fn, grad, state, scalars))
        plain, copied = [], []
        for i in range(36):
            for mode, dst in (("plain", plain), ("copy", copied)) if i % 2 == 0 else (("copy", copied), ("plain", plain)):
                def replay():
                    if mode == "copy":
                        grad.copy_(src)
                    graph.replay()
                for _ in range(3): replay()
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(8): replay()
                b.record(); b.synchronize()
                dst.append(a.elapsed_time(b) * 1000 / 8)
        result[bank] = dict(plain_us=statistics.median(plain), copy_us=statistics.median(copied),
                            plain_samples_us=plain, copy_samples_us=copied)
        print("COPY", bank, {k: round(v, 3) for k, v in result[bank].items()
                              if k.endswith("_us") and not isinstance(v, list)}, flush=True)
    args.output.joinpath("graph_copy.json").write_text(json.dumps(result, indent=2))


def rect_cascade(args, kernels):
    # Best rectangular finalists from the corrected standalone sweep.
    configs = {
        "qk": ((64, 32, 64, 4, 3), (64, 32, 64, 4, 3)),
        "vo": ((64, 128, 128, 4, 3), (64, 128, 128, 4, 3)),
        "mlp": ((64, 128, 128, 4, 3), (64, 128, 128, 4, 3)),
    }
    result = {}
    for bank, shape in sw.SHAPES.items():
        grad0, state0, scalars = sw.inputs(shape)
        op = "xtx" if shape[-2] > shape[-1] else "xxt"
        gram, poly = configs[bank]
        tuned = dict(XXT=rect_launcher(kernels, "xxt", gram),
                     XTX=rect_launcher(kernels, "xtx", gram),
                     ba_plus_cAA=lambda x, alpha, beta, out: rect_launcher(kernels, "poly", poly)(x, out, alpha, beta))
        baseline = dict(XXT=kernels["XXT"], XTX=kernels["XTX"], ba_plus_cAA=kernels["ba_plus_cAA"])
        f0 = sw.load_cascade(args.source / "train_gpt.py", baseline, f"rect_base_{bank}", compiled=True)
        f1 = sw.load_cascade(args.source / "train_gpt.py", tuned, f"rect_tuned_{bank}", compiled=True)
        states = {}
        for label, fn in (("baseline", f0), ("rect", f1)):
            grad, state = grad0.clone(), state0.clone()
            holder = {}
            def call(fn=fn, grad=grad, state=state):
                holder["result"] = sw.invoke(fn, grad, state, scalars)
            graph = graph_call(call)
            states[label] = dict(graph=graph, grad=grad, state=state, result=holder["result"])
        # Matched multi-step numerical test and alternating timing.
        checks = []
        vals = {"baseline": [], "rect": []}
        for step in range(24):
            fresh = torch.randn_like(grad0)
            for label in states:
                states[label]["grad"].copy_(fresh)
                states[label]["graph"].replay()
            if step < 8:
                checks.append(dict(output=sw.metrics(states["rect"]["result"], states["baseline"]["result"]),
                                   state=sw.metrics(states["rect"]["state"], states["baseline"]["state"])))
            for label in ("baseline", "rect") if step % 2 == 0 else ("rect", "baseline"):
                vals[label].extend(measure_one_graph(states[label]["graph"], repeats=1))
        result[bank] = dict(baseline_us=statistics.median(vals["baseline"]),
                            rect_us=statistics.median(vals["rect"]),
                            speedup=statistics.median(vals["baseline"]) / statistics.median(vals["rect"]),
                            checks=checks, configs=configs[bank])
        print("RECT-CASCADE", bank, {k: round(v, 3) for k, v in result[bank].items() if k.endswith("_us") or k == "speedup"}, flush=True)
    args.output.joinpath("rect_cascade.json").write_text(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--phase", choices=("rectangular", "rect_cascade", "streams", "copy"), required=True)
    args = ap.parse_args()
    kernels = sw.load_kernels(args.source / "triton_kernels.py")
    with torch.no_grad():
        if args.phase == "rectangular":
            rectangular(args, kernels)
        elif args.phase == "rect_cascade":
            rect_cascade(args, kernels)
        elif args.phase == "streams":
            streams(args, kernels)
        else:
            graph_copy(args, kernels)


if __name__ == "__main__":
    main()
