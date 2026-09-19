"""Graph-replayed ANVIL Gram tile sweep; leaves production sources untouched.

python profile/sweep_gram_tiles.py --source /path/to/anvil --output /path/to/results
python profile/sweep_gram_tiles.py --source /path/to/anvil --output /path/to/results --phase cascade

Extracts the existing Triton kernels and cascade from source. Only square output
tiles are swept: the existing triangular skip/mirror scheme assumes aligned tiles.
Synthetic inputs are passed through the actual cascade to obtain early/mid/late
operands. These are numerical/performance experiments, not training validation.
"""
import argparse
import ast
import hashlib
import itertools
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

SHAPES = {"qk": (6, 256, 768), "vo": (2, 768, 768), "mlp": (3, 2816, 768)}
BASE = (128, 64, 8, 4)  # output tile, reduction tile, warps, stages


def load_kernels(path):
    names = {"_pid_to_block", "XXT_kernel", "XTX_kernel", "ba_plus_cAA_kernel",
             "XXT", "XTX", "ba_plus_cAA"}
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(__name__=__name__, torch=torch, triton=triton, tl=tl)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    # Dynamo resolves Triton JIT functions through their owning Python module.
    globals().update({name: ns[name] for name in names})
    return ns


def load_cascade(path, kernels, name, compiled=False):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "anvil_cascade")
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    consts = [n for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id in used
                      for target in n.targets for t in ast.walk(target))]
    fn.name = name
    fn.decorator_list = []
    ns = dict(__name__=__name__, torch=torch, **{k: kernels[k] for k in ("XXT", "XTX", "ba_plus_cAA")})
    exec(compile(ast.fix_missing_locations(ast.Module(body=consts + [fn], type_ignores=[])),
                 str(path), "exec"), ns)
    result = ns[name]
    globals()[name] = result
    return torch.compile(result, dynamic=False, fullgraph=True) if compiled else result


def launcher(kernels, op, cfg):
    tile, reduction, warps, stages = cfg
    kernel = kernels[{"xxt": "XXT_kernel", "xtx": "XTX_kernel", "poly": "ba_plus_cAA_kernel"}[op]]

    def run(x, out, alpha=1.0, beta=0.0):
        m, k = x.shape[-2:]
        d = k if op == "xtx" else m
        grid = (x.shape[0] * triton.cdiv(d, tile) ** 2,)
        extra = {"alpha": alpha, "beta": beta} if op == "poly" else {"K": k}
        kernel[grid](A_ptr=x, C_ptr=out, M=m,
                     a_stride_b=x.stride(0), a_stride_r=x.stride(1), a_stride_c=x.stride(2),
                     c_stride_b=out.stride(0), c_stride_r=out.stride(1), c_stride_c=out.stride(2),
                     BLOCK_SIZE_M=tile, BLOCK_SIZE_N=tile, BLOCK_SIZE_K=reduction,
                     GROUP_SIZE_M=8, LOWER_UPPER=1, num_warps=warps, num_stages=stages, **extra)
        return out
    return run


def wrappers(kernels, gram_cfg, poly_cfg):
    xxt = launcher(kernels, "xxt", gram_cfg)
    xtx = launcher(kernels, "xtx", gram_cfg)
    poly = launcher(kernels, "poly", poly_cfg)
    return dict(XXT=xxt, XTX=xtx,
                ba_plus_cAA=lambda x, alpha, beta, out: poly(x, out, alpha, beta))


def metrics(actual, expected):
    a, b = actual.float(), expected.float()
    diff = a - b
    return dict(bitwise=bool(torch.equal(actual, expected)),
                relative_l2=float(diff.norm() / b.norm().clamp_min(1e-30)),
                max_abs=float(diff.abs().max()),
                finite=bool(torch.isfinite(a).all()))


def inputs(shape, seed=123):
    torch.manual_seed(seed)
    grad = torch.randn(shape, device="cuda")
    state = torch.randn((2, *shape), device="cuda")
    scalars = [torch.tensor(v, device="cuda") for v in (0.95, 0.85, 0.4385)]
    return grad, state, scalars


def invoke(fn, grad, state, scalars):
    return fn(grad, state, scalars[0], split_baddbmm=grad.shape[-2] > 1024,
              bimax_bf_t=scalars[1], bimax_w_t=scalars[2])


def operands(source, kernels, shape):
    records = {"gram": [], "poly": []}

    def capture(op, original):
        def wrapped(x, out, **kw):
            saved = x.clone()
            original(x, out=out, **kw)
            records[op].append((saved, out.clone(), kw))
            return out
        return wrapped
    ns = dict(XXT=capture("gram", kernels["XXT"]), XTX=capture("gram", kernels["XTX"]),
              ba_plus_cAA=capture("poly", kernels["ba_plus_cAA"]))
    fn = load_cascade(source, ns, "collect_operands")
    invoke(fn, *inputs(shape))
    return records


def capture_graph(fn, count=64):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(count):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def measure(graph, count=64, repeats=3):
    values = []
    graph.replay()
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (count * 5))
    return values


def emit(path, item):
    with path.open("a") as f:
        f.write(json.dumps(item) + "\n")


def sweep(args, kernels):
    rng = random.Random(924)
    configs = list(itertools.product((32, 64, 128, 256), (32, 64, 128), (4, 8), (2, 3, 4)))
    results = {}
    raw_path = args.output / "sweep.jsonl"
    for bank, shape in SHAPES.items():
        rec = operands(args.source / "train_gpt.py", kernels, shape)
        for kind in ("gram", "poly"):
            op = "poly" if kind == "poly" else ("xtx" if shape[-2] > shape[-1] else "xxt")
            # Include the real coefficient values and operand distributions from all six maps.
            samples = [rec[kind][i] for i in (0, 2, 5)]
            x, expected, kw = samples[1]
            out = torch.empty_like(expected)
            baseline_run = launcher(kernels, op, BASE)
            baseline_run(x, out, **kw)
            assert torch.equal(out, expected), "Baseline launch differs from source wrapper"
            baseline_graph = capture_graph(lambda: baseline_run(x, out, **kw))
            candidates = []
            order = configs.copy()
            rng.shuffle(order)
            print(f"SWEEP {bank}/{kind}: {len(order)} configurations", flush=True)
            for index, cfg in enumerate(order):
                item = dict(bank=bank, kind=kind, config=cfg)
                started = time.monotonic()
                try:
                    run = launcher(kernels, op, cfg)
                    checks = []
                    for sx, sy, skw in samples:
                        run(sx, out, **skw)
                        checks.append(metrics(out, sy))
                    item["checks"] = checks
                    item["bitwise"] = all(c["bitwise"] for c in checks)
                    item["valid"] = all(c["finite"] and c["relative_l2"] < 0.01 for c in checks)
                    if not item["valid"]:
                        raise ValueError("Numerical screening failed")
                    graph = capture_graph(lambda: run(x, out, **kw))
                    before = statistics.median(measure(baseline_graph))
                    times = measure(graph)
                    after = statistics.median(measure(baseline_graph))
                    item.update(us=statistics.median(times), baseline_us=(before + after) / 2,
                                samples_us=times)
                    item["speedup"] = item["baseline_us"] / item["us"]
                    candidates.append(item)
                    del graph
                except Exception as exc:
                    item["error"] = str(exc)[:500]
                item["elapsed_s"] = time.monotonic() - started
                emit(raw_path, item)
                if index % 12 == 0:
                    print(f"  {index + 1}/{len(order)} {cfg} {item.get('us', item.get('error'))}", flush=True)
            # Remeasure finalists in shuffled paired rounds to reduce winner-selection noise.
            finalists = sorted(candidates, key=lambda r: -r["speedup"])[:6]
            exact = sorted((r for r in candidates if r["bitwise"]), key=lambda r: -r["speedup"])
            for r in exact[:2]:
                if r not in finalists:
                    finalists.append(r)
            final_graphs = []
            for r in finalists:
                fn = launcher(kernels, op, r["config"])
                g = capture_graph(lambda fn=fn: fn(x, out, **kw))
                final_graphs.append((r, g, []))
            for round_idx in range(12):
                rng.shuffle(final_graphs)
                for r, g, pairs in final_graphs:
                    if round_idx % 2:
                        t = measure(g, repeats=1)[0]
                        b = measure(baseline_graph, repeats=1)[0]
                    else:
                        b = measure(baseline_graph, repeats=1)[0]
                        t = measure(g, repeats=1)[0]
                    pairs.append(dict(baseline_us=b, candidate_us=t, saving_us=b-t))
            finals = []
            for r, g, pairs in final_graphs:
                entry = dict(r, paired=pairs)
                entry["us"] = statistics.median(p["candidate_us"] for p in pairs)
                entry["baseline_us"] = statistics.median(p["baseline_us"] for p in pairs)
                entry["speedup"] = entry["baseline_us"] / entry["us"]
                finals.append(entry)
            finals.sort(key=lambda r: -r["speedup"])
            best_exact = next((r for r in finals if r["bitwise"]), None)
            results[f"{bank}/{kind}"] = dict(best=finals[0], best_exact=best_exact, finalists=finals,
                                              successful=len(candidates), attempted=len(order))
            (args.output / "summary.json").write_text(json.dumps(results, indent=2))
            print("RESULT", bank, kind, json.dumps({k: finals[0][k] for k in
                  ("config", "us", "baseline_us", "speedup", "bitwise")}), flush=True)
            del baseline_graph, final_graphs


def cascade(args, kernels):
    summary = json.loads((args.output / "summary.json").read_text())
    results = {}
    all_contexts = {}
    initial = {}
    for bank, shape in SHAPES.items():
        contexts = {}
        grad0, state0, scalars = inputs(shape)
        initial[bank] = (grad0, state0)
        for label in ("baseline", "tuned", "exact"):
            if label == "baseline":
                ns = kernels
                cfgs = [BASE, BASE]
            else:
                key = "best" if label == "tuned" else "best_exact"
                cfgs = [(summary[f"{bank}/{kind}"].get(key) or {"config": BASE})["config"]
                        for kind in ("gram", "poly")]
                ns = wrappers(kernels, *cfgs)
            fn = load_cascade(args.source / "train_gpt.py", ns, f"cascade_{bank}_{label}", compiled=True)
            grad, state = grad0.clone(), state0.clone()
            started = time.monotonic()
            for _ in range(3):
                grad.copy_(grad0)
                state.copy_(state0)
                result = invoke(fn, grad, state, scalars)
            torch.cuda.synchronize()
            compile_s = time.monotonic() - started
            holder = {}
            def call():
                holder["result"] = invoke(fn, grad, state, scalars)
            graph = capture_graph(call, count=1)
            contexts[label] = dict(fn=fn, grad=grad, state=state, graph=graph,
                                   result=holder["result"], config=cfgs, compile_s=compile_s)
            print(f"CAPTURED {bank}/{label} compile+warmup={compile_s:.1f}s", flush=True)
        # Matched one-step and multi-step checks; gradients refreshed each step.
        checks = {label: [] for label in ("tuned", "exact")}
        for seed, zero_state, blend in ((123, False, 0.4385), (321, True, 1.0), (456, False, 0.4385)):
            g0, s0, _ = inputs(shape, seed)
            if zero_state:
                s0.zero_()
            scalars[2].fill_(blend)
            for ctx in contexts.values():
                ctx["state"].copy_(s0)
            for step in range(8):
                fresh = torch.randn_like(g0)
                for ctx in contexts.values():
                    ctx["grad"].copy_(fresh)
                    ctx["graph"].replay()
                ref = contexts["baseline"]
                for label in checks:
                    ctx = contexts[label]
                    checks[label].append(dict(seed=seed, step=step, output=metrics(ctx["result"], ref["result"]),
                                               state=metrics(ctx["state"], ref["state"])))
        scalars[2].fill_(0.4385)
        timings = {label: [] for label in contexts}
        for round_idx in range(24):
            labels = list(contexts)
            if round_idx % 2:
                labels.reverse()
            for label in labels:
                ctx = contexts[label]
                ctx["grad"].copy_(grad0)
                ctx["state"].copy_(state0)
                timings[label].append(measure(ctx["graph"], count=1, repeats=1)[0])
        results[bank] = {label: dict(us=statistics.median(times), samples_us=times,
                                     config=contexts[label]["config"], compile_s=contexts[label]["compile_s"],
                                     checks=checks.get(label)) for label, times in timings.items()}
        (args.output / "cascade.json").write_text(json.dumps(results, indent=2))
        print("CASCADE", bank, json.dumps({label: round(v["us"], 3) for label,v in results[bank].items()}), flush=True)
        all_contexts[bank] = contexts
    # Each bank remains a separate graph, replayed serially as in the source.
    class BankSequence:
        def __init__(self, label):
            self.label = label

        def replay(self):
            for bank in SHAPES:
                all_contexts[bank][self.label]["graph"].replay()

    sequence = {label: [] for label in ("baseline", "tuned", "exact")}
    for round_idx in range(36):
        labels = list(sequence)
        if round_idx % 2:
            labels.reverse()
        for label in labels:
            for bank, contexts in all_contexts.items():
                contexts[label]["grad"].copy_(initial[bank][0])
                contexts[label]["state"].copy_(initial[bank][1])
            sequence[label].append(measure(BankSequence(label), count=1, repeats=1)[0])
    seq_result = {label: dict(us=statistics.median(values), samples_us=values)
                  for label, values in sequence.items()}
    (args.output / "sequential.json").write_text(json.dumps(seq_result, indent=2))
    print("SEQUENTIAL", json.dumps({label: v["us"] for label,v in seq_result.items()}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--phase", choices=("sweep", "cascade"), default="sweep")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Freeze exact source bytes; subsequent phases read the frozen copies.
    snapshot = args.output / "source"
    snapshot.mkdir(exist_ok=True)
    for name in ("train_gpt.py", "triton_kernels.py"):
        dest = snapshot / name
        if not dest.exists():
            dest.write_bytes((args.source / name).read_bytes())
    args.source = snapshot
    manifest = dict(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), shapes=SHAPES,
                    hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()},
                    nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True))
    (args.output / f"manifest_{args.phase}.json").write_text(json.dumps(manifest, indent=2))
    kernels = load_kernels(snapshot / "triton_kernels.py")
    print(json.dumps(manifest), flush=True)
    with torch.no_grad():
        (sweep if args.phase == "sweep" else cascade)(args, kernels)


if __name__ == "__main__":
    main()
