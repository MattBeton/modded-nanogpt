#!/usr/bin/env python3
"""Independently check that the swept tile configs change no arithmetic.

GRAM_SWEEP.md reports bitwise agreement on sampled operands. That is plausible here but is a
property of the DATA, not of the transformation: changing BLOCK_SIZE_K regroups the fp32
partial sums in the kernel's `for k in range(cdiv(K, BLOCK_SIZE_K))` accumulation, which is
only exact while every partial sum stays exactly representable (bf16 x bf16 products carry a
16-bit mantissa into a 24-bit fp32 accumulator). It would stop being exact if operand dynamic
range widened. So re-run this whenever the cascade's scaling changes.

Run once per tile state and compare the two dumps:

  python profile/verify_gram_tiles.py --src <tree>/train_gpt.py --out before.pt
  python profile/gram_tiles.py --file <tree>/triton_kernels.py
  python profile/verify_gram_tiles.py --src <tree>/train_gpt.py --out after.pt
  python profile/verify_gram_tiles.py --compare before.pt after.pt
"""
import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_polar_express import SHAPES_BY_VARIANT, call_cascade, load_cascade, make_inputs  # noqa: E402


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def collect(src, steps=4):
    """Run each bank for several sequential steps so state evolution is covered, not just step 1."""
    device = torch.device("cuda")
    cascade, variant = load_cascade(Path(src))
    out = {}
    for label, shape in SHAPES_BY_VARIANT[variant].items():
        torch.manual_seed(1234)
        grad, state, scalars = make_inputs(shape, device, variant)
        split = shape[-2] > 1024
        for step in range(steps):
            v = call_cascade(cascade, variant, grad.clone(), state, scalars, split)
            out[f"{label}/step{step}/out"] = v.detach().clone()
            out[f"{label}/step{step}/state"] = state.detach().clone()
    return out, variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    if args.compare:
        a, b = (torch.load(p, map_location="cpu") for p in args.compare)
        assert a.keys() == b.keys(), "dumps cover different tensors"
        bad = []
        for k in a:
            if not torch.equal(a[k], b[k]):
                d = (a[k].float() - b[k].float()).abs()
                bad.append((k, d.max().item(), d.mean().item()))
        print(f"compared {len(a)} tensors across {len({k.split('/')[0] for k in a})} banks")
        if not bad:
            print("RESULT: BITWISE IDENTICAL")
            return 0
        print(f"RESULT: {len(bad)} tensors DIFFER")
        for k, mx, mn in bad[:10]:
            print(f"   {k:<34} max|d|={mx:.3e} mean|d|={mn:.3e}")
        return 1

    out, variant = collect(args.src, args.steps)
    torch.save(out, args.out)
    print(f"variant={variant}  wrote {len(out)} tensors -> {args.out}")
    for k in sorted(out):
        if k.endswith("/out") and "step0" in k:
            print(f"   {k:<34} {tuple(out[k].shape)}  sha={digest(out[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
