#!/usr/bin/env python3
"""Apply (or revert) the swept Gram/polynomial tile configurations in a triton_kernels.py.

The production launchers pick their Triton config from K alone, so QK (M=256, batch 6) and VO
(M=768, batch 2) are forced to share one config -- hardcoded from an autotune done against the
older, larger banks. profile/GRAM_SWEEP.md swept per-shape and found real headroom on the small
banks. This rewrites the config blocks to be shape-aware. Kernel bodies are untouched.

Selected configs, as (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps):

  XXT         K==768        (64, 64, 128, 3, 4)   QK Gram 1.77x, VO Gram 1.13x
  XTX         unchanged                           MLP Gram measured 1.001x == noise
  ba_plus_cAA M==256        (64, 64,  64, 4, 4)   QK polynomial 1.71x
  ba_plus_cAA otherwise     (64, 64, 128, 3, 4)   VO 1.23x, MLP 1.10x

All tiles are square, preserving the alignment assumption of the triangular skip/mirror logic
in the kernels (rectangular tiling needs a separate correctness review -- see GRAM_SWEEP.md).

  python profile/gram_tiles.py --file <tree>/triton_kernels.py          # apply
  python profile/gram_tiles.py --file <tree>/triton_kernels.py --revert # restore from .orig
  python profile/gram_tiles.py --file <tree>/triton_kernels.py --check  # report state only
"""
import argparse
import shutil
import sys
from pathlib import Path

# (anchor, original block, tuned block). Anchors are unique so a silent mis-patch is impossible.
PATCHES = [
    (
        "XXT",
        """    # Hardcoded configs based on H100 autotuning
    if K == 768:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
        num_stages, num_warps = 4, 8
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 128
        num_stages, num_warps = 4, 8

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)""",
        """    # Swept per-shape on H100 (profile/GRAM_SWEEP.md); QK and VO both take the K==768 path.
    if K == 768:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 128
        num_stages, num_warps = 3, 4
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 128
        num_stages, num_warps = 4, 8

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)""",
    ),
    (
        "ba_plus_cAA",
        """    # Hardcoded config based on H100 autotuning (M=768)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
    num_stages, num_warps = 4, 8""",
        """    # Swept per-shape on H100 (profile/GRAM_SWEEP.md). M==256 is the QK bank's Gram.
    if M == 256:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 64
        num_stages, num_warps = 4, 4
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 128
        num_stages, num_warps = 3, 4""",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="path to the triton_kernels.py to patch")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    orig = path.with_suffix(".py.orig")
    text = path.read_text()

    applied = [name for name, _, tuned in PATCHES if tuned in text]
    pending = [name for name, base, _ in PATCHES if base in text]

    if args.check:
        print(f"{path}: tuned={applied or '-'}  original={pending or '-'}  backup={'yes' if orig.exists() else 'no'}")
        return 0

    if args.revert:
        if not orig.exists():
            print(f"no backup at {orig}; nothing to revert", file=sys.stderr)
            return 1
        shutil.copy2(orig, path)
        print(f"reverted {path} from {orig}")
        return 0

    if applied and not pending:
        print(f"already tuned: {applied}")
        return 0

    missing = [name for name, base, _ in PATCHES if base not in text]
    if missing:
        print(f"ERROR: config block not found for {missing} in {path}.\n"
              f"The launcher source differs from what was swept -- re-read it before patching.",
              file=sys.stderr)
        return 1

    if not orig.exists():
        shutil.copy2(path, orig)
    for name, base, tuned in PATCHES:
        assert text.count(base) == 1, f"{name}: expected exactly one config block"
        text = text.replace(base, tuned)
    path.write_text(text)
    print(f"patched {path} ({', '.join(n for n, _, _ in PATCHES)}); backup at {orig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
