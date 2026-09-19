#!/usr/bin/env python3
"""Rename the process/thread tracks in a Kineto trace so the GPU is findable in Perfetto.

Kineto names every process "python", including the GPU devices (which get pid = device index),
so a trace opens as several identical "python" groups and the kernels look missing. This rewrites
the process_name / thread_name metadata to say what each track actually is. Nothing else changes.

  python profile/relabel_trace.py profiles/pe_trace.json            # -> profiles/pe_trace.labeled.json
  python profile/relabel_trace.py profiles/*.json
"""
import json
import sys
from pathlib import Path


def relabel(path: Path) -> Path:
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]

    # A pid is a GPU device iff its events are GPU-side (kernels/memcpy/memset live only there).
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
    gpu_pids, cpu_pids = set(), set()
    for e in events:
        if e.get("ph") != "X":
            continue
        (gpu_pids if e.get("cat") in gpu_cats else cpu_pids).add(e.get("pid"))
    gpu_pids -= {None}
    cpu_pids -= gpu_pids | {None, -1, "Spans"}

    n = 0
    for e in events:
        if e.get("ph") != "M":
            continue
        pid, args = e.get("pid"), e.get("args", {})
        if e.get("name") == "process_name":
            if pid in gpu_pids:
                args["name"] = f"GPU {pid} -- KERNELS HERE"
            elif pid in cpu_pids:
                args["name"] = f"CPU python (pid {pid})"
            else:
                continue
            n += 1
        elif e.get("name") == "thread_name" and pid in gpu_pids:
            args["name"] = f"GPU {pid} / {args.get('name', '').strip()}"
            n += 1

    # Drop the empty device pids Kineto declares for GPUs that were never used; they add
    # nothing but identically-named collapsed rows.
    used = gpu_pids | cpu_pids | {-1, "Spans"}
    trace["traceEvents"] = [e for e in events
                            if e.get("ph") != "M" or e.get("pid") in used]

    out = path.with_suffix(".labeled.json")
    out.write_text(json.dumps(trace))
    print(f"{path.name}: GPU pids {sorted(gpu_pids)}, CPU pids {sorted(cpu_pids)}, "
          f"{n} tracks renamed -> {out}")
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["profiles/pe_trace.json"]:
        relabel(Path(arg))
