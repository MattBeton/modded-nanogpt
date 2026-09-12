"""Generate a small synthetic Kineto-style chrome trace to test analyze_trace.py without a GPU."""
import json
import sys

COMPUTE, NCCL = 7, 20
events, corr_ctr = [], [0]
cpu_t = [1000.0]        # CPU clock (us)
gpu_t = {COMPUTE: 1200.0, NCCL: 1200.0}  # GPU stream clocks


def annot(name, tid=1):
    class _A:
        def __enter__(self):
            self.ts = cpu_t[0]
            return self
        def __exit__(self, *a):
            events.append(dict(ph="X", cat="user_annotation", name=name, pid=1, tid=tid, ts=self.ts, dur=cpu_t[0] - self.ts, args={}))
    return _A()


def launch(name, dur, stream=COMPUTE, cpu_cost=4.0, gpu_gap=0.0, cat="kernel"):
    corr_ctr[0] += 1
    c = corr_ctr[0]
    events.append(dict(ph="X", cat="cuda_runtime", name="cudaLaunchKernel", pid=1, tid=1, ts=cpu_t[0], dur=cpu_cost, args=dict(correlation=c)))
    cpu_t[0] += cpu_cost + 1.0
    start = max(gpu_t[stream] + gpu_gap, cpu_t[0] + 5.0)  # kernel can't start before launch
    if stream == NCCL:
        start = max(start, gpu_t[COMPUTE])  # collectives wait on the compute stream (like real NCCL event deps)
    events.append(dict(ph="X", cat=cat, name=name, pid=0, tid=stream, ts=start, dur=dur, args=dict(correlation=c, stream=stream)))
    gpu_t[stream] = start + dur


def sync_to_gpu():
    cpu_t[0] = max(cpu_t[0], max(gpu_t.values()))


for step in range(4):
    with annot("step"):
        with annot("dataload"):
            cpu_t[0] += 200
        with annot("fwd"):
            for i in range(6):
                launch("nvjet_tst_gemm_bf16", 300)
                launch("flash_fwd_kernel", 250)
                launch("triton_poi_fused_add_mul_3", 40)
        with annot("bwd"):
            for i in range(6):
                launch("flash_bwd_kernel", 500)
                launch("nvjet_tst_gemm_bf16", 300)
                launch("triton_red_fused_sum_1", 60)
        with annot("opt"):
            for label in ["scalars", "mlp_bank", "qk_bank", "vo_bank", "lm_head"]:
                with annot(f"opt/scatter/{label}"):
                    launch("ncclDevKernel_ReduceScatter_Sum_bf16", 400 if "bank" in label else 50, stream=NCCL)
            for label in ["scalars", "qk_bank", "vo_bank", "mlp_bank", "lm_head"]:
                with annot(f"opt/wait/{label}"):
                    cpu_t[0] += 3
                with annot(f"opt/update/{label}"):
                    if "bank" in label:
                        with annot("pe"):
                            launch("triton_poi_fused_lerp_0", 30)
                            for it in range(5):
                                launch("XXT_kernel", 25, gpu_gap=8)
                                launch("ba_plus_cAA_kernel", 20, gpu_gap=8)
                                launch("nvjet_tst_gemm_bf16", 60, gpu_gap=8)
                        with annot("normuon_vr"):
                            launch("triton_per_fused_mean_2", 15)
                        with annot("cautious_update"):
                            launch("triton_poi_fused_bitwise_5", 40)
                    else:
                        launch("triton_poi_fused_adam_0", 20)
                with annot(f"opt/gather/{label}"):
                    launch("ncclDevKernel_AllGather", 350 if "bank" in label else 40, stream=NCCL)
            with annot("opt/finalize"):
                sync_to_gpu()
                launch("_transpose_copy_kernel", 80)
        with annot("fp8_quant"):
            launch("quantize_transpose_mlp_down_weights_kernel", 120)
        sync_to_gpu()

json.dump(dict(traceEvents=events), open(sys.argv[1], "w"))
print(f"wrote {len(events)} events to {sys.argv[1]}")
