# Profiling image

One image with everything needed to run and profile the speedrun: CUDA 12.8 toolkit (no cuDNN layer; torch brings its own), torch 2.10 (cu128),
triton, the `kernels` FA3 build pre-fetched, `ncu` + `nsys`, sshd, and this repo (including `profile/`)
at `/opt/modded-nanogpt`.

## Build & push

Automatic: `.github/workflows/docker-image.yml` builds and pushes `<DOCKERHUB_USERNAME>/modded-nanogpt-prof:latest`
on every push to the `profiling` branch of your fork (also runnable manually from the Actions tab).
Set the repo secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (Docker Hub → Account settings →
Personal access tokens, read/write). First build ~15-20 min; later ones reuse the cached layers, and
only the final `COPY . .` layer changes when you edit code.

Manual, from any machine with Docker and ~40 GB free disk:

```bash
docker login
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t <dockerhub-user>/modded-nanogpt-prof:latest --push .
```

## RunPod template

* Container image: `<dockerhub-user>/modded-nanogpt-prof:latest`
* Container start command: leave empty (image `CMD` runs `docker/start.sh`)
* Expose TCP port `22`; RunPod injects your SSH key as `$PUBLIC_KEY`, `start.sh` installs it and starts sshd
* Volume mount path: `/workspace` (a network volume is worth it: repo copy, fineweb shards, traces and the
  inductor/triton caches — i.e. the ~7 min compile — all persist there)
* Optional env: `DOWNLOAD_SHARDS=8` to fetch the first 8 fineweb shards at boot if they are missing

On first boot the repo is copied to `/workspace/modded-nanogpt` (symlinked at `~/modded-nanogpt`);
later boots keep whatever is on the volume. Pick an **8×H100** pod for the real step profile; a 1×H100
pod is enough for the Polar Express microbenchmark / ncu work.

## Recipes (inside the pod)

```bash
cd ~/modded-nanogpt
python data/cached_fineweb10B.py 8                          # if not using DOWNLOAD_SHARDS

# 8 GPU: profiler trace of steps 20-25 (stage 1) and 900-905 (stage 3), exits after
PROFILE_STEPS="20:26,900:906" ./run.sh
python profile/analyze_trace.py profiles/<run_id>/ --pe-detail

# inductor-generated code for every compiled graph (count kernels, see what polar_express became)
TORCH_LOGS="output_code" torchrun --standalone --nproc_per_node=8 train_gpt.py 2> inductor.log
grep -c "^def triton_" inductor.log        # rough kernel count

# single GPU: Polar Express timing / launch gaps / hardware counters
python profile/bench_polar_express.py --trace pe.json
profile/ncu_polar_express.sh              # needs counter access, see below

# nsys timeline of the same (no counters needed)
nsys profile -o pe --trace=cuda,nvtx python profile/bench_polar_express.py
```

`ncu` needs `NVreg_RestrictProfilingToAdminUsers=0` on the *host* driver; on shared clouds it often
fails with `ERR_NVGPUCTRPERM`. Secure-cloud / bare-metal style pods usually allow it, community pods
often don't. `nsys` and `torch.profiler` do not need it.

## Local sanity check (no GPU)

```bash
docker run --rm modded-nanogpt-prof:latest python -c "import torch, triton, kernels; print(torch.__version__)"
docker run --rm modded-nanogpt-prof:latest python profile/analyze_trace.py --help
```
