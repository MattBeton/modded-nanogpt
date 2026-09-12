# Profiling-ready image for the nanogpt speedrun (RunPod-compatible).
#   docker buildx build --platform linux/amd64 -f docker/Dockerfile -t <user>/modded-nanogpt-prof:latest .
# See docker/README.md for RunPod template settings and the run recipes.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PATH=/usr/local/cuda/bin:/usr/local/cuda/nsight-compute:${PATH}

# Ubuntu 24.04 ships Python 3.12. ncu / nsys come from the CUDA apt repo already configured in the base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git curl rsync vim htop tmux less openssh-server ca-certificates \
        cuda-nsight-compute-12-8 cuda-nsight-systems-12-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && ln -sf "$(ls -d /usr/local/cuda/nsight-compute-*/ncu | head -1)" /usr/local/bin/ncu \
    && ln -sf "$(ls -d /usr/local/cuda/nsight-systems-*/bin/nsys | head -1)" /usr/local/bin/nsys \
    && ncu --version | tail -1 && nsys --version

# torch pinned to the CUDA 12.8 build so torch.version.cuda matches the toolkit in this image
# (train_gpt.py's torch.cuda._compile_kernel uses /usr/local/cuda/include).
WORKDIR /opt/modded-nanogpt
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch==2.10 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt \
    && python -c "import torch, triton; v = torch.version.cuda; print(torch.__version__, triton.__version__, v); assert v.startswith('12.8'), v"

# Pre-fetch the Flash Attention 3 build used by train_gpt.py so a fresh pod does not hit the HF hub at start.
RUN python -c "from kernels import get_kernel; get_kernel('kernels-community/flash-attn3', version=1); print('flash-attn3 cached')"

COPY . .
RUN chmod +x docker/start.sh profile/ncu_polar_express.sh

# Persistent caches live on the RunPod volume (/workspace) when present, see docker/start.sh.
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor \
    TRITON_CACHE_DIR=/workspace/.cache/triton

EXPOSE 22
CMD ["/opt/modded-nanogpt/docker/start.sh"]
