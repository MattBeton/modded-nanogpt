#!/bin/bash
# Container entrypoint (RunPod convention): sshd keyed from $PUBLIC_KEY, code copied to the persistent
# volume on first boot, container kept alive. Also fine to run interactively: docker run ... bash
set -euo pipefail

# --- ssh ---
mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A >/dev/null
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
# Container env vars are not inherited by ssh sessions; export them for login shells.
env | grep -E '^(PATH|LD_LIBRARY_PATH|HF_HOME|TORCHINDUCTOR_CACHE_DIR|TRITON_CACHE_DIR|PIP_BREAK_SYSTEM_PACKAGES|CUDA_HOME|NVIDIA_[A-Z_]+)=' \
    | sed 's/^/export /' > /etc/profile.d/modded-nanogpt.sh
/usr/sbin/sshd

# --- persistent volume ---
# On RunPod /workspace is the (network) volume; the repo is copied there once so edits, data, traces and
# the inductor/triton caches (the ~7 min compile) survive pod restarts. Without a volume the same paths
# simply live in the container filesystem.
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
if [ ! -d /workspace/modded-nanogpt ]; then
    cp -r /opt/modded-nanogpt /workspace/modded-nanogpt
    echo "copied repo to /workspace/modded-nanogpt"
fi
ln -sfn /workspace/modded-nanogpt /root/modded-nanogpt

# Optional: DOWNLOAD_SHARDS=8 downloads the first N fineweb shards at boot if missing.
if [ -n "${DOWNLOAD_SHARDS:-}" ]; then
    (cd /root/modded-nanogpt && python data/cached_fineweb10B.py "$DOWNLOAD_SHARDS" > data_download.log 2>&1 &)
fi

echo "modded-nanogpt profiling image ready: $(nvidia-smi -L 2>/dev/null | wc -l) GPU(s)"
exec sleep infinity
