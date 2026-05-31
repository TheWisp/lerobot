#!/usr/bin/env bash
# One-shot installer for the host-side prereqs of the LeRobot training-deploy
# prototype: Docker Engine + nvidia-container-toolkit. Tested on Ubuntu 22.04
# and 24.04. Idempotent — re-running is safe.
#
# Run via:  sudo bash scripts/training/install_prereqs.sh
#
# After install:
#   - Log out and back in (so the docker group takes effect for your user), OR
#   - Run `newgrp docker` in your current shell.

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "This script needs root. Re-run with: sudo bash $0" >&2
    exit 1
fi

# Resolve the real user (so we can add them to the docker group, not root)
TARGET_USER="${SUDO_USER:-$USER}"

log() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }

# ── Docker Engine ────────────────────────────────────────────────────────────

if command -v docker >/dev/null 2>&1; then
    log "Docker already installed: $(docker --version)"
else
    log "Installing Docker Engine via the official Docker apt repo..."

    apt-get update
    apt-get install -y ca-certificates curl gnupg

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    UBUNTU_CODENAME=$(. /etc/os-release && echo "${VERSION_CODENAME}")
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME} stable" \
        > /etc/apt/sources.list.d/docker.list

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

    systemctl enable --now docker
    log "Docker installed: $(docker --version)"
fi

# Add target user to docker group so they don't need sudo for docker
if ! id -nG "${TARGET_USER}" | grep -qw docker; then
    usermod -aG docker "${TARGET_USER}"
    log "Added ${TARGET_USER} to the docker group."
    log "  → You must log out and back in (or run 'newgrp docker') for this to take effect."
else
    log "${TARGET_USER} is already in the docker group."
fi

# ── nvidia-container-toolkit (GPU pass-through) ──────────────────────────────

if dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q '^ii'; then
    log "nvidia-container-toolkit already installed."
else
    log "Installing nvidia-container-toolkit..."

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update
    apt-get install -y nvidia-container-toolkit

    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    log "nvidia-container-toolkit installed and Docker runtime configured."
fi

# ── Sanity check ─────────────────────────────────────────────────────────────

log "Verifying GPU access from a container..."
# Use the target user's docker socket access if possible; falls back to root.
SUDO_RUN="sudo -u ${TARGET_USER}"
if ! ${SUDO_RUN} docker info >/dev/null 2>&1; then
    log "  (target user can't reach docker yet — they need to log out/in;"
    log "   running the smoke check as root to verify the toolkit instead)"
    SUDO_RUN=""
fi

if ${SUDO_RUN} docker run --rm --gpus all \
        nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi --query-gpu=name \
        --format=csv,noheader >/dev/null 2>&1; then
    log "✓ GPU access from container works."
else
    log "✗ Container could not access the GPU. Check the NVIDIA driver version"
    log "  (must be ≥ 525.60 for CUDA 12.4). Run 'nvidia-smi' on the host."
    exit 1
fi

log ""
log "Done. Next steps:"
log "  1. If you weren't already in the docker group, log out and back in now."
log "  2. Then run:  bash scripts/training/deploy_local.sh"
