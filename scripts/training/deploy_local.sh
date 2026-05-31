#!/usr/bin/env bash
# Prototype: fully-automated localhost deploy of a LeRobot training run.
#
# Builds (or reuses) the lerobot-training Docker image, then runs a
# container with --gpus all and bind-mounts the user's HF cache and a
# host-side outputs directory. The training command is forwarded verbatim.
#
# Goals demonstrated:
#   - One artifact (the image) carries every dependency
#   - No host-Python venv state leaks into the run
#   - Checkpoints land on the host (bind-mount) so they survive container exit
#   - HF cache shared with the host so datasets and tokens are reused
#   - GPU pass-through is automatic via --gpus all + nvidia-container-toolkit
#   - tini PID-1 in the image means SIGTERM reaches lerobot-train cleanly
#
# Example:
#   ./scripts/training/deploy_local.sh \
#     lerobot-train --policy.type=act --dataset.repo_id=lerobot/pusht --steps=10
#
# Or skip args and the script runs the bundled smoke-test command.

set -euo pipefail

# ── Constants ────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="lerobot-training"
IMAGE_TAG="dev"
IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
DOCKERFILE="${REPO_ROOT}/docker/Dockerfile.training"
HF_CACHE_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"
OUTPUTS_HOST="${REPO_ROOT}/outputs"

# ── Helpers ──────────────────────────────────────────────────────────────────

log() { printf '\033[1;36m[deploy]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[deploy]\033[0m %s\n' "$*" >&2; }

# Check for required host-side prerequisites. Prints a clear remediation
# message and exits if anything is missing — no silent fallback.
check_prereqs() {
    local missing=0

    if ! command -v docker >/dev/null 2>&1; then
        err "docker not found on this host."
        err "Install Docker + nvidia-container-toolkit:"
        err "    sudo bash ${REPO_ROOT}/scripts/training/install_prereqs.sh"
        missing=1
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        err "nvidia-smi not found — no NVIDIA driver detected on this host."
        err "This prototype requires a CUDA-capable GPU on localhost."
        missing=1
    fi

    if [[ $missing -eq 1 ]]; then
        exit 2
    fi

    # Verify Docker can reach the daemon (catches "permission denied" before
    # we waste time on a build).
    if ! docker info >/dev/null 2>&1; then
        err "Docker daemon is unreachable. Try:"
        err "    sudo systemctl start docker"
        err "    sudo usermod -aG docker \$USER  # then log out and back in"
        exit 2
    fi

    # Verify nvidia-container-toolkit is wired into Docker. The simplest
    # probe is to ask Docker to run a stock cuda image with --gpus all;
    # if the runtime isn't registered, docker errors out clearly.
    if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 \
            nvidia-smi >/dev/null 2>&1; then
        err "Docker can't use the GPU. Likely missing nvidia-container-toolkit."
        err "Install it via:"
        err "    sudo bash ${REPO_ROOT}/scripts/training/install_prereqs.sh"
        exit 2
    fi
}

# Build the training image if it isn't already present, or if the
# Dockerfile or pyproject.toml has been modified since the image was built.
# Uses Docker's BuildKit cache to keep re-builds cheap.
build_image_if_needed() {
    local need_build=0

    if ! docker image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
        log "Image ${IMAGE_REF} not present — building..."
        need_build=1
    else
        # Compare the timestamp of pyproject.toml vs the image's creation
        # time. If pyproject changed since the image was built, rebuild.
        local image_created
        image_created=$(docker image inspect "${IMAGE_REF}" \
            --format '{{.Created}}' | xargs -I {} date -d {} +%s)
        local pyproject_mtime
        pyproject_mtime=$(stat -c %Y "${REPO_ROOT}/pyproject.toml")
        if [[ $pyproject_mtime -gt $image_created ]]; then
            log "pyproject.toml newer than image — rebuilding..."
            need_build=1
        fi
    fi

    if [[ $need_build -eq 1 ]]; then
        DOCKER_BUILDKIT=1 docker build \
            -f "${DOCKERFILE}" \
            -t "${IMAGE_REF}" \
            "${REPO_ROOT}"
        log "Build complete: ${IMAGE_REF}"
    else
        log "Image ${IMAGE_REF} is up-to-date (skipping build)"
    fi
}

# Forward the training command into a fresh container with the right
# mounts and environment. The container is removed on exit; outputs and
# HF cache survive on the host via bind-mounts.
run_training() {
    mkdir -p "${HF_CACHE_HOST}" "${OUTPUTS_HOST}"

    # If user is logged into HF on the host, forward their token. The
    # container's `huggingface-cli login`-equivalent is the env var.
    local hf_token_args=()
    if [[ -f "${HF_CACHE_HOST}/token" ]]; then
        local hf_token
        hf_token=$(cat "${HF_CACHE_HOST}/token")
        hf_token_args+=(-e "HF_TOKEN=${hf_token}")
    fi

    log "Running training in container..."
    log "  Image:   ${IMAGE_REF}"
    log "  HF cache: ${HF_CACHE_HOST}  →  /home/user_lerobot/.cache/huggingface"
    log "  Outputs:  ${OUTPUTS_HOST}    →  /lerobot/outputs"
    log "  Command:  $*"

    # --rm removes the container on exit (we keep outputs via the bind mount)
    # --gpus all exposes all NVIDIA GPUs via nvidia-container-toolkit
    # -t allocates a TTY so tqdm progress bars render usefully if attached
    # --network host keeps HF Hub access fast and avoids Docker bridge overhead
    docker run --rm \
        --gpus all \
        --network host \
        -t \
        -v "${HF_CACHE_HOST}:/home/user_lerobot/.cache/huggingface" \
        -v "${OUTPUTS_HOST}:/lerobot/outputs" \
        "${hf_token_args[@]}" \
        "${IMAGE_REF}" \
        "$@"
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    check_prereqs
    build_image_if_needed

    if [[ $# -eq 0 ]]; then
        log "No command given; running bundled smoke-test."
        log "(Override by passing your own: ./deploy_local.sh lerobot-train --policy.type=...)"
        run_training \
            lerobot-train \
            --policy.type=act \
            --dataset.repo_id=lerobot/pusht \
            --output_dir=/lerobot/outputs/smoke_act_pusht \
            --job_name=smoke_act_pusht \
            --steps=10 \
            --save_freq=10 \
            --batch_size=2 \
            --num_workers=0 \
            --policy.device=cuda \
            --wandb.enable=false
    else
        run_training "$@"
    fi

    log "Done. Outputs at ${OUTPUTS_HOST}"
}

main "$@"
