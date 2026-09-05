#!/usr/bin/env bash
# A throwaway SSH host, for exercising the remote-training path without a rig.
#
#   scripts/training/ssh_test_host.sh up      # start it, print the host spec
#   scripts/training/ssh_test_host.sh status  # is it up, and does sudo behave
#   scripts/training/ssh_test_host.sh down    # remove it
#
# It is an Ubuntu container running sshd, reachable at tester@127.0.0.1:2299
# with your SSH key. Nothing about it is special to this machine: the GUI
# connects to it the way it connects to any host, which is the point — the SSH
# path can then be driven end to end without a second machine, and without
# anyone's real credentials.
#
# It is built to look like a workstation someone set up themselves, because that
# is the case the rig represents and the one the code finds hardest:
#
#   - login by key only, no password over SSH;
#   - a Docker the SSH user can actually use, with the NVIDIA runtime;
#   - a GPU visible to `nvidia-smi` (passed through, when this host has one);
#   - sudo that works but demands a password.
#
# "Can actually use" is the point, and the first version of this file got it
# wrong: it mounted the docker CLI and created a group named `docker`, which
# satisfied every check the provisioning script made while the host could not
# start a single container. A fixture built against the checks rather than the
# requirement only proves the checks agree with themselves. It now shares this
# machine's docker daemon, so `docker info` and `docker run --gpus all` answer
# for real.
#
# That socket mount gives anything inside the container root-equivalent control
# of this machine's Docker. It is a throwaway fixture bound to loopback, and
# that is the trade being made deliberately.
#
# That last property is what a test host normally cannot give you: a password
# you may know. `up` generates one and prints it, so the accepted-password path
# can be driven without holding a real operator's credential.
#
# What it does NOT cover: installing Docker or the NVIDIA toolkit, which is the
# branch of install_prereqs.sh that this host skips. Those steps end in
# `systemctl`, and there is no init inside the container. A host that needs them
# is still a real-machine test.
#
# The Docker CLI is bind-mounted from this machine rather than installed, so
# `command -v docker` succeeds. There is no daemon inside the container and none
# is needed: provisioning skips the container GPU smoke, and nothing else here
# runs a container.

set -euo pipefail

NAME="lerobot-ssh-test"
PORT="${LEROBOT_SSH_TEST_PORT:-2299}"
LOGIN="tester"
PUBKEY="${LEROBOT_SSH_TEST_KEY:-${HOME}/.ssh/id_ed25519.pub}"
IMAGE="ubuntu:24.04"

log() { printf '\033[1;36m[ssh-test-host]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[ssh-test-host]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# A password is generated per `up` rather than fixed in this file. A constant
# would be a credential checked into the repository — harmless for a container
# bound to loopback, but it would train the secret scanners to ignore exactly
# the shape they exist to catch.
#
# Read a fixed number of bytes and trim afterwards, rather than letting `head`
# close the pipe on a reader of /dev/urandom: that raises SIGPIPE upstream, and
# under `pipefail` it takes the script with it.
generate_password() {
    local hex
    hex=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
    printf '%s' "${hex:0:20}"
}

require_docker() {
    command -v docker >/dev/null 2>&1 || die "docker is not installed on this machine."
    docker info >/dev/null 2>&1 || die "the Docker daemon is not reachable."
}

# --gpus all fails outright where no NVIDIA runtime is configured, and the
# provisioning script's host GPU check then fails inside the container. Warn
# rather than pretend: everything up to that check is still exercised.
gpu_args() {
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
        printf -- '--gpus all'
    fi
}

cmd_up() {
    local password=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --password) password="${2:-}"; shift 2 ;;
            *) die "unknown option for up: $1" ;;
        esac
    done
    require_docker
    [[ -f ${PUBKEY} ]] || die "no public key at ${PUBKEY}. Set LEROBOT_SSH_TEST_KEY."
    [[ -n ${password} ]] || password=$(generate_password)

    local gpus
    gpus=$(gpu_args)
    if [[ -z ${gpus} ]]; then
        log "No NVIDIA container runtime here — the host GPU check will fail."
    fi

    DOCKER_GID=$(stat -c %g /var/run/docker.sock)
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    docker run -d --name "${NAME}" ${gpus} -p "127.0.0.1:${PORT}:22" \
        -v /usr/bin/docker:/usr/bin/docker:ro \
        -v /var/run/docker.sock:/var/run/docker.sock \
        "${IMAGE}" sleep infinity >/dev/null

    log "Installing sshd and the toolkit inside the container..."
    docker exec -e DOCKER_GID="${DOCKER_GID}" "${NAME}" bash -c "
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq openssh-server sudo curl gnupg ca-certificates
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            > /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update -qq
        apt-get install -y -qq nvidia-container-toolkit
        # The group the provisioning script adds the user to. Absent in a
        # container that never installed Docker's package, and its absence
        # fails the script well after the point under test.
        # The group must carry the socket's real gid, or the user still cannot
        # reach the daemon and the fixture is lying again.
        groupadd -g ${DOCKER_GID} docker 2>/dev/null || groupadd -f docker
        useradd -m -s /bin/bash ${LOGIN}
        usermod -aG sudo,docker ${LOGIN}
        mkdir -p /home/${LOGIN}/.ssh /run/sshd
        chmod 700 /home/${LOGIN}/.ssh
    " >/dev/null

    printf '%s:%s' "${LOGIN}" "${password}" | docker exec -i "${NAME}" chpasswd
    docker cp "${PUBKEY}" "${NAME}:/home/${LOGIN}/.ssh/authorized_keys" >/dev/null
    docker exec "${NAME}" bash -c "
        chown -R ${LOGIN}:${LOGIN} /home/${LOGIN}/.ssh
        chmod 600 /home/${LOGIN}/.ssh/authorized_keys
        /usr/sbin/sshd
    "

    # Each `up` mints a new host key. Without this the stale entry makes ssh
    # refuse the host outright, which reads like a connection bug.
    ssh-keygen -q -f "${HOME}/.ssh/known_hosts" -R "[127.0.0.1]:${PORT}" >/dev/null 2>&1 || true

    log "Ready."
    printf '\n  host spec   %s@127.0.0.1:%s\n' "${LOGIN}" "${PORT}"
    printf '  password    %s\n' "${password}"
    printf '  workdir     /home/%s/.lerobot-training\n\n' "${LOGIN}"
    log "Add it in the GUI under Model → hosts, then start a run against it."
    log "The password is not stored anywhere. Re-run 'up' for a new one."
}

cmd_status() {
    require_docker
    if ! docker ps --filter "name=^/${NAME}$" --format '{{.Names}}' | grep -q .; then
        log "not running. Start it with: $0 up"
        return 1
    fi
    log "container up, checking what an SSH client sees..."
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "${PORT}" "${LOGIN}@127.0.0.1" '
        echo "  login as     $(whoami)"
        echo "  docker       $(docker --version 2>/dev/null || echo MISSING)"
        echo "  gpu          $(nvidia-smi -L 2>/dev/null | head -1 || echo "not visible")"
        echo "  toolkit      $(dpkg -l nvidia-container-toolkit 2>/dev/null | grep -c ^ii) installed"
        echo "  docker info  $(docker info >/dev/null 2>&1 && echo usable || echo UNUSABLE)"
        echo "  sudo -n      $(sudo -n true 2>&1 | head -1 || true)"
    '
}

cmd_down() {
    require_docker
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
    ssh-keygen -q -f "${HOME}/.ssh/known_hosts" -R "[127.0.0.1]:${PORT}" >/dev/null 2>&1 || true
    log "removed."
}

case "${1:-}" in
    up) shift; cmd_up "$@" ;;
    status) cmd_status ;;
    down) cmd_down ;;
    -h|--help|"") usage 0 ;;
    *) usage 1 ;;
esac
