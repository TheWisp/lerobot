# Docker

This directory contains Dockerfiles for running LeRobot in containerized environments. Both images are **built nightly from `main`** and published to Docker Hub with the full environment pre-baked — no dependency setup required.

## Pre-built Images

```bash
# CPU-only image (based on Dockerfile.user)
docker pull huggingface/lerobot-cpu:latest

# GPU image with CUDA support (based on Dockerfile.internal)
docker pull huggingface/lerobot-gpu:latest
```

## Quick Start

The fastest way to start training is to pull the GPU image and run `lerobot-train` directly. This is the same environment used for all of our CI, so it is a well-tested, batteries-included setup.

```bash
docker run -it --rm --gpus all --shm-size 16gb huggingface/lerobot-gpu:latest

# inside the container:
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
```

## Dockerfiles

### `Dockerfile.user` (CPU)

A lightweight image based on `python:3.12-slim`. Includes all Python dependencies and system libraries but does not include CUDA — there is no GPU support. Useful for exploring the codebase, running scripts, or working with robots, but not practical for training.

### `Dockerfile.internal` (GPU)

A CUDA-enabled image based on `nvidia/cuda`. This is the image for training — mostly used for internal interactions with the GPU cluster.

### `Dockerfile.training` (GPU, fork-only)

The image every training run actually uses. CI publishes it to GHCR on each push
to `main`, the GUI can build it locally as `lerobot-training:dev-local`, and
`scripts/training/setup_host.sh` pulls or builds it on a new host. What the image
must contain is declared once, by the fork-only `training-image` extra in
`pyproject.toml` — adding a policy's dependency is a line there, not here.

#### The layer order is a contract, not a style choice

Dependencies install **before** any project source is copied:

```dockerfile
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --extra training-image --no-cache --no-install-project

COPY setup.py README.md MANIFEST.in ./
COPY src/ src/
RUN uv sync --locked --extra training-image --no-cache
```

Docker keys a layer on the content of what it copies. Anything copied above that
first `uv sync` re-couples the dependency install — the single most expensive
step in the build — to files that change on every commit.

`--no-install-project` is what makes the split possible: uv installs the full
dependency closure without the project's own source present. The resulting layer
therefore contains **no first-party code at all**, which is why it can never be
stale with respect to yours. The second sync installs the project *editable* (a
`.pth` pointing at `/lerobot/src`), so the copied source is what executes.

**Measured** (see issue #98 for the harness): with source copied first, appending
one comment to a Python file cost **1427s of a 1440s rebuild**. With the split,
the same edit rebuilds in **30s**. The dependency layer is also ~8.4 GB, so the
ordering governs republish and re-pull cost too — a code-only bump moves tens of
megabytes instead of gigabytes, on CI and on every host that pulls.

Two invariants follow, both enforced by
`tests/gui/training/test_image_layers.py` and, where Docker is available, by
`tests/gui/training/test_image_layer_cache.py` (which builds twice and compares
layer identity rather than trusting a parse):

- **Only `pyproject.toml` and `uv.lock` may be copied before the dependency
  sync.** Ordering alone is not sufficient — a `COPY . .` above it satisfies
  every ordering check while destroying the property.
- **Dependencies must be declared only in those two files.** `setup.py` here is a
  real build script (it parses the version and rewrites README media links), but
  nothing install-relevant may move into it, or the dependency layer will stop
  rebuilding when it should.

#### Why `uv sync --no-cache`

Not to be confused with `docker build --no-cache`. This one stops uv writing its
wheel cache, which — because the image sets `HOME=/home/user_lerobot` — would
otherwise land inside the layer and add gigabytes. uv's documented alternative,
`--mount=type=cache`, requires BuildKit; this file builds with the classic
builder. The cost is a re-download whenever dependencies genuinely change, which
is the rare case now that the layer is no longer invalidated by code edits.

#### Build-time policy import check

The build runs `check_policy_registry_health` before it finishes, so a
mis-pinned transitive dependency fails the build in seconds of CI rather than on
a GPU pod after a pull. The same probe backs the GUI's startup health check.

#### No `ENTRYPOINT`

Signal handling comes from `docker run --init`, which makes Docker's tini PID 1
so SIGTERM reaches the trainer and DataLoader workers get reaped. A nested tini
inside the image produced a "Tini is not running as PID 1" warning, so the
launcher owns this, not the image. Callers that spawn this image must pass
`--init`.

## Usage

### Running a pre-built image

```bash
# CPU
docker run -it --rm huggingface/lerobot-cpu:latest

# GPU
docker run -it --rm --gpus all --shm-size 16gb huggingface/lerobot-gpu:latest
```

### Building locally

From the repo root:

```bash
# CPU
docker build -f docker/Dockerfile.user -t lerobot-user .
docker run -it --rm lerobot-user

# GPU
docker build -f docker/Dockerfile.internal -t lerobot-internal .
docker run -it --rm --gpus all --shm-size 16gb lerobot-internal
```

### Multi-GPU training

To select specific GPUs, set `CUDA_VISIBLE_DEVICES` when launching the container:

```bash
# Use 4 GPUs
docker run -it --rm --gpus all --shm-size 16gb \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  huggingface/lerobot-gpu:latest
```

### USB device access (e.g. robots, cameras)

```bash
docker run -it --device=/dev/ -v /dev/:/dev/ --rm huggingface/lerobot-cpu:latest
```
