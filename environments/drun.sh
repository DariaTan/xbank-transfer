#!/usr/bin/env bash
# Start the xbank-transfer container as a persistent background service
# (kept alive via `tail -f /dev/null`, not an interactive shell) so a
# training run survives an SSH disconnect. Use dexec.sh to get a shell in it.
#
# Mounts:
#   repo root                               -> /app                    (project code, editable live)
#   /mnt/storage/d.tanyushkina/transactions -> /app/data               (read-write, so downstream embeds can be
#                                                                        written under /app/data/embeds; be careful
#                                                                        not to touch the raw parquet files here)
#   /mnt/storage/d.tanyushkina/hf_cache     -> /hf_cache               (read-write, survives container recreation)
#
# The HF cache mount matters for Chronos-2: it's a real pretrained
# checkpoint pulled from the Hub (Apache-2.0, public weights, not
# proprietary data), and without this mount it silently re-downloads
# every time the container is recreated. Mounted at /hf_cache rather
# than the default /root/.cache/huggingface because the container now
# runs as the host user (--user below), and /root is mode 700 -- not
# even traversable by a non-root uid, which would silently break HF_HOME.
#
# --user "$(id -u):$(id -g)" runs the container as the host user instead
# of root (added 2026-09-14, after repeatedly finding root-owned files
# under /app/data from earlier root-run containers, which then blocked
# writes once ownership was fixed manually) -- every new file the
# container creates under /app/data or /hf_cache now lands owned by the
# host user from the start, no manual chown needed going forward.
# HOME=/tmp for the same reason: the base image bakes in HOME=/root, and
# that env var isn't reset just because --user changes the running uid --
# left as /root, any library that writes cache/config to $HOME (matplotlib
# font cache, ~/.local, ~/.triton, ...) would hit permission-denied since
# /root is mode 700. /tmp is container-local (doesn't survive a recreate,
# unlike the two bind mounts above) but that only costs re-warming a few
# harmless caches, never real data.
#
# PYTHONPATH=/app/src makes `data.*`/`models.*`/`training.*` importable
# from anywhere in the container (no per-script sys.path hack needed) --
# src/training/ holds the smoke_*.py / train_*.py entry-point scripts
# directly, run as e.g. `python src/training/train_thp.py`.
#
# Port 6006 (host) -> 6006 (container) for TensorBoard -- see TRAINING.md
# for the access command (docker exec + SSH port forward).
#
# This host has 2x RTX A5000 (24GB each) shared with other users' containers.
# Override GPUS to grab just one, e.g.:  GPUS='"device=0"' ./drun.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="/mnt/storage/d.tanyushkina/transactions"
HF_CACHE_DIR="/mnt/storage/d.tanyushkina/hf_cache"
IMAGE_NAME="${IMAGE_NAME:-xbank-transfer:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"
GPUS="${GPUS:-all}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' is already running."
    exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' exists but is stopped -- starting it."
    docker start "${CONTAINER_NAME}"
    exit 0
fi

mkdir -p "${HF_CACHE_DIR}"

docker run -d \
    --name "${CONTAINER_NAME}" \
    --user "$(id -u):$(id -g)" \
    --gpus "${GPUS}" \
    --shm-size=16g \
    -v "${REPO_DIR}:/app" \
    -v "${DATA_DIR}:/app/data" \
    -v "${HF_CACHE_DIR}:/hf_cache" \
    -e HF_HOME=/hf_cache \
    -e HOME=/tmp \
    -e PYTHONPATH=/app/src \
    -p 6006:6006 \
    -w /app \
    "${IMAGE_NAME}" \
    tail -f /dev/null

echo "Started '${CONTAINER_NAME}'. Use ./environments/dexec.sh for a shell."
