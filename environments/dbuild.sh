#!/usr/bin/env bash
# Build the xbank-transfer image. Run from anywhere; resolves the repo root
# relative to this script's own location so it works regardless of cwd.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-xbank-transfer:latest}"

cd "${REPO_DIR}"
docker build -f environments/Dockerfile -t "${IMAGE_NAME}" .

echo "Built ${IMAGE_NAME}"
docker images "${IMAGE_NAME%%:*}"
