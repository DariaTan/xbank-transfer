#!/usr/bin/env bash
# Attach an interactive shell to the running xbank-transfer container.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"
docker exec -it "${CONTAINER_NAME}" bash
