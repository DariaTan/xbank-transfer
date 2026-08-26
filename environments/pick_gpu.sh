#!/usr/bin/env bash
# Prints current free memory / utilization for both host GPUs -- a manual
# aid for picking CUDA_VISIBLE_DEVICES before launching a training job.
# Both GPUs are shared with other users' containers on this host, so this
# reflects everyone's usage, not just ours; run it fresh before each launch
# rather than trusting a stale reading. Run on the host, not via docker
# exec -- nvidia-smi shows the same physical-GPU-wide numbers either way,
# host is just one less hop.
set -euo pipefail

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
while IFS=',' read -r idx used total util; do
    idx=$(echo "$idx" | xargs)
    used=$(echo "$used" | xargs)
    total=$(echo "$total" | xargs)
    util=$(echo "$util" | xargs)
    free=$((total - used))
    echo "GPU ${idx}: ${free} MiB free / ${total} MiB total, ${util}% util"
done
