#!/bin/bash
set -e

mkdir -p artifacts/images/eps artifacts/images/png

OBSERVATIONS="${1:-1000000}"
CLUSTERS="${2:-5}"
ALGORITHM="${3:-s}"

if [ "$ALGORITHM" = "g" ]; then
    OMP_TARGET_OFFLOAD=MANDATORY \
    KMEANS_OMP_GPU_ALLOW_CPU_FALLBACK=0 \
    time ./artifacts/executables/kmeans "$OBSERVATIONS" "$CLUSTERS" "$ALGORITHM"
else
    time ./artifacts/executables/kmeans "$OBSERVATIONS" "$CLUSTERS" "$ALGORITHM"
fi
# convert artifacts/images/eps/image.eps artifacts/images/png/image.png
