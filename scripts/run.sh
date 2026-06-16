#!/bin/bash
set -e

mkdir -p artifacts/images/eps artifacts/images/png

OBSERVATIONS="${1:-1000000}"
CLUSTERS="${2:-5}"
ALGORITHM="${3:-s}"

time ./artifacts/executables/kmeans_seq "$OBSERVATIONS" "$CLUSTERS" "$ALGORITHM"
convert artifacts/images/eps/image.eps artifacts/images/png/image.png
