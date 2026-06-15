#!/bin/bash
set -e

mkdir -p artifacts/images/eps artifacts/images/png

time ./artifacts/executables/kmeans_seq "${1:-1000000}" "${2:-5}"
convert artifacts/images/eps/image.eps artifacts/images/png/image.png
