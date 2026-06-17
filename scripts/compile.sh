#!/bin/bash
set -e

mkdir -p artifacts/executables

CFLAGS="${CFLAGS:--O3 -mtune=native -fno-math-errno}"

gcc $CFLAGS code/runner.c \
    code/k_means_clustering_utils.c \
    code/k_means_clustering_sequencial.c \
    code/k_means_clustering_openmp.c \
    code/k_means_clustering_openmp_gpu.c \
    code/k_means_clustering_cuda.c \
    -Icode/headers \
    -fopenmp \
    -o artifacts/executables/kmeans \
    -lm

cp artifacts/executables/kmeans artifacts/executables/kmeans_seq
