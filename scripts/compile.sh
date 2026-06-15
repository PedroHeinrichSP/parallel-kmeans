#!/bin/bash
set -e

mkdir -p artifacts/executables

gcc code/runner.c \
    code/k_means_clustering_utils.c \
    code/k_means_clustering_sequencial.c \
    code/k_means_clustering_openmp.c \
    code/k_means_clustering_openmp_gpu.c \
    code/k_means_clustering_cuda.c \
    -Icode/headers \
    -o artifacts/executables/kmeans_seq \
    -lm
