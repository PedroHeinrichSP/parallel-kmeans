#!/bin/bash
set -e

mkdir -p artifacts/executables artifacts/objects

CC=${CC:-gcc}
NVCC=${NVCC:-nvcc}
CUDA_ARCH=${CUDA_ARCH:-sm_61}
CFLAGS="${CFLAGS:--O3 -mtune=native -fno-math-errno}"

if command -v "$NVCC" >/dev/null 2>&1; then
    "$CC" $CFLAGS -fopenmp -Icode/headers -c code/runner.c \
        -o artifacts/objects/runner.o
    "$CC" $CFLAGS -fopenmp -Icode/headers -c code/k_means_clustering_utils.c \
        -o artifacts/objects/k_means_clustering_utils.o
    "$CC" $CFLAGS -fopenmp -Icode/headers -c code/k_means_clustering_sequencial.c \
        -o artifacts/objects/k_means_clustering_sequencial.o
    "$CC" $CFLAGS -fopenmp -Icode/headers -c code/k_means_clustering_openmp.c \
        -o artifacts/objects/k_means_clustering_openmp.o
    "$CC" $CFLAGS -fopenmp -Icode/headers -c code/k_means_clustering_openmp_gpu.c \
        -o artifacts/objects/k_means_clustering_openmp_gpu.o
    "$NVCC" -arch="$CUDA_ARCH" -std=c++11 -Icode/headers \
        -c code/k_means_clustering_cuda.cu \
        -o artifacts/objects/k_means_clustering_cuda.o
    "$NVCC" -Xcompiler -fopenmp artifacts/objects/runner.o \
        artifacts/objects/k_means_clustering_utils.o \
        artifacts/objects/k_means_clustering_sequencial.o \
        artifacts/objects/k_means_clustering_openmp.o \
        artifacts/objects/k_means_clustering_openmp_gpu.o \
        artifacts/objects/k_means_clustering_cuda.o \
        -o artifacts/executables/kmeans \
        -lm
else
    "$CC" $CFLAGS code/runner.c \
        code/k_means_clustering_utils.c \
        code/k_means_clustering_sequencial.c \
        code/k_means_clustering_openmp.c \
        code/k_means_clustering_openmp_gpu.c \
        code/k_means_clustering_cuda.c \
        -Icode/headers \
        -fopenmp \
        -o artifacts/executables/kmeans \
        -lm
fi

cp artifacts/executables/kmeans artifacts/executables/kmeans_seq
