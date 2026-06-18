#!/bin/bash
set -euo pipefail

mkdir -p artifacts/executables artifacts/objects

CFLAGS="${CFLAGS:--O3 -mtune=native -fno-math-errno}"
OPENMP_CFLAGS="${OPENMP_CFLAGS:-}"
OPENMP_LDFLAGS="${OPENMP_LDFLAGS:-}"

pick_macos_compiler() {
    if [ -n "${CC:-}" ]; then
        printf "%s" "$CC"
        return 0
    fi

    for candidate in gcc-14 gcc-13 gcc-12; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf "%s" "$candidate"
            return 0
        fi
    done

    if command -v brew >/dev/null 2>&1; then
        local llvm_prefix libomp_prefix
        llvm_prefix="$(brew --prefix llvm 2>/dev/null || true)"
        libomp_prefix="$(brew --prefix libomp 2>/dev/null || true)"

        if [ -n "$llvm_prefix" ] && [ -x "$llvm_prefix/bin/clang" ] && [ -n "$libomp_prefix" ]; then
            OPENMP_CFLAGS="-Xpreprocessor -fopenmp -I$libomp_prefix/include"
            OPENMP_LDFLAGS="-L$libomp_prefix/lib -lomp"
            export OPENMP_CFLAGS OPENMP_LDFLAGS
            printf "%s" "$llvm_prefix/bin/clang"
            return 0
        fi
    fi

    return 1
}

CC_BIN="$(pick_macos_compiler)" || {
    echo "Erro: nao encontrei um compilador com OpenMP no macOS." >&2
    echo "Instale 'gcc' ou 'llvm' + 'libomp' via Homebrew, ou defina CC/OPENMP_CFLAGS/OPENMP_LDFLAGS." >&2
    exit 1
}

"$CC_BIN" $CFLAGS \
    code/runner.c \
    code/k_means_clustering_utils.c \
    code/k_means_clustering_sequencial.c \
    code/k_means_clustering_openmp.c \
    code/k_means_clustering_openmp_gpu.c \
    code/k_means_clustering_cuda.c \
    -Icode/headers \
    $OPENMP_CFLAGS \
    -o artifacts/executables/kmeans \
    -lm \
    $OPENMP_LDFLAGS
