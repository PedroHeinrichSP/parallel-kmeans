#!/bin/bash
set -euo pipefail

mkdir -p artifacts/images/eps artifacts/images/png

OBSERVATIONS="${1:-1000000}"
CLUSTERS="${2:-5}"
ALGORITHM="${3:-s}"
EXECUTABLE="./artifacts/executables/kmeans"

if [ ! -x "$EXECUTABLE" ]; then
    echo "Erro: executavel nao encontrado em $EXECUTABLE. Execute ./scripts/compile_macos.sh primeiro." >&2
    exit 1
fi

time "$EXECUTABLE" "$OBSERVATIONS" "$CLUSTERS" "$ALGORITHM"
