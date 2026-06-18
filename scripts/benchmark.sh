#!/bin/bash
set -euo pipefail

OBSERVATIONS="${1:-1000000}"
CLUSTERS="${2:-5}"
RUNS="${3:-3}"
THREADS_CSV="${4:-1,2,4,8,16,32}"
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"

EXECUTABLE="./artifacts/executables/kmeans"

if ! command -v /usr/bin/time >/dev/null 2>&1; then
    echo "Erro: /usr/bin/time nao esta disponivel." >&2
    exit 1
fi

if ! [[ "$OBSERVATIONS" =~ ^[0-9]+$ && "$CLUSTERS" =~ ^[0-9]+$ && "$RUNS" =~ ^[0-9]+$ ]]; then
    echo "Uso: $0 [observacoes] [clusters] [repeticoes] [threads_csv]" >&2
    echo "Exemplo: $0 1000000 5 3 1,2,4,8" >&2
    exit 1
fi

IFS=',' read -r -a THREADS <<< "$THREADS_CSV"
if [ "${#THREADS[@]}" -eq 0 ]; then
    echo "Erro: informe ao menos uma contagem de threads." >&2
    exit 1
fi

for thread_count in "${THREADS[@]}"; do
    if ! [[ "$thread_count" =~ ^[0-9]+$ ]] || [ "$thread_count" -lt 1 ]; then
        echo "Erro: valor invalido na lista de threads: $thread_count" >&2
        exit 1
    fi
done

echo "Compilando o projeto..."
./scripts/compile.sh >/dev/null

TMP_RESULTS="$(mktemp)"
TMP_FAILURES="$(mktemp)"
trap 'rm -f "$TMP_RESULTS" "$TMP_FAILURES"' EXIT

measure_case() {
    local label="$1"
    local algorithm="$2"
    local threads="$3"
    local display_threads="${4:-$threads}"
    local run_index elapsed time_file stderr_file wait_policy note_printed target_offload gpu_fallback exit_code

    wait_policy="ACTIVE"
    note_printed=0
    target_offload="DEFAULT"
    gpu_fallback="1"
    if [ "$threads" -gt "$CPU_COUNT" ]; then
        wait_policy="PASSIVE"
    fi

    if [ "$algorithm" = "g" ]; then
        target_offload="MANDATORY"
        gpu_fallback="0"
    fi

    time_file="$(mktemp)"
    stderr_file="$(mktemp)"

    echo
    echo "Executando $label"

    for run_index in $(seq 1 "$RUNS"); do
        set +e
        KMEANS_SKIP_EPS=1 \
        OMP_DYNAMIC=FALSE \
        OMP_PROC_BIND=TRUE \
        OMP_PLACES=cores \
        OMP_PROC_BIND=spread \
        OMP_WAIT_POLICY="$wait_policy" \
        OMP_NUM_THREADS="$threads" \
        OMP_TARGET_OFFLOAD="$target_offload" \
        KMEANS_OMP_GPU_ALLOW_CPU_FALLBACK="$gpu_fallback" \
        /usr/bin/time -f "%e" -o "$time_file" \
        "$EXECUTABLE" "$OBSERVATIONS" "$CLUSTERS" "$algorithm" \
        >/dev/null 2>"$stderr_file"
        exit_code=$?
        set -e

        elapsed="$(tr -d "[:space:]" < "$time_file" 2>/dev/null || true)"

        if [ -s "$stderr_file" ]; then
            while IFS= read -r line; do
                printf "  nota: %s\n" "$line"
            done < "$stderr_file"
            note_printed=1
        fi

        if [ "$exit_code" -ne 0 ]; then
            printf "%s\t%s\t%d\n" "$label" "$display_threads" "$exit_code" >>"$TMP_FAILURES"
            printf "  falhou no run %d/%d (exit=%d)\n" "$run_index" "$RUNS" "$exit_code"
            break
        fi

        printf "%s\t%s\t%s\t%s\n" "$label" "$display_threads" "$run_index" "$elapsed" >>"$TMP_RESULTS"
        printf "  run %d/%d: %ss\n" "$run_index" "$RUNS" "$elapsed"
    done

    rm -f "$time_file"
    rm -f "$stderr_file"
}

measure_case "seq" "s" "1"

for thread_count in "${THREADS[@]}"; do
    measure_case "omp-${thread_count}t" "o" "$thread_count"
done

measure_case "omp-gpu" "g" "1" "gpu"

if command -v "${NVCC:-nvcc}" >/dev/null 2>&1; then
    measure_case "cuda" "c" "1" "gpu"
else
    echo
    echo "Pulando cuda: nvcc nao esta disponivel; o build usou o placeholder."
fi

echo
echo "Resumo"
printf "%-10s %-8s %-10s %-10s %-10s %-10s\n" "modo" "threads" "media(s)" "min(s)" "max(s)" "speedup"

awk '
BEGIN {
    FS = "\t";
}
{
    label = $1;
    threads[label] = $2;
    value = $4 + 0.0;

    sum[label] += value;
    count[label] += 1;

    if (!(label in min) || value < min[label]) {
        min[label] = value;
    }
    if (!(label in max) || value > max[label]) {
        max[label] = value;
    }

    if (!(label in order)) {
        ordered_labels[++total] = label;
        order[label] = total;
    }
}
END {
    seq_avg = sum["seq"] / count["seq"];

    for (i = 1; i <= total; i++) {
        label = ordered_labels[i];
        avg = sum[label] / count[label];
        speedup_str = "n/a";

        if (avg > 0 && seq_avg > 0) {
            speedup = seq_avg / avg;
            speedup_str = sprintf("%.2fx", speedup);
        }

        printf "%-10s %-8s %-10.6f %-10.6f %-10.6f %-10s\n",
               label, threads[label], avg, min[label], max[label], speedup_str;
    }
}
' "$TMP_RESULTS"

if [ -s "$TMP_FAILURES" ]; then
    echo
    echo "Falhas"
    awk -F "\t" '{ printf "%-10s %-8s exit=%s\n", $1, $2, $3 }' "$TMP_FAILURES"
fi

echo
echo "Parametros usados: observacoes=$OBSERVATIONS clusters=$CLUSTERS repeticoes=$RUNS threads=$THREADS_CSV"
