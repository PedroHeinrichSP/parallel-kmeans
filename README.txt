parallel-kmeans
===============

Aplicacao
---------
Este projeto executa o algoritmo K-Means para agrupar observacoes bidimensionais
em k clusters. A versao sequencial vem do codigo-base disponibilizado e a versao
CPU paralela usa OpenMP para acelerar as duas partes mais custosas do algoritmo:
o calculo dos centroides e a reatribuicao de cada observacao ao centroide mais
proximo.

A entrada padrao e gerada pelo proprio programa como pontos 2D em uma regiao
circular. Para atender ao requisito da disciplina, escolha uma quantidade de
observacoes grande o suficiente para que a versao sequencial execute por pelo
menos 10 segundos no servidor de teste usado pelo grupo.

Requisitos
----------
- gcc com suporte a OpenMP
- bash
- ImageMagick, apenas se desejar converter a saida EPS para PNG via script run.sh

Windows:
- PowerShell
- gcc no PATH (por exemplo, via MinGW-w64/MSYS2)

macOS:
- bash
- um compilador com OpenMP
- opcionalmente Homebrew com `gcc`, ou `llvm` + `libomp`

Compilacao
----------
Execute:

    ./scripts/compile.sh

O executavel sera criado em:

    artifacts/executables/kmeans

Windows:

    .\scripts\compile.ps1

macOS:

    ./scripts/compile_macos.sh

Execucao
--------
Formato geral:

    ./scripts/run.sh <quantidade_de_observacoes> <quantidade_de_clusters> <algoritmo>

Algoritmos aceitos:

    s  versao sequencial
    o  versao OpenMP em CPU
    g  versao OpenMP GPU
    c  versao CUDA ou fallback placeholder sem nvcc

Exemplos:

    ./scripts/run.sh 1000000 5 s
    OMP_NUM_THREADS=8 ./scripts/run.sh 1000000 5 o

Windows:

    .\scripts\run.ps1 1000000 5 s

macOS:

    ./scripts/run_macos.sh 1000000 5 s

Para medir 1, 2, 4, 8, 16 e 32 threads:

    OMP_NUM_THREADS=1  ./scripts/run.sh 1000000 5 o
    OMP_NUM_THREADS=2  ./scripts/run.sh 1000000 5 o
    OMP_NUM_THREADS=4  ./scripts/run.sh 1000000 5 o
    OMP_NUM_THREADS=8  ./scripts/run.sh 1000000 5 o
    OMP_NUM_THREADS=16 ./scripts/run.sh 1000000 5 o
    OMP_NUM_THREADS=32 ./scripts/run.sh 1000000 5 o

Saidas
------
A execucao gera:

    artifacts/images/eps/image.eps
    artifacts/images/png/image.png

O arquivo EPS e escrito pelo programa. O PNG e gerado pelo ImageMagick no script
de execucao.

Observacao sobre benchmarks
---------------------------
Os tempos abaixo foram obtidos com os scripts de benchmark do projeto.

Scripts de benchmark:

    ./scripts/benchmark.sh
    .\scripts\benchmark.ps1
    ./scripts/benchmark_macos.sh

Resumo
modo       threads  media(s)   min(s)     max(s)     speedup
seq        1        14.690000  11.370000  16.140000  1.00x
omp-1t     1        12.530000  11.470000  16.620000  1.17x
omp-2t     2        11.376000  11.350000  11.430000  1.29x
omp-4t     4        6.600000   6.340000   7.460000   2.23x
omp-8t     8        6.428000   5.920000   7.250000   2.29x
omp-16t    16       5.866000   5.830000   5.920000   2.50x
omp-32t    32       6.208000   5.810000   7.400000   2.37x
cuda       gpu      6.072000   5.970000   6.270000   2.42x
