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

Compilacao
----------
Execute:

    ./scripts/compile.sh

O executavel sera criado em:

    artifacts/executables/kmeans_seq

Execucao
--------
Formato geral:

    ./scripts/run.sh <quantidade_de_observacoes> <quantidade_de_clusters> <algoritmo>

Algoritmos aceitos:

    s  versao sequencial
    o  versao OpenMP em CPU
    g  placeholder OpenMP GPU
    c  placeholder CUDA

Exemplos:

    ./scripts/run.sh 1000000 5 s
    OMP_NUM_THREADS=8 ./scripts/run.sh 1000000 5 o

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
Depois de medir no servidor de teste, preencha o bloco inicial de
code/k_means_clustering_openmp.c com os tempos da versao sequencial e das
execucoes OpenMP com 1, 2, 4, 8, 16 e 32 threads.
