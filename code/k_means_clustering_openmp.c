/**
 * @file k_means_clustering_openmp.c
 * @brief Versao paralela em CPU do algoritmo K-Means usando OpenMP.
 *
 * Mudancas realizadas para a paralelizacao:
 * - A atribuicao inicial dos grupos usa um mapeamento deterministico por
 *   indice, eliminando a dependencia de rand() e permitindo bootstrap paralelo.
 * - O loop principal do K-Means roda dentro de uma regiao paralela persistente,
 *   reduzindo o overhead de abrir multiplas regioes a cada iteracao.
 * - O calculo dos centroides usa acumuladores privados por thread e merge
 *   paralelo por cluster, mantendo o laco quente livre de atomics.
 * - A reatribuicao das observacoes aos centroides acumula mudancas por thread,
 *   consolidadas ao fim de cada iteracao para decidir a parada.
 * - O calculo do centroide unico (k <= 1) tambem usa reduction para somar x/y.
 *
 * Tempos de execucao medidos no servidor de teste (preencher apos benchmark):
 * Base de dados: __________________________
 * Parametros: observacoes = __________, clusters = __________
 * Sequencial: __________ s
 * OpenMP  1 thread : __________ s
 * OpenMP  2 threads: __________ s
 * OpenMP  4 threads: __________ s
 * OpenMP  8 threads: __________ s
 * OpenMP 16 threads: __________ s
 * OpenMP 32 threads: __________ s
 */

#include "headers/k_means_clustering_openmp.h"

#include <float.h>
#include <stdint.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

static unsigned int mixIndexToGroupSeed(size_t index, int k)
{
    uint64_t value = (uint64_t)index + 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    value ^= value >> 31;

    return (unsigned int)(value % (uint64_t)k);
}

int calculateNearstOpenMP(observation *o, cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;

    /* Mantem a busca interna sequencial: a paralelizacao ocorre no laco externo
       que classifica varias observacoes ao mesmo tempo. */
    for (int i = 0; i < k; i++)
    {
        dist = (clusters[i].x - o->x) * (clusters[i].x - o->x) +
               (clusters[i].y - o->y) * (clusters[i].y - o->y);
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }

    return index;
}

void calculateCentroidOpenMP(observation observations[], size_t size, cluster *centroid)
{
    double sumX = 0;
    double sumY = 0;

    centroid->x = 0;
    centroid->y = 0;
    centroid->count = size;

    /* Paralelizacao: cada thread soma uma faixa do vetor e a clausula reduction
       combina os valores parciais em sumX/sumY sem condicao de corrida. */
    #pragma omp parallel for reduction(+ : sumX, sumY) schedule(static)
    for (size_t i = 0; i < size; i++)
    {
        sumX += observations[i].x;
        sumY += observations[i].y;
        observations[i].group = 0;
    }

    if (centroid->count > 0)
    {
        centroid->x = sumX / centroid->count;
        centroid->y = sumY / centroid->count;
    }
}

cluster *kMeansOpenMP(observation observations[], size_t size, int k)
{
    cluster *clusters = NULL;

    if (k <= 1)
    {
        clusters = (cluster *)malloc(sizeof(cluster));
        if (clusters == NULL)
        {
            return NULL;
        }
        memset(clusters, 0, sizeof(cluster));
        calculateCentroidOpenMP(observations, size, clusters);
    }
    else if ((size_t)k < size)
    {
        int maxThreads = omp_get_max_threads();
        double *localX = NULL;
        double *localY = NULL;
        size_t *localCount = NULL;
        size_t *changedPerThread = NULL;
        size_t changed = 0;
        size_t minAcceptedError = size / 10000;
        int keepRunning = 1;

        clusters = (cluster *)malloc(sizeof(cluster) * k);
        if (clusters == NULL)
        {
            return NULL;
        }
        memset(clusters, 0, k * sizeof(cluster));

        localX = (double *)calloc((size_t)maxThreads * k, sizeof(double));
        localY = (double *)calloc((size_t)maxThreads * k, sizeof(double));
        localCount = (size_t *)calloc((size_t)maxThreads * k, sizeof(size_t));
        changedPerThread = (size_t *)calloc((size_t)maxThreads, sizeof(size_t));
        if (localX == NULL || localY == NULL || localCount == NULL ||
            changedPerThread == NULL)
        {
            free(localX);
            free(localY);
            free(localCount);
            free(changedPerThread);
            free(clusters);
            return NULL;
        }

        /* Bootstrap deterministico e paralelizavel: remove dependencia do
           estado global de rand() e mantem a distribuicao inicial por indice. */
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = (int)mixIndexToGroupSeed(j, k);
        }

        #pragma omp parallel shared(changed, keepRunning, clusters, localX, localY, localCount, changedPerThread)
        {
            int tid = omp_get_thread_num();
            double *threadX = localX + ((size_t)tid * k);
            double *threadY = localY + ((size_t)tid * k);
            size_t *threadCount = localCount + ((size_t)tid * k);

            while (keepRunning)
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < k; i++)
                {
                    clusters[i].x = 0;
                    clusters[i].y = 0;
                    clusters[i].count = 0;
                }

                #pragma omp for schedule(static)
                for (int i = 0; i < k; i++)
                {
                    threadX[i] = 0;
                    threadY[i] = 0;
                    threadCount[i] = 0;
                }

                changedPerThread[tid] = 0;

                /* Cada thread acumula apenas em seu bloco local por cluster. */
                #pragma omp for schedule(static)
                for (size_t j = 0; j < size; j++)
                {
                    int group = observations[j].group;
                    threadX[group] += observations[j].x;
                    threadY[group] += observations[j].y;
                    threadCount[group]++;
                }

                /* Merge paralelo por cluster: cada centroide agrega todas as
                   contribuicoes thread-local antes da normalizacao. */
                #pragma omp for schedule(static)
                for (int i = 0; i < k; i++)
                {
                    double sumX = 0;
                    double sumY = 0;
                    size_t count = 0;

                    for (int owner = 0; owner < maxThreads; owner++)
                    {
                        size_t index = (size_t)owner * k + i;
                        sumX += localX[index];
                        sumY += localY[index];
                        count += localCount[index];
                    }

                    clusters[i].count = count;
                    if (count > 0)
                    {
                        clusters[i].x = sumX / count;
                        clusters[i].y = sumY / count;
                    }
                }

                /* Reclassificacao paralela; cada thread soma suas mudancas
                   localmente para evitar atomics no caminho quente. */
                #pragma omp for schedule(static)
                for (size_t j = 0; j < size; j++)
                {
                    int nearest = calculateNearstOpenMP(observations + j, clusters, k);
                    if (nearest != observations[j].group)
                    {
                        changedPerThread[tid]++;
                        observations[j].group = nearest;
                    }
                }

                #pragma omp single
                {
                    changed = 0;
                    for (int owner = 0; owner < maxThreads; owner++)
                    {
                        changed += changedPerThread[owner];
                    }
                    keepRunning = changed > minAcceptedError;
                }
            }
        }

        free(localX);
        free(localY);
        free(localCount);
        free(changedPerThread);
    }
    else
    {
        clusters = (cluster *)malloc(sizeof(cluster) * k);
        if (clusters == NULL)
        {
            return NULL;
        }
        memset(clusters, 0, k * sizeof(cluster));

        /* Caso trivial: cada observacao vira seu proprio cluster. Tambem pode
           ser paralelizado porque cada indice e escrito independentemente. */
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < (int)size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = j;
        }
    }

    return clusters;
}
