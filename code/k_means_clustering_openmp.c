/**
 * @file k_means_clustering_openmp.c
 * @brief Versao paralela em CPU do algoritmo K-Means usando OpenMP.
 *
 * Mudancas realizadas para a paralelizacao:
 * - A atribuicao inicial dos grupos replica a versao sequencial com rand(),
 *   mantendo o mesmo ponto de partida para uma comparacao justa de tempo.
 * - O loop principal do K-Means roda dentro de uma regiao paralela persistente,
 *   reduzindo o overhead de abrir multiplas regioes a cada iteracao.
 * - O calculo dos centroides usa acumuladores privados por thread e merge
 *   paralelo por cluster, mantendo o laco quente livre de atomics.
 * - A reatribuicao das observacoes acumula mudancas por thread,
 *   evitando reducoes caras no laco quente.
 * - O calculo do centroide unico (k <= 1) tambem usa reduction para somar x/y.
 * - A implementacao nao assume valores fixos para quantidade de pontos ou
 *   clusters; apenas espera que a seed global de rand() seja configurada
 *   como 42 pelo runner para reproducibilidade.
 */

#include "headers/k_means_clustering_openmp.h"

#include <float.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define CACHE_LINE_BYTES 64

/* Retorna um stride arredondado para linha de cache. Isso separa os
   acumuladores de threads diferentes e reduz falso compartilhamento. */
static size_t paddedStride(size_t elements, size_t elementSize)
{
    size_t elementsPerLine = CACHE_LINE_BYTES / elementSize;

    if (elementsPerLine == 0)
    {
        elementsPerLine = 1;
    }

    return ((elements + elementsPerLine - 1) / elementsPerLine) *
           elementsPerLine;
}

/* Aloca memoria alinhada a linha de cache e inicializada com zero para os
   buffers thread-local usados no laco principal. */
static void *alignedCalloc(size_t count, size_t size)
{
    void *memory = NULL;

    if (posix_memalign(&memory, CACHE_LINE_BYTES, count * size) != 0)
    {
        return NULL;
    }

    memset(memory, 0, count * size);
    return memory;
}

/* Calcula o centroide mais proximo de uma observacao. O laco permanece
   generico em relacao a k; o paralelismo acontece no laco externo que chama
   esta funcao para muitas observacoes. */
static inline int calculateNearestIndex(const observation *restrict o,
                                        const cluster *restrict clusters, int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    double ox = o->x;
    double oy = o->y;

    for (int i = 0; i < k; i++)
    {
        double dx = clusters[i].x - ox;
        double dy = clusters[i].y - oy;
        dist = dx * dx + dy * dy;
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }

    return index;
}

int calculateNearstOpenMP(observation *o, cluster clusters[], int k)
{
    return calculateNearestIndex(o, clusters, k);
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
        size_t clusterStride = paddedStride((size_t)k, sizeof(double));
        size_t countStride = paddedStride((size_t)k, sizeof(size_t));
        size_t changedStride = paddedStride(1, sizeof(size_t));
        int maxThreads = omp_get_max_threads();
        int usedThreads = maxThreads;
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
        for (int i = 0; i < k; i++)
        {
            clusters[i].x = observations[i].x;
            clusters[i].y = observations[i].y;
            clusters[i].count = 0;
        }

        localX = (double *)alignedCalloc((size_t)maxThreads * clusterStride,
                                         sizeof(double));
        localY = (double *)alignedCalloc((size_t)maxThreads * clusterStride,
                                         sizeof(double));
        localCount = (size_t *)alignedCalloc((size_t)maxThreads * countStride,
                                             sizeof(size_t));
        changedPerThread = (size_t *)alignedCalloc(
            (size_t)maxThreads * changedStride, sizeof(size_t));
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

        /* Mantem a mesma inicializacao da versao sequencial para que o
           benchmark compare o mesmo numero de iteracoes do K-Means. */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }

        #pragma omp parallel shared(changed, keepRunning, clusters, localX, localY, localCount, changedPerThread, usedThreads)
        {
            int tid = omp_get_thread_num();
            double *threadX = localX + ((size_t)tid * clusterStride);
            double *threadY = localY + ((size_t)tid * clusterStride);
            size_t *threadCount = localCount + ((size_t)tid * countStride);
            size_t *threadChanged =
                changedPerThread + ((size_t)tid * changedStride);

            #pragma omp single
            {
                usedThreads = omp_get_num_threads();
            }
            while (keepRunning)
            {
                /* Cada thread deve limpar todo seu proprio bloco local. Usar
                   omp for aqui dividiria indices entre threads e deixaria lixo
                   de iteracoes anteriores em parte dos acumuladores. */
                for (int i = 0; i < k; i++)
                {
                    threadX[i] = 0;
                    threadY[i] = 0;
                    threadCount[i] = 0;
                }

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

                    for (int owner = 0; owner < usedThreads; owner++)
                    {
                        size_t valueIndex =
                            (size_t)owner * clusterStride + (size_t)i;
                        size_t countIndex =
                            (size_t)owner * countStride + (size_t)i;
                        sumX += localX[valueIndex];
                        sumY += localY[valueIndex];
                        count += localCount[countIndex];
                    }

                    clusters[i].count = count;
                    if (count > 0)
                    {
                        clusters[i].x = sumX / count;
                        clusters[i].y = sumY / count;
                    }
                    /* Cluster vazio: mantem o centroide anterior em vez de
                       empurrar artificialmente o cluster para a origem. */
                }

                /* Reclassificacao paralela: cada thread conta suas mudancas
                   localmente e publica um unico contador ao final do bloco. */
                size_t changedLocal = 0;
                #pragma omp for schedule(static)
                for (size_t j = 0; j < size; j++)
                {
                    int nearest = calculateNearestIndex(observations + j, clusters, k);
                    if (nearest != observations[j].group)
                    {
                        changedLocal++;
                        observations[j].group = nearest;
                    }
                }
                *threadChanged = changedLocal;

                #pragma omp barrier
                #pragma omp single
                {
                    changed = 0;
                    for (int owner = 0; owner < usedThreads; owner++)
                    {
                        changed += changedPerThread[(size_t)owner * changedStride];
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
