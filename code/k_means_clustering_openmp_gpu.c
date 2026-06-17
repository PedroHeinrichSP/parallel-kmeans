/**
 * @file k_means_clustering_openmp_gpu.c
 * @brief Versao em GPU do algoritmo K-Means usando OpenMP target offload.
 *
 * A implementacao segue a mesma semantica da versao OpenMP em CPU:
 * - A inicializacao dos grupos usa rand() no host para preservar a comparacao
 *   com a versao sequencial/OpenMP CPU.
 * - Os lacos pesados de soma dos centroides, normalizacao e reatribuicao dos
 *   pontos usam OpenMP target teams distribute parallel for.
 * - Clusters vazios mantem o centroide anterior, evitando divisao por zero e
 *   acompanhando a decisao da versao OpenMP CPU otimizada.
 * - Quando o compilador/runtime nao possui device de offload, OpenMP executa a
 *   regiao target no host, mantendo a implementacao funcional.
 */

#include "headers/k_means_clustering_openmp_gpu.h"

#include <float.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#pragma omp declare target
static inline int calculateNearestIndexGPU(const observation *o,
                                           const cluster *clusters, int k)
{
    double minD = DBL_MAX;
    int index = -1;
    double ox = o->x;
    double oy = o->y;

    for (int i = 0; i < k; i++)
    {
        double dx = clusters[i].x - ox;
        double dy = clusters[i].y - oy;
        double dist = dx * dx + dy * dy;

        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }

    return index;
}
#pragma omp end declare target

int calculateNearstOpenMP_GPU(observation *o, cluster clusters[], int k)
{
    return calculateNearestIndexGPU(o, clusters, k);
}

void calculateCentroidOpenMP_GPU(observation observations[], size_t size,
                                 cluster *centroid)
{
    double sumX = 0;
    double sumY = 0;

    centroid->x = 0;
    centroid->y = 0;
    centroid->count = size;

    #pragma omp target teams distribute parallel for reduction(+ : sumX, sumY) \
        map(tofrom : observations[0:size]) map(to : size)
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

cluster *kMeansOpenMP_GPU(observation observations[], size_t size, int k)
{
    cluster *clusters = NULL;

    if (observations == NULL || size == 0 || k <= 0)
    {
        return NULL;
    }

    if (k <= 1)
    {
        clusters = (cluster *)malloc(sizeof(cluster));
        if (clusters == NULL)
        {
            return NULL;
        }
        memset(clusters, 0, sizeof(cluster));
        calculateCentroidOpenMP_GPU(observations, size, clusters);
    }
    else if ((size_t)k < size)
    {
        size_t changed = 0;
        size_t minAcceptedError = size / 10000;
        int keepRunning = 1;
        double *sumX = NULL;
        double *sumY = NULL;
        size_t *counts = NULL;

        clusters = (cluster *)malloc(sizeof(cluster) * (size_t)k);
        if (clusters == NULL)
        {
            return NULL;
        }

        sumX = (double *)calloc((size_t)k, sizeof(double));
        sumY = (double *)calloc((size_t)k, sizeof(double));
        counts = (size_t *)calloc((size_t)k, sizeof(size_t));
        if (sumX == NULL || sumY == NULL || counts == NULL)
        {
            free(sumX);
            free(sumY);
            free(counts);
            free(clusters);
            return NULL;
        }

        for (int i = 0; i < k; i++)
        {
            clusters[i].x = observations[i].x;
            clusters[i].y = observations[i].y;
            clusters[i].count = 0;
        }

        /* Mantem a inicializacao por rand() no host para reproducibilidade com
           as outras versoes chamadas pelo runner apos srand(42). */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }

        #pragma omp target data map(tofrom : observations[0:size]) \
            map(tofrom : clusters[0:k]) map(alloc : sumX[0:k], sumY[0:k], counts[0:k]) \
            map(to : size, k)
        {
            while (keepRunning)
            {
                #pragma omp target teams distribute parallel for
                for (int i = 0; i < k; i++)
                {
                    sumX[i] = 0;
                    sumY[i] = 0;
                    counts[i] = 0;
                }

                #pragma omp target teams distribute parallel for
                for (size_t j = 0; j < size; j++)
                {
                    int group = observations[j].group;
                    double x = observations[j].x;
                    double y = observations[j].y;

                    #pragma omp atomic update
                    sumX[group] += x;
                    #pragma omp atomic update
                    sumY[group] += y;
                    #pragma omp atomic update
                    counts[group]++;
                }

                #pragma omp target teams distribute parallel for
                for (int i = 0; i < k; i++)
                {
                    size_t count = counts[i];
                    clusters[i].count = count;
                    if (count > 0)
                    {
                        clusters[i].x = sumX[i] / count;
                        clusters[i].y = sumY[i] / count;
                    }
                }

                changed = 0;
                #pragma omp target teams distribute parallel for \
                    reduction(+ : changed)
                for (size_t j = 0; j < size; j++)
                {
                    int nearest = calculateNearestIndexGPU(observations + j,
                                                           clusters, k);
                    if (nearest != observations[j].group)
                    {
                        changed++;
                        observations[j].group = nearest;
                    }
                }

                keepRunning = changed > minAcceptedError;
            }
        }

        free(sumX);
        free(sumY);
        free(counts);
    }
    else
    {
        clusters = (cluster *)malloc(sizeof(cluster) * (size_t)k);
        if (clusters == NULL)
        {
            return NULL;
        }
        memset(clusters, 0, sizeof(cluster) * (size_t)k);

        #pragma omp target teams distribute parallel for \
            map(tofrom : observations[0:size], clusters[0:k]) map(to : size)
        for (size_t j = 0; j < size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = (int)j;
        }
    }

    return clusters;
}
