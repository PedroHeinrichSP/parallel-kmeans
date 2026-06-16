/**
 * @file k_means_clustering_openmp.c
 * @brief Versao paralela em CPU do algoritmo K-Means usando OpenMP.
 *
 * Mudancas realizadas para a paralelizacao:
 * - A atribuicao inicial dos grupos permanece sequencial porque usa rand(), que
 *   possui estado global e nao e thread-safe.
 * - O calculo dos centroides foi paralelizado com acumuladores privados por
 *   thread; ao final de cada iteracao, os acumuladores sao reduzidos para o
 *   vetor compartilhado de clusters.
 * - A reatribuicao das observacoes aos centroides foi paralelizada com
 *   reduction(+:changed), pois cada thread atualiza apenas a observacao do seu
 *   proprio indice e soma localmente a quantidade de mudancas.
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
#include <omp.h>
#include <stdlib.h>
#include <string.h>

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
        size_t changed = 0;
        size_t minAcceptedError = size / 10000;

        clusters = (cluster *)malloc(sizeof(cluster) * k);
        if (clusters == NULL)
        {
            return NULL;
        }
        memset(clusters, 0, k * sizeof(cluster));

        localX = (double *)calloc((size_t)maxThreads * k, sizeof(double));
        localY = (double *)calloc((size_t)maxThreads * k, sizeof(double));
        localCount = (size_t *)calloc((size_t)maxThreads * k, sizeof(size_t));
        if (localX == NULL || localY == NULL || localCount == NULL)
        {
            free(localX);
            free(localY);
            free(localCount);
            free(clusters);
            return NULL;
        }

        /* Igual a versao sequencial: rand() fica fora da regiao paralela para
           evitar uso concorrente do gerador pseudoaleatorio global. */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }

        do
        {
            memset(localX, 0, (size_t)maxThreads * k * sizeof(double));
            memset(localY, 0, (size_t)maxThreads * k * sizeof(double));
            memset(localCount, 0, (size_t)maxThreads * k * sizeof(size_t));

            /* Paralelizacao do STEP 2: cada thread acumula x, y e count em uma
               area privada indexada por tid, removendo atomics no laco grande. */
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                double *threadX = localX + ((size_t)tid * k);
                double *threadY = localY + ((size_t)tid * k);
                size_t *threadCount = localCount + ((size_t)tid * k);

                #pragma omp for schedule(static)
                for (size_t j = 0; j < size; j++)
                {
                    int group = observations[j].group;
                    threadX[group] += observations[j].x;
                    threadY[group] += observations[j].y;
                    threadCount[group]++;
                }
            }

            for (int i = 0; i < k; i++)
            {
                clusters[i].x = 0;
                clusters[i].y = 0;
                clusters[i].count = 0;
            }

            /* Reducao manual dos acumuladores privados para os centroides
               compartilhados. Este trecho e pequeno: maxThreads * k. */
            for (int tid = 0; tid < maxThreads; tid++)
            {
                for (int i = 0; i < k; i++)
                {
                    size_t index = (size_t)tid * k + i;
                    clusters[i].x += localX[index];
                    clusters[i].y += localY[index];
                    clusters[i].count += localCount[index];
                }
            }

            for (int i = 0; i < k; i++)
            {
                if (clusters[i].count > 0)
                {
                    clusters[i].x /= clusters[i].count;
                    clusters[i].y /= clusters[i].count;
                }
            }

            changed = 0;

            /* Paralelizacao dos STEPs 3 e 4: cada observacao e independente na
               busca pelo centroide mais proximo; reduction soma as mudancas. */
            #pragma omp parallel for reduction(+ : changed) schedule(static)
            for (size_t j = 0; j < size; j++)
            {
                int nearest = calculateNearstOpenMP(observations + j, clusters, k);
                if (nearest != observations[j].group)
                {
                    changed++;
                    observations[j].group = nearest;
                }
            }
        } while (changed > minAcceptedError);

        free(localX);
        free(localY);
        free(localCount);
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
