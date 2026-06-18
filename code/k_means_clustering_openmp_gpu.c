/**
 * @file k_means_clustering_openmp_gpu.c
 * @brief Versao em GPU do algoritmo K-Means usando OpenMP target offload.
 *
 * Comparacao com k_means_clustering_sequencial.c:
 * - A semantica do K-Means continua a mesma: inicializacao, recomputacao dos
 *   centroides e reclassificacao ate a convergencia.
 * - A principal diferenca e que os lacos dominantes deixam a CPU e passam a
 *   executar em um dispositivo OpenMP target, tipicamente a GPU.
 *
 * Mudancas principais e por que elas sao necessarias:
 * - A inicializacao com rand() continua no host para manter reproducibilidade
 *   com as versoes sequencial e OpenMP CPU.
 * - Os lacos pesados foram convertidos para
 *   target teams distribute parallel for, que e o mecanismo de offload do
 *   OpenMP para explorar paralelismo massivo na GPU.
 * - Como os dados agora atravessam a fronteira host/dispositivo, a implementacao
 *   precisa mapear explicitamente observacoes, clusters e buffers auxiliares.
 * - Antes de offload, o codigo verifica se existe GPU acessivel e se o dataset
 *   cabe no limite configurado. Isso evita iniciar uma execucao que falharia
 *   por indisponibilidade de dispositivo ou excesso de memoria.
 * - A acumulacao por cluster na GPU usa atomics. Diferente da CPU, onde ha
 *   buffers privados por thread, aqui essa escolha simplifica a consolidacao
 *   das somas dentro do dispositivo.
 * - Clusters vazios mantem o centroide anterior para evitar divisao por zero,
 *   problema que a formula sequencial deixa implicito quando count > 0.
 */

#include "headers/k_means_clustering_openmp_gpu.h"
#include "headers/k_means_clustering_openmp.h"

#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KMEANS_OMP_GPU_DEFAULT_MAX_BYTES (1024ULL * 1024ULL * 1024ULL)

static size_t getOpenMPGPUByteLimit(void)
{
    const char *limitEnv = getenv("KMEANS_OMP_GPU_MAX_BYTES");

    if (limitEnv != NULL && limitEnv[0] != '\0')
    {
        char *end = NULL;
        unsigned long long parsed = strtoull(limitEnv, &end, 10);

        if (end != limitEnv && *end == '\0')
        {
            return (size_t)parsed;
        }
    }

    return (size_t)KMEANS_OMP_GPU_DEFAULT_MAX_BYTES;
}

static int shouldAllowOpenMPGPUCPUFallback(void)
{
    const char *fallbackEnv = getenv("KMEANS_OMP_GPU_ALLOW_CPU_FALLBACK");

    return fallbackEnv != NULL && strcmp(fallbackEnv, "1") == 0;
}

static void reportOpenMPGPUFailure(const char *reason)
{
    fprintf(stderr, "omp-gpu: %s.\n", reason);
}

static int openMPTargetIsOffloading(void)
{
    int runningOnHost = 1;

    if (omp_get_num_devices() <= 0)
    {
        return 0;
    }

    /* Esta pequena regiao target confirma se o runtime realmente saiu do host.
       Sem essa verificacao, a "versao GPU" poderia executar silenciosamente na
       CPU e invalidar a comparacao com a implementacao sequencial. */
    #pragma omp target map(from : runningOnHost)
    {
        runningOnHost = omp_is_initial_device();
    }

    return runningOnHost == 0;
}

static int openMPGPUCanRun(size_t size, int k)
{
    size_t requiredBytes = 0;
    size_t clusterCount = (size_t)((k > 0) ? k : 0);
    size_t byteLimit = getOpenMPGPUByteLimit();

    /* Diferente das versoes CPU, aqui precisamos validar o ambiente de
       execucao antes de entrar no algoritmo. */
    if (!openMPTargetIsOffloading())
    {
        reportOpenMPGPUFailure(
            "runtime/compilacao sem offload real de OpenMP target");
        return 0;
    }

    /* O sequencial trabalha diretamente na memoria principal. Com offload,
       estimamos a memoria a ser mapeada para evitar exceder a capacidade
       configurada do dispositivo. */
    requiredBytes += size * sizeof(observation);
    requiredBytes += clusterCount * sizeof(cluster);
    requiredBytes += clusterCount * sizeof(double) * 2U;
    requiredBytes += clusterCount * sizeof(size_t);

    if (requiredBytes > byteLimit)
    {
        reportOpenMPGPUFailure(
            "dataset grande demais para o limite configurado de memoria do offload");
        return 0;
    }

    return 1;
}

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
/* Esta funcao precisa estar em declare target porque sera chamada de dentro de
   regioes executadas no dispositivo. A versao sequencial/CPU nao precisa dessa
   anotacao, pois todo o codigo roda no mesmo espaco de execucao. */

int calculateNearstOpenMP_GPU(observation *o, cluster clusters[], int k)
{
    return calculateNearestIndexGPU(o, clusters, k);
}

void calculateCentroidOpenMP_GPU(observation observations[], size_t size,
                                 cluster *centroid)
{
    if (!openMPGPUCanRun(size, 1))
    {
        if (shouldAllowOpenMPGPUCPUFallback())
        {
            fprintf(stderr,
                    "omp-gpu: fallback em CPU habilitado por KMEANS_OMP_GPU_ALLOW_CPU_FALLBACK=1.\n");
            calculateCentroidOpenMP(observations, size, centroid);
        }
        else
        {
            centroid->x = 0;
            centroid->y = 0;
            centroid->count = 0;
        }
        return;
    }

    double sumX = 0;
    double sumY = 0;

    centroid->x = 0;
    centroid->y = 0;
    centroid->count = size;

    /* A soma do centroide unico segue a mesma formula da versao sequencial,
       mas o laco e enviado para a GPU e a reduction combina os resultados
       parciais produzidos pelas equipes/threads do dispositivo. */
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

    if (!openMPGPUCanRun(size, k))
    {
        if (shouldAllowOpenMPGPUCPUFallback())
        {
            fprintf(stderr,
                    "omp-gpu: fallback em CPU habilitado por KMEANS_OMP_GPU_ALLOW_CPU_FALLBACK=1.\n");
            return kMeansOpenMP(observations, size, k);
        }
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

        /* A inicializacao segue no host para manter a mesma base de comparacao
           das versoes sequencial e OpenMP CPU. */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }

        /* O bloco target data mantem os vetores residentes no dispositivo
           entre iteracoes. Sem isso, cada laco faria novas transferencias de
           dados entre host e GPU, reduzindo bastante o ganho do offload. */
        #pragma omp target data map(tofrom : observations[0:size]) \
            map(tofrom : clusters[0:k]) map(alloc : sumX[0:k], sumY[0:k], counts[0:k]) \
            map(to : size, k)
        {
            while (keepRunning)
            {
                /* Equivale ao reset dos acumuladores na versao sequencial,
                   mas agora acontece diretamente na memoria do dispositivo. */
                #pragma omp target teams distribute parallel for
                for (int i = 0; i < k; i++)
                {
                    sumX[i] = 0;
                    sumY[i] = 0;
                    counts[i] = 0;
                }

                /* Este e o STEP 2 do sequencial adaptado para GPU. Como muitas
                   threads podem atualizar o mesmo cluster, usamos atomics para
                   proteger sumX/sumY/counts. */
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

                /* Depois das somas, a normalizacao recompõe os centroides. */
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
                    /* Se um cluster ficar vazio, preservamos o centroide
                       anterior em vez de dividir por zero. */
                }

                changed = 0;
                /* Equivale ao STEP 3/4 do sequencial: cada ponto procura o
                   centroide mais proximo. A reduction agrega quantos pontos
                   trocaram de grupo nesta iteracao. */
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

        /* Caso trivial: cada observacao ocupa um cluster proprio. Assim como
           na versao CPU, cada iteracao escreve em um indice independente. */
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
