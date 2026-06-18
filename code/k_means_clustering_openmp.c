/**
 * @file k_means_clustering_openmp.c
 * @brief Versao paralela em CPU do algoritmo K-Means usando OpenMP.
 *
 * Comparacao com k_means_clustering_sequencial.c:
 * - A logica do algoritmo e a mesma: inicializa grupos, recalcula centroides
 *   e reclassifica os pontos ate estabilizar.
 * - A diferenca e que os lacos que percorrem todas as observacoes deixam de ser
 *   executados por um unico fluxo e passam a ser divididos entre varias
 *   threads.
 *
 * Mudancas principais e por que elas sao necessarias:
 * - A inicializacao com rand() foi mantida no host para preservar a mesma
 *   sensibilidade do algoritmo ao estado inicial e permitir comparacao justa
 *   com a versao sequencial.
 * - O loop principal foi envolvido por uma regiao paralela persistente para
 *   evitar o custo de criar/destruir threads a cada iteracao do K-Means.
 * - As somas de centroides nao podem mais escrever diretamente em
 *   clusters[group], como na versao sequencial, porque varias threads podem
 *   atingir o mesmo grupo ao mesmo tempo. Por isso usamos acumuladores
 *   privados por thread e um merge posterior.
 * - Os acumuladores privados foram alinhados e preenchidos com padding para
 *   reduzir falso compartilhamento entre threads.
 * - O numero de mudancas de grupo tambem passa a ser computado localmente por
 *   thread e consolidado depois, evitando contencao em um contador global.
 * - O caso especial k <= 1 usa reduction, que preserva a mesma formula da
 *   versao sequencial sem serializar a soma de todos os pontos.
 */

#include "headers/k_means_clustering_openmp.h"

#include <float.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define CACHE_LINE_BYTES 64

/* A versao sequencial nao precisa disso porque so existe um escritor.
   Em paralelo, arredondar o stride para a linha de cache ajuda a evitar
   falso compartilhamento entre acumuladores de threads diferentes. */
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

/* A versao sequencial usa apenas o vetor final de clusters. Aqui surgem
   buffers thread-local para transformar atualizacoes concorrentes em
   acumulacoes privadas seguidas de merge. */
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

/* A regra de distancia e igual a da versao sequencial; o ganho vem do fato de
   que varias observacoes chamam esta rotina ao mesmo tempo em threads
   diferentes. */
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

    /* Na versao sequencial um unico laco acumula x/y. Em OpenMP, a reduction
       cria somas privadas por thread e combina tudo no fim sem corrida. */
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

        /* Mantemos a mesma inicializacao da versao sequencial porque o numero
           de iteracoes depende do chute inicial dos grupos. */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }

        /* A regiao paralela persistente e uma diferenca importante em relacao
           ao sequencial: ela amortiza o overhead de gerenciar threads ao longo
           de todas as iteracoes do algoritmo. */
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
                /* Na versao sequencial bastaria zerar clusters[i]. Aqui cada
                   thread precisa limpar o proprio buffer local inteiro; usar
                   omp for nesse passo deixaria partes antigas sem limpeza. */
                for (int i = 0; i < k; i++)
                {
                    threadX[i] = 0;
                    threadY[i] = 0;
                    threadCount[i] = 0;
                }

                /* Este trecho corresponde ao STEP 2 da versao sequencial.
                   A diferenca e que o acumulador agora e privado por thread,
                   evitando corrida quando muitos pontos caem no mesmo grupo. */
                #pragma omp for schedule(static)
                for (size_t j = 0; j < size; j++)
                {
                    int group = observations[j].group;
                    threadX[group] += observations[j].x;
                    threadY[group] += observations[j].y;
                    threadCount[group]++;
                }

                /* O merge recompõe o mesmo resultado numerico esperado da
                   versao sequencial, mas separa a fase de escrita concorrente
                   da fase de consolidacao. */
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
                    /* Se count == 0, nao podemos dividir. Manter o centroide
                       anterior evita instabilidade numerica e clusters falsos
                       na origem. */
                }

                /* Este trecho corresponde ao STEP 3/4 da versao sequencial.
                   O contador de mudancas vira local por thread para eliminar
                   sincronizacao a cada ponto reclassificado. */
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

                /* A barreira garante que todas as threads terminaram de
                   publicar seus contadores locais antes da decisao de
                   convergencia. */
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

        /* No caso trivial, cada iteracao escreve em uma posicao independente.
           Por isso a paralelizacao e direta, sem buffers auxiliares. */
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
