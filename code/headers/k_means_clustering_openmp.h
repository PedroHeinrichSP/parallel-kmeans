#ifndef K_MEANS_CLUSTERING_OPENMP_H
#define K_MEANS_CLUSTERING_OPENMP_H

#include <stddef.h>
#include "k_means_clustering_utils.h"

/*
 * Versoes OpenMP das mesmas funcoes expostas pela implementacao sequencial.
 * As mudancas de paralelizacao estao documentadas em k_means_clustering_openmp.c.
 */
int calculateNearstOpenMP(observation* o, cluster clusters[], int k);
void calculateCentroidOpenMP(observation observations[], size_t size, cluster* centroid);
cluster* kMeansOpenMP(observation observations[], size_t size, int k);

#endif
