#ifndef K_MEANS_CLUSTERING_OPENMP_H
#define K_MEANS_CLUSTERING_OPENMP_H

#include "k_means_clustering_utils.h"
#

/* Placeholder: OpenMP implementation pending. */
int calculateNearstOpenMP(observation* o, cluster clusters[], int k);
void calculateCentroidOpenMP(observation observations[], size_t size, cluster* centroid);
cluster* kMeansOpenMP(observation observations[], size_t size, int k);

#endif
