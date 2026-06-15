#ifndef K_MEANS_CLUSTERING_OPENMP_GPU_H
#define K_MEANS_CLUSTERING_OPENMP_GPU_H

#include "k_means_clustering_utils.h"

/* Placeholder: OpenMP GPU implementation pending. */
int calculateNearstOpenMP_GPU(observation* o, cluster clusters[], int k);
void calculateCentroidOpenMP_GPU(observation observations[], size_t size, cluster* centroid);
cluster* kMeansOpenMP_GPU(observation observations[], size_t size, int k);

#endif
