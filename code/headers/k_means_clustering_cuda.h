#ifndef K_MEANS_CLUSTERING_CUDA_H
#define K_MEANS_CLUSTERING_CUDA_H

#include "k_means_clustering_utils.h"

/* Placeholder: CUDA implementation pending. */
int calculateNearstCUDA(observation* o, cluster clusters[], int k);
void calculateCentroidCUDA(observation observations[], size_t size, cluster* centroid);
cluster* kMeansCUDA(observation observations[], size_t size, int k);

#endif
