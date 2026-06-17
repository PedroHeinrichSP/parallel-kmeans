#ifndef K_MEANS_CLUSTERING_CUDA_H
#define K_MEANS_CLUSTERING_CUDA_H

#include "k_means_clustering_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

int calculateNearstCUDA(observation* o, cluster clusters[], int k);
void calculateCentroidCUDA(observation observations[], size_t size, cluster* centroid);
cluster* kMeansCUDA(observation observations[], size_t size, int k);

#ifdef __cplusplus
}
#endif

#endif
