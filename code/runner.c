#include "headers/k_means_clustering_sequencial.h"
#include "headers/k_means_clustering_openmp.h"
#include "headers/k_means_clustering_openmp_gpu.h"
#include "headers/k_means_clustering_cuda.h"
#include "headers/k_means_clustering_utils.h"

#include <stdlib.h>

int main(int argc, char* argv[]) {
    srand(42);

    size_t size = 100000L;
    int k = 5;
    kmeans_algorithm algorithm = kMeans;

    if (argc > 1) {
        size = (size_t)atoi(argv[1]);
    }
    if (argc > 2) {
        k = atoi(argv[2]);
    }
    if (argc > 3) {
        switch (argv[3][0]) {
        case 'o':
            algorithm = kMeansOpenMP;
            break;
        case 'g':
            algorithm = kMeansOpenMP_GPU;
            break;
        case 'c':
            algorithm = kMeansCUDA;
            break;
        case 's':
        default:
            algorithm = kMeans;
            break;
        }
    }

    return runKMeans(size, k, algorithm);
}
