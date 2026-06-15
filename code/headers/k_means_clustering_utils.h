#ifndef K_MEANS_CLUSTERING_UTILS_H
#define K_MEANS_CLUSTERING_UTILS_H

#include <stddef.h>

typedef struct observation
{
    double x;
    double y;
    int group;
} observation;

typedef struct cluster
{
    double x;
    double y;
    size_t count;
} cluster;

/*!
 * Defines a common signature for K-Means implementations
 * used by the runner utility
 *
 * @param observations  an array of observations to cluster
 * @param size  size of observations array
 * @param k  no of clusters to be made
 *
 * @returns pointer to cluster object
 */
typedef cluster* (*kmeans_algorithm)(observation observations[], size_t size, int k);

/*!
 * Generate observations in a circle of
 * radius 20.0 with center at (0,0)
 *
 * @param observations  an array of observations to fill
 * @param size  size of observations array
 */
void generateObservations(observation observations[], size_t size);

/*!
 * Runs a K-Means implementation with generated observations
 * and prints the result in EPS format
 *
 * @param size  size of observations array
 * @param k  no of clusters to be made
 * @param algorithm  K-Means implementation to execute
 */
void runKMeans(size_t size, int k, kmeans_algorithm algorithm);

/*!
 * A function to print observations and clusters
 * The code is taken from
 * http://rosettacode.org/wiki/K-means%2B%2B_clustering.
 * Even the K Means code is also inspired from it
 *
 * @note Creates artifacts/images/eps/image.eps directly.
 *
 * @param observations  observations array
 * @param len  size of observation array
 * @param cent  clusters centroid's array
 * @param k  size of cent array
 */
void printEPS(observation pts[], size_t len, cluster cent[], int k);

#endif
