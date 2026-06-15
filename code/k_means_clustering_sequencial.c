/**
 * @file k_means_clustering.c
 * @brief K Means Clustering Algorithm implemented
 * @details
 * This file has K Means algorithm implemmented
 * It prints test output in eps format
 *
 * Note:
 * Though the code for clustering works for all the
 * 2D data points and can be extended for any size vector
 * by making the required changes, but note that
 * the output method i.e. printEPS is only good for
 * polar data points i.e. in a circle and both test
 * use the same.
 * @author [Lakhan Nad](https://github.com/Lakhan-Nad)
 */

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <stdio.h>        /* printf */
#include <stdlib.h>       /* rand */
#include <string.h>       /* memset */
#include <time.h>         /* time */
#include "k_means_clustering_sequencial.h"

/*!
 * @addtogroup machine_learning Machine Learning Algorithms
 * @{
 * @addtogroup k_means K-Means Clustering Algorithm
 * @{
 */


int calculateNearst(observation* o, cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    int i = 0;
    for (; i < k; i++)
    {
        /* Calculate Squared Distance*/
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

void calculateCentroid(observation observations[], size_t size,
                       cluster* centroid)
{
    size_t i = 0;
    centroid->x = 0;
    centroid->y = 0;
    centroid->count = size;
    for (; i < size; i++)
    {
        centroid->x += observations[i].x;
        centroid->y += observations[i].y;
        observations[i].group = 0;
    }
    centroid->x /= centroid->count;
    centroid->y /= centroid->count;
}

cluster* kMeans(observation observations[], size_t size, int k)
{
    cluster* clusters = NULL;
    if (k <= 1)
    {
        /*
        If we have to cluster them only in one group
        then calculate centroid of observations and
        that will be a ingle cluster
        */
        clusters = (cluster*)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        calculateCentroid(observations, size, clusters);
    }
    else if (k < size)
    {
        clusters = malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        /* STEP 1 */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }
        size_t changed = 0;
        size_t minAcceptedError =
            size /
            10000;  // Do until 99.99 percent points are in correct cluster
        int t = 0;
        do
        {
            /* Initialize clusters */
            for (int i = 0; i < k; i++)
            {
                clusters[i].x = 0;
                clusters[i].y = 0;
                clusters[i].count = 0;
            }
            /* STEP 2*/
            for (size_t j = 0; j < size; j++)
            {
                t = observations[j].group;
                clusters[t].x += observations[j].x;
                clusters[t].y += observations[j].y;
                clusters[t].count++;
            }
            for (int i = 0; i < k; i++)
            {
                clusters[i].x /= clusters[i].count;
                clusters[i].y /= clusters[i].count;
            }
            /* STEP 3 and 4 */
            changed = 0;  // this variable stores change in clustering
            for (size_t j = 0; j < size; j++)
            {
                t = calculateNearst(observations + j, clusters, k);
                if (t != observations[j].group)
                {
                    changed++;
                    observations[j].group = t;
                }
            }
        } while (changed > minAcceptedError);  // Keep on grouping until we have
                                               // got almost best clustering
    }
    else
    {
        /* If no of clusters is more than observations
           each observation can be its own cluster
        */
        clusters = (cluster*)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        for (int j = 0; j < size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = j;
        }
    }
    return clusters;
}

