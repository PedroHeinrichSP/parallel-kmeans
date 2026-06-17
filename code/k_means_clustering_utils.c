#define _USE_MATH_DEFINES

#include "headers/k_means_clustering_utils.h"

#include <float.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void generateObservations(observation observations[], size_t size)
{
    double maxRadius = 20.00;

    for (size_t i = 0; i < size; i++)
    {
        double radius = maxRadius * ((double)rand() / RAND_MAX);
        double ang = 2 * M_PI * ((double)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
        observations[i].group = 0;
    }
}

void runKMeans(size_t size, int k, kmeans_algorithm algorithm)
{
    observation* observations = malloc(sizeof(observation) * size);
    if (observations == NULL || algorithm == NULL)
    {
        free(observations);
        return;
    }

    generateObservations(observations, size);

    cluster* clusters = algorithm(observations, size, k);
    if (clusters != NULL)
    {
        const char* skipEPS = getenv("KMEANS_SKIP_EPS");
        if (skipEPS == NULL || strcmp(skipEPS, "1") != 0)
        {
            printEPS(observations, size, clusters, k);
        }
    }

    free(observations);
    free(clusters);
}

void printEPS(observation pts[], size_t len, cluster cent[], int k)
{
    const char* outputPath = "artifacts/images/eps/image.eps";
    int W = 400, H = 400;
    double min_x = DBL_MAX, max_x = DBL_MIN, min_y = DBL_MAX, max_y = DBL_MIN;
    double scale = 0, cx = 0, cy = 0;
    double* colors = (double*)malloc(sizeof(double) * (k * 3));
    FILE* epsFile = NULL;
    int i;
    size_t j;
    double kd = k * 1.0;

    if (colors == NULL)
    {
        return;
    }

    if ((mkdir("artifacts", 0777) != 0 && errno != EEXIST) ||
        (mkdir("artifacts/images", 0777) != 0 && errno != EEXIST) ||
        (mkdir("artifacts/images/eps", 0777) != 0 && errno != EEXIST))
    {
        perror("Could not create EPS output directory");
        free(colors);
        return;
    }

    epsFile = fopen(outputPath, "w");
    if (epsFile == NULL)
    {
        perror("Could not create EPS output file");
        free(colors);
        return;
    }

    /* Generate a deterministic color palette for each cluster id. */
    for (i = 0; i < k; i++)
    {
        *(colors + 3 * i) = (3 * (i + 1) % k) / kd;
        *(colors + 3 * i + 1) = (7 * i % k) / kd;
        *(colors + 3 * i + 2) = (9 * i % k) / kd;
    }

    /* Find the bounding box so the EPS output can be scaled to the canvas. */
    for (j = 0; j < len; j++)
    {
        if (max_x < pts[j].x)
        {
            max_x = pts[j].x;
        }
        if (min_x > pts[j].x)
        {
            min_x = pts[j].x;
        }
        if (max_y < pts[j].y)
        {
            max_y = pts[j].y;
        }
        if (min_y > pts[j].y)
        {
            min_y = pts[j].y;
        }
    }

    scale = W / (max_x - min_x);
    if (scale > (H / (max_y - min_y)))
    {
        scale = H / (max_y - min_y);
    };
    cx = (max_x + min_x) / 2;
    cy = (max_y + min_y) / 2;

    fprintf(epsFile, "%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: -5 -5 %d %d\n",
            W + 10, H + 10);
    fprintf(
        epsFile,
        "/l {rlineto} def /m {rmoveto} def\n"
        "/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n"
        "/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath "
        "\tgsave 1 setgray fill grestore gsave 3 setlinewidth"
        " 1 setgray stroke grestore 0 setgray stroke }def\n");

    /* Print each observation using the color of its assigned cluster. */
    for (int i = 0; i < k; i++)
    {
        fprintf(epsFile, "%g %g %g setrgbcolor\n", *(colors + 3 * i),
                *(colors + 3 * i + 1), *(colors + 3 * i + 2));
        for (j = 0; j < len; j++)
        {
            if (pts[j].group != i)
            {
                continue;
            }
            fprintf(epsFile, "%.3f %.3f c\n",
                    (pts[j].x - cx) * scale + W / 2,
                    (pts[j].y - cy) * scale + H / 2);
        }
        fprintf(epsFile, "\n0 setgray %g %g s\n",
                (cent[i].x - cx) * scale + W / 2,
                (cent[i].y - cy) * scale + H / 2);
    }
    fprintf(epsFile, "\n%%%%EOF");

    fclose(epsFile);
    free(colors);
}
