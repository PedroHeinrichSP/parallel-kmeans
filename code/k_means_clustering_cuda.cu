// kmeans_cuda_balanced.cu
// Primeira versão CUDA do K-means 2D usando orientação a dados (SoA).
// Objetivo: base clara, funcional e otimzável.
//
// Compilar exemplo:
//   nvcc -O3 -std=c++11 -DKMEANS_CUDA_DEMO_MAIN kmeans_cuda_balanced.cu -o kmeans_cuda
//
// Executar:
//   ./kmeans_cuda

#include "headers/k_means_clustering_cuda.h"

#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <vector>

using real_t = double;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(err__));       \
            throw std::runtime_error(cudaGetErrorString(err__));               \
        }                                                                      \
    } while (0)

static void* cuda_malloc_bytes(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

// Versão global: não usa shared memory para centroides.
// Serve como fallback quando K é grande demais para shared memory.
__global__ void assign_clusters_global_kernel(
    const real_t* __restrict__ x,
    const real_t* __restrict__ y,
    int* __restrict__ label,
    const real_t* __restrict__ cx,
    const real_t* __restrict__ cy,
    int* __restrict__ changed,
    int n,
    int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    real_t px = x[i];
    real_t py = y[i];

    real_t best_dist = DBL_MAX;
    int best = 0;

    for (int c = 0; c < k; ++c) {
        real_t dx = cx[c] - px;
        real_t dy = cy[c] - py;
        real_t dist = dx * dx + dy * dy;
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }

    int old = label[i];
    changed[i] = (old != best) ? 1 : 0;
    label[i] = best;
}

// Versão shared: carrega todos os centroides do K atual em shared memory.
// Boa para K pequeno/médio. Para 2D, o custo é 2 * K * sizeof(real_t) por bloco.
__global__ void assign_clusters_shared_kernel(
    const real_t* __restrict__ x,
    const real_t* __restrict__ y,
    int* __restrict__ label,
    const real_t* __restrict__ cx,
    const real_t* __restrict__ cy,
    int* __restrict__ changed,
    int n,
    int k)
{
    extern __shared__ real_t shared_centroids[];
    real_t* scx = shared_centroids;
    real_t* scy = shared_centroids + k;

    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        scx[c] = cx[c];
        scy[c] = cy[c];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    real_t px = x[i];
    real_t py = y[i];

    real_t best_dist = DBL_MAX;
    int best = 0;

    for (int c = 0; c < k; ++c) {
        real_t dx = scx[c] - px;
        real_t dy = scy[c] - py;
        real_t dist = dx * dx + dy * dy;
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }

    int old = label[i];
    changed[i] = (old != best) ? 1 : 0;
    label[i] = best;
}

// Cada ponto soma sua contribuição no centroide atual.
// Esta é a versão simples/balanceada: atomicAdd global.
// Próxima otimização provável: redução por bloco ou sort + segmented reduce.
__global__ void accumulate_centroids_kernel(
    const real_t* __restrict__ x,
    const real_t* __restrict__ y,
    const int* __restrict__ label,
    real_t* __restrict__ sum_x,
    real_t* __restrict__ sum_y,
    int* __restrict__ count,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int c = label[i];
    atomicAdd(&sum_x[c], x[i]);
    atomicAdd(&sum_y[c], y[i]);
    atomicAdd(&count[c], 1);
}

// Um thread por cluster recalcula o centroide.
// Se um cluster ficar vazio, mantemos o centroide anterior.
__global__ void finalize_centroids_kernel(
    real_t* __restrict__ cx,
    real_t* __restrict__ cy,
    const real_t* __restrict__ sum_x,
    const real_t* __restrict__ sum_y,
    const int* __restrict__ count,
    int k)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;

    int cnt = count[c];
    if (cnt > 0) {
        cx[c] = sum_x[c] / static_cast<real_t>(cnt);
        cy[c] = sum_y[c] / static_cast<real_t>(cnt);
    }
}

static void choose_initial_centroids_rand(
    const std::vector<real_t>& hx,
    const std::vector<real_t>& hy,
    std::vector<real_t>& hcx,
    std::vector<real_t>& hcy,
    unsigned seed)
{
    const int n = static_cast<int>(hx.size());
    const int k = static_cast<int>(hcx.size());

    std::srand(seed);

    // Mantém rand(), mas evita escolher o mesmo ponto duas vezes.
    std::vector<int> chosen;
    chosen.reserve(k);

    for (int c = 0; c < k; ++c) {
        int idx = 0;
        bool repeated = false;
        do {
            idx = std::rand() % n;
            repeated = false;
            for (int old : chosen) {
                if (old == idx) {
                    repeated = true;
                    break;
                }
            }
        } while (repeated);

        chosen.push_back(idx);
        hcx[c] = hx[idx];
        hcy[c] = hy[idx];
    }
}

static std::vector<cluster> kmeans_cuda_2d(
    std::vector<observation>& observations,
    int k,
    int max_iters = 300,
    real_t tolerance_fraction = static_cast<real_t>(0.0001),
    unsigned seed = static_cast<unsigned>(std::time(nullptr)))
{
    const int n = static_cast<int>(observations.size());
    if (n <= 0) throw std::invalid_argument("observations vazio");
    if (k <= 0) throw std::invalid_argument("k deve ser positivo");
    if (k > n) throw std::invalid_argument("k nao pode ser maior que n nesta versao");

    // Caso trivial: um cluster só, fica no CPU.
    if (k == 1) {
        real_t sx = 0;
        real_t sy = 0;
        for (auto& p : observations) {
            sx += p.x;
            sy += p.y;
            p.group = 0;
        }
        return std::vector<cluster>{{sx / n, sy / n, static_cast<size_t>(n)}};
    }

    // Host em SoA.
    std::vector<real_t> hx(n), hy(n);
    std::vector<int> hlabel(n, -1);
    for (int i = 0; i < n; ++i) {
        hx[i] = observations[i].x;
        hy[i] = observations[i].y;
    }

    std::vector<real_t> hcx(k), hcy(k);
    std::vector<int> hcount(k, 0);

    // Requisito de negócio: rand() escolhe os centros iniciais.
    choose_initial_centroids_rand(hx, hy, hcx, hcy, seed);

    const size_t n_real_bytes = static_cast<size_t>(n) * sizeof(real_t);
    const size_t n_int_bytes  = static_cast<size_t>(n) * sizeof(int);
    const size_t k_real_bytes = static_cast<size_t>(k) * sizeof(real_t);
    const size_t k_int_bytes  = static_cast<size_t>(k) * sizeof(int);

    real_t* d_x = static_cast<real_t*>(cuda_malloc_bytes(n_real_bytes));
    real_t* d_y = static_cast<real_t*>(cuda_malloc_bytes(n_real_bytes));
    int* d_label = static_cast<int*>(cuda_malloc_bytes(n_int_bytes));

    real_t* d_cx = static_cast<real_t*>(cuda_malloc_bytes(k_real_bytes));
    real_t* d_cy = static_cast<real_t*>(cuda_malloc_bytes(k_real_bytes));

    real_t* d_sum_x = static_cast<real_t*>(cuda_malloc_bytes(k_real_bytes));
    real_t* d_sum_y = static_cast<real_t*>(cuda_malloc_bytes(k_real_bytes));
    int* d_count = static_cast<int*>(cuda_malloc_bytes(k_int_bytes));
    int* d_changed = static_cast<int*>(cuda_malloc_bytes(n_int_bytes));

    try {
        CUDA_CHECK(cudaMemcpy(d_x, hx.data(), n_real_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, hy.data(), n_real_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_label, hlabel.data(), n_int_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cx, hcx.data(), k_real_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cy, hcy.data(), k_real_bytes, cudaMemcpyHostToDevice));

        const int threads = 256;
        const int point_blocks = (n + threads - 1) / threads;
        const int cluster_blocks = (k + threads - 1) / threads;
        const int min_changed = static_cast<int>(n * tolerance_fraction);

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        int max_shared_bytes = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &max_shared_bytes,
            cudaDevAttrMaxSharedMemoryPerBlock,
            device));

        const size_t shared_centroids_bytes = 2ULL * static_cast<size_t>(k) * sizeof(real_t);
        const bool use_shared_centroids = shared_centroids_bytes <= static_cast<size_t>(max_shared_bytes);

        int total_changed = n;
        int iter = 0;

        for (; iter < max_iters; ++iter) {
            // 1) Assignment: cada thread cuida de um ponto.
            if (use_shared_centroids) {
                assign_clusters_shared_kernel<<<point_blocks, threads, shared_centroids_bytes>>>(
                    d_x, d_y, d_label, d_cx, d_cy, d_changed, n, k);
            } else {
                assign_clusters_global_kernel<<<point_blocks, threads>>>(
                    d_x, d_y, d_label, d_cx, d_cy, d_changed, n, k);
            }
            CUDA_CHECK(cudaGetLastError());

            // 2) Zera acumuladores.
            CUDA_CHECK(cudaMemset(d_sum_x, 0, k_real_bytes));
            CUDA_CHECK(cudaMemset(d_sum_y, 0, k_real_bytes));
            CUDA_CHECK(cudaMemset(d_count, 0, k_int_bytes));

            // 3) Acumula soma e contagem por cluster.
            accumulate_centroids_kernel<<<point_blocks, threads>>>(
                d_x, d_y, d_label, d_sum_x, d_sum_y, d_count, n);
            CUDA_CHECK(cudaGetLastError());

            // 4) Finaliza centroides.
            finalize_centroids_kernel<<<cluster_blocks, threads>>>(
                d_cx, d_cy, d_sum_x, d_sum_y, d_count, k);
            CUDA_CHECK(cudaGetLastError());

            // 5) Reduz changed[] para saber se convergiu.
            total_changed = thrust::reduce(
                thrust::device,
                d_changed,
                d_changed + n,
                0,
                thrust::plus<int>());

            if (total_changed <= min_changed) {
                break;
            }
        }

        CUDA_CHECK(cudaMemcpy(hlabel.data(), d_label, n_int_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hcx.data(), d_cx, k_real_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hcy.data(), d_cy, k_real_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hcount.data(), d_count, k_int_bytes, cudaMemcpyDeviceToHost));

        for (int i = 0; i < n; ++i) {
            observations[i].group = hlabel[i];
        }

        std::vector<cluster> result(k);
        for (int c = 0; c < k; ++c) {
            result[c] = cluster{hcx[c], hcy[c], static_cast<size_t>(hcount[c])};
        }

        std::printf("kmeans_cuda_2d: iteracoes=%d, changed_final=%d, shared_centroids=%s\n",
                    iter + 1,
                    total_changed,
                    use_shared_centroids ? "sim" : "nao");

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_label);
        cudaFree(d_cx);
        cudaFree(d_cy);
        cudaFree(d_sum_x);
        cudaFree(d_sum_y);
        cudaFree(d_count);
        cudaFree(d_changed);

        return result;
    } catch (...) {
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_label);
        cudaFree(d_cx);
        cudaFree(d_cy);
        cudaFree(d_sum_x);
        cudaFree(d_sum_y);
        cudaFree(d_count);
        cudaFree(d_changed);
        throw;
    }
}

int calculateNearstCUDA(observation* o, cluster clusters[], int k)
{
    real_t best_dist = DBL_MAX;
    int best = -1;

    for (int c = 0; c < k; ++c) {
        real_t dx = clusters[c].x - o->x;
        real_t dy = clusters[c].y - o->y;
        real_t dist = dx * dx + dy * dy;

        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }

    return best;
}

void calculateCentroidCUDA(observation observations[], size_t size, cluster* centroid)
{
    centroid->x = 0.0;
    centroid->y = 0.0;
    centroid->count = size;

    for (size_t i = 0; i < size; ++i) {
        centroid->x += observations[i].x;
        centroid->y += observations[i].y;
        observations[i].group = 0;
    }

    centroid->x /= centroid->count;
    centroid->y /= centroid->count;
}

cluster* kMeansCUDA(observation observations[], size_t size, int k)
{
    if (observations == NULL || size == 0 || k <= 0) {
        return NULL;
    }

    try {
        std::vector<observation> cuda_observations(observations,
                                                   observations + size);
        std::vector<cluster> cuda_clusters = kmeans_cuda_2d(cuda_observations, k);
        cluster* clusters = static_cast<cluster*>(std::malloc(sizeof(cluster) * k));

        if (clusters == NULL) {
            return NULL;
        }

        for (size_t i = 0; i < size; ++i) {
            observations[i].group = cuda_observations[i].group;
        }

        for (int c = 0; c < k; ++c) {
            clusters[c] = cuda_clusters[c];
        }

        return clusters;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "kMeansCUDA failed: %s\n", error.what());
        return NULL;
    }
}

#ifdef KMEANS_CUDA_DEMO_MAIN
int main() {
    std::vector<observation> points;
    points.reserve(12);

    // Grupo visual 1
    points.push_back({1.0f, 1.0f, -1});
    points.push_back({1.2f, 0.8f, -1});
    points.push_back({0.8f, 1.1f, -1});
    points.push_back({1.1f, 1.3f, -1});

    // Grupo visual 2
    points.push_back({8.0f, 8.0f, -1});
    points.push_back({8.2f, 7.9f, -1});
    points.push_back({7.8f, 8.1f, -1});
    points.push_back({8.1f, 8.3f, -1});

    // Grupo visual 3
    points.push_back({4.0f, 12.0f, -1});
    points.push_back({4.2f, 11.8f, -1});
    points.push_back({3.8f, 12.1f, -1});
    points.push_back({4.1f, 12.3f, -1});

    int k = 3;
    auto clusters = kmeans_cuda_2d(points, k, 100, 0.0f, 42);

    for (int c = 0; c < k; ++c) {
        std::printf("cluster %d: x=%.4f y=%.4f count=%zu\n",
                    c, clusters[c].x, clusters[c].y, clusters[c].count);
    }

    for (size_t i = 0; i < points.size(); ++i) {
        std::printf("ponto %zu: (%.2f, %.2f) -> grupo %d\n",
                    i, points[i].x, points[i].y, points[i].group);
    }

    return 0;
}
#endif
