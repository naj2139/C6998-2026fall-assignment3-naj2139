#ifndef HPML_HW3_PARTC_COMMON_H
#define HPML_HW3_PARTC_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// Part-C problem: one head of scaled dot-product attention.
//
//   Q, K, V each have shape (N, d), row-major, double precision.
//   S = scale * Q * K^T          (N x N), scale = 1 / sqrt(d)
//   P = softmax(S) row-wise       (N x N), safe softmax (subtract row max)
//   O = P * V                     (N x d)
//
// c1_naive materializes S in global memory (the N x N scores, written and
// read back -- the "naive attention" baseline). c2_tiled computes the same
// result with an online softmax over key/value tiles so the N x N score
// matrix never touches global memory (FlashAttention-style tiling).
// c3_cublas builds the same result from cuBLAS GEMMs plus a softmax kernel.
// The grader checks each one element-wise against an independent reference.
//
// Default size (N=2048, d=64) is used for the timing numbers the report
// needs. All programs accept optional positional overrides `N d` so the
// grader can run a tiny size for a fast, exact CPU-reference correctness
// check first. d must be a multiple of 32 is NOT required; kernels handle
// a general d, but d <= 1024 so one head fits a block.
#define DEFAULT_N 2048
#define DEFAULT_D 64

struct AttnDims { int N, d; };

inline AttnDims parseDims(int argc, char **argv) {
    AttnDims a{DEFAULT_N, DEFAULT_D};
    int *fields[2] = {&a.N, &a.d};
    for (int i = 1; i < argc && i <= 2; ++i) *fields[i - 1] = atoi(argv[i]);
    return a;
}

// Deterministic, bounded inputs. Q and K span roughly [-2.5, 2.5], so the
// scaled logits have enough spread that the softmax concentrates weight on a
// few keys per query rather than spreading it uniformly. V is strictly
// positive and varies with the key index j, so the output depends on which
// keys a query attends to. These two properties make the output of a correct
// attention differ from the output of any weight-agnostic shortcut (for
// example averaging all V rows), which the element-wise reference check in
// grade_local.py relies on.
inline double qval(int i, int t) { return (double)(((i * 2 + t) % 11) - 5) / 2.0; }
inline double kval(int j, int t) { return (double)(((j + t * 3) % 11) - 5) / 2.0; }
inline double vval(int j, int t) { return (double)((j % 13) + 1) * (0.05 + 0.02 * t); }

inline double *makeQ(const AttnDims &a) {
    double *Q = (double *)malloc((size_t)a.N * a.d * sizeof(double));
    for (int i = 0; i < a.N; ++i)
        for (int t = 0; t < a.d; ++t)
            Q[(size_t)i * a.d + t] = qval(i, t);
    return Q;
}

inline double *makeK(const AttnDims &a) {
    double *K = (double *)malloc((size_t)a.N * a.d * sizeof(double));
    for (int j = 0; j < a.N; ++j)
        for (int t = 0; t < a.d; ++t)
            K[(size_t)j * a.d + t] = kval(j, t);
    return K;
}

inline double *makeV(const AttnDims &a) {
    double *V = (double *)malloc((size_t)a.N * a.d * sizeof(double));
    for (int j = 0; j < a.N; ++j)
        for (int t = 0; t < a.d; ++t)
            V[(size_t)j * a.d + t] = vval(j, t);
    return V;
}

inline double checksum(const double *O, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += O[i];
    return s;
}

// Dump every element of the output on one line, as
// "OUT,<v0>,<v1>,...", for the grader's element-wise reference check. Only
// called at the small correctness size (gated by a size cap in each main),
// never for the full timing run, so the normal output stays compact.
inline void dumpOutput(const double *O, size_t n) {
    printf("OUT");
    for (size_t i = 0; i < n; ++i) printf(",%.9g", O[i]);
    printf("\n");
}

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

#endif  // HPML_HW3_PARTC_COMMON_H
