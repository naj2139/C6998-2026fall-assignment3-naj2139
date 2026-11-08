#include "common.h"

// TODO (Part C, C1 -- 15 pts): naive scaled dot-product attention.
//
// Compute, for Q, K, V of shape (N, d):
//   1. scores:  S[i,j] = (1/sqrt(d)) * sum_t Q[i,t] * K[j,t]      (N x N)
//   2. softmax: row-wise over S, SAFE (subtract the row max before exp)
//   3. output:  O[i,t] = sum_j P[i,j] * V[j,t]                    (N x d)
//
// This is the "naive" baseline: materialize the full N x N score matrix S
// in global memory and read it back. That HBM traffic is exactly what the
// tiled C2 kernel removes. All three implementations must produce the same
// attention output; the grader checks each one element-wise against an
// independent reference (the printed checksum is only for your report).
//
// Three kernel stubs are provided with signatures (scores, row softmax,
// output); you choose the thread/block decomposition and any shared memory,
// and you set the launch configuration in main(). The stubs are empty and
// the driver zeros the output, so the program compiles and runs but will not
// match the reference until you implement the kernels and their launches.

// Fill S (N x N, row-major) with the scaled scores: S[i,j] must end up equal
// to scale * dot(Q[i,:], K[j,:]), where Q and K are (N, d) row-major. You
// choose the thread/block decomposition.
__global__ void scores_kernel(int N, int d, double scale,
                               const double *Q, const double *K, double *S)
{
    // TODO (C1): compute the scores for the entries this thread owns.
    (void)N; (void)d; (void)scale; (void)Q; (void)K; (void)S;
}

// Replace S in place with its numerically safe row-wise softmax: subtract
// each row's max before exponentiating, then divide by the row sum. You
// choose how to parallelize the per-row reduction.
__global__ void softmax_rows_kernel(int N, double *S)
{
    // TODO (C1): safe row softmax of S.
    (void)N; (void)S;
}

// Fill O (N x d, row-major) with O[i,t] = sum_j P[i,j] * V[j,t], where P is
// the softmax weights (N x N, row-major) and V is (N, d) row-major. You
// choose the thread/block decomposition.
__global__ void output_kernel(int N, int d, const double *P, const double *V, double *O)
{
    // TODO (C1): compute the output entries this thread owns.
    (void)N; (void)d; (void)P; (void)V; (void)O;
}

int main(int argc, char **argv)
{
    AttnDims a = parseDims(argc, argv);
    double scale = 1.0 / sqrt((double)a.d);

    double *hQ = makeQ(a), *hK = makeK(a), *hV = makeV(a);
    size_t qkvBytes = (size_t)a.N * a.d * sizeof(double);
    size_t sBytes = (size_t)a.N * a.N * sizeof(double);

    double *dQ, *dK, *dV, *dS, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dK, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dV, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dS, sBytes));
    CUDA_CHECK(cudaMalloc(&dO, qkvBytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dO, 0, qkvBytes));  // defined output until you launch your kernels

    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);
    cudaEventRecord(startE);
    // TODO (C1): choose the grid/block (and any dynamic shared memory) for
    // each kernel and launch them in order: scores_kernel, then
    // softmax_rows_kernel, then output_kernel. The timing scaffold around
    // these launches is provided; do not change it.
    //   scores_kernel<<<...>>>(a.N, a.d, scale, dQ, dK, dS);
    //   softmax_rows_kernel<<<...>>>(a.N, dS);
    //   output_kernel<<<...>>>(a.N, a.d, dS, dV, dO);
    cudaEventRecord(stopE);
    cudaEventSynchronize(stopE);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startE, stopE);

    double *hO = (double *)malloc(qkvBytes);
    CUDA_CHECK(cudaMemcpy(hO, dO, qkvBytes, cudaMemcpyDeviceToHost));
    double cs = checksum(hO, (size_t)a.N * a.d);

    // Required program output: "checksum,time_ms" with 3 decimals (spec: -3 pts if wrong format).
    printf("%.3f,%.3f\n", cs, (double)ms);
    if ((size_t)a.N * a.d <= 4096) dumpOutput(hO, (size_t)a.N * a.d);
    // Machine-parseable line for the autograder / grade_local.py.
    printf("RESULT,program=c1_naive,N=%d,d=%d,time_ms=%.3f,checksum=%.6f\n",
           a.N, a.d, (double)ms, cs);

    free(hQ); free(hK); free(hV); free(hO);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dS); cudaFree(dO);
    return 0;
}
