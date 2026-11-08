#include "common.h"
#include <cublas_v2.h>

// TODO (Part C, C3 -- 15 pts): the same attention built from cuBLAS GEMMs.
//
// The two matmuls in attention (Q*K^T and P*V) are exactly what a BLAS
// library is tuned for. Call cublasDgemm for both and keep only the softmax
// as a hand-written kernel (BLAS has no softmax). Must produce the same
// checksum as c1_naive / c2_tiled.
//
// Q, K, V, S, O are all row-major. cuBLAS is column-major, so you must work
// out the op flags, operand order, and leading dimensions yourself so that
// each GEMM produces the row-major result the next step expects; your report
// explains that mapping.
//
// The stubs are empty and the driver zeros S and O, so the program compiles
// and runs but will not match the reference until you implement the softmax
// and add the two GEMMs.

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t _s = (x); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// TODO (C3): safe row softmax of S (N x N, row-major) in place. This is the
// same operation as the softmax you wrote in C1; reuse that implementation
// (and launch it with your own configuration in main). The stub leaves S
// unchanged, so C3 will not match the reference until you fill it in.
__global__ void softmax_rows_kernel(int N, double *S)
{
    // TODO (C3): safe row-wise softmax of S.
    (void)N; (void)S;
}

int main(int argc, char **argv)
{
    AttnDims a = parseDims(argc, argv);
    int N = a.N, d = a.d;
    double scale = 1.0 / sqrt((double)d);

    double *hQ = makeQ(a), *hK = makeK(a), *hV = makeV(a);
    size_t qkvBytes = (size_t)N * d * sizeof(double);
    size_t sBytes = (size_t)N * N * sizeof(double);

    double *dQ, *dK, *dV, *dS, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dK, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dV, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dS, sBytes));
    CUDA_CHECK(cudaMalloc(&dO, qkvBytes));
    CUDA_CHECK(cudaMemset(dS, 0, sBytes));
    CUDA_CHECK(cudaMemset(dO, 0, qkvBytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, qkvBytes, cudaMemcpyHostToDevice));

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));

    // alpha/beta scalars for the two GEMMs (the QK^T GEMM uses scale):
    const double alpha_s = scale, beta0 = 0.0, alpha1 = 1.0;
    (void)alpha_s; (void)beta0; (void)alpha1;

    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);
    cudaEventRecord(startE);

    // TODO (C3): implement the three steps in order, inside this timed block:
    //   1. S = scale * Q * K^T   (one cublasDgemm; result row-major N x N in dS)
    //   2. softmax over the rows of S (launch softmax_rows_kernel with your
    //      own grid/block/shared-memory configuration)
    //   3. O = P * V             (one cublasDgemm; result row-major N x d in dO)
    // Do not change the timing scaffold around this block.

    cudaEventRecord(stopE);
    cudaEventSynchronize(stopE);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startE, stopE);

    double *hO = (double *)malloc(qkvBytes);
    CUDA_CHECK(cudaMemcpy(hO, dO, qkvBytes, cudaMemcpyDeviceToHost));
    double cs = checksum(hO, (size_t)N * d);

    printf("%.3f,%.3f\n", cs, (double)ms);
    if ((size_t)N * d <= 4096) dumpOutput(hO, (size_t)N * d);
    printf("RESULT,program=c3_cublas,N=%d,d=%d,time_ms=%.3f,checksum=%.6f\n",
           N, d, (double)ms, cs);

    cublasDestroy(h);
    free(hQ); free(hK); free(hV); free(hO);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dS); cudaFree(dO);
    return 0;
}
