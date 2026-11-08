#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "matmultKernel.h"
#include "timer.h"

// Part-A Q3/Q4 driver (given -- do not modify). Links against EITHER
// matmultKernel00.cu (FOOTPRINT_SIZE=16, via the Makefile) or
// matmultKernel01.cu (FOOTPRINT_SIZE=32) to build matmult00 / matmult01;
// this file does not change between the two.
//
//   ./matmult00 <size_multiplier>   # square matrix side = size_multiplier * 16
//   ./matmult01 <size_multiplier>   # square matrix side = size_multiplier * 32
//
// Running matmult00/01 with multiplier 16/8, 32/16, 64/32 respectively both
// give the 256, 512, 1024 sizes Q3/Q4 ask for.

// Deterministic fill so the CPU reference below is exact and reproducible
// across runs (no RNG seed to manage). Values are kept small (< 13) so
// fp32 accumulation error over up to 1024 terms stays far under the
// correctness tolerance used below.
static void fillMatrices(int n, float *A, float *B) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = (float)((i + j) % 13);
            B[i * n + j] = (float)((i - j + n) % 7);
        }
    }
}

static void cpuMatMul(int n, const float *A, const float *B, float *C) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size_multiplier>  (matrix side = size_multiplier * FOOTPRINT_SIZE)\n", argv[0]);
        return 1;
    }
    int mult = atoi(argv[1]);
    if (mult <= 0) {
        fprintf(stderr, "size_multiplier must be positive\n");
        return 1;
    }
    int n = mult * FOOTPRINT_SIZE;

    size_t bytes = (size_t)n * n * sizeof(float);
    float *hA = (float *)malloc(bytes);
    float *hB = (float *)malloc(bytes);
    float *hC = (float *)malloc(bytes);
    fillMatrices(n, hA, hB);

    Matrix A, B, C;
    A.width = A.height = A.stride = n;
    B.width = B.height = B.stride = n;
    C.width = C.height = C.stride = n;
    cudaMalloc(&A.elements, bytes);
    cudaMalloc(&B.elements, bytes);
    cudaMalloc(&C.elements, bytes);
    cudaMemset(C.elements, 0, bytes);  // defined output if the kernel is unimplemented
    cudaMemcpy(A.elements, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B.elements, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(n / FOOTPRINT_SIZE, n / FOOTPRINT_SIZE);

    GpuTimer timer;
    timer.start();
    MatMulKernel<<<grid, block>>>(A, B, C);
    timer.stop();
    float ms = timer.elapsed_ms();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hC, C.elements, bytes, cudaMemcpyDeviceToHost);

    // Correctness check against a CPU reference. For n=1024 this O(n^3)
    // CPU loop, not the GPU kernel, is the slow part of running this
    // program -- expected, since it exists only to certify correctness.
    float *ref = (float *)malloc(bytes);
    cpuMatMul(n, hA, hB, ref);
    bool correct = true;
    double maxAbsErr = 0.0;
    for (long i = 0; i < (long)n * n; ++i) {
        double err_i = fabs((double)hC[i] - (double)ref[i]);
        if (err_i > maxAbsErr) maxAbsErr = err_i;
        if (err_i > 1e-2) correct = false;  // generous atol for fp32 accumulation over n terms
    }

    const char *prog = argv[0];
    const char *slash = strrchr(prog, '/');
    if (slash) prog = slash + 1;

    printf("Matrix size: %d x %d (FOOTPRINT_SIZE=%d)\n", n, n, FOOTPRINT_SIZE);
    printf("Kernel Time: %.3f milliseconds (ms)\n", ms);
    printf("Max abs error vs CPU reference: %g\n", maxAbsErr);
    printf(correct ? "Test PASSED\n" : "Test FAILED\n");
    printf("RESULT,program=%s,size=%d,footprint=%d,time_ms=%.3f,correct=%d,max_abs_err=%g\n",
           prog, n, FOOTPRINT_SIZE, ms, correct ? 1 : 0, maxAbsErr);

    free(hA); free(hB); free(hC); free(ref);
    cudaFree(A.elements); cudaFree(B.elements); cudaFree(C.elements);
    return correct ? 0 : 1;
}
