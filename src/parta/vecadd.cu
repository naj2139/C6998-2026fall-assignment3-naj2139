#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "vecaddKernel.h"
#include "timer.h"

// Part-A Q1/Q2 driver (given -- do not modify). Links against EITHER
// vecaddKernel00.cu or vecaddKernel01.cu (the Makefile picks which) to
// build vecadd00 / vecadd01 respectively; this file itself never changes
// between the two variants.
//
//   ./vecadd00 <values_per_thread>
//   ./vecadd01 <values_per_thread>
//
// N = THREADS_PER_BLOCK * NUM_BLOCKS * values_per_thread elements per
// vector. A is filled with 1.0f and B with 2.0f so correctness reduces to
// "every element of C equals 3.0f exactly" (no floating-point tolerance
// needed: 1.0f + 2.0f is exact in IEEE-754 single precision).
#define THREADS_PER_BLOCK 512
#define NUM_BLOCKS 240

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <values_per_thread>\n", argv[0]);
        return 1;
    }
    int valuesPerThread = atoi(argv[1]);
    if (valuesPerThread <= 0) {
        fprintf(stderr, "values_per_thread must be positive\n");
        return 1;
    }

    long totalThreads = (long)THREADS_PER_BLOCK * NUM_BLOCKS;
    long N = totalThreads * valuesPerThread;
    size_t bytes = (size_t)N * sizeof(float);

    float *hA = (float *)malloc(bytes);
    float *hB = (float *)malloc(bytes);
    float *hC = (float *)malloc(bytes);
    for (long i = 0; i < N; ++i) {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemset(dC, 0, bytes);  // defined output if the kernel is unimplemented
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.start();
    AddVectors<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dA, dB, dC, valuesPerThread);
    timer.stop();
    float ms = timer.elapsed_ms();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (long i = 0; i < N; ++i) {
        if (hC[i] != 3.0f) { correct = false; break; }
    }

    const char *prog = argv[0];
    const char *slash = strrchr(prog, '/');
    if (slash) prog = slash + 1;

    printf("N=%ld valuesPerThread=%d\n", N, valuesPerThread);
    printf("Kernel Time: %.3f milliseconds (ms)\n", ms);
    printf(correct ? "Test PASSED\n" : "Test FAILED\n");
    printf("RESULT,program=%s,N=%ld,values_per_thread=%d,time_ms=%.3f,correct=%d\n",
           prog, N, valuesPerThread, ms, correct ? 1 : 0);

    free(hA); free(hB); free(hC);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return correct ? 0 : 1;
}
