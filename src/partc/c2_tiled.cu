#include "common.h"

// TODO (Part C, C2 -- 25 pts): tiled, FlashAttention-style attention.
//
// The naive C1 kernel writes the whole N x N score matrix to global memory
// and reads it back. This kernel must NOT materialize it. Compute the same
// output as C1 by processing the keys and values in tiles and carrying the
// softmax statistics across tiles, so the full row of scores is never stored
// at once. You decide what to stage in shared memory. Derive the update that
// lets you fold each new tile of scores into a running result without having
// first seen the whole row; the lecture's FlashAttention material is the
// reference for the idea, but the recurrence is yours to work out.
//
// Choose the block-to-row mapping and the tile width yourself, and place a
// barrier wherever one thread's shared-memory writes must be visible to the
// others before they read. The output must match C1 element by element.
//
// You own the whole implementation: the block-to-row mapping, the tile
// width, the shared-memory layout, and the launch configuration in main().
// The stub is empty and the driver zeros the output, so the program compiles
// and runs but will not match the reference until you implement the kernel
// and its launch.
__global__ void attn_tiled(int N, int d, double scale,
                            const double *Q, const double *K, const double *V, double *O)
{
    // TODO (C2): implement tiled attention for the query row(s) this block
    // owns, carrying the running softmax max, denominator, and output
    // accumulator across key/value tiles. Do not materialize the N x N
    // scores. Place __syncthreads() where shared-memory writes must be
    // visible before other threads read them.
    (void)N; (void)d; (void)scale; (void)Q; (void)K; (void)V; (void)O;
}

int main(int argc, char **argv)
{
    AttnDims a = parseDims(argc, argv);
    double scale = 1.0 / sqrt((double)a.d);

    double *hQ = makeQ(a), *hK = makeK(a), *hV = makeV(a);
    size_t qkvBytes = (size_t)a.N * a.d * sizeof(double);

    double *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dK, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dV, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dO, qkvBytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dO, 0, qkvBytes));  // defined output until you launch your kernel

    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);
    cudaEventRecord(startE);
    // TODO (C2): choose the grid, block, and dynamic shared-memory size for
    // your design and launch attn_tiled here. The timing scaffold around
    // this launch is provided; do not change it.
    //   attn_tiled<<<grid, block, shmemBytes>>>(a.N, a.d, scale, dQ, dK, dV, dO);
    cudaEventRecord(stopE);
    cudaEventSynchronize(stopE);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startE, stopE);

    double *hO = (double *)malloc(qkvBytes);
    CUDA_CHECK(cudaMemcpy(hO, dO, qkvBytes, cudaMemcpyDeviceToHost));
    double cs = checksum(hO, (size_t)a.N * a.d);

    printf("%.3f,%.3f\n", cs, (double)ms);
    if ((size_t)a.N * a.d <= 4096) dumpOutput(hO, (size_t)a.N * a.d);
    printf("RESULT,program=c2_tiled,N=%d,d=%d,time_ms=%.3f,checksum=%.6f\n",
           a.N, a.d, (double)ms, cs);

    free(hQ); free(hK); free(hV); free(hO);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
