#include "vecaddKernel.h"

// vecaddKernel00: the NON-coalesced baseline (given -- this is Q1).
//
// Thread `tid` owns the CONTIGUOUS block of `valuesPerThread` elements
// starting at tid*valuesPerThread ("blocked" partition of the array across
// threads). At any fixed loop iteration j, thread tid touches index
// tid*valuesPerThread + j, so consecutive threads in the same warp (tid,
// tid+1, tid+2, ...) touch addresses that are valuesPerThread FLOATS apart.
// That is far outside a single 128-byte memory transaction, so the memory
// controller must issue one transaction per thread instead of coalescing
// the whole warp into a handful of transactions.
__global__ void AddVectors(const float *A, const float *B, float *C, int valuesPerThread)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * valuesPerThread;
    for (int j = 0; j < valuesPerThread; ++j) {
        int idx = start + j;
        C[idx] = A[idx] + B[idx];
    }
}
