#ifndef HPML_HW3_VECADD_KERNEL_H
#define HPML_HW3_VECADD_KERNEL_H

// Every vecadd kernel variant implements this exact signature so vecadd.cu
// can be linked against either vecaddKernel00.cu (non-coalesced) or
// vecaddKernel01.cu (coalesced) with no other source change.
//
// The grid has gridDim.x * blockDim.x threads total; each thread is
// responsible for `valuesPerThread` elements of the
// N = gridDim.x * blockDim.x * valuesPerThread element vectors A, B, C.
// The two kernels differ ONLY in which N/valuesPerThread indices each
// thread is assigned -- that assignment is what makes one coalesced and
// the other not.
__global__ void AddVectors(const float *A, const float *B, float *C, int valuesPerThread);

#endif  // HPML_HW3_VECADD_KERNEL_H
