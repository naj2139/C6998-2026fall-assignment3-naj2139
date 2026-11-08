#include "vecaddKernel.h"

// TODO (Part A, Q2 -- 8 pts): write the COALESCED version of AddVectors.
//
// vecaddKernel00.cu (given) gives thread `tid` the contiguous block of
// `valuesPerThread` elements starting at tid*valuesPerThread. At a fixed
// loop iteration, consecutive threads in a warp then access addresses that
// are valuesPerThread elements apart, so the memory controller cannot
// coalesce the warp's accesses into a few wide transactions.
//
// Rewrite the index mapping so that at a fixed loop iteration, consecutive
// threads in a warp access consecutive addresses. Each thread must still
// process exactly valuesPerThread elements, so the only thing that changes
// between vecadd00 and vecadd01 is the access pattern. Work out the mapping
// from thread index, loop iteration, and the total number of threads in the
// grid; the Part A questions in the handout describe what to measure and
// report.
//
// The stub below compiles and runs but writes 0 everywhere, so the grader
// reports it as incorrect until you implement the kernel.
__global__ void AddVectors(const float *A, const float *B, float *C, int valuesPerThread)
{
    // TODO (A2): compute C = A + B. Each thread must handle exactly
    // valuesPerThread elements, with an index mapping that makes consecutive
    // threads in a warp access consecutive addresses at each step (coalesced).
    // The stub leaves C unwritten, so it fails the check until you implement it.
    (void)A; (void)B; (void)C; (void)valuesPerThread;
}
