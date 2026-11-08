#include "matmultKernel.h"

#define SUBTILE (FOOTPRINT_SIZE / BLOCK_SIZE)  // = 2

__device__ float GetElement(const Matrix m, int row, int col) {
    return m.elements[row * m.stride + col];
}
__device__ void SetElement(Matrix m, int row, int col, float value) {
    m.elements[row * m.stride + col] = value;
}
__device__ Matrix GetSubMatrix(Matrix m, int row, int col, int size) {
    Matrix sub;
    sub.width = size;
    sub.height = size;
    sub.stride = m.stride;
    sub.elements = &m.elements[m.stride * size * row + size * col];
    return sub;
}

// TODO (Part A, Q4 -- 10 pts): write the improved tiled matmul kernel.
//
// The thread block stays BLOCK_SIZE x BLOCK_SIZE = 16x16 = 256 threads, but
// FOOTPRINT_SIZE is now 32, so each block produces a 32x32 tile of C and
// each thread is responsible for a 2x2 block of four output elements rather
// than one. Decide how the 256 threads cooperatively load the larger
// FOOTPRINT_SIZE x FOOTPRINT_SIZE shared-memory tiles, how each thread
// accumulates its four outputs across the K dimension, and where each
// result is written back.
//
// Why this helps: a 32x32 tile stages larger As/Bs blocks than the 16x16
// kernel, so each element loaded from global memory into shared memory is
// consumed by more multiply-accumulates before it is evicted, and the whole
// matrix is read from global memory fewer times. Quantify that reuse change
// in your report rather than assuming a particular factor.
//
// The driver zeros C before launch, so the unimplemented stub produces
// defined (wrong) output and fails the correctness check until you write it.
// The GetElement/SetElement/GetSubMatrix helpers above are provided
// infrastructure you may use.

// Thread-to-output mapping (shrunk example: BLOCK_SIZE = 2, FOOTPRINT_SIZE = 4;
// the real kernel uses BLOCK_SIZE = 16, FOOTPRINT_SIZE = 32 with the same shape).

// kernel00: 4 threads, 4 boxes. Each thread fills 1 box.
//
//         col0 col1
// row0  [ 1 ][ 2 ]
// row1  [ 3 ][ 4 ]
//
// kernel01: the same 4 threads, but now 16 boxes. Each thread fills 4 boxes.
// The tile splits into 4 quadrants, each the size of the thread block, and
// every thread takes the same relative spot in each quadrant:

//         col0 col1 | col2 col3
// row0  [ 1 ][ 2 ] | [ 1 ][ 2 ]
// row1  [ 3 ][ 4 ] | [ 3 ][ 4 ]
//       ------------+------------
// row2  [ 1 ][ 2 ] | [ 1 ][ 2 ]
// row3  [ 3 ][ 4 ] | [ 3 ][ 4 ]
//
// Thread 1 (row=0, col=0) owns (0,0), (0,2), (2,0), (2,2): its own position
// shifted by BLOCK_SIZE right, down, and both. Shifting by BLOCK_SIZE covers
// every box exactly once and keeps neighbouring threads on neighbouring
// addresses (coalesced).

// matmultKernel01: register-blocked tiled matmul, FOOTPRINT_SIZE = 32.
//
// The thread block stays 16x16 (BLOCK_SIZE), but each block now computes a
// 32x32 tile of C (FOOTPRINT_SIZE). With 4x the outputs and the same number
// of threads, each thread computes a 2x2 group of outputs: its own position
// (row, col) plus that position shifted by BLOCK_SIZE right, down, and both.
// Each value read from shared memory is reused for two multiply-adds instead
// of one, raising arithmetic intensity per byte moved.
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
    // Which 32x32 tile of C this block owns.
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol, FOOTPRINT_SIZE);

    // This thread's position within the 16x16 thread block.
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Four running sums, one per output this thread owns. Kept in registers;
    // C in global memory is only written once, after all tiles are summed.
    float Cvalue00 = 0.0f;   // (row,              col             )
    float Cvalue01 = 0.0f;   // (row,              col + BLOCK_SIZE)
    float Cvalue10 = 0.0f;   // (row + BLOCK_SIZE, col             )
    float Cvalue11 = 0.0f;   // (row + BLOCK_SIZE, col + BLOCK_SIZE)

    // Walk A's row panel and B's column panel in 32-wide chunks.
    int numTiles = A.width / FOOTPRINT_SIZE;
    for (int m = 0; m < numTiles; ++m) {
        // Views of the current A and B chunks (no data copied yet).
        Matrix Asub = GetSubMatrix(A, blockRow, m, FOOTPRINT_SIZE);
        Matrix Bsub = GetSubMatrix(B, m, blockCol, FOOTPRINT_SIZE);

        // Shared-memory staging buffers for the current 32x32 chunks,
        // visible to all 256 threads in the block.
        __shared__ float As[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
        __shared__ float Bs[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

        // Cooperative load: 1024 elements per tile, 256 threads, so each
        // thread loads 4 elements of A's chunk and 4 of B's chunk. Threads
        // with consecutive col read consecutive addresses (coalesced).
        As[row][col]                           = GetElement(Asub, row, col);
        As[row][col + BLOCK_SIZE]              = GetElement(Asub, row, col + BLOCK_SIZE);
        As[row + BLOCK_SIZE][col]              = GetElement(Asub, row + BLOCK_SIZE, col);
        As[row + BLOCK_SIZE][col + BLOCK_SIZE] = GetElement(Asub, row + BLOCK_SIZE, col + BLOCK_SIZE);

        Bs[row][col]                           = GetElement(Bsub, row, col);
        Bs[row][col + BLOCK_SIZE]              = GetElement(Bsub, row, col + BLOCK_SIZE);
        Bs[row + BLOCK_SIZE][col]              = GetElement(Bsub, row + BLOCK_SIZE, col);
        Bs[row + BLOCK_SIZE][col + BLOCK_SIZE] = GetElement(Bsub, row + BLOCK_SIZE, col + BLOCK_SIZE);

        // Wait until both chunks are fully loaded before anyone reads them.
        __syncthreads();

        // Partial dot products over this chunk. Each As value is reused for
        // two outputs (two columns) and each Bs value for two outputs (two
        // rows): 4 shared-memory reads feed 4 multiply-adds per step.
#pragma unroll
        for (int e = 0; e < FOOTPRINT_SIZE; ++e) {
            Cvalue00 += As[row][e]              * Bs[e][col];
            Cvalue01 += As[row][e]              * Bs[e][col + BLOCK_SIZE];
            Cvalue10 += As[row + BLOCK_SIZE][e] * Bs[e][col];
            Cvalue11 += As[row + BLOCK_SIZE][e] * Bs[e][col + BLOCK_SIZE];
        }

        // Wait until everyone is done reading before the next iteration
        // overwrites As and Bs with the next chunk.
        __syncthreads();
    }

    // All chunks summed: write the 4 finished outputs into C through the
    // Csub view (row-major, using C's full stride).
    SetElement(Csub, row,              col,              Cvalue00);
    SetElement(Csub, row,              col + BLOCK_SIZE, Cvalue01);
    SetElement(Csub, row + BLOCK_SIZE, col,              Cvalue10);
    SetElement(Csub, row + BLOCK_SIZE, col + BLOCK_SIZE, Cvalue11);
}