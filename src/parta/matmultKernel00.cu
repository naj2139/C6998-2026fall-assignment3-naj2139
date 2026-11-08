#include "matmultKernel.h"

// Device helpers: element access and sub-matrix carving on a Matrix view.
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

// matmultKernel00: the baseline tiled matmul (given -- this is Q3),
// FOOTPRINT_SIZE=16.
//
// One thread block computes one FOOTPRINT_SIZE x FOOTPRINT_SIZE tile of C.
// Since FOOTPRINT_SIZE == BLOCK_SIZE here, that is exactly BLOCK_SIZE x
// BLOCK_SIZE = 256 threads, each computing ONE output element. The block
// walks across A's row-panel and B's column-panel one BLOCK_SIZE x
// BLOCK_SIZE tile at a time, staging each tile in shared memory (As, Bs) so
// every element loaded from global memory is reused BLOCK_SIZE times
// instead of being re-fetched from global memory by every thread that
// needs it. This is the classic tiled matmul from the CUDA C Programming
// Guide.
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol, FOOTPRINT_SIZE);

    int row = threadIdx.y;
    int col = threadIdx.x;
    float Cvalue = 0.0f;

    int numTiles = A.width / FOOTPRINT_SIZE;
    for (int m = 0; m < numTiles; ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m, FOOTPRINT_SIZE);
        Matrix Bsub = GetSubMatrix(B, m, blockCol, FOOTPRINT_SIZE);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        __syncthreads();

        #pragma unroll
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        __syncthreads();
    }

    SetElement(Csub, row, col, Cvalue);
}
