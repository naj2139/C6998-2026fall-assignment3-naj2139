#ifndef HPML_HW3_MATMULT_KERNEL_H
#define HPML_HW3_MATMULT_KERNEL_H

// A square-ish matrix, row-major, living in device (or host) memory.
// `stride` is the row pitch of the ALLOCATION this Matrix is a view into;
// for the top-level A, B, C matrices stride == width. GetSubMatrix() uses
// stride to carve out FOOTPRINT_SIZE x FOOTPRINT_SIZE tiles of a larger
// matrix without copying.
typedef struct {
    int width;
    int height;
    int stride;
    float *elements;
} Matrix;

// BLOCK_SIZE is the thread-block shape (BLOCK_SIZE x BLOCK_SIZE threads)
// and is the same, 16, for both kernel variants.
//
// FOOTPRINT_SIZE is the width/height, in elements, of the C tile that ONE
// thread block computes; it must be supplied by the Makefile via
// -DFOOTPRINT_SIZE=... :
//   matmultKernel00.cu -> FOOTPRINT_SIZE=16 (== BLOCK_SIZE): 1 output/thread
//   matmultKernel01.cu -> FOOTPRINT_SIZE=32 (== 2*BLOCK_SIZE): 4 outputs/thread
#define BLOCK_SIZE 16
#ifndef FOOTPRINT_SIZE
#error "FOOTPRINT_SIZE must be defined by the Makefile (16 for kernel00, 32 for kernel01)"
#endif

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C);

#endif  // HPML_HW3_MATMULT_KERNEL_H
