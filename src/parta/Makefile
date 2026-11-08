# Part A: vector add (coalescing) and tiled matrix multiply.
#
#   vecadd00   - vector add, NON-coalesced global memory access   (vecaddKernel00.cu, given)
#   vecadd01   - vector add, COALESCED global memory access       (vecaddKernel01.cu, TODO)
#   matmult00  - tiled matmul, FOOTPRINT_SIZE=16, 1 output/thread  (matmultKernel00.cu, given)
#   matmult01  - tiled matmul, FOOTPRINT_SIZE=32, 4 outputs/thread (matmultKernel01.cu, TODO)
#
# `make` builds all four. `make clean` removes binaries and object files.
# Override the target GPU architecture with e.g. `make ARCH=sm_80`
# (T4=sm_75, V100=sm_70, A100=sm_80, H100=sm_90, RTX 30xx=sm_86).

NVCC := nvcc
ARCH ?= sm_75
NVCCFLAGS := -O3 -std=c++14 -arch=$(ARCH) -I../common

BINARIES := vecadd00 vecadd01 matmult00 matmult01

all: $(BINARIES)

timer.o: ../common/timer.cu ../common/timer.h
	$(NVCC) $(NVCCFLAGS) -c ../common/timer.cu -o timer.o

vecadd00: vecadd.cu vecaddKernel00.cu vecaddKernel.h timer.o
	$(NVCC) $(NVCCFLAGS) vecadd.cu vecaddKernel00.cu timer.o -o vecadd00

vecadd01: vecadd.cu vecaddKernel01.cu vecaddKernel.h timer.o
	$(NVCC) $(NVCCFLAGS) vecadd.cu vecaddKernel01.cu timer.o -o vecadd01

matmult00: matmult.cu matmultKernel00.cu matmultKernel.h timer.o
	$(NVCC) $(NVCCFLAGS) -DFOOTPRINT_SIZE=16 matmult.cu matmultKernel00.cu timer.o -o matmult00

matmult01: matmult.cu matmultKernel01.cu matmultKernel.h timer.o
	$(NVCC) $(NVCCFLAGS) -DFOOTPRINT_SIZE=32 matmult.cu matmultKernel01.cu timer.o -o matmult01

.PHONY: all clean
clean:
	rm -f $(BINARIES) *.o
