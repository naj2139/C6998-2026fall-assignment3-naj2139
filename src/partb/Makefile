# Part B: CUDA Unified Memory.
#
#   q1_cpu     - CPU-only vector add, K = 1,5,10,50,100 million elements (TODO)
#   q2_malloc  - GPU vector add via cudaMalloc/cudaMemcpy, 3 launch scenarios (TODO)
#   q3_unified - GPU vector add via cudaMallocManaged, same 3 scenarios (TODO)
#
# q4_plot.py is a plain Python script (no compilation, given); run it after
# grade_local.py has produced results.json, or after redirecting q1_cpu's
# and friends' stdout to log files (see its --help).
#
# Override the target GPU architecture with e.g. `make ARCH=sm_80`.

CXX := g++
CXXFLAGS := -O2 -std=c++14

NVCC := nvcc
ARCH ?= sm_75
NVCCFLAGS := -O3 -std=c++14 -arch=$(ARCH) -I../common

BINARIES := q1_cpu q2_malloc q3_unified

all: $(BINARIES)

q1_cpu: q1_cpu.cpp
	$(CXX) $(CXXFLAGS) q1_cpu.cpp -o q1_cpu

timer.o: ../common/timer.cu ../common/timer.h
	$(NVCC) $(NVCCFLAGS) -c ../common/timer.cu -o timer.o

q2_malloc: q2_malloc.cu timer.o
	$(NVCC) $(NVCCFLAGS) q2_malloc.cu timer.o -o q2_malloc

q3_unified: q3_unified.cu timer.o
	$(NVCC) $(NVCCFLAGS) q3_unified.cu timer.o -o q3_unified

.PHONY: all clean
clean:
	rm -f $(BINARIES) *.o
	rm -rf plots
