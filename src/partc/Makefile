# Part C: scaled dot-product attention in CUDA (naive, tiled, cuBLAS).
#
#   c1_naive  - naive attention, materializes the N x N scores in HBM
#   c2_tiled  - FlashAttention-style tiled attention, online softmax
#   c3_cublas - the two GEMMs via cuBLAS, same checksum as c1/c2
#
# Override the target GPU architecture with e.g. `make ARCH=sm_80`
# (T4=sm_75, V100=sm_70, A100=sm_80, H100=sm_90, RTX 30xx=sm_86).

NVCC := nvcc
ARCH ?= sm_75
NVCCFLAGS := -O3 -std=c++14 -arch=$(ARCH)

all: c1_naive c2_tiled c3_cublas

c1_naive: c1_naive.cu common.h
	$(NVCC) $(NVCCFLAGS) c1_naive.cu -o c1_naive

c2_tiled: c2_tiled.cu common.h
	$(NVCC) $(NVCCFLAGS) c2_tiled.cu -o c2_tiled

# c3_cublas links cuBLAS, which ships with the CUDA toolkit (no extra
# install, unlike cuDNN).
c3_cublas: c3_cublas.cu common.h
	$(NVCC) $(NVCCFLAGS) c3_cublas.cu -lcublas -o c3_cublas

.PHONY: all clean
clean:
	rm -f c1_naive c2_tiled c3_cublas *.o
