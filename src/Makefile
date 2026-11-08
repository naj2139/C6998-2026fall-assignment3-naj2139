# Top-level Makefile: builds Part A, Part B, and Part C from one entry
# point (what grade_local.py and the Gradescope compile-check both call).
#
#   make            build everything (parta + partb + partc)
#   make parta      build only Part A (vecadd00/01, matmult00/01)
#   make partb      build only Part B (q1_cpu, q2_malloc, q3_unified)
#   make partc      build only Part C (c1_naive, c2_tiled, c3_cublas)
#   make clean      clean all three
#
# Override the target GPU architecture with e.g. `make ARCH=sm_80`
# (T4=sm_75, V100=sm_70, A100=sm_80, H100=sm_90, RTX 30xx=sm_86).

ARCH ?= sm_75

.PHONY: all parta partb partc clean

all: parta partb partc

parta:
	$(MAKE) -C parta ARCH=$(ARCH)

partb:
	$(MAKE) -C partb ARCH=$(ARCH)

partc:
	$(MAKE) -C partc ARCH=$(ARCH)

clean:
	$(MAKE) -C parta clean
	$(MAKE) -C partb clean
	$(MAKE) -C partc clean
