# HW3: CUDA Kernels

**High Performance Machine Learning (COMS 6998), Columbia University, Fall 2026**
**Due: see Gradescope and the handout for the deadline**
**Max points: 100 (+10 EC)**

## Learning objectives

This assignment covers the CUDA programming model, memory coalescing,
shared-memory tiling, unified memory, and (as a bonus) writing the same
kernel in Triton. After completing it you should be able to:

- Explain why consecutive threads touching consecutive global-memory
  addresses lets the memory controller coalesce a warp's accesses into far
  fewer, wider transactions, and measure the resulting speedup directly.
- Write a shared-memory tiled kernel where each thread produces more than
  one output element, and explain why that increases arithmetic intensity
  per byte moved from global memory.
- Compare `cudaMalloc`/`cudaMemcpy` against `cudaMallocManaged` (CUDA
  Unified Memory) across launch patterns that stress kernel-launch overhead
  differently, and explain what you observe.
- Implement one head of scaled dot-product attention three ways (naive,
  shared-memory tiled in the FlashAttention style, and via cuBLAS) and show
  that all three produce the same checksum.
- (Bonus) Write the same attention as a Triton kernel and compare it against
  PyTorch's fused attention.

## Environment and setup

You need an NVIDIA GPU with the CUDA 12.x toolkit (`nvcc`) installed: your
own machine, a Columbia or GCP instance, or Colab with a GPU runtime. This
assignment cannot be developed or graded without a GPU. There is no CPU
fallback for the CUDA or cuBLAS kernels (the section "How to run the public
tests" below lists what runs without one).

```bash
# Download HPML_HW3_starter.zip from Courseworks, then:
unzip HPML_HW3_starter.zip
cd HPML_HW3_starter
pip install -r requirements.txt
```

Target architecture defaults to `sm_75` (T4). Override it with `make
ARCH=sm_80` (A100), `ARCH=sm_70` (V100), `ARCH=sm_90` (H100), or
`ARCH=sm_86` (RTX 30xx). Set the value that matches the GPU you run on, or
the kernels will fail to launch.

Part C3 links cuBLAS, which ships with the CUDA toolkit, so no extra install
is needed (unlike cuDNN). The Part C4 Triton bonus needs `torch` and
`triton` in your notebook runtime (Colab/Insomnia or your own GPU box).

## Tasks

All source is under `src/`. Each part has its own `Makefile`, and one
top-level `src/Makefile` builds everything. Every graded program prints a
machine-parseable `RESULT,key=val,...` line in addition to its
human-readable output. Do not remove those prints.

### File map

The table lists every source file, what it does, and what you edit. "Given"
files are complete and must not be changed; the grader rebuilds from the
originals. A driver links against whichever kernel file the Makefile selects,
so the same driver builds both the `00` baseline and the `01` version you
write.

| File | What it is | What you do |
|---|---|---|
| `parta/vecadd.cu` | Vector-add driver: allocates arrays, launches the kernel, times it, checks the result | Given, do not edit |
| `parta/vecaddKernel.h` | Shared kernel signature both vecadd kernels implement | Given, do not edit |
| `parta/vecaddKernel00.cu` | Non-coalesced vector-add kernel (baseline) | Given, do not edit |
| `parta/vecaddKernel01.cu` | Coalesced vector-add kernel | **Edit (8 pts):** change the index mapping so consecutive threads touch consecutive addresses |
| `parta/matmult.cu` | Tiled-matmul driver | Given, do not edit |
| `parta/matmultKernel.h` | `Matrix` type and shared matmul kernel signature | Given, do not edit |
| `parta/matmultKernel00.cu` | Tiled matmul, `FOOTPRINT_SIZE=16`, one output per thread (baseline) | Given, do not edit |
| `parta/matmultKernel01.cu` | Register-blocked matmul | **Edit (10 pts):** set `FOOTPRINT_SIZE=32` so each thread computes a 2x2 subtile |
| `partb/q1_cpu.cpp` | CPU vector-add baseline | **Edit (3 pts):** implement `addArrays` on the CPU |
| `partb/q2_malloc.cu` | GPU vector add with explicit `cudaMalloc`/`cudaMemcpy` | **Edit (5 pts):** implement `runScenario` for the three launch patterns |
| `partb/q3_unified.cu` | Same vector add with `cudaMallocManaged` unified memory | **Edit (4 pts):** implement the three scenarios with no explicit copies |
| `partb/q4_plot.py` | Runs the Part B sweep and generates the plots for your report | Given, run it, do not edit |
| `partc/common.h` | Attention problem setup: input generators, checksum, output dump | Given, do not edit |
| `partc/c1_naive.cu` | Naive attention: materialize the `N x N` scores | **Edit (15 pts):** scores kernel, safe row softmax, output kernel |
| `partc/c2_tiled.cu` | FlashAttention-style tiled attention | **Edit (25 pts):** carry softmax state across key/value tiles, never store the full scores |
| `partc/c3_cublas.cu` | Attention built from cuBLAS GEMMs plus your softmax | **Edit (15 pts):** two `cublasDgemm` calls plus the softmax kernel |
| `partc/c4_triton.ipynb` | The same attention as a Triton kernel | **Edit (bonus +10):** fill in `attn_kernel` and `my_attention` |
| `common/timer.{cu,h}` | Timing helper used by the drivers | Given, do not edit |
| `*/Makefile` | Per-part and top-level build rules | Given, do not edit |

### Part A: coalescing and tiling (30 points)

`src/parta/` contains vector add and tiled matrix multiply.

- **`vecaddKernel00.cu` (given):** the non-coalesced vector add. Thread
  `tid` owns a contiguous run of `valuesPerThread` elements starting at
  `tid*valuesPerThread`, so consecutive threads access far-apart addresses.
- **`vecaddKernel01.cu` (TODO, 8 pts):** the coalesced version. Change the
  index mapping so that at a fixed loop step consecutive threads access
  consecutive addresses, with the same amount of work per thread. Derive the
  mapping yourself; the kernel's comment states the goal. Run both at several
  sizes and measure the speedup.
- **`matmultKernel00.cu` (given):** the shared-memory tiled matmul,
  `FOOTPRINT_SIZE = BLOCK_SIZE = 16`, one output element per thread.
- **`matmultKernel01.cu` (TODO, 10 pts):** set `FOOTPRINT_SIZE = 32` with the
  16x16 thread block unchanged, so each thread computes a 2x2 subtile of four
  output elements. Run both at matrix sizes 256, 512, 1024, measure the
  speedup, and explain in your report why the change affects performance.

Q1 (run `vecadd00`, 3 pts), Q3 (time `matmult00`, 6 pts), and Q5 (rules of
thumb, 3 pts) are report questions; see the handout.

### Part B: CUDA Unified Memory (15 points)

`src/partb/` contains the same vector add, three ways.

- **`q1_cpu.cpp` (TODO, 3 pts):** the plain CPU `addArrays`, no CUDA.
- **`q2_malloc.cu` (TODO, 5 pts):** GPU vector add via explicit
  `cudaMalloc`/`cudaMemcpy`, run under three launch patterns: (1) N
  sequential single-thread launches, (2) N/256 sequential 256-thread
  launches, (3) one launch covering all elements. Implement `runScenario`;
  the kernel is given.
- **`q3_unified.cu` (TODO, 4 pts):** the same three scenarios, but `x`, `y`,
  `z` are allocated with `cudaMallocManaged` unified memory and there is no
  explicit `cudaMemcpy`.

The plots and analysis (Q4) are worth 3 report points. After building Part
B, run the provided plotting script from the repository root, which does the
whole workflow for you and writes into a top-level `plots/` directory:

```bash
python src/partb/q4_plot.py --bindir src/partb --out-dir plots
```

It runs `q1_cpu`, `q2_malloc`, and `q3_unified` at the full sweep (K = 1, 5,
10, 50, 100 million), writes the measurements to `plots/partb_measurements.csv`,
and generates the two charts the handout asks for: one for explicit copies
and one for unified memory, each showing all three launch scenarios plus the
CPU baseline on a log-log scale. Scenario 1 (one launch per element) is
capped at K = 10 million automatically, because launching the kernel tens of
millions of times is too slow at larger K; its curve is shorter than the
others, which is expected and handled by the plot. The script stops with an
error if any program fails to run or reports an incorrect result, so a chart
is only ever built from complete, correct measurements. You do not need to
edit the script or combine results by hand. For a quick manual run, each
program also takes K values on the command line, for example `./q2_malloc 1`.

### Part C: attention in CUDA (55 points)

`src/partc/` contains one head of scaled dot-product attention, three ways,
all of which must produce the same checksum. For `Q`, `K`, `V` of shape
`(N, d)`: `S = (1/sqrt(d)) Q K^T`, `P = softmax(S)` row-wise (safe),
`O = P V`.

- **`c1_naive.cu` (TODO, 15 pts):** materialize the full `N x N` scores in
  global memory, apply a safe row softmax, then compute `O = P V`. A clean
  split is three kernels (scores, row softmax, output).
- **`c2_tiled.cu` (TODO, 25 pts):** FlashAttention-style tiling. Process the
  keys and values in tiles and carry the softmax state across tiles (running
  maximum, running denominator, rescaled accumulator), so the `N x N` scores
  never reach global memory. You decide the block-to-row mapping and what to
  stage in shared memory. This is the heaviest question in the assignment.
- **`c3_cublas.cu` (TODO, 15 pts):** the same attention with `cublasDgemm`
  for both matmuls (`Q K^T` and `P V`) and your own softmax kernel between
  them. cuBLAS is column-major and the tensors are row-major, so choose the
  transpose flags and leading dimensions carefully (see the comment in the
  file).

Run each of the three with no arguments for the full report size
(`N=2048`, `d=64`, the compiled-in defaults) to get your timing numbers. The
grader checks correctness by comparing each implementation's output tensor
element by element against an independent reference at several small sizes
(including sizes that cross tile boundaries), so all three must produce the
correct per-element output, not merely a matching aggregate checksum. The
printed checksum is for your report.

### Part C4: attention in Triton (bonus, +10 EC)

`src/partc/c4_triton.ipynb`: the same attention as a Triton kernel
(`@triton.jit`, `tl.load`/`tl.store` with masking, `tl.program_id`, `tl.dot`).
Fill in `attn_kernel` and `my_attention`; the notebook checks your result
against a PyTorch reference and benchmarks it against
`torch.nn.functional.scaled_dot_product_attention` with `do_bench`.

### Written report

The report questions (Part A Q1/Q3/Q5, Part B Q4) are graded from your PDF
and are already counted in the part totals above. For each part, report your
measured numbers (times, speedups, checksums), include the Part B plots, and
explain the results: the cause of the coalescing and tiling speedups, the
effect of launch overhead across the three Part B scenarios, and why the
tiled attention avoids the `N x N` HBM traffic that the naive version
incurs. These report items are graded against the criteria in the handout.

## How to run the public tests

There is no meaningful CPU-only correctness check for CUDA/cuBLAS kernels,
so "public tests" here means: build everything, and run `grade_local.py`
on your own GPU.

```bash
cd src && make ARCH=sm_75 && cd ..     # build everything; use your GPU's arch
python grade_local.py                  # runs on your GPU, writes results.json
pytest tests/ -v                       # packaging check; no GPU needed
```

`grade_local.py` builds Part A/B/C, runs the binaries (a fast smoke-test
K/size plus a full-size timing pass), checks each Part C implementation
element-wise against an independent Python reference at several sizes,
records host info and a timestamp, and writes
`results.json`. **Include this file in your submission zip.** It is the
evidence your written report and the grading both rely on. Re-run it whenever
you change code; the timestamp in `results.json` should be recent when you
submit.

### What `tests/` is for

The `tests/` folder holds `test_public.py`, a packaging check you run with
`pytest tests/ -v`. It needs no GPU and no `nvcc`, so you can run it on any
machine to confirm your submission is well formed before you upload it. It
checks that every required source file is present and non-empty, that each
Makefile lists the binaries the assignment expects, that `results.schema.json`
is a valid schema, and that `grade_local.py --dry-run` produces a schema-valid
`results.json`. It does **not** check that your kernels are correct or fast:
only a real GPU run of `grade_local.py` does that. Think of it as a check that
nothing is missing or misnamed, not a check that your answers are right.

`python grade_local.py --dry-run` (or running on a machine with no GPU or
`nvcc`) still produces a schema-valid `results.json`, but every GPU-only
field is marked `"skipped: no GPU"`. **This does not satisfy the
assignment.** The `mode` field in a submitted `results.json` must be
`"gpu"`, not `"dry-run"`. Part B's `q1_cpu` is the only program that runs in
dry-run mode; it exists so the harness can be smoke-tested without a GPU,
not so you can skip having one.

## How to submit

Submit both of these to Gradescope before the deadline:

1. **Your code**, as a zip that keeps the given folder layout and file names
   (`src/parta`, `src/partb`, `src/partc`, plus your GPU-generated
   `results.json`). Do not rename or move files; the grader looks for them at
   their given paths.
2. **Your report**, as a PDF with your measurements, plots, and the written
   answers the handout asks for.

On a GPU-enabled Gradescope setup the autograder rebuilds and re-runs your
kernels and scores build + correctness directly; otherwise the TA scores
them from your committed `results.json` and a read of your code.

## Rubric summary

| Component | Points | Graded by |
|---|---|---|
| Part A: run `vecadd00` (Q1, report) | 3 | TA report rubric |
| Part A: coalesced vecadd (`vecaddKernel01.cu`) | 8 | `results.json` + code |
| Part A: time `matmult00` (Q3, report) | 6 | TA report rubric |
| Part A: tiled matmul, FOOTPRINT=32 (`matmultKernel01.cu`) | 10 | `results.json` + code |
| Part A: rules of thumb (Q5, report) | 3 | TA report rubric |
| Part B: CPU vector add (`q1_cpu.cpp`) | 3 | `results.json` + code |
| Part B: explicit-copy launch scenarios (`q2_malloc.cu`) | 5 | `results.json` + code |
| Part B: unified-memory launch scenarios (`q3_unified.cu`) | 4 | `results.json` + code |
| Part B: plots and analysis (Q4, report) | 3 | TA report rubric |
| Part C: naive attention (`c1_naive.cu`) | 15 | `results.json` + code |
| Part C: tiled attention (`c2_tiled.cu`) | 25 | `results.json` + code |
| Part C: cuBLAS attention (`c3_cublas.cu`) | 15 | `results.json` + code |
| **Total** | **100** | |
| Bonus: Triton attention (`c4_triton.ipynb`) | +10 EC | TA rubric |

On a CPU Gradescope container the autograder does not assign points. It
compile-checks your code with `nvcc` and validates your submitted
`results.json` against `results.schema.json` plus sanity bounds (for
example, `mode` must be `"gpu"` and no part's `status` may be `"failed"`).
The TA then assigns the points above by reading that validated
`results.json` and your code and report. On a GPU-enabled Gradescope image
the autograder re-runs the kernels and scores the build and correctness
points directly. The rubric summary above lists how the points break down.

## Academic integrity

This is individual work. You may discuss concepts with classmates, but the
code, measurements, and written analysis you submit must be your own. Cite
any external resources you used. `results.json` records your host,
timestamp, and your own correctness checks; make sure it reflects a run of
your own code on your own GPU.
