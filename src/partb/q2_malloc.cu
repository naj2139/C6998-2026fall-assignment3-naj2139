#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "timer.h"

// Part-B Q2 (given kernel; you implement the launch scenarios):
// vector add on the GPU with EXPLICIT host<->device copies (cudaMalloc +
// cudaMemcpy). Host and device memory are NOT shared here -- that's what
// Q3 (cudaMallocManaged) changes.
//
// One kernel definition serves all three launch scenarios. `offset` is the
// base element index for a launch: the global index a thread works on is
// offset + blockIdx.x*blockDim.x + threadIdx.x. You decide the offset each
// launch needs so that, together, the launches of a scenario cover all n
// elements exactly once. The `n` bound guards the final partial block, since
// N is not always a multiple of 256. This kernel is given; do not change it.
__global__ void addKernel(long n, int offset, float *z, const float *x, const float *y) {
    long i = (long)offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}

// TODO: dispatch on `scenario` and launch addKernel to cover all n elements.
// Call it as addKernel<<<grid, block>>>(n, offset, dz, dx, dy); the kernel's
// `n` bound makes a final partial block safe. The three scenarios differ
// only in how the n elements are split across launches:
//   scenario 1: one thread per launch, so it takes n launches to cover n
//               elements.
//   scenario 2: 256 threads per launch, so it takes about n/256 launches.
//   scenario 3: a single launch of enough 256-thread blocks to cover all n.
// Work out the grid/block arguments and the offset for each case yourself.
//
// Keep the single GpuTimer below wrapped around ALL of a scenario's
// launches, so scenario 1's per-launch overhead (the point of comparing it
// against scenario 3) is included, not hidden.
//
// The stub below compiles and runs but launches nothing, so dz keeps its
// uninitialized contents and the checksum check fails until you implement
// this.
static float runScenario(int scenario, long n, float *dz, const float *dx, const float *dy)
{
    GpuTimer timer;
    timer.start();
    (void)scenario; (void)n; (void)dz; (void)dx; (void)dy;
    // TODO: dispatch on `scenario` and issue the launches described above.
    timer.stop();
    return timer.elapsed_ms();
}

int main(int argc, char **argv)
{
    // Optional CLI args override the K (millions of elements) list, e.g.
    // `./q2_malloc 1` runs only K=1 million -- grade_local.py uses this
    // for a fast correctness/build smoke test. Run with no args to
    // reproduce the full sweep for your report.
    std::vector<int> Ks;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) Ks.push_back(atoi(argv[i]));
    } else {
        Ks = {1, 5, 10, 50, 100};
    }
    for (size_t kidx = 0; kidx < Ks.size(); ++kidx) {
        int K = Ks[kidx];
        long n = (long)K * 1000000L;
        size_t bytes = (size_t)n * sizeof(float);

        float *hx = (float *)malloc(bytes);
        float *hy = (float *)malloc(bytes);
        float *hz = (float *)malloc(bytes);
        for (long i = 0; i < n; ++i) { hx[i] = 1.0f; hy[i] = 2.0f; }

        float *dx, *dy, *dz;
        cudaMalloc(&dx, bytes);
        cudaMalloc(&dy, bytes);
        cudaMalloc(&dz, bytes);

        for (int scenario = 1; scenario <= 3; ++scenario) {
            // Scenario 1 issues one kernel launch per element, so at large K
            // it would launch tens of millions of kernels and take far too
            // long. Cap it at K <= 10 million; that range already shows the
            // per-launch-overhead blowup. Scenarios 2 and 3 run every K.
            if (scenario == 1 && K > 10) continue;

            cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice);

            float ms = runScenario(scenario, n, dz, dx, dy);

            cudaMemcpy(hz, dz, bytes, cudaMemcpyDeviceToHost);
            double checksum = 0.0;
            for (long i = 0; i < n; ++i) checksum += hz[i];
            bool correct = (checksum == 3.0 * (double)n);

            printf("K=%d scenario=%d time_ms=%.3f checksum=%.1f %s\n",
                   K, scenario, ms, checksum, correct ? "PASSED" : "FAILED");
            printf("RESULT,program=q2_malloc,scenario=%d,K=%d,n=%ld,time_ms=%.3f,checksum=%.1f,correct=%d\n",
                   scenario, K, n, ms, checksum, correct ? 1 : 0);
        }

        free(hx); free(hy); free(hz);
        cudaFree(dx); cudaFree(dy); cudaFree(dz);
    }
    return 0;
}
