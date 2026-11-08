#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "timer.h"

// Part-B Q3 (given kernel; you implement the launch scenarios):
// identical to q2_malloc.cu's three scenarios, except x, y, z live in CUDA
// Unified Memory (cudaMallocManaged), so there is a SINGLE pointer the
// host fills directly and the device reads directly -- no cudaMemcpy calls
// at all. The CUDA driver migrates pages between host and device on demand
// the first time each side touches them.
__global__ void addKernel(long n, int offset, float *z, const float *x, const float *y) {
    long i = (long)offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}

// TODO: implement the same three launch scenarios as q2_malloc.cu's
// runScenario (see that file for the description). The logic is identical
// here -- only the memory (managed vs explicit) differs, and that
// difference lives in main(), not in this function.
static float runScenario(int scenario, long n, float *z, const float *x, const float *y)
{
    GpuTimer timer;
    timer.start();
    (void)scenario; (void)n; (void)z; (void)x; (void)y;
    // TODO: dispatch on `scenario` and issue the launches (see q2_malloc.cu).
    timer.stop();
    return timer.elapsed_ms();
}

int main(int argc, char **argv)
{
    // Optional CLI args override the K (millions of elements) list, see
    // q2_malloc.cu for the rationale.
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

        for (int scenario = 1; scenario <= 3; ++scenario) {
            // Scenario 1 launches one kernel per element; cap it at K <= 10
            // million so it does not take far too long at large K. Scenarios
            // 2 and 3 run every K. (Same cap as q2_malloc.)
            if (scenario == 1 && K > 10) continue;
            float *x, *y, *z;
            cudaMallocManaged(&x, bytes);
            cudaMallocManaged(&y, bytes);
            cudaMallocManaged(&z, bytes);
            for (long i = 0; i < n; ++i) { x[i] = 1.0f; y[i] = 2.0f; }

            float ms = runScenario(scenario, n, z, x, y);
            cudaDeviceSynchronize();  // all launches must finish before the host reads z below

            double checksum = 0.0;
            for (long i = 0; i < n; ++i) checksum += z[i];
            bool correct = (checksum == 3.0 * (double)n);

            printf("K=%d scenario=%d time_ms=%.3f checksum=%.1f %s\n",
                   K, scenario, ms, checksum, correct ? "PASSED" : "FAILED");
            printf("RESULT,program=q3_unified,scenario=%d,K=%d,n=%ld,time_ms=%.3f,checksum=%.1f,correct=%d\n",
                   scenario, K, n, ms, checksum, correct ? 1 : 0);

            cudaFree(x); cudaFree(y); cudaFree(z);
        }
    }
    return 0;
}
