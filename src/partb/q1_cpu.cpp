#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

// Part-B Q1: add two K-million-element float arrays on the CPU,
// for K = 1, 5, 10, 50, 100 by default (looped internally so one run of
// this program produces every data point Q1 asks for). Use malloc/free
// (not new/delete). A is filled with 1.0f and B with 2.0f, so the exact
// expected checksum is 3.0*N -- 1.0f+2.0f is exact in fp32, so this is a
// real equality check, not a tolerance check.
//
// Optional CLI args override the K list, e.g. `./q1_cpu 1` runs only
// K=1 million -- grade_local.py uses this for a fast correctness/build
// smoke test; run with no args to reproduce the full sweep for your report.
//
// TODO: implement addArrays so that z becomes the element-wise sum of x and
// y. The stub below compiles and runs but just writes zeros, so it fails the
// checksum check (which expects every z[i] == 3.0) until you implement it.
static void addArrays(long n, const float *x, const float *y, float *z) {
    // TODO (B1): set z to the element-wise sum of x and y (both length n).
    // The placeholder below writes zeros so the program runs; replace it.
    (void)x; (void)y;
    memset(z, 0, (size_t)n * sizeof(float));
}

int main(int argc, char **argv)
{
    std::vector<int> Ks;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) Ks.push_back(atoi(argv[i]));
    } else {
        Ks = {1, 5, 10, 50, 100};
    }
    for (size_t idx = 0; idx < Ks.size(); ++idx) {
        int K = Ks[idx];
        long n = (long)K * 1000000L;

        float *x = (float *)malloc(n * sizeof(float));
        float *y = (float *)malloc(n * sizeof(float));
        float *z = (float *)malloc(n * sizeof(float));
        for (long i = 0; i < n; ++i) { x[i] = 1.0f; y[i] = 2.0f; }

        auto t0 = std::chrono::high_resolution_clock::now();
        addArrays(n, x, y, z);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double checksum = 0.0;
        for (long i = 0; i < n; ++i) checksum += z[i];
        bool correct = (checksum == 3.0 * (double)n);

        printf("K=%d n=%ld time_ms=%.3f checksum=%.1f %s\n",
               K, n, ms, checksum, correct ? "PASSED" : "FAILED");
        printf("RESULT,program=q1_cpu,K=%d,n=%ld,time_ms=%.3f,checksum=%.1f,correct=%d\n",
               K, n, ms, checksum, correct ? 1 : 0);

        free(x); free(y); free(z);
    }
    return 0;
}
