#ifndef HPML_HW3_TIMER_H
#define HPML_HW3_TIMER_H

#include <cuda_runtime.h>

// Minimal CUDA-event stopwatch for timing KERNEL EXECUTION ONLY -- not
// allocation, not host<->device copies, not host-side setup. Call start()
// immediately before a kernel launch (or a short run of launches) and
// stop() immediately after; elapsed_ms() blocks (via cudaEventSynchronize)
// until the device work has actually finished, so the number it returns
// is real elapsed device time, not just launch overhead.
struct GpuTimer {
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    GpuTimer();
    ~GpuTimer();
    void start();
    void stop();
    float elapsed_ms();
};

#endif  // HPML_HW3_TIMER_H
