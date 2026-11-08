#include "timer.h"

GpuTimer::GpuTimer() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
}

GpuTimer::~GpuTimer() {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void GpuTimer::start() {
    cudaEventRecord(startEvent, 0);
}

void GpuTimer::stop() {
    cudaEventRecord(stopEvent, 0);
}

float GpuTimer::elapsed_ms() {
    cudaEventSynchronize(stopEvent);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    return ms;
}
