#include "core/cuda_utils.h"
#include "kernels.h"

#include <cstdint>

#include <cuda_runtime.h>

__global__ void sleep_kernel(uint64_t ms) {
  for (int i = 0; i < ms; i++) {
    __nanosleep(1000000);
  }
}

void pplx::sleepOnStream(double seconds, cudaStream_t stream) {
  uint64_t ms = seconds * 1000;
  void *args[] = {&ms};
  dim3 grid(1);
  dim3 block(1);
  CUDACHECK(cudaLaunchKernel((void *)sleep_kernel, grid, block, args, 0, stream));
}
