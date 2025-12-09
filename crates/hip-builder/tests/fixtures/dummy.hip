// Minimal HIP source file for smoke testing the hip-builder crate.
// This file should compile successfully with hipcc.

#include <hip/hip_runtime.h>

// A trivial device function
__device__ int add_one(int x) {
    return x + 1;
}

// A trivial kernel
__global__ void dummy_kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = add_one(data[idx]);
    }
}

// Export a C function so the library has a symbol
extern "C" int dummy_hip_function() {
    return 42;
}
