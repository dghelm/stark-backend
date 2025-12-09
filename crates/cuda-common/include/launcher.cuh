#pragma once

#include <algorithm>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#if defined(CUDA_DEBUG) || defined(HIP_DEBUG)
#include <cstdio>
#endif

static const size_t MAX_THREADS = 1024;
static const size_t WARP_SIZE = 32;

// HIP/CUDA device function compatibility
#if defined(__HIPCC__)
// HIP uses same intrinsics as CUDA for __brev, __clz, etc.
// __forceinline__ is not available in HIP - use attribute
#define DEVICE_INLINE __device__ __attribute__((always_inline)) inline
#else
#define DEVICE_INLINE __device__ __forceinline__
#endif

inline size_t div_ceil(size_t a, size_t b) { return (a + b - 1) / b; }

inline std::pair<dim3, dim3> kernel_launch_params(
    size_t count,
    size_t threads_per_block = MAX_THREADS
) {
    size_t block = std::min(count, threads_per_block);
    size_t grid = div_ceil(count, block);
    return std::make_pair(dim3(grid, 1, 1), dim3(block, 1, 1));
}

inline std::pair<dim3, dim3> kernel_launch_2d_params(size_t x, size_t y) {
    dim3 block = dim3(std::min(x, WARP_SIZE), std::min(y, WARP_SIZE));
    dim3 grid = dim3(div_ceil(x, block.x), div_ceil(y, block.y));
    return std::make_pair(grid, block);
}

// HIP/CUDA error checking compatibility
#if defined(__HIPCC__)
#define GPU_OK(expr) do {                                   \
    hipError_t err = expr;                                  \
    if (err != hipSuccess) {                                \
        fprintf(stderr, "HIP kernel error at %s:%d: %s\n",  \
            __FILE__, __LINE__, hipGetErrorString(err));    \
    }                                                       \
} while(0)
#define CUDA_OK(expr) GPU_OK(expr)
#else
#define GPU_OK(expr) do {                                   \
    cudaError_t err = expr;                                 \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err));   \
    }                                                       \
} while(0)
#define CUDA_OK(expr) GPU_OK(expr)
#endif

// Debug kernel checking - HIP/CUDA compatible
#if defined(__HIPCC__)
#if defined(HIP_DEBUG) || defined(CUDA_DEBUG)
    inline int gpu_check_kernel(const char* kernel_name) {
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR] Kernel '%s' failed: %s\n",
                    kernel_name, hipGetErrorString(err));
        }
        return err;
    }
#   define CHECK_KERNEL() gpu_check_kernel(__func__)
#else
#   define CHECK_KERNEL() hipGetLastError()
#endif
#else // CUDA path
#ifdef CUDA_DEBUG
    inline int gpu_check_kernel(const char* kernel_name) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "[ERROR] Kernel '%s' failed: %s\n",
                    kernel_name, cudaGetErrorString(err));
        }
        return err;
    }
#   define CHECK_KERNEL() gpu_check_kernel(__func__)
#else
#   define CHECK_KERNEL() cudaGetLastError()
#endif
#endif
