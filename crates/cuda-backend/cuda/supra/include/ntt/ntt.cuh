/*
 * Source: https://github.com/supranational/sppark (tag=v0.1.12)
 * Status: MODIFIED from sppark/ntt/kernels.cu
 * Imported: 2025-08-13 by @gaxiom
 * 
 * LOCAL CHANGES (high level):
 * - 2025-08-13: support multiple rows in bit_rev_permutation & bit_rev_permutation_z
 * - 2025-09-10: only __device__ functions left
 */

// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __NTT_CUH__
#define __NTT_CUH__

#if defined(__HIPCC__)
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif
#include "parameters.cuh"

// HIP/CUDA compatibility for inline qualifier
#if defined(__HIPCC__)
# define NTT_DEVICE_INLINE __device__ __attribute__((always_inline)) inline
// HIP: __syncwarp is not available, use __syncthreads or cooperative_groups
// For warp-level sync in HIP, we use empty function on RDNA (warp=32) since
// instructions within a wavefront execute in lockstep
# define NTT_SYNCWARP() do { } while(0)
// Conditional sync: use block sync when Z_COUNT > warp size, otherwise warp sync
# define NTT_SYNC_CONDITIONAL(z_count) ((z_count) > WARP_SIZE) ? __syncthreads() : (void)0
#else
# define NTT_DEVICE_INLINE __device__ __forceinline__
# define NTT_SYNCWARP() __syncwarp()
# define NTT_SYNC_CONDITIONAL(z_count) ((z_count) > WARP_SIZE) ? __syncthreads() : __syncwarp()
#endif

template<typename T>
NTT_DEVICE_INLINE
T bit_rev(T i, unsigned int nbits)
{
    // HIP and CUDA both provide __brev / __brevll for bit reversal
    if (sizeof(i) == 4 || nbits <= 32)
        return __brev(static_cast<unsigned int>(i)) >> (8*sizeof(unsigned int) - nbits);
    else
        return __brevll(static_cast<unsigned long long>(i)) >> (8*sizeof(unsigned long long) - nbits);
}

NTT_DEVICE_INLINE
fr_t get_intermediate_root(index_t pow, const fr_t (*roots)[WINDOW_SIZE])
{
    unsigned int off = 0;

    fr_t t, root;

    if (sizeof(fr_t) <= 8) {
        root = fr_t::one();
        bool root_set = false;

        #pragma unroll
        for (unsigned int pow_win, i = 0; i < WINDOW_NUM; i++) {
            if (!root_set && (pow_win = pow % WINDOW_SIZE)) {
                root = roots[i][pow_win];
                root_set = true;
            }
            if (!root_set) {
                pow >>= LG_WINDOW_SIZE;
                off++;
            }
        }
    } else {
        if ((pow % WINDOW_SIZE) == 0) {
            pow >>= LG_WINDOW_SIZE;
            off++;
        }
        root = roots[off][pow % WINDOW_SIZE];
    }

    #pragma unroll 1
    while (pow >>= LG_WINDOW_SIZE)
        root *= (t = roots[++off][pow % WINDOW_SIZE]);

    return root;
}

NTT_DEVICE_INLINE
void get_intermediate_roots(fr_t& root0, fr_t& root1,
                            index_t idx0, index_t idx1,
                            const fr_t (*roots)[WINDOW_SIZE])
{
    int win = (WINDOW_NUM - 1) * LG_WINDOW_SIZE;
    int off = (WINDOW_NUM - 1);
    index_t idxo = idx0 | idx1;
    index_t mask = ((index_t)1 << win) - 1;

    root0 = roots[off][idx0 >> win];
    root1 = roots[off][idx1 >> win];
    #pragma unroll 1
    while (off-- && (idxo & mask)) {
        fr_t t;
        win -= LG_WINDOW_SIZE;
        mask >>= LG_WINDOW_SIZE;
        root0 *= (t = roots[off][(idx0 >> win) % WINDOW_SIZE]);
        root1 *= (t = roots[off][(idx1 >> win) % WINDOW_SIZE]);
    }
}

template<int z_count>
NTT_DEVICE_INLINE
void coalesced_load(fr_t r[z_count], const fr_t* inout, index_t idx,
                    const unsigned int stage)
{
    const unsigned int x = threadIdx.x & (z_count - 1);
    idx &= ~((index_t)(z_count - 1) << stage);
    idx += x;

    #pragma unroll
    for (int z = 0; z < z_count; z++, idx += (index_t)1 << stage)
        r[z] = inout[idx];
}

template<int z_count>
NTT_DEVICE_INLINE
void transpose(fr_t r[z_count])
{
    extern __shared__ fr_t shared_exchange[];
    fr_t (*xchg)[z_count] = reinterpret_cast<decltype(xchg)>(shared_exchange);

    const unsigned int x = threadIdx.x & (z_count - 1);
    const unsigned int y = threadIdx.x & ~(z_count - 1);

    #pragma unroll
    for (int z = 0; z < z_count; z++)
        xchg[y + z][x] = r[z];

    NTT_SYNCWARP();

    #pragma unroll
    for (int z = 0; z < z_count; z++)
        r[z] = xchg[y + x][z];
}

template<int z_count>
NTT_DEVICE_INLINE
void coalesced_store(fr_t* inout, index_t idx, const fr_t r[z_count],
                     const unsigned int stage)
{
    const unsigned int x = threadIdx.x & (z_count - 1);
    idx &= ~((index_t)(z_count - 1) << stage);
    idx += x;

    #pragma unroll
    for (int z = 0; z < z_count; z++, idx += (index_t)1 << stage)
        inout[idx] = r[z];
}

#endif /* __NTT_CUH__ */
