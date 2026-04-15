#pragma once

#include "cuda_compat.cuh"
#include "types.cuh"

namespace cellshard::ptx {

__host__ __device__ __forceinline__ types::u32 add_u32(types::u32 lhs, types::u32 rhs) {
#if CELLSHARD_ENABLE_CUDA && defined(__CUDA_ARCH__) && CELLSHARD_CUDA_MODE_NATIVE_EXTREME
    types::u32 out = 0u;
    asm("add.u32 %0, %1, %2;" : "=r"(out) : "r"(lhs), "r"(rhs));
    return out;
#else
    return (types::u32) (lhs + rhs);
#endif
}

__host__ __device__ __forceinline__ types::u32 sub_u32(types::u32 lhs, types::u32 rhs) {
#if CELLSHARD_ENABLE_CUDA && defined(__CUDA_ARCH__) && CELLSHARD_CUDA_MODE_NATIVE_EXTREME
    types::u32 out = 0u;
    asm("sub.u32 %0, %1, %2;" : "=r"(out) : "r"(lhs), "r"(rhs));
    return out;
#else
    return (types::u32) (lhs - rhs);
#endif
}

__host__ __device__ __forceinline__ types::u32 mul_lo_u32(types::u32 lhs, types::u32 rhs) {
#if CELLSHARD_ENABLE_CUDA && defined(__CUDA_ARCH__) && CELLSHARD_CUDA_MODE_NATIVE_EXTREME
    types::u32 out = 0u;
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(out) : "r"(lhs), "r"(rhs));
    return out;
#else
    return (types::u32) (lhs * rhs);
#endif
}

__host__ __device__ __forceinline__ types::u32 mad_lo_u32(types::u32 a, types::u32 b, types::u32 c) {
#if CELLSHARD_ENABLE_CUDA && defined(__CUDA_ARCH__) && CELLSHARD_CUDA_MODE_NATIVE_EXTREME
    types::u32 out = 0u;
    asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
#else
    return (types::u32) (a * b + c);
#endif
}

__host__ __device__ __forceinline__ types::u32 shr_u32(types::u32 value, unsigned int shift) {
#if CELLSHARD_ENABLE_CUDA && defined(__CUDA_ARCH__) && CELLSHARD_CUDA_MODE_NATIVE_EXTREME
    types::u32 out = 0u;
    asm("shr.u32 %0, %1, %2;" : "=r"(out) : "r"(value), "r"(shift));
    return out;
#else
    return (types::u32) (value >> shift);
#endif
}

__device__ __forceinline__ types::u32 global_tid_1d() {
    return mad_lo_u32((types::u32) blockIdx.x, (types::u32) blockDim.x, (types::u32) threadIdx.x);
}

__device__ __forceinline__ types::u32 global_stride_1d() {
    return mul_lo_u32((types::u32) gridDim.x, (types::u32) blockDim.x);
}

__device__ __forceinline__ types::u32 segment_tid_2d() {
    return mad_lo_u32((types::u32) blockIdx.x, (types::u32) blockDim.y, (types::u32) threadIdx.y);
}

__device__ __forceinline__ types::u32 segment_stride_2d() {
    return mul_lo_u32((types::u32) gridDim.x, (types::u32) blockDim.y);
}

__device__ __forceinline__ types::u32 atomic_add_u32(types::u32 *addr, types::u32 value) {
#if CELLSHARD_ENABLE_CUDA && defined(__CUDA_ARCH__) && CELLSHARD_CUDA_MODE_NATIVE_EXTREME
    types::u32 old = 0u;
    asm volatile("atom.global.add.u32 %0, [%1], %2;"
                 : "=r"(old)
                 : "l"((unsigned long long) addr), "r"(value)
                 : "memory");
    return old;
#else
    return atomicAdd(addr, value);
#endif
}

} // namespace cellshard::ptx
