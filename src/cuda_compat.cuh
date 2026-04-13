#pragma once

#ifndef CELLSHARD_ENABLE_CUDA
#define CELLSHARD_ENABLE_CUDA 1
#endif

#if CELLSHARD_ENABLE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#else

#include <cstddef>
#include <cstdint>
#include <cstring>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __shared__
#define __shared__
#endif

#ifndef __forceinline__
#if defined(__GNUC__) || defined(__clang__)
#define __forceinline__ inline __attribute__((always_inline))
#else
#define __forceinline__ inline
#endif
#endif

struct __half {
    std::uint16_t x;
};

using cudaError_t = int;
using cudaStream_t = void *;

static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t cudaErrorHostMemoryAlreadyRegistered = 712;
static constexpr cudaError_t cudaErrorInvalidValue = 11;
static constexpr unsigned int cudaHostRegisterPortable = 0u;

inline const char *cudaGetErrorString(cudaError_t err) {
    switch (err) {
    case cudaSuccess:
        return "cudaSuccess";
    case cudaErrorHostMemoryAlreadyRegistered:
        return "cudaErrorHostMemoryAlreadyRegistered";
    case cudaErrorInvalidValue:
        return "cudaErrorInvalidValue";
    default:
        return "cudaDisabled";
    }
}

inline cudaError_t cudaGetLastError() {
    return cudaSuccess;
}

inline cudaError_t cudaHostRegister(void *, std::size_t, unsigned int) {
    return cudaSuccess;
}

inline cudaError_t cudaHostUnregister(void *) {
    return cudaSuccess;
}

inline float __half2float(__half value) {
    const std::uint32_t sign = (std::uint32_t) (value.x & 0x8000u) << 16u;
    const std::uint32_t exp = (value.x >> 10u) & 0x1fu;
    const std::uint32_t mant = value.x & 0x03ffu;
    std::uint32_t bits = 0u;

    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            std::uint32_t norm = mant;
            std::uint32_t shift = 0u;
            while ((norm & 0x0400u) == 0u) {
                norm <<= 1u;
                ++shift;
            }
            norm &= 0x03ffu;
            bits = sign | ((127u - 15u - shift) << 23u) | (norm << 13u);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13u);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23u) | (mant << 13u);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline __half __float2half(float value) {
    std::uint32_t bits = 0u;
    std::uint32_t sign = 0u;
    std::uint32_t exp = 0u;
    std::uint32_t mant = 0u;
    __half out{};

    std::memcpy(&bits, &value, sizeof(bits));
    sign = (bits >> 16u) & 0x8000u;
    exp = (bits >> 23u) & 0xffu;
    mant = bits & 0x7fffffu;

    if (exp == 0xffu) {
        out.x = (std::uint16_t) (sign | 0x7c00u | (mant != 0u ? 0x0200u : 0u));
        return out;
    }

    if (exp > 142u) {
        out.x = (std::uint16_t) (sign | 0x7c00u);
        return out;
    }

    if (exp < 113u) {
        if (exp < 103u) {
            out.x = (std::uint16_t) sign;
            return out;
        }
        mant |= 0x00800000u;
        {
            const std::uint32_t shift = 125u - exp;
            std::uint32_t half_mant = mant >> shift;
            const std::uint32_t round_bit = 1u << (shift - 1u);
            if ((mant & round_bit) != 0u) half_mant += 1u;
            out.x = (std::uint16_t) (sign | half_mant);
        }
        return out;
    }

    exp = exp - 112u;
    mant = mant + 0x00001000u;
    if ((mant & 0x00800000u) != 0u) {
        mant = 0u;
        ++exp;
    }
    if (exp >= 31u) {
        out.x = (std::uint16_t) (sign | 0x7c00u);
        return out;
    }
    out.x = (std::uint16_t) (sign | (exp << 10u) | (mant >> 13u));
    return out;
}

#endif
