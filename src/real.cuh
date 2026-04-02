#pragma once

#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef CELLSHARD_ENABLE_BF16_TYPES
#define CELLSHARD_ENABLE_BF16_TYPES 0
#endif

#ifndef CELLSHARD_ENABLE_FP8_TYPES
#define CELLSHARD_ENABLE_FP8_TYPES 0
#endif

#if CELLSHARD_ENABLE_BF16_TYPES
#if defined(__has_include)
#if __has_include(<cuda_bf16.h>)
#include <cuda_bf16.h>
#else
#error "CELLSHARD_ENABLE_BF16_TYPES requires <cuda_bf16.h>"
#endif
#else
#include <cuda_bf16.h>
#endif
#endif

#if CELLSHARD_ENABLE_FP8_TYPES
#if defined(__has_include)
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#else
#error "CELLSHARD_ENABLE_FP8_TYPES requires <cuda_fp8.h>"
#endif
#else
#include <cuda_fp8.h>
#endif
#endif

// Scalar policy for the library.
//
// storage_t is the value type persisted on host, moved over I/O paths, and
// staged to device by default. compute_t and accum_t are the promotion targets
// for arithmetic and reductions.
namespace real {

enum {
    value_f16 = 3,
    value_f32 = 4,
    value_f64 = 5,
    value_bf16 = 6,
    value_fp8_e4m3 = 7,
    value_fp8_e5m2 = 8
};

static constexpr int has_bf16_types = CELLSHARD_ENABLE_BF16_TYPES;
static constexpr int has_fp8_types = CELLSHARD_ENABLE_FP8_TYPES;

// Default scalar choices for the current codebase.
using storage_t = __half;
using compute_t = float;
using accum_t = float;

template<typename T>
struct is_real_type {
    enum { value = 0 };
};

template<>
struct is_real_type<__half> {
    enum { value = 1 };
};

template<>
struct is_real_type<float> {
    enum { value = 1 };
};

template<>
struct is_real_type<double> {
    enum { value = 1 };
};

#if CELLSHARD_ENABLE_BF16_TYPES
template<>
struct is_real_type<__nv_bfloat16> {
    enum { value = 1 };
};
#endif

#if CELLSHARD_ENABLE_FP8_TYPES
template<>
struct is_real_type<__nv_fp8_e4m3> {
    enum { value = 1 };
};

template<>
struct is_real_type<__nv_fp8_e5m2> {
    enum { value = 1 };
};
#endif

template<typename T>
struct require_real {
    static_assert(is_real_type<T>::value, "real type required");
    using type = T;
};

template<typename T>
struct code_of;

template<>
struct code_of<__half> {
    enum { code = value_f16 };
};

template<>
struct code_of<float> {
    enum { code = value_f32 };
};

template<>
struct code_of<double> {
    enum { code = value_f64 };
};

#if CELLSHARD_ENABLE_BF16_TYPES
template<>
struct code_of<__nv_bfloat16> {
    enum { code = value_bf16 };
};
#endif

#if CELLSHARD_ENABLE_FP8_TYPES
template<>
struct code_of<__nv_fp8_e4m3> {
    enum { code = value_fp8_e4m3 };
};

template<>
struct code_of<__nv_fp8_e5m2> {
    enum { code = value_fp8_e5m2 };
};
#endif

template<int Code>
struct type_of;

template<>
struct type_of<value_f16> {
    using type = __half;
};

template<>
struct type_of<value_f32> {
    using type = float;
};

template<>
struct type_of<value_f64> {
    using type = double;
};

#if CELLSHARD_ENABLE_BF16_TYPES
template<>
struct type_of<value_bf16> {
    using type = __nv_bfloat16;
};
#endif

#if CELLSHARD_ENABLE_FP8_TYPES
template<>
struct type_of<value_fp8_e4m3> {
    using type = __nv_fp8_e4m3;
};

template<>
struct type_of<value_fp8_e5m2> {
    using type = __nv_fp8_e5m2;
};
#endif

template<typename T>
using require_real_t = typename require_real<T>::type;

template<typename T>
__host__ __device__ __forceinline__ constexpr int is_real_v() {
    return is_real_type<T>::value;
}

} // namespace real
