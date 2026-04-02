#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "real.cuh"

namespace cellshard {
namespace types {

// Integer type policy for metadata and indices.
// Local physical parts stay 32-bit by default. Global stitched coordinates and
// shard offsets stay unsigned long because they span many parts/files.
enum {
    value_u32 = 1,
    value_i32 = 2
};

static constexpr int value_f16 = real::value_f16;
static constexpr int value_f32 = real::value_f32;
static constexpr int value_f64 = real::value_f64;
static constexpr int value_bf16 = real::value_bf16;
static constexpr int value_fp8_e4m3 = real::value_fp8_e4m3;
static constexpr int value_fp8_e5m2 = real::value_fp8_e5m2;

static constexpr int has_bf16_types = real::has_bf16_types;
static constexpr int has_fp8_types = real::has_fp8_types;

using u32 = std::uint32_t;
using i32 = std::int32_t;
using u64 = std::uint64_t;
using i64 = std::int64_t;

// Local matrix dimensions and nnz within one physical part.
// Keep these 32-bit unless a concrete path proves they are too small.
using dim_t = u32;
using nnz_t = u32;
using idx_t = u32;
using ptr_t = u32;

// Global stitched offsets across many parts/shards/files.
using shard_idx_t = unsigned long;

// Default scalar choices for the current codebase.
using storage_value_t = real::storage_t;
using compute_value_t = real::compute_t;
using accum_value_t = real::accum_t;
using count_value_t = u32;

// Bridge integer codes into the same code/type machinery used for real-valued
// storage in real.cuh.
template<typename T>
struct value_code : real::code_of<T> {};

template<>
struct value_code<u32> {
    enum { code = value_u32 };
};

template<>
struct value_code<i32> {
    enum { code = value_i32 };
};

template<int Code>
struct value_type : real::type_of<Code> {};

template<>
struct value_type<value_u32> {
    using type = u32;
};

template<>
struct value_type<value_i32> {
    using type = i32;
};

} // namespace types
} // namespace cellshard
