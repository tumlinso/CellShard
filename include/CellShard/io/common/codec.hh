#pragma once

#include <cstdint>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __forceinline__
#if defined(__GNUC__) || defined(__clang__)
#define __forceinline__ inline __attribute__((always_inline))
#else
#define __forceinline__ inline
#endif
#endif

namespace cellshard {

// Codec families describe how one stored partition payload should be
// interpreted after lightweight container metadata has already been loaded.
enum {
    dataset_codec_family_none = 0u,
    dataset_codec_family_standard_csr = 1u,
    dataset_codec_family_quantized_csr = 2u,
    dataset_codec_family_blocked_ell = 3u,
    dataset_codec_family_sliced_ell = 4u,
    dataset_codec_family_quantized_blocked_ell = 5u
};

enum {
    dataset_quantized_decode_policy_unknown = 0u,
    dataset_quantized_decode_policy_per_gene_affine = 1u,
    dataset_quantized_decode_policy_column_scale_row_offset = 2u
};

enum {
    dataset_codec_flag_direct_device_delivery = 1u << 0,
    dataset_codec_flag_live_fused_decode = 1u << 1,
    dataset_codec_quantized_decode_policy_shift = 8u,
    dataset_codec_quantized_decode_policy_mask = 0xffu << dataset_codec_quantized_decode_policy_shift
};

__host__ __device__ __forceinline__ std::uint32_t dataset_codec_quantized_decode_policy(std::uint32_t flags) {
    return (flags & dataset_codec_quantized_decode_policy_mask) >> dataset_codec_quantized_decode_policy_shift;
}

__host__ __device__ __forceinline__ std::uint32_t set_dataset_codec_quantized_decode_policy(
    std::uint32_t flags,
    std::uint32_t policy) {
    return (flags & ~dataset_codec_quantized_decode_policy_mask)
        | ((policy << dataset_codec_quantized_decode_policy_shift) & dataset_codec_quantized_decode_policy_mask);
}

} // namespace cellshard
