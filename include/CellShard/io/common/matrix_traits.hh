#pragma once

#ifndef CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#define CELLSHARD_ENABLE_CELLERATOR_QUANTIZED 1
#endif

#include "../../formats/compressed.cuh"
#include "../../formats/blocked_ell.cuh"
#if CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#include "../../formats/quantized_blocked_ell.cuh"
#endif
#include "../../formats/sliced_ell.cuh"
#include "../../formats/dense.cuh"
#include "../../formats/diagonal.cuh"
#include "../../formats/triplet.cuh"
#include "codec.hh"
#include "layout.hh"
#include "raw_format.hh"

namespace cellshard {

template<typename MatrixT>
struct matrix_traits;

template<>
struct matrix_traits<dense> {
    static constexpr disk_format raw_disk_format = disk_format_dense;
    static constexpr std::uint32_t matrix_family = dataset_matrix_family_none;
    static constexpr std::uint32_t execution_format = dataset_execution_format_unknown;
    static constexpr std::uint32_t codec_family = dataset_codec_family_none;
    static inline const char *matrix_format_name() { return "dense"; }
    static inline const char *name() { return "dense matrix"; }
};

template<>
struct matrix_traits<sparse::compressed> {
    static constexpr disk_format raw_disk_format = disk_format_compressed;
    static constexpr std::uint32_t matrix_family = dataset_matrix_family_none;
    static constexpr std::uint32_t execution_format = dataset_execution_format_compressed;
    static constexpr std::uint32_t codec_family = dataset_codec_family_standard_csr;
    static inline const char *matrix_format_name() { return "compressed"; }
    static inline const char *name() { return "compressed matrix"; }
};

template<>
struct matrix_traits<sparse::blocked_ell> {
    static constexpr disk_format raw_disk_format = disk_format_blocked_ell;
    static constexpr std::uint32_t matrix_family = dataset_matrix_family_blocked_ell;
    static constexpr std::uint32_t execution_format = dataset_execution_format_blocked_ell;
    static constexpr std::uint32_t bucketed_execution_format = dataset_execution_format_bucketed_blocked_ell;
    static constexpr std::uint32_t codec_family = dataset_codec_family_blocked_ell;
    static inline const char *matrix_format_name() { return "blocked_ell"; }
    static inline const char *name() { return "blocked ell matrix"; }
};

#if CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
template<>
struct matrix_traits<sparse::quantized_blocked_ell> {
    static constexpr disk_format raw_disk_format = disk_format_quantized_blocked_ell;
    static constexpr std::uint32_t matrix_family = dataset_matrix_family_quantized_blocked_ell;
    static constexpr std::uint32_t execution_format = dataset_execution_format_quantized_blocked_ell;
    static constexpr std::uint32_t bucketed_execution_format = dataset_execution_format_quantized_blocked_ell;
    static constexpr std::uint32_t codec_family = dataset_codec_family_quantized_blocked_ell;
    static inline const char *matrix_format_name() { return "quantized_blocked_ell"; }
    static inline const char *name() { return "quantized blocked ell matrix"; }
};
#endif

template<>
struct matrix_traits<sparse::sliced_ell> {
    static constexpr disk_format raw_disk_format = disk_format_sliced_ell;
    static constexpr std::uint32_t matrix_family = dataset_matrix_family_sliced_ell;
    static constexpr std::uint32_t execution_format = dataset_execution_format_sliced_ell;
    static constexpr std::uint32_t bucketed_execution_format = dataset_execution_format_bucketed_sliced_ell;
    static constexpr std::uint32_t codec_family = dataset_codec_family_sliced_ell;
    static inline const char *matrix_format_name() { return "sliced_ell"; }
    static inline const char *name() { return "sliced ell matrix"; }
};

template<>
struct matrix_traits<sparse::coo> {
    static constexpr disk_format raw_disk_format = disk_format_coo;
    static constexpr std::uint32_t matrix_family = dataset_matrix_family_none;
    static constexpr std::uint32_t execution_format = dataset_execution_format_unknown;
    static constexpr std::uint32_t codec_family = dataset_codec_family_none;
    static inline const char *matrix_format_name() { return "coo"; }
    static inline const char *name() { return "coo matrix"; }
};

template<>
struct matrix_traits<sparse::dia> {
    static constexpr disk_format raw_disk_format = disk_format_dia;
    static constexpr std::uint32_t matrix_family = dataset_matrix_family_none;
    static constexpr std::uint32_t execution_format = dataset_execution_format_unknown;
    static constexpr std::uint32_t codec_family = dataset_codec_family_none;
    static inline const char *matrix_format_name() { return "dia"; }
    static inline const char *name() { return "dia matrix"; }
};

inline std::uint32_t default_execution_format_for_matrix_family(std::uint32_t matrix_family, int optimized_blocked_ell) {
    if (matrix_family == dataset_matrix_family_blocked_ell) {
        return optimized_blocked_ell != 0
            ? dataset_execution_format_bucketed_blocked_ell
            : dataset_execution_format_blocked_ell;
    }
#if CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
    if (matrix_family == dataset_matrix_family_quantized_blocked_ell) return dataset_execution_format_quantized_blocked_ell;
#endif
    if (matrix_family == dataset_matrix_family_sliced_ell) return dataset_execution_format_bucketed_sliced_ell;
    return dataset_execution_format_unknown;
}

} // namespace cellshard
