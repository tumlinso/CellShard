#pragma once

#ifndef CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#define CELLSHARD_ENABLE_CELLERATOR_QUANTIZED 1
#endif

#if CELLSHARD_ENABLE_CELLERATOR_QUANTIZED

#include <Cellerator/core/matrix/quantized_blocked_ell.cuh>

namespace cellshard {
namespace sparse {

using ::cellerator::core::matrix::quantized_blocked_ell;
using ::cellerator::core::matrix::quantized_blocked_ell_host_registered;
using ::cellerator::core::matrix::quantized_blocked_ell_invalid_col;
using ::cellerator::core::matrix::quantized_blocked_ell_decode_policy_unknown;
using ::cellerator::core::matrix::quantized_blocked_ell_decode_policy_per_gene_affine;
using ::cellerator::core::matrix::quantized_blocked_ell_decode_policy_column_scale_row_offset;
using ::cellerator::core::matrix::allocate;
using ::cellerator::core::matrix::block_col_idx_count;
using ::cellerator::core::matrix::bytes;
using ::cellerator::core::matrix::clear;
using ::cellerator::core::matrix::ell_width_blocks;
using ::cellerator::core::matrix::host_registered;
using ::cellerator::core::matrix::init;
using ::cellerator::core::matrix::pack_quantized_blocked_ell_aux;
using ::cellerator::core::matrix::packed_value_bytes;
using ::cellerator::core::matrix::pin;
using ::cellerator::core::matrix::quantized_blocked_ell_aligned_row_bytes;
using ::cellerator::core::matrix::quantized_blocked_ell_codes_per_byte;
using ::cellerator::core::matrix::quantized_blocked_ell_row_bytes;
using ::cellerator::core::matrix::row_block_count;
using ::cellerator::core::matrix::unpack_quantized_blocked_ell_bits;
using ::cellerator::core::matrix::unpack_quantized_blocked_ell_block_size;
using ::cellerator::core::matrix::unpack_quantized_blocked_ell_cols;
using ::cellerator::core::matrix::unpack_quantized_blocked_ell_ell_width;
using ::cellerator::core::matrix::unpin;

} // namespace sparse
} // namespace cellshard

#endif
