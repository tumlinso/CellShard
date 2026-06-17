#pragma once

#ifndef CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#define CELLSHARD_ENABLE_CELLERATOR_QUANTIZED 1
#endif

#if CELLSHARD_ENABLE_CELLERATOR_QUANTIZED

#include <Cellerator/matrix/quantized_blocked_ell.cuh>

namespace cellshard {
namespace sparse {

using ::cellerator::matrix::quantized_blocked_ell;
using ::cellerator::matrix::quantized_blocked_ell_host_registered;
using ::cellerator::matrix::quantized_blocked_ell_invalid_col;
using ::cellerator::matrix::quantized_blocked_ell_decode_policy_unknown;
using ::cellerator::matrix::quantized_blocked_ell_decode_policy_per_gene_affine;
using ::cellerator::matrix::quantized_blocked_ell_decode_policy_column_scale_row_offset;
using ::cellerator::matrix::allocate;
using ::cellerator::matrix::block_col_idx_count;
using ::cellerator::matrix::bytes;
using ::cellerator::matrix::clear;
using ::cellerator::matrix::ell_width_blocks;
using ::cellerator::matrix::host_registered;
using ::cellerator::matrix::init;
using ::cellerator::matrix::pack_quantized_blocked_ell_aux;
using ::cellerator::matrix::packed_value_bytes;
using ::cellerator::matrix::pin;
using ::cellerator::matrix::quantized_blocked_ell_aligned_row_bytes;
using ::cellerator::matrix::quantized_blocked_ell_codes_per_byte;
using ::cellerator::matrix::quantized_blocked_ell_row_bytes;
using ::cellerator::matrix::row_block_count;
using ::cellerator::matrix::unpack_quantized_blocked_ell_bits;
using ::cellerator::matrix::unpack_quantized_blocked_ell_block_size;
using ::cellerator::matrix::unpack_quantized_blocked_ell_cols;
using ::cellerator::matrix::unpack_quantized_blocked_ell_ell_width;
using ::cellerator::matrix::unpin;

} // namespace sparse
} // namespace cellshard

#endif
