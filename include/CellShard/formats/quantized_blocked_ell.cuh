#pragma once

#ifndef CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#define CELLSHARD_ENABLE_CELLERATOR_QUANTIZED 1
#endif

#if CELLSHARD_ENABLE_CELLERATOR_QUANTIZED

#include <Cellerator/quantized/quantized_blocked_ell.cuh>

namespace cellshard {
namespace sparse {

using ::cellerator::quantized::formats::quantized_blocked_ell;

using ::cellerator::quantized::formats::quantized_blocked_ell_host_registered;
using ::cellerator::quantized::formats::quantized_blocked_ell_invalid_col;

using ::cellerator::quantized::formats::quantized_blocked_ell_decode_policy_unknown;
using ::cellerator::quantized::formats::quantized_blocked_ell_decode_policy_per_gene_affine;
using ::cellerator::quantized::formats::quantized_blocked_ell_decode_policy_column_scale_row_offset;

using ::cellerator::quantized::formats::allocate;
using ::cellerator::quantized::formats::block_col_idx_count;
using ::cellerator::quantized::formats::bytes;
using ::cellerator::quantized::formats::clear;
using ::cellerator::quantized::formats::ell_width_blocks;
using ::cellerator::quantized::formats::host_registered;
using ::cellerator::quantized::formats::init;
using ::cellerator::quantized::formats::pack_quantized_blocked_ell_aux;
using ::cellerator::quantized::formats::packed_value_bytes;
using ::cellerator::quantized::formats::pin;
using ::cellerator::quantized::formats::quantized_blocked_ell_aligned_row_bytes;
using ::cellerator::quantized::formats::quantized_blocked_ell_codes_per_byte;
using ::cellerator::quantized::formats::quantized_blocked_ell_row_bytes;
using ::cellerator::quantized::formats::row_block_count;
using ::cellerator::quantized::formats::unpack_quantized_blocked_ell_bits;
using ::cellerator::quantized::formats::unpack_quantized_blocked_ell_block_size;
using ::cellerator::quantized::formats::unpack_quantized_blocked_ell_cols;
using ::cellerator::quantized::formats::unpack_quantized_blocked_ell_ell_width;
using ::cellerator::quantized::formats::unpin;

} // namespace sparse
} // namespace cellshard

#endif
