#pragma once

#ifndef CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#define CELLSHARD_ENABLE_CELLERATOR_QUANTIZED 1
#endif

#if CELLSHARD_ENABLE_CELLERATOR_QUANTIZED

#include <Cellerator/interop/formats/affine_blocked_ell.hh>

namespace cellshard {
namespace sparse {

using quantized_blocked_ell = ::cellerator::interop::formats::AffineBlockedEllCodec;
using ::cellerator::interop::formats::quantized_blocked_ell_host_registered;
using ::cellerator::interop::formats::quantized_blocked_ell_invalid_col;
using ::cellerator::interop::formats::quantized_blocked_ell_decode_policy_unknown;
using ::cellerator::interop::formats::quantized_blocked_ell_decode_policy_per_gene_affine;
using ::cellerator::interop::formats::quantized_blocked_ell_decode_policy_column_scale_row_offset;
using ::cellerator::interop::formats::allocate;
using ::cellerator::interop::formats::block_col_idx_count;
using ::cellerator::interop::formats::bytes;
using ::cellerator::interop::formats::clear;
using ::cellerator::interop::formats::ell_width_blocks;
using ::cellerator::interop::formats::host_registered;
using ::cellerator::interop::formats::init;
using ::cellerator::interop::formats::pack_quantized_blocked_ell_aux;
using ::cellerator::interop::formats::packed_value_bytes;
using ::cellerator::interop::formats::pin;
using ::cellerator::interop::formats::quantized_blocked_ell_aligned_row_bytes;
using ::cellerator::interop::formats::quantized_blocked_ell_codes_per_byte;
using ::cellerator::interop::formats::quantized_blocked_ell_row_bytes;
using ::cellerator::interop::formats::row_block_count;
using ::cellerator::interop::formats::unpack_quantized_blocked_ell_bits;
using ::cellerator::interop::formats::unpack_quantized_blocked_ell_block_size;
using ::cellerator::interop::formats::unpack_quantized_blocked_ell_cols;
using ::cellerator::interop::formats::unpack_quantized_blocked_ell_ell_width;
using ::cellerator::interop::formats::unpin;

} // namespace sparse
} // namespace cellshard

#endif
