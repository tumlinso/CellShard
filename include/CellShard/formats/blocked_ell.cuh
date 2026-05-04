#pragma once

#include <Cellerator/core/matrix/blocked_ell.cuh>

namespace cellshard {
namespace sparse {

using ::cellerator::core::matrix::blocked_ell;
using ::cellerator::core::matrix::blocked_ell_host_registered;
using ::cellerator::core::matrix::blocked_ell_invalid_col;
using ::cellerator::core::matrix::allocate;
using ::cellerator::core::matrix::at;
using ::cellerator::core::matrix::bytes;
using ::cellerator::core::matrix::clear;
using ::cellerator::core::matrix::col_block_count;
using ::cellerator::core::matrix::ell_width_blocks;
using ::cellerator::core::matrix::host_registered;
using ::cellerator::core::matrix::init;
using ::cellerator::core::matrix::pack_blocked_ell_aux;
using ::cellerator::core::matrix::pin;
using ::cellerator::core::matrix::row_block_count;
using ::cellerator::core::matrix::unpack_blocked_ell_block_size;
using ::cellerator::core::matrix::unpack_blocked_ell_cols;
using ::cellerator::core::matrix::unpack_blocked_ell_ell_width;
using ::cellerator::core::matrix::unpin;

} // namespace sparse
} // namespace cellshard
