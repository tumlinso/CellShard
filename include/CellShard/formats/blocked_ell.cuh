#pragma once

#include <Cellerator/matrix/blocked_ell.cuh>

namespace cellshard {
namespace sparse {

using ::cellerator::matrix::blocked_ell;
using ::cellerator::matrix::blocked_ell_host_registered;
using ::cellerator::matrix::blocked_ell_invalid_col;
using ::cellerator::matrix::allocate;
using ::cellerator::matrix::at;
using ::cellerator::matrix::bytes;
using ::cellerator::matrix::clear;
using ::cellerator::matrix::col_block_count;
using ::cellerator::matrix::ell_width_blocks;
using ::cellerator::matrix::host_registered;
using ::cellerator::matrix::init;
using ::cellerator::matrix::pack_blocked_ell_aux;
using ::cellerator::matrix::pin;
using ::cellerator::matrix::row_block_count;
using ::cellerator::matrix::unpack_blocked_ell_block_size;
using ::cellerator::matrix::unpack_blocked_ell_cols;
using ::cellerator::matrix::unpack_blocked_ell_ell_width;
using ::cellerator::matrix::unpin;

} // namespace sparse
} // namespace cellshard
