#pragma once

#include <Cellerator/interop/formats/blocked_ell.cuh>

namespace cellshard {
namespace sparse {

using blocked_ell = ::cellerator::interop::formats::BlockedEllPayload;
using ::cellerator::interop::formats::blocked_ell_host_registered;
using ::cellerator::interop::formats::blocked_ell_invalid_col;
using ::cellerator::interop::formats::allocate;
using ::cellerator::interop::formats::at;
using ::cellerator::interop::formats::bytes;
using ::cellerator::interop::formats::clear;
using ::cellerator::interop::formats::col_block_count;
using ::cellerator::interop::formats::ell_width_blocks;
using ::cellerator::interop::formats::host_registered;
using ::cellerator::interop::formats::init;
using ::cellerator::interop::formats::pack_blocked_ell_aux;
using ::cellerator::interop::formats::pin;
using ::cellerator::interop::formats::row_block_count;
using ::cellerator::interop::formats::unpack_blocked_ell_block_size;
using ::cellerator::interop::formats::unpack_blocked_ell_cols;
using ::cellerator::interop::formats::unpack_blocked_ell_ell_width;
using ::cellerator::interop::formats::unpin;

} // namespace sparse
} // namespace cellshard
