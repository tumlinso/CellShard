#pragma once

#include <Cellerator/core/matrix/sliced_ell.cuh>

namespace cellshard {
namespace sparse {

using ::cellerator::core::matrix::sliced_ell;
using ::cellerator::core::matrix::sliced_ell_host_registered;
using ::cellerator::core::matrix::sliced_ell_invalid_col;
using ::cellerator::core::matrix::allocate;
using ::cellerator::core::matrix::at;
using ::cellerator::core::matrix::bytes;
using ::cellerator::core::matrix::clear;
using ::cellerator::core::matrix::find_slice;
using ::cellerator::core::matrix::host_registered;
using ::cellerator::core::matrix::init;
using ::cellerator::core::matrix::pack_sliced_ell_aux;
using ::cellerator::core::matrix::pin;
using ::cellerator::core::matrix::row_nnz;
using ::cellerator::core::matrix::slice_slot_base;
using ::cellerator::core::matrix::total_slots;
using ::cellerator::core::matrix::uniform_slice_rows;
using ::cellerator::core::matrix::unpack_sliced_ell_slice_count;
using ::cellerator::core::matrix::unpack_sliced_ell_total_slots;
using ::cellerator::core::matrix::unpin;

} // namespace sparse
} // namespace cellshard
