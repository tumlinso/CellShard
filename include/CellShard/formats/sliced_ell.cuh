#pragma once

#include <Cellerator/matrix/sliced_ell.cuh>

namespace cellshard {
namespace sparse {

using ::cellerator::matrix::sliced_ell;
using ::cellerator::matrix::sliced_ell_host_registered;
using ::cellerator::matrix::sliced_ell_invalid_col;
using ::cellerator::matrix::allocate;
using ::cellerator::matrix::at;
using ::cellerator::matrix::bytes;
using ::cellerator::matrix::clear;
using ::cellerator::matrix::find_slice;
using ::cellerator::matrix::host_registered;
using ::cellerator::matrix::init;
using ::cellerator::matrix::pack_sliced_ell_aux;
using ::cellerator::matrix::pin;
using ::cellerator::matrix::row_nnz;
using ::cellerator::matrix::slice_slot_base;
using ::cellerator::matrix::total_slots;
using ::cellerator::matrix::uniform_slice_rows;
using ::cellerator::matrix::unpack_sliced_ell_slice_count;
using ::cellerator::matrix::unpack_sliced_ell_total_slots;
using ::cellerator::matrix::unpin;

} // namespace sparse
} // namespace cellshard
