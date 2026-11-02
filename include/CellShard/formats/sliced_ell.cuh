#pragma once

#include <Cellerator/interop/formats/sliced_ell.cuh>

namespace cellshard {
namespace sparse {

using sliced_ell = ::cellerator::interop::formats::SlicedEllPayload;
using ::cellerator::interop::formats::sliced_ell_host_registered;
using ::cellerator::interop::formats::sliced_ell_invalid_col;
using ::cellerator::interop::formats::allocate;
using ::cellerator::interop::formats::at;
using ::cellerator::interop::formats::bytes;
using ::cellerator::interop::formats::clear;
using ::cellerator::interop::formats::find_slice;
using ::cellerator::interop::formats::host_registered;
using ::cellerator::interop::formats::init;
using ::cellerator::interop::formats::pack_sliced_ell_aux;
using ::cellerator::interop::formats::pin;
using ::cellerator::interop::formats::row_nnz;
using ::cellerator::interop::formats::slice_slot_base;
using ::cellerator::interop::formats::total_slots;
using ::cellerator::interop::formats::uniform_slice_rows;
using ::cellerator::interop::formats::unpack_sliced_ell_slice_count;
using ::cellerator::interop::formats::unpack_sliced_ell_total_slots;
using ::cellerator::interop::formats::unpin;

} // namespace sparse
} // namespace cellshard
