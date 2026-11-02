#pragma once

#include <Cellerator/interop/formats/dense.cuh>

namespace cellshard {

using dense = ::cellerator::interop::formats::DensePayload;
using ::cellerator::interop::formats::dense_col_major;
using ::cellerator::interop::formats::dense_host_registered;
using ::cellerator::interop::formats::dense_row_major;
using ::cellerator::interop::formats::allocate;
using ::cellerator::interop::formats::at;
using ::cellerator::interop::formats::attach;
using ::cellerator::interop::formats::bytes;
using ::cellerator::interop::formats::clear;
using ::cellerator::interop::formats::host_registered;
using ::cellerator::interop::formats::init;
using ::cellerator::interop::formats::offset;
using ::cellerator::interop::formats::packed_stride;
using ::cellerator::interop::formats::payload_bytes;
using ::cellerator::interop::formats::payload_elements;
using ::cellerator::interop::formats::pin;
using ::cellerator::interop::formats::unpin;

__host__ __device__ __forceinline__ int dense_is_packed_row_major(const dense * __restrict__ m) {
    return m != 0 && m->order == dense_row_major && m->stride == m->cols;
}

} // namespace cellshard
