#pragma once

#include <Cellerator/core/matrix/dense.cuh>

namespace cellshard {

using ::cellerator::core::matrix::dense;
using ::cellerator::core::matrix::dense_col_major;
using ::cellerator::core::matrix::dense_host_registered;
using ::cellerator::core::matrix::dense_row_major;
using ::cellerator::core::matrix::allocate;
using ::cellerator::core::matrix::at;
using ::cellerator::core::matrix::attach;
using ::cellerator::core::matrix::bytes;
using ::cellerator::core::matrix::clear;
using ::cellerator::core::matrix::host_registered;
using ::cellerator::core::matrix::init;
using ::cellerator::core::matrix::offset;
using ::cellerator::core::matrix::packed_stride;
using ::cellerator::core::matrix::payload_bytes;
using ::cellerator::core::matrix::payload_elements;
using ::cellerator::core::matrix::pin;
using ::cellerator::core::matrix::unpin;

__host__ __device__ __forceinline__ int dense_is_packed_row_major(const dense * __restrict__ m) {
    return m != 0 && m->order == dense_row_major && m->stride == m->cols;
}

} // namespace cellshard
