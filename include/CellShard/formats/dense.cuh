#pragma once

#include <Cellerator/matrix/dense.cuh>

namespace cellshard {

using ::cellerator::matrix::dense;
using ::cellerator::matrix::dense_col_major;
using ::cellerator::matrix::dense_host_registered;
using ::cellerator::matrix::dense_row_major;
using ::cellerator::matrix::allocate;
using ::cellerator::matrix::at;
using ::cellerator::matrix::attach;
using ::cellerator::matrix::bytes;
using ::cellerator::matrix::clear;
using ::cellerator::matrix::host_registered;
using ::cellerator::matrix::init;
using ::cellerator::matrix::offset;
using ::cellerator::matrix::packed_stride;
using ::cellerator::matrix::payload_bytes;
using ::cellerator::matrix::payload_elements;
using ::cellerator::matrix::pin;
using ::cellerator::matrix::unpin;

__host__ __device__ __forceinline__ int dense_is_packed_row_major(const dense * __restrict__ m) {
    return m != 0 && m->order == dense_row_major && m->stride == m->cols;
}

} // namespace cellshard
