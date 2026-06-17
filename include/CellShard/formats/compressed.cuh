#pragma once

#include <Cellerator/matrix/compressed.cuh>

namespace cellshard {
namespace sparse {

using ::cellerator::matrix::compressed;
using ::cellerator::matrix::compressed_by_col;
using ::cellerator::matrix::compressed_by_row;
using ::cellerator::matrix::compressed_host_registered;
using ::cellerator::matrix::allocate;
using ::cellerator::matrix::at;
using ::cellerator::matrix::bytes;
using ::cellerator::matrix::clear;
using ::cellerator::matrix::host_registered;
using ::cellerator::matrix::init;
using ::cellerator::matrix::major_dim;
using ::cellerator::matrix::minor_dim;
using ::cellerator::matrix::pin;
using ::cellerator::matrix::unpin;

} // namespace sparse
} // namespace cellshard
