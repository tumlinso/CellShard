#pragma once

#include <Cellerator/core/matrix/compressed.cuh>

namespace cellshard {
namespace sparse {

using ::cellerator::core::matrix::compressed;
using ::cellerator::core::matrix::compressed_by_col;
using ::cellerator::core::matrix::compressed_by_row;
using ::cellerator::core::matrix::compressed_host_registered;
using ::cellerator::core::matrix::allocate;
using ::cellerator::core::matrix::at;
using ::cellerator::core::matrix::bytes;
using ::cellerator::core::matrix::clear;
using ::cellerator::core::matrix::host_registered;
using ::cellerator::core::matrix::init;
using ::cellerator::core::matrix::major_dim;
using ::cellerator::core::matrix::minor_dim;
using ::cellerator::core::matrix::pin;
using ::cellerator::core::matrix::unpin;

} // namespace sparse
} // namespace cellshard
