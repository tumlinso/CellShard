#pragma once

#include <Cellerator/matrix/coo.cuh>

namespace cellshard {
namespace sparse {

using ::cellerator::matrix::coo;
using ::cellerator::matrix::allocate;
using ::cellerator::matrix::append_rows;
using ::cellerator::matrix::at;
using ::cellerator::matrix::bytes;
using ::cellerator::matrix::clear;
using ::cellerator::matrix::concatenate_rows;
using ::cellerator::matrix::init;

} // namespace sparse
} // namespace cellshard
