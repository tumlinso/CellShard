#pragma once

#include <Cellerator/interop/formats/coo.cuh>

namespace cellshard {
namespace sparse {

using coo = ::cellerator::interop::formats::CooPayload;
using ::cellerator::interop::formats::allocate;
using ::cellerator::interop::formats::append_rows;
using ::cellerator::interop::formats::at;
using ::cellerator::interop::formats::bytes;
using ::cellerator::interop::formats::clear;
using ::cellerator::interop::formats::concatenate_rows;
using ::cellerator::interop::formats::init;

} // namespace sparse
} // namespace cellshard
