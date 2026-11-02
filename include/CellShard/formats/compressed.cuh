#pragma once

#include <Cellerator/interop/formats/compressed.cuh>

namespace cellshard {
namespace sparse {

using compressed = ::cellerator::interop::formats::CompressedPayload;
using ::cellerator::interop::formats::compressed_by_col;
using ::cellerator::interop::formats::compressed_by_row;
using ::cellerator::interop::formats::compressed_host_registered;
using ::cellerator::interop::formats::allocate;
using ::cellerator::interop::formats::at;
using ::cellerator::interop::formats::bytes;
using ::cellerator::interop::formats::clear;
using ::cellerator::interop::formats::host_registered;
using ::cellerator::interop::formats::init;
using ::cellerator::interop::formats::major_dim;
using ::cellerator::interop::formats::minor_dim;
using ::cellerator::interop::formats::pin;
using ::cellerator::interop::formats::unpin;

} // namespace sparse
} // namespace cellshard
