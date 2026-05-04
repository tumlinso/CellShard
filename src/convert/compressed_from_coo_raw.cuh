#pragma once

#include <cub/cub.cuh>

#include <Cellerator/compute/matrix/convert/compressed.cuh>

namespace cellshard {
namespace convert {

// Compatibility surface: generic sparse conversion math is implemented by
// CelleratorCompute. CellShard keeps these names while storage, pack, and
// ingest callers migrate to Cellerator headers directly.
using ::cellerator::compute::matrix::convert::build_compressed_from_coo_custom_raw;
using ::cellerator::compute::matrix::convert::build_compressed_from_coo_library_raw;
using ::cellerator::compute::matrix::convert::build_compressed_from_coo_raw;
using ::cellerator::compute::matrix::convert::build_compressed_from_coo_sorted_raw;
using ::cellerator::compute::matrix::convert::build_compressed_from_sorted_coo_custom_raw;
using ::cellerator::compute::matrix::convert::build_cs_from_coo_raw;
using ::cellerator::compute::matrix::convert::compressed_from_coo_library_workspace_bytes;
using ::cellerator::compute::matrix::convert::compressed_from_coo_sorted_workspace_bytes;

} // namespace convert
} // namespace cellshard
