#pragma once

#include <Cellerator/compute/matrix/convert/compressed.cuh>

namespace cellshard {
namespace convert {

// Compatibility surface for CelleratorCore-owned transpose and COO scatter
// helpers. CellShard callers keep stable names during the ownership migration.
using ::cellerator::compute::matrix::convert::build_compressed_transpose_raw;
using ::cellerator::compute::matrix::convert::build_compressed_transpose_custom_raw;
using ::cellerator::compute::matrix::convert::build_compressed_transpose_library_raw;
using ::cellerator::compute::matrix::convert::build_transpose_cs_from_cs_raw;
using ::cellerator::compute::matrix::convert::compressed_transpose_library_workspace_bytes;
using ::cellerator::compute::matrix::convert::transpose_coo_entries_raw;

} // namespace convert
} // namespace cellshard
