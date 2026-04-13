#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cellshard {
namespace convert {
namespace kernels {

// Expand CSR/CSC-style pointer ranges into COO compressed-axis indices.
// One block walks one or more compressed-axis ranges and writes contiguous runs.
// This is the preferred path when the compressed axis is wide enough to expose
// enough blocks.
#include "csExpand/csExpandToCoo.cuh"

// Fallback path when the compressed axis is too small to expose enough blocks.
// Each thread resolves its own COO compressed-axis index by upper-bound search.
// This trades extra pointer-table searches for better parallel exposure when
// cDim is small.
#include "csExpand/csSearchToCoo.cuh"

} // namespace kernels
} // namespace convert
} // namespace cellshard
