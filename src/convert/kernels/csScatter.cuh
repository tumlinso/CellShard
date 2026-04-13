#pragma once

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cellshard {
namespace convert {
namespace kernels {

// Count COO entries per compressed-axis bucket into ptr[1..cDim].
// Atomic pressure depends directly on how skewed the compressed-axis index
// distribution is.
#include "csScatter/shift_ptr_idx_count.cuh"

// Copy scanned pointer starts into mutable scatter heads.
// heads is intentionally mutable because scatter consumes it with atomicAdd.
#include "csScatter/init_cs_scatter_heads.cuh"

// Scatter uncompressed-axis indices and values into CSR/CSC order.
// AtomicAdd on heads chooses the final slot for each COO entry.
#include "csScatter/csScatter.cuh"

} // namespace kernels
} // namespace convert
} // namespace cellshard
