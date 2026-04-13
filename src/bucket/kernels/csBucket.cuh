#pragma once

#include "../../offset_span.cuh"
#include "../../types.cuh"

#include <cuda_runtime.h>

namespace cellshard {
namespace bucket {
namespace kernels {

static inline void setup_cs_bucket_major_launch(types::dim_t major_dim, dim3 *grid, dim3 *block) {
    block->x = 256;
    block->y = 1;
    block->z = 1;
    grid->x = (unsigned int) (((unsigned long) major_dim + 255ul) >> 8);
    grid->y = 1;
    grid->z = 1;
    if (grid->x < 1u) grid->x = 1u;
    if (grid->x > 4096u) grid->x = 4096u;
}

#include "csBucket/count_major_nnz.cuh"

#include "csBucket/init_major_identity.cuh"

#include "csBucket/fill_equal_count_bucket_offsets.cuh"

#include "csBucket/gather_major_nnz_by_order.cuh"

#include "csBucket/gather_shifted_major_counts_from_order.cuh"

#include "csBucket/scatter_inverse_major_order.cuh"

#include "csBucket/reorder_compressed_major_segments.cuh"

#include "csBucket/count_shard_major_nnz.cuh"

#include "csBucket/gather_shifted_shard_major_counts_from_order.cuh"

#include "csBucket/reorder_shard_major_segments.cuh"

} // namespace kernels
} // namespace bucket
} // namespace cellshard
