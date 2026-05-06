#pragma once

#include "../../../include/CellShard/io/csh5/api.cuh"

namespace cellshard {

struct sliced_execution_cuda_layout {
    const std::uint32_t *row_order;
    const std::uint32_t *segment_row_offsets;
    const std::uint32_t *segment_widths;
    std::uint32_t segment_count;
};

int fill_bucketed_sliced_execution_partition_cuda(bucketed_sliced_ell_partition *out,
                                                  const sparse::sliced_ell *part,
                                                  const sliced_execution_cuda_layout *layout);

} // namespace cellshard
