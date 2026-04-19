#pragma once

#include "../convert/blocked_ell_from_compressed.cuh"
#include "../formats/blocked_ell.cuh"
#include "../formats/triplet.cuh"

#include <cstdint>

namespace cellshard {
namespace bucket {

int blocked_ell_cuda_current_device(int *device_out);

int blocked_ell_bipartite_build_column_order_cuda(const sparse::coo *sampled,
                                                  const std::uint32_t *sample_row_rank,
                                                  int device,
                                                  std::uint32_t *exec_to_canonical_cols,
                                                  std::uint32_t *canonical_to_exec_cols);

int blocked_ell_bipartite_build_row_order_cuda(const sparse::coo *src,
                                               const std::uint32_t *canonical_to_exec_cols,
                                               int device,
                                               std::uint32_t *exec_to_canonical_rows,
                                               std::uint32_t *canonical_to_exec_rows);

int blocked_ell_from_coo_cuda_auto_bridge(const sparse::coo *src,
                                          types::dim_t cols,
                                          const types::u32 *feature_to_global,
                                          const unsigned int *candidates,
                                          unsigned int candidate_count,
                                          sparse::blocked_ell *dst,
                                          int device,
                                          convert::blocked_ell_tune_result *picked);

} // namespace bucket
} // namespace cellshard
