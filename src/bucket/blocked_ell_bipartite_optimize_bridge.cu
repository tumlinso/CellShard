#include "blocked_ell_bipartite_optimize_host.cuh"
#include "operators/blocked_ell_bipartite_optimize.cuh"

#include "../convert/blocked_ell_from_coo_cuda.cuh"

#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace cellshard {
namespace bucket {

int blocked_ell_cuda_current_device(int *device_out) {
    if (device_out == nullptr) return 0;
    return cudaGetDevice(device_out) == cudaSuccess;
}

int blocked_ell_bipartite_build_column_order_cuda(const sparse::coo *sampled,
                                                  const std::uint32_t *sample_row_rank,
                                                  int device,
                                                  std::uint32_t *exec_to_canonical_cols,
                                                  std::uint32_t *canonical_to_exec_cols) {
    std::vector<std::uint32_t> exec_to_canonical;
    std::vector<std::uint32_t> canonical_to_exec;
    if (sampled == nullptr || (sampled->cols != 0u && (exec_to_canonical_cols == nullptr || canonical_to_exec_cols == nullptr))) return 0;
    if (!build_column_order_from_sampled_coo(sampled,
                                             sample_row_rank,
                                             device,
                                             &exec_to_canonical,
                                             &canonical_to_exec)) {
        return 0;
    }
    if (!exec_to_canonical.empty()) {
        std::memcpy(exec_to_canonical_cols,
                    exec_to_canonical.data(),
                    exec_to_canonical.size() * sizeof(std::uint32_t));
        std::memcpy(canonical_to_exec_cols,
                    canonical_to_exec.data(),
                    canonical_to_exec.size() * sizeof(std::uint32_t));
    }
    return 1;
}

int blocked_ell_bipartite_build_row_order_cuda(const sparse::coo *src,
                                               const std::uint32_t *canonical_to_exec_cols,
                                               int device,
                                               std::uint32_t *exec_to_canonical_rows,
                                               std::uint32_t *canonical_to_exec_rows) {
    std::vector<std::uint32_t> exec_to_canonical;
    std::vector<std::uint32_t> canonical_to_exec;
    if (src == nullptr || (src->rows != 0u && (exec_to_canonical_rows == nullptr || canonical_to_exec_rows == nullptr))) return 0;
    if (!build_row_order_from_coo(src,
                                  canonical_to_exec_cols,
                                  device,
                                  &exec_to_canonical,
                                  &canonical_to_exec)) {
        return 0;
    }
    if (!exec_to_canonical.empty()) {
        std::memcpy(exec_to_canonical_rows,
                    exec_to_canonical.data(),
                    exec_to_canonical.size() * sizeof(std::uint32_t));
        std::memcpy(canonical_to_exec_rows,
                    canonical_to_exec.data(),
                    canonical_to_exec.size() * sizeof(std::uint32_t));
    }
    return 1;
}

int blocked_ell_from_coo_cuda_auto_bridge(const sparse::coo *src,
                                          types::dim_t cols,
                                          const types::u32 *feature_to_global,
                                          const unsigned int *candidates,
                                          unsigned int candidate_count,
                                          sparse::blocked_ell *dst,
                                          int device,
                                          convert::blocked_ell_tune_result *picked) {
    return convert::blocked_ell_from_coo_cuda_auto(src,
                                                   cols,
                                                   feature_to_global,
                                                   candidates,
                                                   candidate_count,
                                                   dst,
                                                   device,
                                                   picked);
}

} // namespace bucket
} // namespace cellshard
