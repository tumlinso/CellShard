#pragma once

#include "../../formats/triplet.cuh"

#include <cub/cub.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cellshard {
namespace bucket {

struct blocked_ell_bipartite_optimize_workspace {
    int device;
    cudaStream_t stream;
    int owns_stream;

    std::size_t nnz_capacity;
    std::size_t row_capacity;
    std::size_t col_capacity;
    std::size_t entity_capacity;
    std::size_t sort_bytes;

    types::idx_t *d_row_idx;
    types::idx_t *d_col_idx;
    types::u32 *d_rank_map;
    float *d_entity_sum;
    types::u32 *d_entity_count;
    types::u64 *d_sort_keys_in;
    types::u64 *d_sort_keys_out;
    types::u32 *d_order_in;
    types::u32 *d_order_out;
    void *d_sort_tmp;
};

inline void init(blocked_ell_bipartite_optimize_workspace *ws) {
    if (ws == nullptr) return;
    ws->device = -1;
    ws->stream = (cudaStream_t) 0;
    ws->owns_stream = 0;
    ws->nnz_capacity = 0u;
    ws->row_capacity = 0u;
    ws->col_capacity = 0u;
    ws->entity_capacity = 0u;
    ws->sort_bytes = 0u;
    ws->d_row_idx = nullptr;
    ws->d_col_idx = nullptr;
    ws->d_rank_map = nullptr;
    ws->d_entity_sum = nullptr;
    ws->d_entity_count = nullptr;
    ws->d_sort_keys_in = nullptr;
    ws->d_sort_keys_out = nullptr;
    ws->d_order_in = nullptr;
    ws->d_order_out = nullptr;
    ws->d_sort_tmp = nullptr;
}

inline void clear(blocked_ell_bipartite_optimize_workspace *ws) {
    if (ws == nullptr) return;
    if (ws->device >= 0) cudaSetDevice(ws->device);
    if (ws->d_sort_tmp != nullptr) cudaFree(ws->d_sort_tmp);
    if (ws->d_order_out != nullptr) cudaFree(ws->d_order_out);
    if (ws->d_order_in != nullptr) cudaFree(ws->d_order_in);
    if (ws->d_sort_keys_out != nullptr) cudaFree(ws->d_sort_keys_out);
    if (ws->d_sort_keys_in != nullptr) cudaFree(ws->d_sort_keys_in);
    if (ws->d_entity_count != nullptr) cudaFree(ws->d_entity_count);
    if (ws->d_entity_sum != nullptr) cudaFree(ws->d_entity_sum);
    if (ws->d_rank_map != nullptr) cudaFree(ws->d_rank_map);
    if (ws->d_col_idx != nullptr) cudaFree(ws->d_col_idx);
    if (ws->d_row_idx != nullptr) cudaFree(ws->d_row_idx);
    if (ws->owns_stream && ws->stream != (cudaStream_t) 0) cudaStreamDestroy(ws->stream);
    init(ws);
}

inline int optimize_cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "cellshard: CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

inline int setup(blocked_ell_bipartite_optimize_workspace *ws,
                 int device,
                 cudaStream_t stream = (cudaStream_t) 0) {
    if (ws == nullptr) return 0;
    clear(ws);
    ws->device = device;
    if (!optimize_cuda_check(cudaSetDevice(device), "cudaSetDevice blocked_ell_bipartite")) return 0;
    if (stream == (cudaStream_t) 0) {
        if (!optimize_cuda_check(cudaStreamCreateWithFlags(&ws->stream, cudaStreamNonBlocking),
                                 "cudaStreamCreateWithFlags blocked_ell_bipartite")) {
            clear(ws);
            return 0;
        }
        ws->owns_stream = 1;
    } else {
        ws->stream = stream;
        ws->owns_stream = 0;
    }
    return 1;
}

inline int reserve(blocked_ell_bipartite_optimize_workspace *ws,
                   std::size_t nnz,
                   std::size_t rows,
                   std::size_t cols,
                   std::size_t entities) {
    std::size_t sort_bytes = 0u;
    if (ws == nullptr) return 0;
    if (!optimize_cuda_check(cudaSetDevice(ws->device >= 0 ? ws->device : 0),
                             "cudaSetDevice blocked_ell_bipartite reserve")) {
        return 0;
    }
    if (nnz > ws->nnz_capacity) {
        if (ws->d_row_idx != nullptr) cudaFree(ws->d_row_idx);
        if (ws->d_col_idx != nullptr) cudaFree(ws->d_col_idx);
        ws->d_row_idx = nullptr;
        ws->d_col_idx = nullptr;
        if (nnz != 0u) {
            if (!optimize_cuda_check(cudaMalloc((void **) &ws->d_row_idx, nnz * sizeof(types::idx_t)),
                                     "cudaMalloc bipartite row_idx")) {
                return 0;
            }
            if (!optimize_cuda_check(cudaMalloc((void **) &ws->d_col_idx, nnz * sizeof(types::idx_t)),
                                     "cudaMalloc bipartite col_idx")) {
                return 0;
            }
        }
        ws->nnz_capacity = nnz;
    }
    if (std::max(rows, cols) > ws->col_capacity) {
        if (ws->d_rank_map != nullptr) cudaFree(ws->d_rank_map);
        ws->d_rank_map = nullptr;
        if (std::max(rows, cols) != 0u) {
            if (!optimize_cuda_check(cudaMalloc((void **) &ws->d_rank_map, std::max(rows, cols) * sizeof(types::u32)),
                                     "cudaMalloc bipartite rank_map")) {
                return 0;
            }
        }
        ws->col_capacity = std::max(rows, cols);
    }
    if (entities > ws->entity_capacity) {
        if (ws->d_entity_sum != nullptr) cudaFree(ws->d_entity_sum);
        if (ws->d_entity_count != nullptr) cudaFree(ws->d_entity_count);
        if (ws->d_sort_keys_in != nullptr) cudaFree(ws->d_sort_keys_in);
        if (ws->d_sort_keys_out != nullptr) cudaFree(ws->d_sort_keys_out);
        if (ws->d_order_in != nullptr) cudaFree(ws->d_order_in);
        if (ws->d_order_out != nullptr) cudaFree(ws->d_order_out);
        ws->d_entity_sum = nullptr;
        ws->d_entity_count = nullptr;
        ws->d_sort_keys_in = nullptr;
        ws->d_sort_keys_out = nullptr;
        ws->d_order_in = nullptr;
        ws->d_order_out = nullptr;
        if (entities != 0u) {
            if (!optimize_cuda_check(cudaMalloc((void **) &ws->d_entity_sum, entities * sizeof(float)),
                                     "cudaMalloc bipartite entity_sum")) {
                return 0;
            }
            if (!optimize_cuda_check(cudaMalloc((void **) &ws->d_entity_count, entities * sizeof(types::u32)),
                                     "cudaMalloc bipartite entity_count")) {
                return 0;
            }
            if (!optimize_cuda_check(cudaMalloc((void **) &ws->d_sort_keys_in, entities * sizeof(types::u64)),
                                     "cudaMalloc bipartite sort_keys_in")) {
                return 0;
            }
            if (!optimize_cuda_check(cudaMalloc((void **) &ws->d_sort_keys_out, entities * sizeof(types::u64)),
                                     "cudaMalloc bipartite sort_keys_out")) {
                return 0;
            }
            if (!optimize_cuda_check(cudaMalloc((void **) &ws->d_order_in, entities * sizeof(types::u32)),
                                     "cudaMalloc bipartite order_in")) {
                return 0;
            }
            if (!optimize_cuda_check(cudaMalloc((void **) &ws->d_order_out, entities * sizeof(types::u32)),
                                     "cudaMalloc bipartite order_out")) {
                return 0;
            }
        }
        ws->entity_capacity = entities;
    }
    if (cub::DeviceRadixSort::SortPairs(nullptr,
                                        sort_bytes,
                                        (const types::u64 *) nullptr,
                                        (types::u64 *) nullptr,
                                        (const types::u32 *) nullptr,
                                        (types::u32 *) nullptr,
                                        entities,
                                        0,
                                        64,
                                        ws->stream) != cudaSuccess) {
        return 0;
    }
    if (sort_bytes > ws->sort_bytes) {
        if (ws->d_sort_tmp != nullptr) cudaFree(ws->d_sort_tmp);
        ws->d_sort_tmp = nullptr;
        if (sort_bytes != 0u
            && !optimize_cuda_check(cudaMalloc(&ws->d_sort_tmp, sort_bytes),
                                    "cudaMalloc bipartite sort_tmp")) {
            return 0;
        }
        ws->sort_bytes = sort_bytes;
    }
    ws->row_capacity = rows;
    return 1;
}

namespace detail {

constexpr std::uint32_t invalid_rank = 0xffffffffu;

__device__ inline std::uint32_t float_to_ordered_u32_(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t mask = (bits & 0x80000000u) != 0u ? 0xffffffffu : 0x80000000u;
    return bits ^ mask;
}

__global__ inline void init_identity_order_kernel_(std::uint32_t count,
                                                   std::uint32_t *order) {
    const std::uint32_t tid = (std::uint32_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const std::uint32_t stride = (std::uint32_t) (gridDim.x * blockDim.x);
    for (std::uint32_t i = tid; i < count; i += stride) order[i] = i;
}

__global__ inline void accumulate_column_score_kernel_(std::uint32_t nnz,
                                                       const types::idx_t *row_idx,
                                                       const types::idx_t *col_idx,
                                                       const std::uint32_t *row_rank,
                                                       float *col_sum,
                                                       std::uint32_t *col_count) {
    const std::uint32_t tid = (std::uint32_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const std::uint32_t stride = (std::uint32_t) (gridDim.x * blockDim.x);
    for (std::uint32_t i = tid; i < nnz; i += stride) {
        const std::uint32_t row = row_idx[i];
        const std::uint32_t col = col_idx[i];
        const std::uint32_t rank = row_rank[row];
        if (rank == invalid_rank) continue;
        atomicAdd(col_sum + col, (float) rank);
        atomicAdd(col_count + col, 1u);
    }
}

__global__ inline void accumulate_row_score_kernel_(std::uint32_t nnz,
                                                    const types::idx_t *row_idx,
                                                    const types::idx_t *col_idx,
                                                    const std::uint32_t *col_rank,
                                                    float *row_sum,
                                                    std::uint32_t *row_count) {
    const std::uint32_t tid = (std::uint32_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const std::uint32_t stride = (std::uint32_t) (gridDim.x * blockDim.x);
    for (std::uint32_t i = tid; i < nnz; i += stride) {
        const std::uint32_t row = row_idx[i];
        const std::uint32_t col = col_idx[i];
        const std::uint32_t rank = col_rank[col];
        if (rank == invalid_rank) continue;
        atomicAdd(row_sum + row, (float) rank);
        atomicAdd(row_count + row, 1u);
    }
}

__global__ inline void build_entity_sort_keys_kernel_(std::uint32_t count,
                                                      const float *sum,
                                                      const std::uint32_t *weight,
                                                      std::uint64_t *keys) {
    const std::uint32_t tid = (std::uint32_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const std::uint32_t stride = (std::uint32_t) (gridDim.x * blockDim.x);
    for (std::uint32_t i = tid; i < count; i += stride) {
        const std::uint32_t w = weight[i];
        const float barycenter = w != 0u ? sum[i] / (float) w : FLT_MAX;
        keys[i] = ((std::uint64_t) float_to_ordered_u32_(barycenter) << 32u) | (std::uint64_t) i;
    }
}

} // namespace detail

inline int upload_coo(blocked_ell_bipartite_optimize_workspace *ws,
                      const sparse::coo *src) {
    if (ws == nullptr || src == nullptr) return 0;
    if (!reserve(ws, src->nnz, src->rows, src->cols, std::max<std::size_t>(src->rows, src->cols))) return 0;
    if (src->nnz != 0u) {
        if (!optimize_cuda_check(cudaMemcpyAsync(ws->d_row_idx,
                                                 src->rowIdx,
                                                 (std::size_t) src->nnz * sizeof(types::idx_t),
                                                 cudaMemcpyHostToDevice,
                                                 ws->stream),
                                 "cudaMemcpyAsync bipartite row_idx")) {
            return 0;
        }
        if (!optimize_cuda_check(cudaMemcpyAsync(ws->d_col_idx,
                                                 src->colIdx,
                                                 (std::size_t) src->nnz * sizeof(types::idx_t),
                                                 cudaMemcpyHostToDevice,
                                                 ws->stream),
                                 "cudaMemcpyAsync bipartite col_idx")) {
            return 0;
        }
    }
    return 1;
}

inline int sort_entities(blocked_ell_bipartite_optimize_workspace *ws,
                         std::uint32_t entity_count,
                         std::vector<std::uint32_t> *exec_to_canonical) {
    dim3 block(256u, 1u, 1u);
    dim3 grid(entity_count == 0u ? 1u : (std::min<std::uint32_t>)(4096u, (entity_count + 255u) / 256u), 1u, 1u);
    if (ws == nullptr || exec_to_canonical == nullptr) return 0;
    exec_to_canonical->assign((std::size_t) entity_count, 0u);
    if (entity_count == 0u) return 1;
    detail::init_identity_order_kernel_<<<grid, block, 0, ws->stream>>>(entity_count, ws->d_order_in);
    detail::build_entity_sort_keys_kernel_<<<grid, block, 0, ws->stream>>>(
        entity_count,
        ws->d_entity_sum,
        ws->d_entity_count,
        ws->d_sort_keys_in);
    if (!optimize_cuda_check(cudaGetLastError(), "blocked_ell_bipartite build_entity_sort_keys")) return 0;
    if (cub::DeviceRadixSort::SortPairs(ws->d_sort_tmp,
                                        ws->sort_bytes,
                                        ws->d_sort_keys_in,
                                        ws->d_sort_keys_out,
                                        ws->d_order_in,
                                        ws->d_order_out,
                                        entity_count,
                                        0,
                                        64,
                                        ws->stream) != cudaSuccess) {
        return 0;
    }
    if (!optimize_cuda_check(cudaMemcpyAsync(exec_to_canonical->data(),
                                             ws->d_order_out,
                                             (std::size_t) entity_count * sizeof(std::uint32_t),
                                             cudaMemcpyDeviceToHost,
                                             ws->stream),
                             "cudaMemcpyAsync bipartite order_out")) {
        return 0;
    }
    return optimize_cuda_check(cudaStreamSynchronize(ws->stream), "cudaStreamSynchronize blocked_ell_bipartite sort");
}

inline int invert_permutation(const std::vector<std::uint32_t> &exec_to_canonical,
                              std::vector<std::uint32_t> *canonical_to_exec) {
    if (canonical_to_exec == nullptr) return 0;
    canonical_to_exec->assign(exec_to_canonical.size(), detail::invalid_rank);
    for (std::size_t exec = 0u; exec < exec_to_canonical.size(); ++exec) {
        const std::uint32_t canonical = exec_to_canonical[exec];
        if (canonical >= canonical_to_exec->size()) return 0;
        (*canonical_to_exec)[canonical] = (std::uint32_t) exec;
    }
    return 1;
}

inline int build_column_order_from_sampled_coo(const sparse::coo *sampled,
                                               const std::uint32_t *sample_row_rank,
                                               int device,
                                               std::vector<std::uint32_t> *exec_to_canonical_cols,
                                               std::vector<std::uint32_t> *canonical_to_exec_cols,
                                               blocked_ell_bipartite_optimize_workspace *workspace = nullptr,
                                               cudaStream_t stream = (cudaStream_t) 0) {
    blocked_ell_bipartite_optimize_workspace local_ws;
    blocked_ell_bipartite_optimize_workspace *ws = workspace;
    int owns_ws = 0;
    dim3 block(256u, 1u, 1u);
    dim3 grid(sampled != nullptr && sampled->nnz != 0u
                  ? (std::min<std::uint32_t>)(4096u, ((std::uint32_t) sampled->nnz + 255u) / 256u)
                  : 1u,
              1u,
              1u);
    if (sampled == nullptr || exec_to_canonical_cols == nullptr || canonical_to_exec_cols == nullptr) return 0;
    if (ws == nullptr) {
        init(&local_ws);
        if (!setup(&local_ws, device, stream)) return 0;
        ws = &local_ws;
        owns_ws = 1;
    } else if (ws->device != device || (stream != (cudaStream_t) 0 && ws->stream != stream)) {
        if (!setup(ws, device, stream)) return 0;
    }
    if (!upload_coo(ws, sampled)) {
        if (owns_ws) clear(ws);
        return 0;
    }
    if (sampled->rows != 0u) {
        if (!optimize_cuda_check(cudaMemcpyAsync(ws->d_rank_map,
                                                 sample_row_rank,
                                                 (std::size_t) sampled->rows * sizeof(std::uint32_t),
                                                 cudaMemcpyHostToDevice,
                                                 ws->stream),
                                 "cudaMemcpyAsync bipartite sample_row_rank")) {
            if (owns_ws) clear(ws);
            return 0;
        }
    }
    if (!optimize_cuda_check(cudaMemsetAsync(ws->d_entity_sum, 0, (std::size_t) sampled->cols * sizeof(float), ws->stream),
                             "cudaMemsetAsync bipartite column sum")
        || !optimize_cuda_check(cudaMemsetAsync(ws->d_entity_count, 0, (std::size_t) sampled->cols * sizeof(types::u32), ws->stream),
                                "cudaMemsetAsync bipartite column count")) {
        if (owns_ws) clear(ws);
        return 0;
    }
    detail::accumulate_column_score_kernel_<<<grid, block, 0, ws->stream>>>(
        sampled->nnz,
        ws->d_row_idx,
        ws->d_col_idx,
        ws->d_rank_map,
        ws->d_entity_sum,
        ws->d_entity_count);
    if (!optimize_cuda_check(cudaGetLastError(), "blocked_ell_bipartite accumulate_column_score")) {
        if (owns_ws) clear(ws);
        return 0;
    }
    if (!sort_entities(ws, sampled->cols, exec_to_canonical_cols)
        || !invert_permutation(*exec_to_canonical_cols, canonical_to_exec_cols)) {
        if (owns_ws) clear(ws);
        return 0;
    }
    if (owns_ws) clear(ws);
    return 1;
}

inline int build_row_order_from_coo(const sparse::coo *src,
                                    const std::uint32_t *canonical_to_exec_cols,
                                    int device,
                                    std::vector<std::uint32_t> *exec_to_canonical_rows,
                                    std::vector<std::uint32_t> *canonical_to_exec_rows,
                                    blocked_ell_bipartite_optimize_workspace *workspace = nullptr,
                                    cudaStream_t stream = (cudaStream_t) 0) {
    blocked_ell_bipartite_optimize_workspace local_ws;
    blocked_ell_bipartite_optimize_workspace *ws = workspace;
    int owns_ws = 0;
    dim3 block(256u, 1u, 1u);
    dim3 grid(src != nullptr && src->nnz != 0u
                  ? (std::min<std::uint32_t>)(4096u, ((std::uint32_t) src->nnz + 255u) / 256u)
                  : 1u,
              1u,
              1u);
    if (src == nullptr || canonical_to_exec_cols == nullptr
        || exec_to_canonical_rows == nullptr || canonical_to_exec_rows == nullptr) {
        return 0;
    }
    if (ws == nullptr) {
        init(&local_ws);
        if (!setup(&local_ws, device, stream)) return 0;
        ws = &local_ws;
        owns_ws = 1;
    } else if (ws->device != device || (stream != (cudaStream_t) 0 && ws->stream != stream)) {
        if (!setup(ws, device, stream)) return 0;
    }
    if (!upload_coo(ws, src)) {
        if (owns_ws) clear(ws);
        return 0;
    }
    if (src->cols != 0u) {
        if (!optimize_cuda_check(cudaMemcpyAsync(ws->d_rank_map,
                                                 canonical_to_exec_cols,
                                                 (std::size_t) src->cols * sizeof(std::uint32_t),
                                                 cudaMemcpyHostToDevice,
                                                 ws->stream),
                                 "cudaMemcpyAsync bipartite canonical_to_exec_cols")) {
            if (owns_ws) clear(ws);
            return 0;
        }
    }
    if (!reserve(ws, src->nnz, src->rows, src->cols, src->rows)) {
        if (owns_ws) clear(ws);
        return 0;
    }
    if (!optimize_cuda_check(cudaMemsetAsync(ws->d_entity_sum, 0, (std::size_t) src->rows * sizeof(float), ws->stream),
                             "cudaMemsetAsync bipartite row sum")
        || !optimize_cuda_check(cudaMemsetAsync(ws->d_entity_count, 0, (std::size_t) src->rows * sizeof(types::u32), ws->stream),
                                "cudaMemsetAsync bipartite row count")) {
        if (owns_ws) clear(ws);
        return 0;
    }
    detail::accumulate_row_score_kernel_<<<grid, block, 0, ws->stream>>>(
        src->nnz,
        ws->d_row_idx,
        ws->d_col_idx,
        ws->d_rank_map,
        ws->d_entity_sum,
        ws->d_entity_count);
    if (!optimize_cuda_check(cudaGetLastError(), "blocked_ell_bipartite accumulate_row_score")) {
        if (owns_ws) clear(ws);
        return 0;
    }
    if (!sort_entities(ws, src->rows, exec_to_canonical_rows)
        || !invert_permutation(*exec_to_canonical_rows, canonical_to_exec_rows)) {
        if (owns_ws) clear(ws);
        return 0;
    }
    if (owns_ws) clear(ws);
    return 1;
}

} // namespace bucket
} // namespace cellshard
