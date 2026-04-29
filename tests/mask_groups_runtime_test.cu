#include <CellShard/runtime/mask_groups.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>

namespace csr = ::cellshard::runtime;
namespace csd = ::cellshard::device;

namespace {

int check_cuda(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

int close_enough(float got, float expected, float tol = 1.0e-3f) {
    return std::fabs(got - expected) <= tol;
}

int require(int condition, const char *label) {
    if (condition) return 1;
    std::fprintf(stderr, "%s\n", label);
    return 0;
}

std::uint32_t group_bit(unsigned int group) {
    return group < csr::CELLSHARD_MAX_MASK_GROUPS ? (1u << group) : 0u;
}

int copy_reduce(csr::sparse_group_reduce_workspace *ws,
                const csr::sparse_group_reduce_result &result,
                float *totals,
                unsigned int *detected,
                unsigned char *keep,
                float *groups,
                float *pct) {
    if (!check_cuda(cudaStreamSynchronize(ws->stream), "sync reduce")) return 0;
    if (!check_cuda(cudaMemcpy(totals, result.row_totals, (std::size_t) result.rows * sizeof(float), cudaMemcpyDeviceToHost), "copy totals")) return 0;
    if (!check_cuda(cudaMemcpy(detected, result.detected_features, (std::size_t) result.rows * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy detected")) return 0;
    if (!check_cuda(cudaMemcpy(keep, result.row_keep, (std::size_t) result.rows, cudaMemcpyDeviceToHost), "copy keep")) return 0;
    if (result.group_count != 0u) {
        const std::size_t count = (std::size_t) result.rows * result.group_count;
        if (!check_cuda(cudaMemcpy(groups, result.group_counts, count * sizeof(float), cudaMemcpyDeviceToHost), "copy groups")) return 0;
        if (!check_cuda(cudaMemcpy(pct, result.group_percentages, count * sizeof(float), cudaMemcpyDeviceToHost), "copy pct")) return 0;
    }
    return 1;
}

} // namespace

int main() {
    const unsigned int rows = 4u, cols = 5u;
    const std::uint32_t group_masks[5] = {
        group_bit(0u),
        group_bit(1u),
        group_bit(0u) | group_bit(1u),
        0u,
        group_bit(2u)
    };
    const unsigned char row_keep_host[4] = {1u, 0u, 1u, 1u};
    const unsigned char feature_keep_host[5] = {1u, 1u, 0u, 1u, 1u};

    unsigned char *row_keep = nullptr;
    unsigned char *feature_keep = nullptr;
    if (!check_cuda(cudaMalloc((void **) &row_keep, sizeof(row_keep_host)), "cudaMalloc row_keep")) return 1;
    if (!check_cuda(cudaMalloc((void **) &feature_keep, sizeof(feature_keep_host)), "cudaMalloc feature_keep")) return 2;
    if (!check_cuda(cudaMemcpy(row_keep, row_keep_host, sizeof(row_keep_host), cudaMemcpyHostToDevice), "copy row_keep")) return 3;
    if (!check_cuda(cudaMemcpy(feature_keep, feature_keep_host, sizeof(feature_keep_host), cudaMemcpyHostToDevice), "copy feature_keep")) return 4;

    csr::sparse_group_reduce_workspace ws;
    csr::init(&ws);
    if (!csr::setup(&ws, 0)) return 5;
    if (!csr::upload_feature_group_masks(&ws, cols, group_masks)) return 5;

    csr::group_mask_config_view groups{3u, nullptr, ws.feature_group_masks};
    csr::row_feature_mask_view masks{row_keep, feature_keep, nullptr, nullptr};
    csr::sparse_group_filter_params filter{1.0f, 1u, 0.80f, 0u};
    csr::sparse_group_reduce_result result{};
    float totals[4] = {}, group_counts[12] = {}, group_pct[12] = {};
    unsigned int detected[4] = {};
    unsigned char keep[4] = {};

    unsigned int block_cols_host[8] = {0u, 1u, 0u, 1u, 0u, 1u, 1u, 2u};
    const float blocked_values_fp32[16] = {
        5.0f, 3.0f, 7.0f, 0.0f,
        9.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 4.0f, 6.0f, 2.0f,
        0.0f, 8.0f, 0.0f, 10.0f
    };
    __half blocked_values_host[16];
    for (unsigned int i = 0u; i < 16u; ++i) blocked_values_host[i] = __float2half(blocked_values_fp32[i]);
    unsigned int *block_cols = nullptr;
    __half *blocked_values = nullptr;
    if (!check_cuda(cudaMalloc((void **) &block_cols, sizeof(block_cols_host)), "cudaMalloc block_cols")) return 6;
    if (!check_cuda(cudaMalloc((void **) &blocked_values, sizeof(blocked_values_host)), "cudaMalloc blocked_values")) return 7;
    if (!check_cuda(cudaMemcpy(block_cols, block_cols_host, sizeof(block_cols_host), cudaMemcpyHostToDevice), "copy block cols")) return 8;
    if (!check_cuda(cudaMemcpy(blocked_values, blocked_values_host, sizeof(blocked_values_host), cudaMemcpyHostToDevice), "copy blocked values")) return 9;

    csd::blocked_ell_view blocked{};
    blocked.rows = rows;
    blocked.cols = cols;
    blocked.nnz = 10u;
    blocked.block_size = 2u;
    blocked.ell_cols = 4u;
    blocked.blockColIdx = block_cols;
    blocked.val = blocked_values;
    if (!csr::compute_sparse_group_reduce(&blocked, &ws, &groups, &masks, &filter, &result)) return 10;
    if (!copy_reduce(&ws, result, totals, detected, keep, group_counts, group_pct)) return 11;
    if (!close_enough(totals[0], 8.0f) || !close_enough(totals[1], 0.0f) || !close_enough(totals[2], 7.0f) || !close_enough(totals[3], 18.0f)) return 12;
    if (detected[0] != 2u || detected[1] != 0u || detected[2] != 3u || detected[3] != 2u) return 13;
    if (keep[0] == 0u || keep[1] != 0u || keep[2] == 0u || keep[3] != 0u) return 14;
    if (!close_enough(group_counts[0], 5.0f) || !close_enough(group_counts[1], 3.0f) || !close_enough(group_counts[2], 0.0f)) return 15;
    if (!close_enough(group_counts[6], 1.0f) || !close_enough(group_counts[7], 4.0f) || !close_enough(group_counts[8], 2.0f)) return 16;
    if (!close_enough(group_pct[0], 62.5f) || !close_enough(group_pct[4], 0.0f)) return 17;

    csr::group_mask_config_view no_groups{0u, nullptr, nullptr};
    if (!csr::compute_sparse_group_reduce(&blocked, &ws, &no_groups, nullptr, &filter, &result)) return 17;
    if (!copy_reduce(&ws, result, totals, detected, keep, group_counts, group_pct)) return 17;
    if (result.group_count != 0u || !close_enough(totals[0], 15.0f) || detected[0] != 3u) return 17;

    unsigned int compressed_ptr_host[5] = {0u, 3u, 4u, 8u, 10u};
    unsigned int compressed_idx_host[10] = {0u, 1u, 2u, 0u, 0u, 1u, 2u, 4u, 1u, 4u};
    const float compressed_values_fp32[10] = {5.0f, 3.0f, 7.0f, 9.0f, 1.0f, 4.0f, 6.0f, 2.0f, 8.0f, 10.0f};
    __half compressed_values_host[10];
    for (unsigned int i = 0u; i < 10u; ++i) compressed_values_host[i] = __float2half(compressed_values_fp32[i]);
    unsigned int *compressed_ptr = nullptr;
    unsigned int *compressed_idx = nullptr;
    __half *compressed_values = nullptr;
    if (!check_cuda(cudaMalloc((void **) &compressed_ptr, sizeof(compressed_ptr_host)), "cudaMalloc compressed ptr")) return 18;
    if (!check_cuda(cudaMalloc((void **) &compressed_idx, sizeof(compressed_idx_host)), "cudaMalloc compressed idx")) return 19;
    if (!check_cuda(cudaMalloc((void **) &compressed_values, sizeof(compressed_values_host)), "cudaMalloc compressed values")) return 20;
    if (!check_cuda(cudaMemcpy(compressed_ptr, compressed_ptr_host, sizeof(compressed_ptr_host), cudaMemcpyHostToDevice), "copy compressed ptr")) return 21;
    if (!check_cuda(cudaMemcpy(compressed_idx, compressed_idx_host, sizeof(compressed_idx_host), cudaMemcpyHostToDevice), "copy compressed idx")) return 22;
    if (!check_cuda(cudaMemcpy(compressed_values, compressed_values_host, sizeof(compressed_values_host), cudaMemcpyHostToDevice), "copy compressed values")) return 23;
    csd::compressed_view compressed{rows, cols, 10u, ::cellshard::sparse::compressed_by_row, compressed_ptr, compressed_idx, compressed_values};
    if (!csr::compute_sparse_group_reduce_compressed_fallback(&compressed, &ws, &groups, &masks, &filter, &result)) return 24;
    if (!copy_reduce(&ws, result, totals, detected, keep, group_counts, group_pct)) return 25;
    if (!close_enough(totals[2], 7.0f) || !close_enough(group_counts[8], 2.0f)) return 26;

    unsigned int slice_offsets_host[2] = {0u, rows};
    unsigned int slice_widths_host[1] = {3u};
    unsigned int slice_slot_offsets_host[1] = {0u};
    unsigned int sliced_idx_host[12] = {
        0u, 1u, 2u,
        0u, ::cellshard::sparse::sliced_ell_invalid_col, ::cellshard::sparse::sliced_ell_invalid_col,
        0u, 1u, 4u,
        1u, 4u, ::cellshard::sparse::sliced_ell_invalid_col
    };
    const float sliced_values_fp32[12] = {
        5.0f, 3.0f, 7.0f,
        9.0f, 0.0f, 0.0f,
        1.0f, 4.0f, 2.0f,
        8.0f, 10.0f, 0.0f
    };
    __half sliced_values_host[12];
    for (unsigned int i = 0u; i < 12u; ++i) sliced_values_host[i] = __float2half(sliced_values_fp32[i]);
    unsigned int *slice_offsets = nullptr, *slice_widths = nullptr, *slice_slot_offsets = nullptr, *sliced_idx = nullptr;
    __half *sliced_values = nullptr;
    if (!check_cuda(cudaMalloc((void **) &slice_offsets, sizeof(slice_offsets_host)), "cudaMalloc slice offsets")) return 27;
    if (!check_cuda(cudaMalloc((void **) &slice_widths, sizeof(slice_widths_host)), "cudaMalloc slice widths")) return 28;
    if (!check_cuda(cudaMalloc((void **) &slice_slot_offsets, sizeof(slice_slot_offsets_host)), "cudaMalloc slice slot offsets")) return 29;
    if (!check_cuda(cudaMalloc((void **) &sliced_idx, sizeof(sliced_idx_host)), "cudaMalloc sliced idx")) return 30;
    if (!check_cuda(cudaMalloc((void **) &sliced_values, sizeof(sliced_values_host)), "cudaMalloc sliced values")) return 31;
    if (!check_cuda(cudaMemcpy(slice_offsets, slice_offsets_host, sizeof(slice_offsets_host), cudaMemcpyHostToDevice), "copy slice offsets")) return 32;
    if (!check_cuda(cudaMemcpy(slice_widths, slice_widths_host, sizeof(slice_widths_host), cudaMemcpyHostToDevice), "copy slice widths")) return 33;
    if (!check_cuda(cudaMemcpy(slice_slot_offsets, slice_slot_offsets_host, sizeof(slice_slot_offsets_host), cudaMemcpyHostToDevice), "copy slice slot offsets")) return 34;
    if (!check_cuda(cudaMemcpy(sliced_idx, sliced_idx_host, sizeof(sliced_idx_host), cudaMemcpyHostToDevice), "copy sliced idx")) return 35;
    if (!check_cuda(cudaMemcpy(sliced_values, sliced_values_host, sizeof(sliced_values_host), cudaMemcpyHostToDevice), "copy sliced values")) return 36;
    csd::sliced_ell_view sliced{rows, cols, 9u, 1u, rows, slice_offsets, slice_widths, slice_slot_offsets, sliced_idx, sliced_values};
    if (!csr::compute_sparse_group_reduce(&sliced, &ws, &groups, &masks, &filter, &result)) return 37;
    if (!copy_reduce(&ws, result, totals, detected, keep, group_counts, group_pct)) return 38;
    if (!close_enough(totals[0], 8.0f) || !close_enough(totals[3], 18.0f)) return 39;

    csr::masked_sparse_reoptimize_workspace reopt_ws;
    csr::init(&reopt_ws);
    if (!csr::setup(&reopt_ws, 0)) return 40;
    unsigned int row_remap_host[4] = {0u, 0xffffffffu, 1u, 2u};
    unsigned int feature_remap_host[5] = {0u, 1u, 0xffffffffu, 2u, 3u};
    unsigned int *row_remap = nullptr;
    unsigned int *feature_remap = nullptr;
    if (!check_cuda(cudaMalloc((void **) &row_remap, sizeof(row_remap_host)), "cudaMalloc row remap")) return 41;
    if (!check_cuda(cudaMalloc((void **) &feature_remap, sizeof(feature_remap_host)), "cudaMalloc feature remap")) return 42;
    if (!check_cuda(cudaMemcpy(row_remap, row_remap_host, sizeof(row_remap_host), cudaMemcpyHostToDevice), "copy row remap")) return 43;
    if (!check_cuda(cudaMemcpy(feature_remap, feature_remap_host, sizeof(feature_remap_host), cudaMemcpyHostToDevice), "copy feature remap")) return 44;
    csr::row_feature_mask_view reopt_masks{row_keep, feature_keep, row_remap, feature_remap};
    csr::masked_sparse_reoptimize_config reopt_cfg{3u, 4u, 0u, 2u, csr::sparse_mask_layout_compressed};
    csr::masked_sparse_reoptimize_result reopt_result{};
    if (!csr::manual_reoptimize_masked_sparse(&blocked, &reopt_masks, &reopt_cfg, &reopt_ws, &reopt_result)) return 45;
    if (!check_cuda(cudaStreamSynchronize(reopt_ws.stream), "sync reopt")) return 46;
    if (!require(reopt_result.metadata.kept_rows == 3u, "bad kept rows")) return 47;
    if (!require(reopt_result.metadata.kept_features == 4u, "bad kept features")) return 48;
    if (!require(reopt_result.metadata.live_nnz == 7u, "bad live nnz")) return 49;
    if (!require(reopt_result.metadata.output_layout == csr::sparse_mask_layout_compressed, "bad output layout")) return 50;

    csr::clear(&reopt_ws);
    csr::clear(&ws);
    cudaFree(feature_remap);
    cudaFree(row_remap);
    cudaFree(sliced_values);
    cudaFree(sliced_idx);
    cudaFree(slice_slot_offsets);
    cudaFree(slice_widths);
    cudaFree(slice_offsets);
    cudaFree(compressed_values);
    cudaFree(compressed_idx);
    cudaFree(compressed_ptr);
    cudaFree(blocked_values);
    cudaFree(block_cols);
    cudaFree(feature_keep);
    cudaFree(row_keep);
    return 0;
}
