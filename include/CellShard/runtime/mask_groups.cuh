#pragma once

#include "device.hh"
#include "distributed.hh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace cellshard {
namespace runtime {

static constexpr unsigned int CELLSHARD_MAX_MASK_GROUPS = 32u;

enum sparse_mask_layout : std::uint32_t {
    sparse_mask_layout_unknown = 0u,
    sparse_mask_layout_blocked_ell = 1u,
    sparse_mask_layout_sliced_ell = 2u,
    sparse_mask_layout_compressed = 3u
};

struct alignas(16) group_mask_config_view {
    unsigned int group_count;
    const char * const *group_names;
    const std::uint32_t *feature_group_masks;
};

struct alignas(16) row_feature_mask_view {
    const unsigned char *row_keep;
    const unsigned char *feature_keep;
    const unsigned int *row_remap;
    const unsigned int *feature_remap;
};

struct alignas(16) sparse_group_filter_params {
    float min_total;
    unsigned int min_detected_features;
    float max_group_fraction;
    unsigned int fraction_group_index;
};

struct alignas(16) sparse_group_reduce_result {
    unsigned int rows;
    unsigned int group_count;
    float *row_totals;
    unsigned int *detected_features;
    float *max_values;
    unsigned char *row_keep;
    float *group_counts;
    float *group_percentages;
};

struct alignas(16) sparse_group_reduce_workspace {
    int device;
    cudaStream_t stream;
    int owns_stream;

    unsigned int rows_capacity;
    unsigned int cols_capacity;
    unsigned int group_capacity;
    unsigned int values_capacity;

    void *row_block;
    void *feature_block;
    void *scratch_block;
    std::size_t scratch_bytes;

    float *row_totals;
    unsigned int *detected_features;
    float *max_values;
    unsigned char *row_keep;
    float *group_counts;
    float *group_percentages;
    std::uint32_t *feature_group_masks;
};

struct alignas(16) sparse_group_reduce_fleet_result {
    unsigned int slot_count;
    unsigned int leader_index;
    sparse_group_reduce_result *slot_results;
};

struct alignas(16) sparse_group_reduce_fleet_workspace {
    distributed::local_context local;
    unsigned int slot_count;
    unsigned int *slots;
    sparse_group_reduce_workspace *devices;
    sparse_group_reduce_result *results;
    void **reduce_scratch;
    std::size_t *reduce_scratch_bytes;
#if CELLSHARD_HAS_NCCL
    distributed::nccl_communicator ranked_nccl;
#endif
};

struct alignas(16) ranked_nccl_config_view {
    int world_size;
    const int *local_world_ranks;
    const void *unique_id;
    std::size_t unique_id_bytes;
};

struct alignas(16) sparse_group_reduce_fleet_config {
    const int *device_ids;
    unsigned int device_count;
    unsigned int enable_peer_access;
    unsigned int stream_flags;
    const ranked_nccl_config_view *ranked_nccl;
};

struct alignas(16) masked_sparse_reoptimize_config {
    unsigned int output_rows;
    unsigned int output_cols;
    unsigned int requested_bucket_count;
    unsigned int output_block_size;
    sparse_mask_layout requested_layout;
};

struct alignas(16) masked_sparse_reoptimize_metadata {
    unsigned int kept_rows;
    unsigned int kept_features;
    unsigned int live_nnz;
    unsigned long long bytes_before;
    unsigned long long bytes_after;
    sparse_mask_layout output_layout;
};

struct alignas(16) masked_sparse_reoptimize_result {
    device::compressed_view compressed;
    device::blocked_ell_view blocked_ell;
    masked_sparse_reoptimize_metadata metadata;
};

struct alignas(16) masked_sparse_reoptimize_workspace {
    int device;
    cudaStream_t stream;
    int owns_stream;
    void *private_workspace;
};

struct alignas(16) auto_reoptimize_prediction_input {
    float density_change;
    float row_drop_ratio;
    float feature_drop_ratio;
    float layout_padding_change;
    float estimated_rebuild_cost_ms;
    unsigned int expected_reuse_count;
};

struct alignas(16) auto_reoptimize_prediction {
    unsigned int should_reoptimize;
    float estimated_net_gain_ms;
};

// TODO(auto-reoptimize): use density change, row/feature drop ratios, layout
// padding change, rebuild cost, and expected repeated-use count to predict
// when a masked layout rebuild should be triggered automatically.
int predict_masked_sparse_reoptimization(const auto_reoptimize_prediction_input *input,
                                         auto_reoptimize_prediction *out);

void init(sparse_group_reduce_workspace *workspace);
void init(sparse_group_reduce_fleet_workspace *fleet);
void init(masked_sparse_reoptimize_workspace *workspace);

void clear(sparse_group_reduce_workspace *workspace);
void clear(sparse_group_reduce_fleet_workspace *fleet);
void clear(masked_sparse_reoptimize_workspace *workspace);

int setup(sparse_group_reduce_workspace *workspace, int device, cudaStream_t stream = (cudaStream_t) 0);
int setup_fleet(sparse_group_reduce_fleet_workspace *fleet,
                const sparse_group_reduce_fleet_config *config = nullptr);
int setup(masked_sparse_reoptimize_workspace *workspace, int device, cudaStream_t stream = (cudaStream_t) 0);

int reserve(sparse_group_reduce_workspace *workspace,
            unsigned int rows,
            unsigned int cols,
            unsigned int values,
            unsigned int group_count);

int upload_feature_group_masks(sparse_group_reduce_workspace *workspace,
                               unsigned int cols,
                               const std::uint32_t *host_masks);

int compute_sparse_group_reduce(const device::blocked_ell_view *src,
                                sparse_group_reduce_workspace *workspace,
                                const group_mask_config_view *groups,
                                const row_feature_mask_view *masks,
                                const sparse_group_filter_params *filter,
                                sparse_group_reduce_result *out);

int compute_sparse_group_reduce(const device::sliced_ell_view *src,
                                sparse_group_reduce_workspace *workspace,
                                const group_mask_config_view *groups,
                                const row_feature_mask_view *masks,
                                const sparse_group_filter_params *filter,
                                sparse_group_reduce_result *out);

int compute_sparse_group_reduce_compressed_fallback(const device::compressed_view *src,
                                                   sparse_group_reduce_workspace *workspace,
                                                   const group_mask_config_view *groups,
                                                   const row_feature_mask_view *masks,
                                                   const sparse_group_filter_params *filter,
                                                   sparse_group_reduce_result *out);

int compute_sparse_group_reduce_fleet(const device::blocked_ell_view *src_by_slot,
                                      sparse_group_reduce_fleet_workspace *fleet,
                                      const group_mask_config_view *groups,
                                      const row_feature_mask_view *masks,
                                      const sparse_group_filter_params *filter,
                                      sparse_group_reduce_fleet_result *out);

int compute_sparse_group_reduce_fleet(const device::sliced_ell_view *src_by_slot,
                                      sparse_group_reduce_fleet_workspace *fleet,
                                      const group_mask_config_view *groups,
                                      const row_feature_mask_view *masks,
                                      const sparse_group_filter_params *filter,
                                      sparse_group_reduce_fleet_result *out);

int manual_reoptimize_masked_sparse(const device::blocked_ell_view *src,
                                    const row_feature_mask_view *masks,
                                    const masked_sparse_reoptimize_config *config,
                                    masked_sparse_reoptimize_workspace *workspace,
                                    masked_sparse_reoptimize_result *out);

} // namespace runtime
} // namespace cellshard
