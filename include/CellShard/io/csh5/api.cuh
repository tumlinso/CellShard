#pragma once

#include "../../formats/compressed.cuh"
#include "../../formats/blocked_ell.cuh"
#include "../../formats/quantized_blocked_ell.cuh"
#include "../../formats/sliced_ell.cuh"
#include "../../core/real.cuh"
#include "../../runtime/layout/shard_paths.cuh"
#include "../../runtime/layout/sharded.cuh"

#include <cstdint>
#include <cstdlib>

namespace cellshard {

enum {
    dataset_h5_schema_version = 3u
};

// Codec families describe how one stored partition payload should be
// interpreted after the lightweight dataset metadata has already been loaded.
// Values 1 and 2 are reserved for unsupported legacy compressed `.csh5`
// datasets; new forward `.csh5` output is Blocked-ELL-first.
enum {
    dataset_codec_family_none = 0u,
    dataset_codec_family_standard_csr = 1u,
    dataset_codec_family_quantized_csr = 2u,
    dataset_codec_family_blocked_ell = 3u,
    dataset_codec_family_sliced_ell = 4u,
    dataset_codec_family_quantized_blocked_ell = 5u
};

enum {
    dataset_execution_format_unknown = 0u,
    // Reserved legacy value for unsupported compressed `.csh5` datasets.
    dataset_execution_format_compressed = 1u,
    dataset_execution_format_blocked_ell = 2u,
    dataset_execution_format_mixed = 3u,
    dataset_execution_format_bucketed_blocked_ell = 4u,
    dataset_execution_format_sliced_ell = 5u,
    dataset_execution_format_bucketed_sliced_ell = 6u,
    dataset_execution_format_quantized_blocked_ell = 7u
};

enum {
    dataset_quantized_decode_policy_unknown = 0u,
    dataset_quantized_decode_policy_per_gene_affine = 1u,
    dataset_quantized_decode_policy_column_scale_row_offset = 2u
};

enum {
    dataset_codec_flag_direct_device_delivery = 1u << 0,
    dataset_codec_flag_live_fused_decode = 1u << 1,
    dataset_codec_quantized_decode_policy_shift = 8u,
    dataset_codec_quantized_decode_policy_mask = 0xffu << dataset_codec_quantized_decode_policy_shift
};

enum {
    dataset_runtime_service_mode_unknown = 0u,
    dataset_runtime_service_mode_local_cache = 1u,
    dataset_runtime_service_mode_owner_hosted = 2u,
    dataset_runtime_service_mode_executor_client = 3u
};

enum {
    dataset_live_write_mode_unknown = 0u,
    dataset_live_write_mode_read_only = 1u,
    dataset_live_write_mode_append_only = 2u,
    dataset_live_write_mode_maintenance = 3u
};

struct dataset_codec_descriptor {
    std::uint32_t codec_id;
    std::uint32_t family;
    std::uint32_t value_code;
    std::uint32_t scale_value_code;
    std::uint32_t bits;
    std::uint32_t flags;
};

__host__ __device__ __forceinline__ std::uint32_t dataset_codec_quantized_decode_policy(std::uint32_t flags) {
    return (flags & dataset_codec_quantized_decode_policy_mask) >> dataset_codec_quantized_decode_policy_shift;
}

__host__ __device__ __forceinline__ std::uint32_t set_dataset_codec_quantized_decode_policy(
    std::uint32_t flags,
    std::uint32_t policy) {
    return (flags & ~dataset_codec_quantized_decode_policy_mask)
        | ((policy << dataset_codec_quantized_decode_policy_shift) & dataset_codec_quantized_decode_policy_mask);
}

struct dataset_text_column_view {
    std::uint32_t count;
    std::uint32_t bytes;
    const std::uint32_t *offsets;
    const char *data;
};

struct dataset_dataset_table_view {
    std::uint32_t count;
    dataset_text_column_view dataset_ids;
    dataset_text_column_view matrix_paths;
    dataset_text_column_view feature_paths;
    dataset_text_column_view barcode_paths;
    dataset_text_column_view metadata_paths;
    const std::uint32_t *formats;
    const std::uint64_t *row_begin;
    const std::uint64_t *row_end;
    const std::uint64_t *rows;
    const std::uint64_t *cols;
    const std::uint64_t *nnz;
};

// Provenance tables are metadata-only views used at file-build time. They can
// be large on the host, but they are not part of the steady-state fetch path.
struct dataset_provenance_view {
    dataset_text_column_view global_barcodes;
    const std::uint32_t *cell_dataset_ids;
    const std::uint64_t *cell_local_indices;

    dataset_text_column_view feature_ids;
    dataset_text_column_view feature_names;
    dataset_text_column_view feature_types;
    const std::uint32_t *feature_dataset_ids;
    const std::uint64_t *feature_local_indices;

    const std::uint64_t *dataset_feature_offsets;
    const std::uint32_t *dataset_feature_to_global;
};

struct dataset_metadata_table_view {
    std::uint32_t rows;
    std::uint32_t cols;
    dataset_text_column_view column_names;
    dataset_text_column_view field_values;
    const std::uint32_t *row_offsets;
};

struct dataset_embedded_metadata_view {
    std::uint32_t count;
    const std::uint32_t *dataset_indices;
    const std::uint64_t *global_row_begin;
    const std::uint64_t *global_row_end;
    const dataset_metadata_table_view *tables;
};

enum {
    dataset_observation_metadata_type_none = 0u,
    dataset_observation_metadata_type_text = 1u,
    dataset_observation_metadata_type_float32 = 2u,
    dataset_observation_metadata_type_uint8 = 3u
};

struct dataset_observation_metadata_column_view {
    const char *name;
    std::uint32_t type;
    dataset_text_column_view text_values;
    const float *float32_values;
    const std::uint8_t *uint8_values;
};

struct dataset_annotation_view {
    std::uint64_t extent;
    std::uint32_t cols;
    const dataset_observation_metadata_column_view *columns;
};

struct dataset_observation_metadata_view {
    std::uint64_t rows;
    std::uint32_t cols;
    const dataset_observation_metadata_column_view *columns;
};

struct dataset_feature_metadata_view {
    std::uint64_t cols;
    std::uint32_t annotation_count;
    const dataset_observation_metadata_column_view *annotations;
};

struct dataset_user_attribute_view {
    std::uint32_t count;
    dataset_text_column_view keys;
    dataset_text_column_view values;
};

struct dataset_browse_cache_view {
    std::uint32_t selected_feature_count;
    const std::uint32_t *selected_feature_indices;
    const float *gene_sum;
    const float *gene_detected;
    const float *gene_sq_sum;

    std::uint32_t dataset_count;
    const float *dataset_feature_mean;

    std::uint32_t shard_count;
    const float *shard_feature_mean;

    std::uint32_t partition_count;
    std::uint32_t sample_rows_per_partition;
    const std::uint32_t *partition_sample_row_offsets;
    const std::uint64_t *partition_sample_global_rows;
    const float *partition_sample_values;
};

struct dataset_preprocess_view {
    const char *assay;
    const char *matrix_orientation;
    const char *matrix_state;
    const char *pipeline_scope;
    const char *raw_matrix_name;
    const char *active_matrix_name;
    const char *feature_namespace;
    const char *mito_prefix;

    std::uint32_t raw_counts_available;
    std::uint32_t processed_matrix_available;
    std::uint32_t normalized_log1p_metrics;
    std::uint32_t hvg_available;
    std::uint32_t mark_mito_from_feature_names;

    std::uint64_t rows;
    std::uint32_t cols;
    std::uint64_t nnz;
    std::uint32_t partitions_processed;
    std::uint32_t mito_feature_count;

    float target_sum;
    float min_counts;
    std::uint32_t min_genes;
    float max_mito_fraction;
    float min_gene_sum;
    float min_detected_cells;
    float min_variance;
    double kept_cells;
    std::uint32_t kept_genes;
    double gene_sum_checksum;

    const float *cell_total_counts;
    const float *cell_mito_counts;
    const float *cell_max_counts;
    const std::uint32_t *cell_detected_genes;
    const std::uint8_t *cell_keep;

    const float *gene_sum;
    const float *gene_sq_sum;
    const float *gene_detected_cells;
    const std::uint8_t *gene_keep;
    const std::uint8_t *gene_flags;
};

struct dataset_layout_view {
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t nnz;
    std::uint64_t num_partitions;
    std::uint64_t num_shards;

    const std::uint64_t *partition_rows;
    const std::uint64_t *partition_nnz;
    const std::uint32_t *partition_axes;
    const std::uint64_t *partition_aux;
    const std::uint64_t *partition_row_offsets;
    const std::uint32_t *partition_dataset_ids;
    const std::uint32_t *partition_codec_ids;
    const std::uint64_t *shard_offsets;

    const dataset_codec_descriptor *codecs;
    std::uint32_t num_codecs;
};

struct dataset_execution_view {
    std::uint32_t partition_count;
    const std::uint32_t *partition_execution_formats;
    const std::uint32_t *partition_blocked_ell_block_sizes;
    const std::uint32_t *partition_blocked_ell_bucket_counts;
    const float *partition_blocked_ell_fill_ratios;
    const std::uint64_t *partition_execution_bytes;
    const std::uint64_t *partition_blocked_ell_bytes;
    const std::uint64_t *partition_bucketed_blocked_ell_bytes;
    const std::uint32_t *partition_sliced_ell_slice_counts;
    const std::uint32_t *partition_sliced_ell_slice_rows;
    const std::uint64_t *partition_sliced_ell_bytes;
    const std::uint64_t *partition_bucketed_sliced_ell_bytes;

    std::uint32_t shard_count;
    const std::uint32_t *shard_execution_formats;
    const std::uint32_t *shard_blocked_ell_block_sizes;
    const std::uint32_t *shard_bucketed_partition_counts;
    const std::uint32_t *shard_bucketed_segment_counts;
    const float *shard_blocked_ell_fill_ratios;
    const std::uint64_t *shard_execution_bytes;
    const std::uint64_t *shard_bucketed_blocked_ell_bytes;
    const std::uint32_t *shard_sliced_ell_slice_counts;
    const std::uint32_t *shard_sliced_ell_slice_rows;
    const std::uint64_t *shard_bucketed_sliced_ell_bytes;
    const std::uint32_t *shard_preferred_pair_ids;
    const std::uint32_t *shard_owner_node_ids;
    const std::uint32_t *shard_owner_rank_ids;

    std::uint32_t preferred_base_format;
};

struct dataset_runtime_service_view {
    std::uint32_t service_mode;
    std::uint32_t live_write_mode;
    std::uint32_t prefer_pack_delivery;
    std::uint32_t remote_pack_delivery;
    std::uint32_t single_reader_coordinator;
    std::uint32_t maintenance_lock_blocks_overwrite;
    std::uint64_t canonical_generation;
    std::uint64_t execution_plan_generation;
    std::uint64_t pack_generation;
    std::uint64_t service_epoch;
    std::uint64_t active_read_generation;
    std::uint64_t staged_write_generation;
};

inline void init(dataset_runtime_service_view *view) {
    if (view == nullptr) return;
    view->service_mode = dataset_runtime_service_mode_unknown;
    view->live_write_mode = dataset_live_write_mode_unknown;
    view->prefer_pack_delivery = 0u;
    view->remote_pack_delivery = 0u;
    view->single_reader_coordinator = 0u;
    view->maintenance_lock_blocks_overwrite = 0u;
    view->canonical_generation = 0u;
    view->execution_plan_generation = 0u;
    view->pack_generation = 0u;
    view->service_epoch = 0u;
    view->active_read_generation = 0u;
    view->staged_write_generation = 0u;
}

struct bucketed_blocked_ell_partition {
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t nnz;
    std::uint32_t segment_count;
    sparse::blocked_ell *segments;
    std::uint32_t *segment_row_offsets;
    std::uint32_t *exec_to_canonical_rows;
    std::uint32_t *canonical_to_exec_rows;
    std::uint32_t *exec_to_canonical_cols;
    std::uint32_t *canonical_to_exec_cols;
};

inline void init(bucketed_blocked_ell_partition *part) {
    if (part == nullptr) return;
    part->rows = 0u;
    part->cols = 0u;
    part->nnz = 0u;
    part->segment_count = 0u;
    part->segments = nullptr;
    part->segment_row_offsets = nullptr;
    part->exec_to_canonical_rows = nullptr;
    part->canonical_to_exec_rows = nullptr;
    part->exec_to_canonical_cols = nullptr;
    part->canonical_to_exec_cols = nullptr;
}

inline void clear(bucketed_blocked_ell_partition *part) {
    std::uint32_t i = 0u;
    if (part == nullptr) return;
    if (part->segments != nullptr) {
        for (i = 0u; i < part->segment_count; ++i) sparse::clear(part->segments + i);
    }
    std::free(part->segments);
    std::free(part->segment_row_offsets);
    std::free(part->exec_to_canonical_rows);
    std::free(part->canonical_to_exec_rows);
    std::free(part->exec_to_canonical_cols);
    std::free(part->canonical_to_exec_cols);
    init(part);
}

struct bucketed_blocked_ell_shard {
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t nnz;
    std::uint32_t partition_count;
    bucketed_blocked_ell_partition *partitions;
    std::uint32_t *partition_row_offsets;
    std::uint32_t *exec_to_canonical_cols;
    std::uint32_t *canonical_to_exec_cols;
};

inline void init(bucketed_blocked_ell_shard *shard) {
    if (shard == nullptr) return;
    shard->rows = 0u;
    shard->cols = 0u;
    shard->nnz = 0u;
    shard->partition_count = 0u;
    shard->partitions = nullptr;
    shard->partition_row_offsets = nullptr;
    shard->exec_to_canonical_cols = nullptr;
    shard->canonical_to_exec_cols = nullptr;
}

inline void clear(bucketed_blocked_ell_shard *shard) {
    std::uint32_t i = 0u;
    if (shard == nullptr) return;
    if (shard->partitions != nullptr) {
        for (i = 0u; i < shard->partition_count; ++i) clear(shard->partitions + i);
    }
    std::free(shard->partitions);
    std::free(shard->partition_row_offsets);
    std::free(shard->exec_to_canonical_cols);
    std::free(shard->canonical_to_exec_cols);
    init(shard);
}

struct bucketed_sliced_ell_partition {
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t nnz;
    std::uint32_t segment_count;
    sparse::sliced_ell *segments;
    std::uint32_t *segment_row_offsets;
    std::uint32_t *exec_to_canonical_rows;
    std::uint32_t *canonical_to_exec_rows;
};

inline void init(bucketed_sliced_ell_partition *part) {
    if (part == nullptr) return;
    part->rows = 0u;
    part->cols = 0u;
    part->nnz = 0u;
    part->segment_count = 0u;
    part->segments = nullptr;
    part->segment_row_offsets = nullptr;
    part->exec_to_canonical_rows = nullptr;
    part->canonical_to_exec_rows = nullptr;
}

inline void clear(bucketed_sliced_ell_partition *part) {
    std::uint32_t i = 0u;
    if (part == nullptr) return;
    if (part->segments != nullptr) {
        for (i = 0u; i < part->segment_count; ++i) sparse::clear(part->segments + i);
    }
    std::free(part->segments);
    std::free(part->segment_row_offsets);
    std::free(part->exec_to_canonical_rows);
    std::free(part->canonical_to_exec_rows);
    init(part);
}

struct bucketed_sliced_ell_shard {
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t nnz;
    std::uint32_t partition_count;
    bucketed_sliced_ell_partition *partitions;
    std::uint32_t *partition_row_offsets;
};

inline void init(bucketed_sliced_ell_shard *shard) {
    if (shard == nullptr) return;
    shard->rows = 0u;
    shard->cols = 0u;
    shard->nnz = 0u;
    shard->partition_count = 0u;
    shard->partitions = nullptr;
    shard->partition_row_offsets = nullptr;
}

inline void clear(bucketed_sliced_ell_shard *shard) {
    std::uint32_t i = 0u;
    if (shard == nullptr) return;
    if (shard->partitions != nullptr) {
        for (i = 0u; i < shard->partition_count; ++i) clear(shard->partitions + i);
    }
    std::free(shard->partitions);
    std::free(shard->partition_row_offsets);
    init(shard);
}

#if CELLSHARD_ENABLE_CUDA
namespace device {
template<typename MatrixT>
struct partition_record;
}

struct dataset_sliced_execution_device_partition_view {
    unsigned long partition_id;
    std::uint32_t segment_count;
    const bucketed_sliced_ell_partition *host_partition;
    const device::partition_record<sparse::sliced_ell> *device_segments;
    std::uint64_t resident_bytes;
    std::uint64_t execution_plan_generation;
    std::uint64_t pack_generation;
    std::uint64_t service_epoch;
};

inline void init(dataset_sliced_execution_device_partition_view *view) {
    if (view == nullptr) return;
    view->partition_id = 0u;
    view->segment_count = 0u;
    view->host_partition = nullptr;
    view->device_segments = nullptr;
    view->resident_bytes = 0u;
    view->execution_plan_generation = 0u;
    view->pack_generation = 0u;
    view->service_epoch = 0u;
}
#endif

// Create/append helpers are whole-file synchronous HDF5 operations.
int create_dataset_blocked_ell_h5(const char *filename,
                                 const dataset_layout_view *layout,
                                 const dataset_dataset_table_view *datasets,
                                 const dataset_provenance_view *provenance);
int create_dataset_quantized_blocked_ell_h5(const char *filename,
                                            const dataset_layout_view *layout,
                                            const dataset_dataset_table_view *datasets,
                                            const dataset_provenance_view *provenance);
int create_dataset_sliced_ell_h5(const char *filename,
                                 const dataset_layout_view *layout,
                                 const dataset_dataset_table_view *datasets,
                                 const dataset_provenance_view *provenance);

int append_dataset_embedded_metadata_h5(const char *filename,
                                       const dataset_embedded_metadata_view *metadata);

int append_dataset_observation_annotations_h5(const char *filename,
                                              const dataset_annotation_view *metadata);
int append_dataset_observation_metadata_h5(const char *filename,
                                          const dataset_observation_metadata_view *metadata);
int append_dataset_feature_metadata_h5(const char *filename,
                                       const dataset_feature_metadata_view *metadata);
int append_dataset_user_attributes_h5(const char *filename,
                                      const dataset_user_attribute_view *attributes);

int append_dataset_browse_cache_h5(const char *filename,
                                  const dataset_browse_cache_view *browse);

int append_dataset_preprocess_h5(const char *filename,
                                const dataset_preprocess_view *preprocess);
int finalize_preprocessed_blocked_ell_dataset_h5(const char *filename,
                                                 const std::uint8_t *cell_keep,
                                                 const std::uint8_t *gene_keep,
                                                 const dataset_embedded_metadata_view *embedded_metadata,
                                                 const dataset_annotation_view *observation_metadata,
                                                 const dataset_feature_metadata_view *feature_metadata,
                                                 const dataset_user_attribute_view *attributes,
                                                 const dataset_preprocess_view *preprocess,
                                                 const char *working_root,
                                                 std::uint64_t *rows_out,
                                                 std::uint64_t *cols_out,
                                                 std::uint64_t *nnz_out);

int create_dataset_optimized_blocked_ell_h5(const char *filename,
                                            const dataset_layout_view *layout,
                                            const dataset_dataset_table_view *datasets,
                                            const dataset_provenance_view *provenance);
int append_dataset_execution_h5(const char *filename,
                               const dataset_execution_view *execution);
int append_dataset_runtime_service_h5(const char *filename,
                                     const dataset_runtime_service_view *runtime_service);
int bind_dataset_h5_builder(shard_storage *s, const char *path);
int bind_dataset_h5_owner_runtime(shard_storage *s, const char *path);
int append_blocked_ell_partition_h5(const char *filename,
                               unsigned long partition_id,
                               const sparse::blocked_ell *part);
int append_quantized_blocked_ell_partition_h5(const char *filename,
                                              unsigned long partition_id,
                                              const sparse::quantized_blocked_ell *part);
int append_sliced_ell_partition_h5(const char *filename,
                                   unsigned long partition_id,
                                   const sparse::sliced_ell *part);
int append_bucketed_blocked_ell_shard_h5(const char *filename,
                                         unsigned long shard_id,
                                         const bucketed_blocked_ell_shard *shard);
int append_bucketed_sliced_ell_shard_h5(const char *filename,
                                        unsigned long shard_id,
                                        const bucketed_sliced_ell_shard *shard);

// Header load binds a lazy shard_storage backend; fetch/prefetch calls are the
// points that actually materialize partition payloads or populate local caches.
int bind_dataset_h5(shard_storage *s, const char *path);
int adopt_dataset_h5_executor_role(shard_storage *s);
int bind_dataset_h5_cache(shard_storage *s, const char *cache_root);
int get_dataset_h5_execution_metadata(const shard_storage *s,
                                     dataset_execution_view *execution);
int get_dataset_h5_runtime_service(const shard_storage *s,
                                  dataset_runtime_service_view *runtime_service);
int set_dataset_h5_cache_budget_bytes(shard_storage *s, std::uint64_t bytes);
int set_dataset_h5_cache_predictor_enabled(shard_storage *s, int enabled);
int pin_dataset_h5_cache_shard(shard_storage *s, unsigned long shard_id);
int unpin_dataset_h5_cache_shard(shard_storage *s, unsigned long shard_id);
int evict_dataset_h5_cache_shard(shard_storage *s, unsigned long shard_id);
int invalidate_dataset_h5_cache(shard_storage *s);
int load_dataset_blocked_ell_h5_header(const char *filename,
                                      sharded<sparse::blocked_ell> *m,
                                      shard_storage *s);
int load_dataset_quantized_blocked_ell_h5_header(const char *filename,
                                                 sharded<sparse::quantized_blocked_ell> *m,
                                                 shard_storage *s);
int load_dataset_sliced_ell_h5_header(const char *filename,
                                      sharded<sparse::sliced_ell> *m,
                                      shard_storage *s);
int prefetch_dataset_blocked_ell_h5_partition_cache(const sharded<sparse::blocked_ell> *m,
                                              shard_storage *s,
                                              unsigned long partition_id);
int prefetch_dataset_blocked_ell_h5_shard_cache(const sharded<sparse::blocked_ell> *m,
                                               shard_storage *s,
                                               unsigned long shard_id);
int prefetch_dataset_quantized_blocked_ell_h5_partition_cache(const sharded<sparse::quantized_blocked_ell> *m,
                                                              shard_storage *s,
                                                              unsigned long partition_id);
int prefetch_dataset_quantized_blocked_ell_h5_shard_cache(const sharded<sparse::quantized_blocked_ell> *m,
                                                          shard_storage *s,
                                                          unsigned long shard_id);
int prefetch_dataset_sliced_ell_h5_partition_cache(const sharded<sparse::sliced_ell> *m,
                                                   shard_storage *s,
                                                   unsigned long partition_id);
int prefetch_dataset_sliced_ell_h5_shard_cache(const sharded<sparse::sliced_ell> *m,
                                               shard_storage *s,
                                               unsigned long shard_id);
int warm_dataset_blocked_ell_h5_cache_range(const char *filename,
                                           const char *cache_root,
                                           unsigned long shard_begin,
                                           unsigned long shard_end);
int warm_dataset_blocked_ell_h5_cache(const char *filename,
                                     const char *cache_root);
int warm_dataset_quantized_blocked_ell_h5_cache_range(const char *filename,
                                                      const char *cache_root,
                                                      unsigned long shard_begin,
                                                      unsigned long shard_end);
int warm_dataset_quantized_blocked_ell_h5_cache(const char *filename,
                                                const char *cache_root);
int warm_dataset_sliced_ell_h5_cache_range(const char *filename,
                                           const char *cache_root,
                                           unsigned long shard_begin,
                                           unsigned long shard_end);
int warm_dataset_sliced_ell_h5_cache(const char *filename,
                                     const char *cache_root);
int warm_dataset_blocked_ell_h5_execution_cache_range(const char *filename,
                                                     const char *cache_root,
                                                     unsigned long shard_begin,
                                                     unsigned long shard_end);
int warm_dataset_blocked_ell_h5_execution_cache(const char *filename,
                                               const char *cache_root);
int fetch_dataset_blocked_ell_h5_execution_partition(bucketed_blocked_ell_partition *out,
                                                    const sharded<sparse::blocked_ell> *m,
                                                    const shard_storage *s,
                                                    unsigned long partition_id);
int fetch_dataset_sliced_ell_h5_execution_partition(bucketed_sliced_ell_partition *out,
                                                    const sharded<sparse::sliced_ell> *m,
                                                    const shard_storage *s,
                                                    unsigned long partition_id);
#if CELLSHARD_ENABLE_CUDA
int acquire_dataset_sliced_ell_h5_execution_partition_device(dataset_sliced_execution_device_partition_view *out,
                                                             const sharded<sparse::sliced_ell> *m,
                                                             const shard_storage *s,
                                                             unsigned long partition_id,
                                                             int device_id,
                                                             std::uint64_t cache_budget_bytes);
int release_dataset_sliced_ell_h5_execution_partition_device(dataset_sliced_execution_device_partition_view *view);
int clear_dataset_sliced_ell_h5_device_cache(const char *source_path,
                                             int device_id);
int clear_all_dataset_sliced_ell_h5_device_caches();
#endif
int fetch_dataset_blocked_ell_h5_partition(sharded<sparse::blocked_ell> *m,
                                     const shard_storage *s,
                                     unsigned long partition_id);
int fetch_dataset_blocked_ell_h5_shard(sharded<sparse::blocked_ell> *m,
                                      const shard_storage *s,
                                      unsigned long shard_id);
int fetch_dataset_quantized_blocked_ell_h5_partition(sharded<sparse::quantized_blocked_ell> *m,
                                                     const shard_storage *s,
                                                     unsigned long partition_id);
int fetch_dataset_quantized_blocked_ell_h5_shard(sharded<sparse::quantized_blocked_ell> *m,
                                                 const shard_storage *s,
                                                 unsigned long shard_id);
int fetch_dataset_sliced_ell_h5_partition(sharded<sparse::sliced_ell> *m,
                                          const shard_storage *s,
                                          unsigned long partition_id);
int fetch_dataset_sliced_ell_h5_shard(sharded<sparse::sliced_ell> *m,
                                      const shard_storage *s,
                                      unsigned long shard_id);
int build_bucketed_blocked_ell_partition(bucketed_blocked_ell_partition *out,
                                         const sparse::blocked_ell *part,
                                         std::uint32_t requested_bucket_count,
                                         std::uint64_t *bucketed_bytes_out);
int build_bucketed_sliced_ell_partition(bucketed_sliced_ell_partition *out,
                                        const sparse::sliced_ell *part,
                                        std::uint32_t requested_bucket_count,
                                        std::uint64_t *bucketed_bytes_out);

// Temporary compatibility wrappers for repo-internal callers while the new
// cache manager surface propagates through the tree.
inline int bind_dataset_h5_partition_cache(shard_storage *s, const char *cache_dir) {
    return bind_dataset_h5_cache(s, cache_dir);
}

inline int prefetch_dataset_blocked_ell_h5_partition_to_cache(const sharded<sparse::blocked_ell> *m,
                                                        const shard_storage *s,
                                                        unsigned long partition_id) {
    return prefetch_dataset_blocked_ell_h5_partition_cache(m, const_cast<shard_storage *>(s), partition_id);
}

inline int prefetch_dataset_blocked_ell_h5_shard_to_cache(const sharded<sparse::blocked_ell> *m,
                                                         const shard_storage *s,
                                                         unsigned long shard_id) {
    return prefetch_dataset_blocked_ell_h5_shard_cache(m, const_cast<shard_storage *>(s), shard_id);
}

inline int prefetch_dataset_quantized_blocked_ell_h5_partition_to_cache(const sharded<sparse::quantized_blocked_ell> *m,
                                                                        const shard_storage *s,
                                                                        unsigned long partition_id) {
    return prefetch_dataset_quantized_blocked_ell_h5_partition_cache(m, const_cast<shard_storage *>(s), partition_id);
}

inline int prefetch_dataset_quantized_blocked_ell_h5_shard_to_cache(const sharded<sparse::quantized_blocked_ell> *m,
                                                                    const shard_storage *s,
                                                                    unsigned long shard_id) {
    return prefetch_dataset_quantized_blocked_ell_h5_shard_cache(m, const_cast<shard_storage *>(s), shard_id);
}

} // namespace cellshard
