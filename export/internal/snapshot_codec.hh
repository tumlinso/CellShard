#pragma once

#include "common.hh"

#include <cstring>

namespace cellshard::exporting::detail {

struct byte_reader {
    const std::uint8_t *cur = nullptr;
    const std::uint8_t *end = nullptr;
};

inline bool checked_size_from_u64(std::uint64_t value, std::size_t *out) {
    if (out == nullptr || value > (std::uint64_t) std::numeric_limits<std::size_t>::max()) return false;
    *out = (std::size_t) value;
    return true;
}

inline void append_bytes(std::vector<std::uint8_t> *out, const void *data, std::size_t bytes) {
    const auto *begin = static_cast<const std::uint8_t *>(data);
    if (out == nullptr || bytes == 0u) return;
    out->insert(out->end(), begin, begin + bytes);
}

template<typename T>
inline void append_pod(std::vector<std::uint8_t> *out, const T &value) {
    append_bytes(out, &value, sizeof(T));
}

template<typename T>
inline void append_pod_vector(std::vector<std::uint8_t> *out, const std::vector<T> &values) {
    const std::uint64_t count = (std::uint64_t) values.size();
    append_pod(out, count);
    if (!values.empty()) append_bytes(out, values.data(), values.size() * sizeof(T));
}

inline void append_string(std::vector<std::uint8_t> *out, const std::string &value) {
    const std::uint64_t bytes = (std::uint64_t) value.size();
    append_pod(out, bytes);
    if (!value.empty()) append_bytes(out, value.data(), value.size());
}

inline void append_string_vector(std::vector<std::uint8_t> *out, const std::vector<std::string> &values) {
    const std::uint64_t count = (std::uint64_t) values.size();
    append_pod(out, count);
    for (const std::string &value : values) append_string(out, value);
}

template<typename T>
inline bool read_pod(byte_reader *reader, T *out, std::string *error, const char *label) {
    if (reader == nullptr || out == nullptr || reader->cur == nullptr || reader->end == nullptr
        || (std::size_t) (reader->end - reader->cur) < sizeof(T)) {
        set_error(error, std::string("truncated metadata snapshot while reading ") + (label != nullptr ? label : "field"));
        return false;
    }
    std::memcpy(out, reader->cur, sizeof(T));
    reader->cur += sizeof(T);
    return true;
}

inline bool read_string(byte_reader *reader, std::string *out, std::string *error, const char *label) {
    std::uint64_t size = 0u;
    std::size_t bytes = 0u;
    if (out == nullptr || !read_pod(reader, &size, error, label)) return false;
    if (!checked_size_from_u64(size, &bytes)) {
        set_error(error, std::string("metadata snapshot string is too large for ") + (label != nullptr ? label : "field"));
        return false;
    }
    if ((std::size_t) (reader->end - reader->cur) < bytes) {
        set_error(error, std::string("truncated metadata snapshot while reading ") + (label != nullptr ? label : "string"));
        return false;
    }
    out->assign(reinterpret_cast<const char *>(reader->cur), bytes);
    reader->cur += bytes;
    return true;
}

inline bool read_string_vector(byte_reader *reader,
                               std::vector<std::string> *out,
                               std::string *error,
                               const char *label) {
    std::uint64_t count = 0u;
    std::size_t size = 0u;
    if (out == nullptr || !read_pod(reader, &count, error, label)) return false;
    if (!checked_size_from_u64(count, &size)) {
        set_error(error, std::string("metadata snapshot string vector is too large for ") + (label != nullptr ? label : "field"));
        return false;
    }
    out->clear();
    out->reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        std::string value;
        if (!read_string(reader, &value, error, label)) return false;
        out->push_back(std::move(value));
    }
    return true;
}

template<typename T>
inline bool read_pod_vector(byte_reader *reader,
                            std::vector<T> *out,
                            std::string *error,
                            const char *label) {
    std::uint64_t count = 0u;
    std::size_t size = 0u;
    if (out == nullptr || !read_pod(reader, &count, error, label)) return false;
    if (!checked_size_from_u64(count, &size)) {
        set_error(error, std::string("metadata snapshot vector is too large for ") + (label != nullptr ? label : "field"));
        return false;
    }
    const std::size_t bytes = size * sizeof(T);
    if (size != 0u && bytes / sizeof(T) != size) {
        set_error(error, std::string("metadata snapshot vector size overflow for ") + (label != nullptr ? label : "field"));
        return false;
    }
    if ((std::size_t) (reader->end - reader->cur) < bytes) {
        set_error(error, std::string("truncated metadata snapshot while reading ") + (label != nullptr ? label : "vector"));
        return false;
    }
    out->assign(size, T{});
    if (bytes != 0u) {
        std::memcpy(out->data(), reader->cur, bytes);
        reader->cur += bytes;
    }
    return true;
}

inline std::uint64_t fnv1a64(const std::vector<std::uint8_t> &bytes) {
    std::uint64_t hash = 1469598103934665603ull;
    for (std::uint8_t value : bytes) {
        hash ^= (std::uint64_t) value;
        hash *= 1099511628211ull;
    }
    return hash;
}

inline void append_source_dataset_summary(std::vector<std::uint8_t> *out, const source_dataset_summary &value) {
    append_string(out, value.dataset_id);
    append_string(out, value.matrix_path);
    append_string(out, value.feature_path);
    append_string(out, value.barcode_path);
    append_string(out, value.metadata_path);
    append_pod(out, value.format);
    append_pod(out, value.row_begin);
    append_pod(out, value.row_end);
    append_pod(out, value.rows);
    append_pod(out, value.cols);
    append_pod(out, value.nnz);
}

inline bool read_source_dataset_summary(byte_reader *reader,
                                        source_dataset_summary *out,
                                        std::string *error) {
    return out != nullptr
        && read_string(reader, &out->dataset_id, error, "dataset_id")
        && read_string(reader, &out->matrix_path, error, "matrix_path")
        && read_string(reader, &out->feature_path, error, "feature_path")
        && read_string(reader, &out->barcode_path, error, "barcode_path")
        && read_string(reader, &out->metadata_path, error, "metadata_path")
        && read_pod(reader, &out->format, error, "dataset_format")
        && read_pod(reader, &out->row_begin, error, "dataset_row_begin")
        && read_pod(reader, &out->row_end, error, "dataset_row_end")
        && read_pod(reader, &out->rows, error, "dataset_rows")
        && read_pod(reader, &out->cols, error, "dataset_cols")
        && read_pod(reader, &out->nnz, error, "dataset_nnz");
}

inline void append_dataset_codec_summary(std::vector<std::uint8_t> *out, const dataset_codec_summary &value) {
    append_pod(out, value.codec_id);
    append_pod(out, value.family);
    append_pod(out, value.value_code);
    append_pod(out, value.scale_value_code);
    append_pod(out, value.bits);
    append_pod(out, value.flags);
}

inline bool read_dataset_codec_summary(byte_reader *reader,
                                       dataset_codec_summary *out,
                                       std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->codec_id, error, "codec_id")
        && read_pod(reader, &out->family, error, "codec_family")
        && read_pod(reader, &out->value_code, error, "codec_value_code")
        && read_pod(reader, &out->scale_value_code, error, "codec_scale_value_code")
        && read_pod(reader, &out->bits, error, "codec_bits")
        && read_pod(reader, &out->flags, error, "codec_flags");
}

inline void append_dataset_partition_summary(std::vector<std::uint8_t> *out, const dataset_partition_summary &value) {
    append_pod(out, value.partition_id);
    append_pod(out, value.row_begin);
    append_pod(out, value.row_end);
    append_pod(out, value.rows);
    append_pod(out, value.nnz);
    append_pod(out, value.aux);
    append_pod(out, value.dataset_id);
    append_pod(out, value.axis);
    append_pod(out, value.codec_id);
}

inline bool read_dataset_partition_summary(byte_reader *reader,
                                           dataset_partition_summary *out,
                                           std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->partition_id, error, "partition_id")
        && read_pod(reader, &out->row_begin, error, "partition_row_begin")
        && read_pod(reader, &out->row_end, error, "partition_row_end")
        && read_pod(reader, &out->rows, error, "partition_rows")
        && read_pod(reader, &out->nnz, error, "partition_nnz")
        && read_pod(reader, &out->aux, error, "partition_aux")
        && read_pod(reader, &out->dataset_id, error, "partition_dataset_id")
        && read_pod(reader, &out->axis, error, "partition_axis")
        && read_pod(reader, &out->codec_id, error, "partition_codec_id");
}

inline void append_dataset_shard_summary(std::vector<std::uint8_t> *out, const dataset_shard_summary &value) {
    append_pod(out, value.shard_id);
    append_pod(out, value.partition_begin);
    append_pod(out, value.partition_end);
    append_pod(out, value.row_begin);
    append_pod(out, value.row_end);
}

inline bool read_dataset_shard_summary(byte_reader *reader,
                                       dataset_shard_summary *out,
                                       std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->shard_id, error, "shard_id")
        && read_pod(reader, &out->partition_begin, error, "shard_partition_begin")
        && read_pod(reader, &out->partition_end, error, "shard_partition_end")
        && read_pod(reader, &out->row_begin, error, "shard_row_begin")
        && read_pod(reader, &out->row_end, error, "shard_row_end");
}

inline void append_dataset_summary(std::vector<std::uint8_t> *out,
                                   const dataset_summary &value,
                                   bool include_path) {
    if (include_path) append_string(out, value.path);
    append_string(out, value.matrix_format);
    append_string(out, value.payload_layout);
    append_pod(out, value.rows);
    append_pod(out, value.cols);
    append_pod(out, value.nnz);
    append_pod(out, value.num_partitions);
    append_pod(out, value.num_shards);
    append_pod(out, value.num_datasets);
    append_pod(out, (std::uint64_t) value.datasets.size());
    for (const source_dataset_summary &dataset : value.datasets) append_source_dataset_summary(out, dataset);
    append_pod(out, (std::uint64_t) value.partitions.size());
    for (const dataset_partition_summary &partition : value.partitions) append_dataset_partition_summary(out, partition);
    append_pod(out, (std::uint64_t) value.shards.size());
    for (const dataset_shard_summary &shard : value.shards) append_dataset_shard_summary(out, shard);
    append_pod(out, (std::uint64_t) value.codecs.size());
    for (const dataset_codec_summary &codec : value.codecs) append_dataset_codec_summary(out, codec);
    append_string_vector(out, value.obs_names);
    append_string_vector(out, value.var_ids);
    append_string_vector(out, value.var_names);
    append_string_vector(out, value.var_types);
    append_pod(out, value.observation_annotations.available ? 1u : 0u);
    append_pod(out, value.observation_annotations.extent);
    append_string_vector(out, value.observation_annotations.names);
    append_pod_vector(out, value.observation_annotations.types);
    append_pod(out, value.feature_annotations.available ? 1u : 0u);
    append_pod(out, value.feature_annotations.extent);
    append_string_vector(out, value.feature_annotations.names);
    append_pod_vector(out, value.feature_annotations.types);
    append_pod(out, value.dataset_attributes.available ? 1u : 0u);
    append_string_vector(out, value.dataset_attributes.keys);
}

inline bool read_dataset_summary(byte_reader *reader,
                                 dataset_summary *out,
                                 std::string *error,
                                 bool include_path) {
    std::uint64_t count = 0u;
    std::size_t size = 0u;
    std::uint32_t available = 0u;
    if (out == nullptr) return false;
    *out = dataset_summary{};
    if (include_path && !read_string(reader, &out->path, error, "summary_path")) return false;
    if (!read_string(reader, &out->matrix_format, error, "matrix_format")
        || !read_string(reader, &out->payload_layout, error, "payload_layout")
        || !read_pod(reader, &out->rows, error, "rows")
        || !read_pod(reader, &out->cols, error, "cols")
        || !read_pod(reader, &out->nnz, error, "nnz")
        || !read_pod(reader, &out->num_partitions, error, "num_partitions")
        || !read_pod(reader, &out->num_shards, error, "num_shards")
        || !read_pod(reader, &out->num_datasets, error, "num_datasets")) return false;

    if (!read_pod(reader, &count, error, "dataset_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "metadata snapshot dataset_count is too large");
        return false;
    }
    out->datasets.assign(size, source_dataset_summary{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_source_dataset_summary(reader, out->datasets.data() + i, error)) return false;
    }

    if (!read_pod(reader, &count, error, "partition_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "metadata snapshot partition_count is too large");
        return false;
    }
    out->partitions.assign(size, dataset_partition_summary{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_dataset_partition_summary(reader, out->partitions.data() + i, error)) return false;
    }

    if (!read_pod(reader, &count, error, "shard_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "metadata snapshot shard_count is too large");
        return false;
    }
    out->shards.assign(size, dataset_shard_summary{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_dataset_shard_summary(reader, out->shards.data() + i, error)) return false;
    }

    if (!read_pod(reader, &count, error, "codec_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "metadata snapshot codec_count is too large");
        return false;
    }
    out->codecs.assign(size, dataset_codec_summary{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_dataset_codec_summary(reader, out->codecs.data() + i, error)) return false;
    }

    return read_string_vector(reader, &out->obs_names, error, "obs_names")
        && read_string_vector(reader, &out->var_ids, error, "var_ids")
        && read_string_vector(reader, &out->var_names, error, "var_names")
        && read_string_vector(reader, &out->var_types, error, "var_types")
        && read_pod(reader, &available, error, "observation_annotations_available")
        && (out->observation_annotations.available = available != 0u, true)
        && read_pod(reader, &out->observation_annotations.extent, error, "observation_annotations_extent")
        && read_string_vector(reader, &out->observation_annotations.names, error, "observation_annotation_names")
        && read_pod_vector(reader, &out->observation_annotations.types, error, "observation_annotation_types")
        && read_pod(reader, &available, error, "feature_annotations_available")
        && (out->feature_annotations.available = available != 0u, true)
        && read_pod(reader, &out->feature_annotations.extent, error, "feature_annotations_extent")
        && read_string_vector(reader, &out->feature_annotations.names, error, "feature_annotation_names")
        && read_pod_vector(reader, &out->feature_annotations.types, error, "feature_annotation_types")
        && read_pod(reader, &available, error, "dataset_attributes_available")
        && (out->dataset_attributes.available = available != 0u, true)
        && read_string_vector(reader, &out->dataset_attributes.keys, error, "dataset_attribute_keys");
}

inline void append_execution_partition_metadata(std::vector<std::uint8_t> *out,
                                                const execution_partition_metadata &value) {
    append_pod(out, value.partition_id);
    append_pod(out, value.row_begin);
    append_pod(out, value.row_end);
    append_pod(out, value.rows);
    append_pod(out, value.nnz);
    append_pod(out, value.aux);
    append_pod(out, value.dataset_id);
    append_pod(out, value.axis);
    append_pod(out, value.codec_id);
    append_pod(out, value.execution_format);
    append_pod(out, value.blocked_ell_block_size);
    append_pod(out, value.blocked_ell_bucket_count);
    append_pod(out, value.blocked_ell_fill_ratio);
    append_pod(out, value.execution_bytes);
    append_pod(out, value.blocked_ell_bytes);
    append_pod(out, value.bucketed_blocked_ell_bytes);
}

inline bool read_execution_partition_metadata(byte_reader *reader,
                                              execution_partition_metadata *out,
                                              std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->partition_id, error, "exec_partition_id")
        && read_pod(reader, &out->row_begin, error, "exec_partition_row_begin")
        && read_pod(reader, &out->row_end, error, "exec_partition_row_end")
        && read_pod(reader, &out->rows, error, "exec_partition_rows")
        && read_pod(reader, &out->nnz, error, "exec_partition_nnz")
        && read_pod(reader, &out->aux, error, "exec_partition_aux")
        && read_pod(reader, &out->dataset_id, error, "exec_partition_dataset_id")
        && read_pod(reader, &out->axis, error, "exec_partition_axis")
        && read_pod(reader, &out->codec_id, error, "exec_partition_codec_id")
        && read_pod(reader, &out->execution_format, error, "exec_partition_format")
        && read_pod(reader, &out->blocked_ell_block_size, error, "exec_partition_block_size")
        && read_pod(reader, &out->blocked_ell_bucket_count, error, "exec_partition_bucket_count")
        && read_pod(reader, &out->blocked_ell_fill_ratio, error, "exec_partition_fill_ratio")
        && read_pod(reader, &out->execution_bytes, error, "exec_partition_execution_bytes")
        && read_pod(reader, &out->blocked_ell_bytes, error, "exec_partition_blocked_ell_bytes")
        && read_pod(reader, &out->bucketed_blocked_ell_bytes, error, "exec_partition_bucketed_blocked_ell_bytes");
}

inline void append_execution_shard_metadata(std::vector<std::uint8_t> *out,
                                            const execution_shard_metadata &value) {
    append_pod(out, value.shard_id);
    append_pod(out, value.partition_begin);
    append_pod(out, value.partition_end);
    append_pod(out, value.row_begin);
    append_pod(out, value.row_end);
    append_pod(out, value.execution_format);
    append_pod(out, value.blocked_ell_block_size);
    append_pod(out, value.bucketed_partition_count);
    append_pod(out, value.bucketed_segment_count);
    append_pod(out, value.blocked_ell_fill_ratio);
    append_pod(out, value.execution_bytes);
    append_pod(out, value.bucketed_blocked_ell_bytes);
    append_pod(out, value.preferred_pair);
    append_pod(out, value.owner_node_id);
    append_pod(out, value.owner_rank_id);
}

inline bool read_execution_shard_metadata(byte_reader *reader,
                                          execution_shard_metadata *out,
                                          std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->shard_id, error, "exec_shard_id")
        && read_pod(reader, &out->partition_begin, error, "exec_shard_partition_begin")
        && read_pod(reader, &out->partition_end, error, "exec_shard_partition_end")
        && read_pod(reader, &out->row_begin, error, "exec_shard_row_begin")
        && read_pod(reader, &out->row_end, error, "exec_shard_row_end")
        && read_pod(reader, &out->execution_format, error, "exec_shard_format")
        && read_pod(reader, &out->blocked_ell_block_size, error, "exec_shard_block_size")
        && read_pod(reader, &out->bucketed_partition_count, error, "exec_shard_bucketed_partition_count")
        && read_pod(reader, &out->bucketed_segment_count, error, "exec_shard_bucketed_segment_count")
        && read_pod(reader, &out->blocked_ell_fill_ratio, error, "exec_shard_fill_ratio")
        && read_pod(reader, &out->execution_bytes, error, "exec_shard_execution_bytes")
        && read_pod(reader, &out->bucketed_blocked_ell_bytes, error, "exec_shard_bucketed_blocked_ell_bytes")
        && read_pod(reader, &out->preferred_pair, error, "exec_shard_preferred_pair")
        && read_pod(reader, &out->owner_node_id, error, "exec_shard_owner_node_id")
        && read_pod(reader, &out->owner_rank_id, error, "exec_shard_owner_rank_id");
}

inline void append_runtime_service_metadata(std::vector<std::uint8_t> *out,
                                            const runtime_service_metadata &value) {
    append_pod(out, value.service_mode);
    append_pod(out, value.live_write_mode);
    append_pod(out, value.prefer_pack_delivery);
    append_pod(out, value.remote_pack_delivery);
    append_pod(out, value.single_reader_coordinator);
    append_pod(out, value.maintenance_lock_blocks_overwrite);
    append_pod(out, value.canonical_generation);
    append_pod(out, value.execution_plan_generation);
    append_pod(out, value.pack_generation);
    append_pod(out, value.service_epoch);
    append_pod(out, value.active_read_generation);
    append_pod(out, value.staged_write_generation);
}

inline bool read_runtime_service_metadata(byte_reader *reader,
                                          runtime_service_metadata *out,
                                          std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->service_mode, error, "runtime_service_mode")
        && read_pod(reader, &out->live_write_mode, error, "runtime_live_write_mode")
        && read_pod(reader, &out->prefer_pack_delivery, error, "runtime_prefer_pack_delivery")
        && read_pod(reader, &out->remote_pack_delivery, error, "runtime_remote_pack_delivery")
        && read_pod(reader, &out->single_reader_coordinator, error, "runtime_single_reader_coordinator")
        && read_pod(reader, &out->maintenance_lock_blocks_overwrite, error, "runtime_maintenance_lock_blocks_overwrite")
        && read_pod(reader, &out->canonical_generation, error, "runtime_canonical_generation")
        && read_pod(reader, &out->execution_plan_generation, error, "runtime_execution_plan_generation")
        && read_pod(reader, &out->pack_generation, error, "runtime_pack_generation")
        && read_pod(reader, &out->service_epoch, error, "runtime_service_epoch")
        && read_pod(reader, &out->active_read_generation, error, "runtime_active_read_generation")
        && read_pod(reader, &out->staged_write_generation, error, "runtime_staged_write_generation");
}

inline bool serialize_global_metadata_snapshot_payload(const global_metadata_snapshot &snapshot,
                                                       std::vector<std::uint8_t> *out,
                                                       bool include_snapshot_id,
                                                       bool include_path,
                                                       std::string *error) {
    static const char magic[] = {'C', 'S', 'G', 'M'};
    const std::uint32_t version = 2u;
    if (out == nullptr) {
        set_error(error, "metadata snapshot output buffer is null");
        return false;
    }
    out->clear();
    append_bytes(out, magic, sizeof(magic));
    append_pod(out, version);
    if (include_snapshot_id) append_pod(out, snapshot.snapshot_id);
    append_dataset_summary(out, snapshot.summary, include_path);
    append_pod(out, (std::uint64_t) snapshot.execution_partitions.size());
    for (const execution_partition_metadata &partition : snapshot.execution_partitions) {
        append_execution_partition_metadata(out, partition);
    }
    append_pod(out, (std::uint64_t) snapshot.execution_shards.size());
    for (const execution_shard_metadata &shard : snapshot.execution_shards) append_execution_shard_metadata(out, shard);
    append_runtime_service_metadata(out, snapshot.runtime_service);
    return true;
}

inline std::uint64_t compute_snapshot_id(const global_metadata_snapshot &snapshot) {
    global_metadata_snapshot normalized = snapshot;
    std::vector<std::uint8_t> bytes;
    normalized.snapshot_id = 0u;
    (void) serialize_global_metadata_snapshot_payload(normalized, &bytes, false, false, nullptr);
    return fnv1a64(bytes);
}

inline client_snapshot_ref build_client_snapshot_ref(const global_metadata_snapshot &snapshot) {
    client_snapshot_ref ref;
    ref.snapshot_id = compute_snapshot_id(snapshot);
    ref.canonical_generation = snapshot.runtime_service.canonical_generation;
    ref.execution_plan_generation = snapshot.runtime_service.execution_plan_generation;
    ref.pack_generation = snapshot.runtime_service.pack_generation;
    ref.service_epoch = snapshot.runtime_service.service_epoch;
    return ref;
}

inline void copy_execution_partition_metadata(const dataset_summary &summary,
                                              const cellshard::dataset_execution_view &execution,
                                              std::vector<execution_partition_metadata> *out) {
    const std::size_t count = summary.partitions.size();
    if (out == nullptr) return;
    out->assign(count, execution_partition_metadata{});
    for (std::size_t i = 0; i < count; ++i) {
        (*out)[i] = execution_partition_metadata{
            summary.partitions[i].partition_id,
            summary.partitions[i].row_begin,
            summary.partitions[i].row_end,
            summary.partitions[i].rows,
            summary.partitions[i].nnz,
            summary.partitions[i].aux,
            summary.partitions[i].dataset_id,
            summary.partitions[i].axis,
            summary.partitions[i].codec_id,
            i < execution.partition_count && execution.partition_execution_formats != nullptr
                ? execution.partition_execution_formats[i] : 0u,
            i < execution.partition_count && execution.partition_blocked_ell_block_sizes != nullptr
                ? execution.partition_blocked_ell_block_sizes[i] : 0u,
            i < execution.partition_count && execution.partition_blocked_ell_bucket_counts != nullptr
                ? execution.partition_blocked_ell_bucket_counts[i] : 0u,
            i < execution.partition_count && execution.partition_blocked_ell_fill_ratios != nullptr
                ? execution.partition_blocked_ell_fill_ratios[i] : 0.0f,
            i < execution.partition_count && execution.partition_execution_bytes != nullptr
                ? execution.partition_execution_bytes[i] : 0u,
            i < execution.partition_count && execution.partition_blocked_ell_bytes != nullptr
                ? execution.partition_blocked_ell_bytes[i] : 0u,
            i < execution.partition_count && execution.partition_bucketed_blocked_ell_bytes != nullptr
                ? execution.partition_bucketed_blocked_ell_bytes[i] : 0u
        };
    }
}

inline void copy_execution_shard_metadata(const dataset_summary &summary,
                                          const cellshard::dataset_execution_view &execution,
                                          std::vector<execution_shard_metadata> *out) {
    const std::size_t count = summary.shards.size();
    if (out == nullptr) return;
    out->assign(count, execution_shard_metadata{});
    for (std::size_t i = 0; i < count; ++i) {
        (*out)[i] = execution_shard_metadata{
            summary.shards[i].shard_id,
            summary.shards[i].partition_begin,
            summary.shards[i].partition_end,
            summary.shards[i].row_begin,
            summary.shards[i].row_end,
            i < execution.shard_count && execution.shard_execution_formats != nullptr
                ? execution.shard_execution_formats[i] : 0u,
            i < execution.shard_count && execution.shard_blocked_ell_block_sizes != nullptr
                ? execution.shard_blocked_ell_block_sizes[i] : 0u,
            i < execution.shard_count && execution.shard_bucketed_partition_counts != nullptr
                ? execution.shard_bucketed_partition_counts[i] : 0u,
            i < execution.shard_count && execution.shard_bucketed_segment_counts != nullptr
                ? execution.shard_bucketed_segment_counts[i] : 0u,
            i < execution.shard_count && execution.shard_blocked_ell_fill_ratios != nullptr
                ? execution.shard_blocked_ell_fill_ratios[i] : 0.0f,
            i < execution.shard_count && execution.shard_execution_bytes != nullptr
                ? execution.shard_execution_bytes[i] : 0u,
            i < execution.shard_count && execution.shard_bucketed_blocked_ell_bytes != nullptr
                ? execution.shard_bucketed_blocked_ell_bytes[i] : 0u,
            i < execution.shard_count && execution.shard_preferred_pair_ids != nullptr
                ? execution.shard_preferred_pair_ids[i] : 0u,
            i < execution.shard_count && execution.shard_owner_node_ids != nullptr
                ? execution.shard_owner_node_ids[i] : 0u,
            i < execution.shard_count && execution.shard_owner_rank_ids != nullptr
                ? execution.shard_owner_rank_ids[i] : 0u
        };
    }
}

inline void copy_runtime_service_metadata(const cellshard::dataset_runtime_service_view &view,
                                          runtime_service_metadata *out) {
    if (out == nullptr) return;
    *out = runtime_service_metadata{
        view.service_mode,
        view.live_write_mode,
        view.prefer_pack_delivery,
        view.remote_pack_delivery,
        view.single_reader_coordinator,
        view.maintenance_lock_blocks_overwrite,
        view.canonical_generation,
        view.execution_plan_generation,
        view.pack_generation,
        view.service_epoch,
        view.active_read_generation,
        view.staged_write_generation
    };
}

} // namespace cellshard::exporting::detail
