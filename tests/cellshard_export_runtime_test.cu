#include "../export/dataset_export.hh"
#include "../src/CellShard.hh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace cs = ::cellshard;
namespace cse = ::cellshard::exporting;

namespace {

struct owned_text_column {
    std::vector<std::uint32_t> offsets;
    std::vector<char> data;

    cs::dataset_text_column_view view() const {
        cs::dataset_text_column_view out{};
        out.count = offsets.empty() ? 0u : (std::uint32_t) offsets.size() - 1u;
        out.bytes = (std::uint32_t) data.size();
        out.offsets = offsets.empty() ? nullptr : offsets.data();
        out.data = data.empty() ? nullptr : data.data();
        return out;
    }
};

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

bool close_float(float lhs, float rhs, float tol = 1.0e-4f) {
    return std::fabs(lhs - rhs) <= tol;
}

owned_text_column make_text_column(const std::vector<const char *> &values) {
    owned_text_column out;
    std::uint32_t cursor = 0u;
    out.offsets.resize(values.size() + 1u, 0u);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const char *value = values[i] != nullptr ? values[i] : "";
        const std::size_t len = std::strlen(value);
        out.offsets[i] = cursor;
        out.data.insert(out.data.end(), value, value + len);
        out.data.push_back('\0');
        cursor += (std::uint32_t) len + 1u;
    }
    out.offsets[values.size()] = cursor;
    return out;
}

void fill_blocked_ell_part(cs::sparse::blocked_ell *part) {
    cs::sparse::init(part, 2u, 4u, 4u, 2u, 4u);
    require(cs::sparse::allocate(part) != 0, "blocked-ELL allocate failed");
    part->blockColIdx[0] = 0u;
    part->blockColIdx[1] = 1u;
    part->val[0] = __float2half(1.0f);
    part->val[1] = __float2half(0.0f);
    part->val[2] = __float2half(2.0f);
    part->val[3] = __float2half(0.0f);
    part->val[4] = __float2half(0.0f);
    part->val[5] = __float2half(3.0f);
    part->val[6] = __float2half(0.0f);
    part->val[7] = __float2half(4.0f);
}

} // namespace

int main() {
    char path[] = "/tmp/cellshard_export_runtimeXXXXXX.csh5";
    const int fd = ::mkstemps(path, 5);
    require(fd >= 0, "mkstemps failed");
    ::close(fd);
    std::remove(path);

    cs::sparse::blocked_ell part;
    cs::sparse::init(&part);

    const owned_text_column dataset_ids = make_text_column({"dataset0"});
    const owned_text_column matrix_paths = make_text_column({"matrix.mtx"});
    const owned_text_column feature_paths = make_text_column({"features.tsv"});
    const owned_text_column barcode_paths = make_text_column({"barcodes.tsv"});
    const owned_text_column metadata_paths = make_text_column({"obs.tsv"});
    const std::uint32_t dataset_formats[] = { 3u };
    const std::uint64_t dataset_row_begin[] = { 0u };
    const std::uint64_t dataset_row_end[] = { 2u };
    const std::uint64_t dataset_rows[] = { 2u };
    const std::uint64_t dataset_cols[] = { 4u };
    const std::uint64_t dataset_nnz[] = { 4u };
    const cs::dataset_dataset_table_view datasets{
        1u,
        dataset_ids.view(),
        matrix_paths.view(),
        feature_paths.view(),
        barcode_paths.view(),
        metadata_paths.view(),
        dataset_formats,
        dataset_row_begin,
        dataset_row_end,
        dataset_rows,
        dataset_cols,
        dataset_nnz
    };

    const owned_text_column global_barcodes = make_text_column({"cell_a", "cell_b"});
    const owned_text_column feature_ids = make_text_column({"gene0", "gene1", "gene2", "gene3"});
    const owned_text_column feature_names = make_text_column({"G0", "G1", "G2", "G3"});
    const owned_text_column feature_types = make_text_column({"gene", "gene", "gene", "gene"});
    const std::uint32_t cell_dataset_ids[] = { 0u, 0u };
    const std::uint64_t cell_local_indices[] = { 0u, 1u };
    const std::uint32_t feature_dataset_ids[] = { 0u, 0u, 0u, 0u };
    const std::uint64_t feature_local_indices[] = { 0u, 1u, 2u, 3u };
    const std::uint64_t dataset_feature_offsets[] = { 0u, 4u };
    const std::uint32_t dataset_feature_to_global[] = { 0u, 1u, 2u, 3u };
    const cs::dataset_provenance_view provenance{
        global_barcodes.view(),
        cell_dataset_ids,
        cell_local_indices,
        feature_ids.view(),
        feature_names.view(),
        feature_types.view(),
        feature_dataset_ids,
        feature_local_indices,
        dataset_feature_offsets,
        dataset_feature_to_global
    };

    const std::uint64_t partition_rows[] = { 2u };
    const std::uint64_t partition_nnz[] = { 4u };
    const std::uint64_t partition_aux[] = {
        (std::uint64_t) cs::sparse::pack_blocked_ell_aux(2u, 2ul)
    };
    const std::uint32_t partition_axes[] = { 0u };
    const std::uint64_t partition_row_offsets[] = { 0u, 2u };
    const std::uint32_t partition_dataset_ids[] = { 0u };
    const std::uint32_t partition_codec_ids[] = { 0u };
    const std::uint64_t shard_offsets[] = { 0u, 2u };
    cs::dataset_codec_descriptor codec{};
    codec.codec_id = 0u;
    codec.family = cs::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    const cs::dataset_layout_view layout{
        2u,
        4u,
        4u,
        1u,
        1u,
        partition_rows,
        partition_nnz,
        partition_axes,
        partition_aux,
        partition_row_offsets,
        partition_dataset_ids,
        partition_codec_ids,
        shard_offsets,
        &codec,
        1u
    };

    const owned_text_column obs_batch_values = make_text_column({"batch0", "batch1"});
    const float obs_quality_values[] = { 1.25f, 2.5f };
    const cs::dataset_observation_metadata_column_view obs_columns[] = {
        { "batch", cs::dataset_observation_metadata_type_text, obs_batch_values.view(), nullptr, nullptr },
        { "quality", cs::dataset_observation_metadata_type_float32, {}, obs_quality_values, nullptr }
    };
    const cs::dataset_observation_metadata_view obs_metadata{
        2u,
        2u,
        obs_columns
    };
    const owned_text_column embedded_column_names = make_text_column({"stage", "batch"});
    const owned_text_column embedded_field_values = make_text_column({"E8.5", "A", "P0", "B"});
    const std::uint32_t embedded_row_offsets[] = { 0u, 2u, 4u };
    const std::uint32_t embedded_dataset_indices[] = { 0u };
    const std::uint64_t embedded_row_begin[] = { 0u };
    const std::uint64_t embedded_row_end[] = { 2u };
    const cs::dataset_metadata_table_view embedded_table{
        2u,
        2u,
        embedded_column_names.view(),
        embedded_field_values.view(),
        embedded_row_offsets
    };
    const cs::dataset_embedded_metadata_view embedded_metadata{
        1u,
        embedded_dataset_indices,
        embedded_row_begin,
        embedded_row_end,
        &embedded_table
    };
    const std::uint32_t partition_execution_formats[] = { cs::dataset_execution_format_bucketed_blocked_ell };
    const std::uint32_t partition_block_sizes[] = { 2u };
    const std::uint32_t partition_bucket_counts[] = { 1u };
    const float partition_fill_ratios[] = { 0.50f };
    const std::uint64_t partition_execution_bytes[] = { 64u };
    const std::uint64_t partition_blocked_ell_bytes[] = { 64u };
    const std::uint64_t partition_bucketed_blocked_ell_bytes[] = { 80u };
    const std::uint32_t shard_execution_formats[] = { cs::dataset_execution_format_bucketed_blocked_ell };
    const std::uint32_t shard_block_sizes[] = { 2u };
    const std::uint32_t shard_bucketed_partition_counts[] = { 1u };
    const std::uint32_t shard_bucketed_segment_counts[] = { 1u };
    const float shard_fill_ratios[] = { 0.50f };
    const std::uint64_t shard_execution_bytes[] = { 80u };
    const std::uint64_t shard_bucketed_blocked_ell_bytes[] = { 80u };
    const std::uint32_t shard_preferred_pair_ids[] = { 2u };
    const std::uint32_t shard_owner_node_ids[] = { 7u };
    const std::uint32_t shard_owner_rank_ids[] = { 3u };
    cs::dataset_execution_view execution{};
    execution.partition_count = 1u;
    execution.partition_execution_formats = partition_execution_formats;
    execution.partition_blocked_ell_block_sizes = partition_block_sizes;
    execution.partition_blocked_ell_bucket_counts = partition_bucket_counts;
    execution.partition_blocked_ell_fill_ratios = partition_fill_ratios;
    execution.partition_execution_bytes = partition_execution_bytes;
    execution.partition_blocked_ell_bytes = partition_blocked_ell_bytes;
    execution.partition_bucketed_blocked_ell_bytes = partition_bucketed_blocked_ell_bytes;
    execution.shard_count = 1u;
    execution.shard_execution_formats = shard_execution_formats;
    execution.shard_blocked_ell_block_sizes = shard_block_sizes;
    execution.shard_bucketed_partition_counts = shard_bucketed_partition_counts;
    execution.shard_bucketed_segment_counts = shard_bucketed_segment_counts;
    execution.shard_blocked_ell_fill_ratios = shard_fill_ratios;
    execution.shard_execution_bytes = shard_execution_bytes;
    execution.shard_bucketed_blocked_ell_bytes = shard_bucketed_blocked_ell_bytes;
    execution.shard_preferred_pair_ids = shard_preferred_pair_ids;
    execution.shard_owner_node_ids = shard_owner_node_ids;
    execution.shard_owner_rank_ids = shard_owner_rank_ids;
    execution.preferred_base_format = cs::dataset_execution_format_bucketed_blocked_ell;
    cs::dataset_runtime_service_view runtime_service{};

    fill_blocked_ell_part(&part);
    cs::init(&runtime_service);
    runtime_service.service_mode = cs::dataset_runtime_service_mode_owner_hosted;
    runtime_service.live_write_mode = cs::dataset_live_write_mode_append_only;
    runtime_service.prefer_pack_delivery = 1u;
    runtime_service.remote_pack_delivery = 0u;
    runtime_service.single_reader_coordinator = 1u;
    runtime_service.maintenance_lock_blocks_overwrite = 1u;
    runtime_service.canonical_generation = 11u;
    runtime_service.execution_plan_generation = 12u;
    runtime_service.pack_generation = 13u;
    runtime_service.service_epoch = 14u;
    runtime_service.active_read_generation = 15u;
    runtime_service.staged_write_generation = 16u;

    require(cs::create_dataset_blocked_ell_h5(path, &layout, &datasets, &provenance) != 0, "create_dataset_blocked_ell_h5 failed");
    require(cs::append_blocked_ell_partition_h5(path, 0ul, &part) != 0, "append_blocked_ell_partition_h5 failed");
    require(cs::append_dataset_embedded_metadata_h5(path, &embedded_metadata) != 0, "append_dataset_embedded_metadata_h5 failed");
    require(cs::append_dataset_observation_metadata_h5(path, &obs_metadata) != 0, "append_dataset_observation_metadata_h5 failed");
    require(cs::append_dataset_execution_h5(path, &execution) != 0, "append_dataset_execution_h5 failed");
    require(cs::append_dataset_runtime_service_h5(path, &runtime_service) != 0, "append_dataset_runtime_service_h5 failed");

    cse::dataset_summary summary;
    std::string error;
    require(cse::load_dataset_summary(path, &summary, &error), error.c_str());
    require(summary.matrix_format == "blocked_ell", "summary matrix_format mismatch");
    require(summary.payload_layout == "shard_packed", "summary payload_layout mismatch");
    require(summary.rows == 2u && summary.cols == 4u && summary.nnz == 4u, "summary shape mismatch");
    require(summary.datasets.size() == 1u, "summary dataset count mismatch");
    require(summary.datasets[0].dataset_id == "dataset0", "summary dataset id mismatch");
    require(summary.partitions.size() == 1u && summary.partitions[0].aux == partition_aux[0], "summary partition metadata mismatch");
    require(summary.shards.size() == 1u && summary.shards[0].row_end == 2u, "summary shard metadata mismatch");
    require(summary.obs_names.size() == 2u && summary.obs_names[1] == "cell_b", "summary obs names mismatch");
    require(summary.var_names.size() == 4u && summary.var_names[2] == "G2", "summary var names mismatch");

    cse::csr_matrix_export csr;
    error.clear();
    require(cse::load_dataset_as_csr(path, &csr, &error), error.c_str());
    require(csr.rows == 2u && csr.cols == 4u, "csr shape mismatch");
    require(csr.indptr == std::vector<std::int64_t>({0, 2, 4}), "csr indptr mismatch");
    require(csr.indices == std::vector<std::int64_t>({0, 2, 1, 3}), "csr indices mismatch");
    require(csr.data.size() == 4u, "csr value count mismatch");
    require(close_float(csr.data[0], 1.0f), "csr data[0] mismatch");
    require(close_float(csr.data[1], 2.0f), "csr data[1] mismatch");
    require(close_float(csr.data[2], 3.0f), "csr data[2] mismatch");
    require(close_float(csr.data[3], 4.0f), "csr data[3] mismatch");

    const std::uint64_t selected_rows[] = {1u, 0u, 1u};
    cse::csr_matrix_export row_subset;
    error.clear();
    require(cse::load_dataset_rows_as_csr(path, selected_rows, 3u, &row_subset, &error), error.c_str());
    require(row_subset.rows == 3u && row_subset.cols == 4u, "row subset shape mismatch");
    require(row_subset.indptr == std::vector<std::int64_t>({0, 2, 4, 6}), "row subset indptr mismatch");
    require(row_subset.indices == std::vector<std::int64_t>({1, 3, 0, 2, 1, 3}), "row subset indices mismatch");
    require(row_subset.data.size() == 6u, "row subset value count mismatch");
    require(close_float(row_subset.data[0], 3.0f), "row subset data[0] mismatch");
    require(close_float(row_subset.data[1], 4.0f), "row subset data[1] mismatch");
    require(close_float(row_subset.data[2], 1.0f), "row subset data[2] mismatch");
    require(close_float(row_subset.data[3], 2.0f), "row subset data[3] mismatch");
    require(close_float(row_subset.data[4], 3.0f), "row subset data[4] mismatch");
    require(close_float(row_subset.data[5], 4.0f), "row subset data[5] mismatch");

    cse::anndata_export snapshot;
    error.clear();
    require(cse::load_dataset_for_anndata(path, &snapshot, &error), error.c_str());
    require(snapshot.obs_columns.size() == 2u, "anndata obs column count mismatch");
    require(snapshot.obs_columns[0].name == "batch", "anndata batch column missing");
    require(snapshot.obs_columns[0].text_values[0] == "batch0", "anndata batch values mismatch");
    require(snapshot.obs_columns[1].name == "quality", "anndata quality column missing");
    require(snapshot.obs_columns[1].float32_values.size() == 2u, "anndata quality values missing");
    require(close_float(snapshot.obs_columns[1].float32_values[1], 2.5f), "anndata quality value mismatch");
    require(snapshot.x.indptr == csr.indptr, "anndata csr indptr mismatch");
    require(snapshot.x.indices == csr.indices, "anndata csr indices mismatch");
    require(snapshot.x.data.size() == csr.data.size(), "anndata csr values mismatch");

    cse::global_metadata_snapshot owner_snapshot;
    error.clear();
    require(cse::load_dataset_global_metadata_snapshot(path, &owner_snapshot, &error), error.c_str());
    require(owner_snapshot.snapshot_id != 0u, "owner snapshot id missing");
    require(owner_snapshot.embedded_metadata.size() == 1u, "owner embedded metadata count mismatch");
    require(owner_snapshot.embedded_metadata[0].column_names.size() == 2u, "owner embedded metadata columns missing");
    require(owner_snapshot.embedded_metadata[0].field_values.size() == 4u, "owner embedded metadata values missing");
    require(owner_snapshot.embedded_metadata[0].field_values[2] == "P0", "owner embedded metadata contents mismatch");
    require(owner_snapshot.observation_metadata_rows == 2u, "owner observation metadata rows mismatch");
    require(owner_snapshot.observation_metadata.size() == 2u, "owner observation metadata count mismatch");
    require(owner_snapshot.execution_partitions.size() == 1u, "owner execution partition count mismatch");
    require(owner_snapshot.execution_partitions[0].execution_format == cs::dataset_execution_format_bucketed_blocked_ell,
            "owner execution partition format mismatch");
    require(owner_snapshot.execution_shards.size() == 1u, "owner execution shard count mismatch");
    require(owner_snapshot.execution_shards[0].owner_node_id == 7u, "owner shard owner node mismatch");
    require(owner_snapshot.execution_shards[0].owner_rank_id == 3u, "owner shard owner rank mismatch");
    require(owner_snapshot.runtime_service.canonical_generation == 11u, "owner runtime canonical generation mismatch");
    require(owner_snapshot.runtime_service.pack_generation == 13u, "owner runtime pack generation mismatch");

    const cse::client_snapshot_ref request_ref = cse::make_client_snapshot_ref(owner_snapshot);
    error.clear();
    require(cse::validate_client_snapshot_ref(owner_snapshot, request_ref, &error), error.c_str());
    cse::client_snapshot_ref stale_ref = request_ref;
    stale_ref.pack_generation += 1u;
    error.clear();
    require(!cse::validate_client_snapshot_ref(owner_snapshot, stale_ref, &error), "stale request ref unexpectedly validated");
    require(error.find("pack_generation") != std::string::npos, "stale request ref error mismatch");

    std::vector<std::uint8_t> serialized_snapshot;
    cse::global_metadata_snapshot decoded_snapshot;
    error.clear();
    require(cse::serialize_global_metadata_snapshot(owner_snapshot, &serialized_snapshot, &error), error.c_str());
    require(!serialized_snapshot.empty(), "serialized owner snapshot is empty");
    error.clear();
    require(cse::deserialize_global_metadata_snapshot(serialized_snapshot.data(),
                                                      serialized_snapshot.size(),
                                                      &decoded_snapshot,
                                                      &error),
            error.c_str());
    require(decoded_snapshot.snapshot_id == owner_snapshot.snapshot_id, "decoded snapshot id mismatch");
    require(decoded_snapshot.runtime_service.service_epoch == owner_snapshot.runtime_service.service_epoch,
            "decoded snapshot runtime service mismatch");
    require(decoded_snapshot.observation_metadata.size() == owner_snapshot.observation_metadata.size(),
            "decoded snapshot observation metadata mismatch");
    require(decoded_snapshot.embedded_metadata[0].row_offsets == owner_snapshot.embedded_metadata[0].row_offsets,
            "decoded snapshot embedded metadata row offsets mismatch");

    cs::sparse::clear(&part);
    std::remove(path);
    return 0;
}
