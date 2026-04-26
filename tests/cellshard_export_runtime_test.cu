#include <CellShard/export/dataset_export.hh>
#include <CellShard/CellShard.hh>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
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
    const std::string derived_path = std::string(path) + ".derived.csh5";
    const std::string derived_cache_root = std::string(path) + ".derived.cache";
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
    const owned_text_column var_chr_values = make_text_column({"chr1", "chr2", "chr3", "chr4"});
    const owned_text_column var_short_values = make_text_column({"G0", "G1", "G2", "G3"});
    const cs::dataset_observation_metadata_column_view var_columns[] = {
        { "chr", cs::dataset_observation_metadata_type_text, var_chr_values.view(), nullptr, nullptr },
        { "gene_short_name", cs::dataset_observation_metadata_type_text, var_short_values.view(), nullptr, nullptr }
    };
    const cs::dataset_feature_metadata_view feature_metadata{
        4u,
        2u,
        var_columns
    };
    const owned_text_column attribute_keys = make_text_column({"preprocess.pipeline_scope", "study"});
    const owned_text_column attribute_values = make_text_column({"qc_only", "demo"});
    const cs::dataset_user_attribute_view dataset_attributes{
        2u,
        attribute_keys.view(),
        attribute_values.view()
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
    cs::bucketed_blocked_ell_shard optimized_shard{};
    cs::bucketed_blocked_ell_partition bucketed_part{};

    fill_blocked_ell_part(&part);
    cs::init(&runtime_service);
    cs::init(&optimized_shard);
    cs::init(&bucketed_part);
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
    require(cs::build_bucketed_blocked_ell_partition(&bucketed_part, &part, 1u, nullptr) != 0,
            "build_bucketed_blocked_ell_partition failed");
    bucketed_part.exec_to_canonical_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    bucketed_part.canonical_to_exec_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    require(bucketed_part.exec_to_canonical_cols != nullptr && bucketed_part.canonical_to_exec_cols != nullptr,
            "optimized blocked partition col-map allocation failed");
    optimized_shard.rows = 2u;
    optimized_shard.cols = 4u;
    optimized_shard.nnz = 4u;
    optimized_shard.partition_count = 1u;
    optimized_shard.partitions = (cs::bucketed_blocked_ell_partition *) std::calloc(1u, sizeof(cs::bucketed_blocked_ell_partition));
    optimized_shard.partition_row_offsets = (std::uint32_t *) std::calloc(2u, sizeof(std::uint32_t));
    optimized_shard.exec_to_canonical_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    optimized_shard.canonical_to_exec_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    require(optimized_shard.partitions != nullptr
                && optimized_shard.partition_row_offsets != nullptr
                && optimized_shard.exec_to_canonical_cols != nullptr
                && optimized_shard.canonical_to_exec_cols != nullptr,
            "optimized blocked shard allocation failed");
    optimized_shard.partitions[0] = bucketed_part;
    std::memset(&bucketed_part, 0, sizeof(bucketed_part));
    optimized_shard.partition_row_offsets[0] = 0u;
    optimized_shard.partition_row_offsets[1] = 2u;
    for (std::uint32_t col = 0u; col < 4u; ++col) {
        optimized_shard.partitions[0].exec_to_canonical_cols[col] = col;
        optimized_shard.partitions[0].canonical_to_exec_cols[col] = col;
        optimized_shard.exec_to_canonical_cols[col] = col;
        optimized_shard.canonical_to_exec_cols[col] = col;
    }
    require(cs::append_bucketed_blocked_ell_shard_h5(path, 0ul, &optimized_shard) != 0,
            "append_bucketed_blocked_ell_shard_h5 failed");
    require(cs::append_dataset_embedded_metadata_h5(path, &embedded_metadata) != 0, "append_dataset_embedded_metadata_h5 failed");
    require(cs::append_dataset_observation_metadata_h5(path, &obs_metadata) != 0, "append_dataset_observation_metadata_h5 failed");
    require(cs::append_dataset_feature_metadata_h5(path, &feature_metadata) != 0, "append_dataset_feature_metadata_h5 failed");
    require(cs::append_dataset_user_attributes_h5(path, &dataset_attributes) != 0, "append_dataset_user_attributes_h5 failed");
    require(cs::append_dataset_execution_h5(path, &execution) != 0, "append_dataset_execution_h5 failed");
    require(cs::append_dataset_runtime_service_h5(path, &runtime_service) != 0, "append_dataset_runtime_service_h5 failed");

    cse::dataset_summary summary;
    std::string error;
    require(cse::load_dataset_summary(path, &summary, &error), error.c_str());
    require(summary.matrix_format == "blocked_ell", "summary matrix_format mismatch");
    require(summary.payload_layout == "optimized_bucketed_blocked_ell", "summary payload_layout mismatch");
    require(summary.rows == 2u && summary.cols == 4u && summary.nnz == 4u, "summary shape mismatch");
    require(summary.datasets.size() == 1u, "summary dataset count mismatch");
    require(summary.datasets[0].dataset_id == "dataset0", "summary dataset id mismatch");
    require(summary.partitions.size() == 1u && summary.partitions[0].aux == partition_aux[0], "summary partition metadata mismatch");
    require(summary.shards.size() == 1u && summary.shards[0].row_span.row_end == 2u, "summary shard metadata mismatch");
    require(summary.obs_names.size() == 2u && summary.obs_names[1] == "cell_b", "summary obs names mismatch");
    require(summary.var_names.size() == 4u && summary.var_names[2] == "G2", "summary var names mismatch");
    require(summary.observation_annotations.available, "summary observation annotation summary missing");
    require(summary.observation_annotations.extent == 2u, "summary observation annotation extent mismatch");
    require(summary.feature_annotations.available, "summary feature annotation summary missing");
    require(summary.feature_annotations.extent == 4u, "summary feature annotation extent mismatch");
    require(summary.dataset_attributes.available, "summary dataset attributes missing");
    require(summary.dataset_attributes.keys.size() == 2u, "summary dataset attribute key count mismatch");

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
    require(snapshot.var_columns.size() == 2u, "anndata var column count mismatch");
    require(snapshot.var_columns[0].name == "chr", "anndata chr column missing");
    require(snapshot.var_columns[0].text_values[3] == "chr4", "anndata chr values mismatch");
    require(snapshot.uns.size() == 2u, "anndata uns entry count mismatch");
    require(snapshot.x.indptr == csr.indptr, "anndata csr indptr mismatch");
    require(snapshot.x.indices == csr.indices, "anndata csr indices mismatch");
    require(snapshot.x.data.size() == csr.data.size(), "anndata csr values mismatch");

    std::vector<cse::observation_metadata_column> loaded_obs_columns;
    error.clear();
    require(cse::load_observation_metadata(path, &loaded_obs_columns, &error), error.c_str());
    require(loaded_obs_columns.size() == 2u, "explicit observation metadata load mismatch");
    std::vector<cse::annotation_column> loaded_var_columns;
    error.clear();
    require(cse::load_feature_metadata(path, &loaded_var_columns, &error), error.c_str());
    require(loaded_var_columns.size() == 2u, "explicit feature metadata load mismatch");
    std::vector<cse::dataset_attribute> loaded_attributes;
    error.clear();
    require(cse::load_dataset_attributes(path, &loaded_attributes, &error), error.c_str());
    require(loaded_attributes.size() == 2u, "explicit dataset attribute load mismatch");
    require(loaded_attributes[1].value == "demo", "explicit dataset attribute value mismatch");

    cse::derived_materialization_request derive_request;
    derive_request.output_path = derived_path;
    derive_request.cache_root = derived_cache_root;
    derive_request.derived_pack_name = "manual_group";
    derive_request.row_indices = {1u, 0u};
    derive_request.feature_indices = {3u, 2u, 0u};
    derive_request.row_groups = {
        {"late", 0u, 1u},
        {"early", 1u, 2u}
    };
    derive_request.feature_groups = {
        {"selected", 0u, 3u}
    };
    derive_request.materialize_dataset = true;
    derive_request.materialize_pack = true;
    cse::derived_materialization_result derive_result;
    error.clear();
    require(cse::materialize_derived_dataset(path, derive_request, &derive_result, &error), error.c_str());
    require(derive_result.materialized_dataset, "derived dataset output missing");
    require(derive_result.materialized_pack, "derived cspack output missing");
    require(derive_result.rows == 2u && derive_result.cols == 3u, "derived shape mismatch");

    cse::dataset_summary derived_summary;
    error.clear();
    require(cse::load_dataset_summary(derived_path.c_str(), &derived_summary, &error), error.c_str());
    require(derived_summary.rows == 2u && derived_summary.cols == 3u, "derived summary shape mismatch");
    require(derived_summary.observation_annotations.available, "derived observation annotations missing");
    require(derived_summary.feature_annotations.available, "derived feature annotations missing");
    require(derived_summary.dataset_attributes.available, "derived dataset attributes missing");

    cse::csr_matrix_export derived_csr;
    error.clear();
    require(cse::load_dataset_as_csr(derived_path.c_str(), &derived_csr, &error), error.c_str());
    require(derived_csr.rows == 2u && derived_csr.cols == 3u, "derived csr shape mismatch");
    require(derived_csr.indptr == std::vector<std::int64_t>({0, 1, 3}), "derived csr indptr mismatch");
    require(derived_csr.indices == std::vector<std::int64_t>({0, 1, 2}), "derived csr indices mismatch");
    require(derived_csr.data.size() == 3u, "derived csr data count mismatch");
    require(close_float(derived_csr.data[0], 4.0f), "derived csr data[0] mismatch");
    require(close_float(derived_csr.data[1], 2.0f), "derived csr data[1] mismatch");
    require(close_float(derived_csr.data[2], 1.0f), "derived csr data[2] mismatch");

    std::vector<cse::observation_metadata_column> derived_obs_columns;
    error.clear();
    require(cse::load_observation_metadata(derived_path.c_str(), &derived_obs_columns, &error), error.c_str());
    require(derived_obs_columns.size() == 3u, "derived observation metadata column count mismatch");
    require(derived_obs_columns[2].name == "derived.row_group", "derived row-group column missing");
    require(derived_obs_columns[2].text_values == std::vector<std::string>({"late", "early"}),
            "derived row-group values mismatch");

    std::vector<cse::annotation_column> derived_var_columns;
    error.clear();
    require(cse::load_feature_metadata(derived_path.c_str(), &derived_var_columns, &error), error.c_str());
    require(derived_var_columns.size() == 3u, "derived feature metadata column count mismatch");
    require(derived_var_columns[2].name == "derived.feature_group", "derived feature-group column missing");

    std::vector<cse::dataset_attribute> derived_attributes;
    error.clear();
    require(cse::load_dataset_attributes(derived_path.c_str(), &derived_attributes, &error), error.c_str());
    bool found_parent_path = false;
    bool found_pack_name = false;
    for (const cse::dataset_attribute &entry : derived_attributes) {
        if (entry.key == "derived.parent_path" && entry.value == path) found_parent_path = true;
        if (entry.key == "derived.pack_name" && entry.value == "manual_group") found_pack_name = true;
    }
    require(found_parent_path, "derived parent path attribute missing");
    require(found_pack_name, "derived pack-name attribute missing");

    {
        cs::sharded<cs::sparse::blocked_ell> derived_matrix;
        cs::shard_storage derived_storage;
        cs::bucketed_blocked_ell_partition derived_exec_part;
        cs::dataset_execution_view derived_execution{};
        cs::init(&derived_matrix);
        cs::init(&derived_storage);
        cs::init(&derived_exec_part);
        require(cs::load_dataset_blocked_ell_h5_header(derived_path.c_str(), &derived_matrix, &derived_storage) != 0,
                "load derived blocked header failed");
        require(cs::bind_dataset_h5_cache(&derived_storage, derived_cache_root.c_str()) != 0,
                "bind derived cache failed");
        require(cs::get_dataset_h5_execution_metadata(&derived_storage, &derived_execution) != 0,
                "get derived execution metadata failed");
        require(cs::fetch_dataset_blocked_ell_h5_pack_partition(&derived_exec_part, &derived_matrix, &derived_storage, 0u) != 0,
                "fetch derived pack partition failed");
        require(derived_execution.partition_count == 1u && derived_execution.shard_count == 1u,
                "derived execution metadata counts mismatch");
        require(derived_exec_part.rows == 2u && derived_exec_part.cols == 3u,
                "derived pack partition shape mismatch");
        cs::clear(&derived_exec_part);
        cs::clear(&derived_storage);
        cs::clear(&derived_matrix);
    }

    cse::global_metadata_snapshot owner_snapshot;
    error.clear();
    require(cse::load_dataset_global_metadata_snapshot(path, &owner_snapshot, &error), error.c_str());
    require(owner_snapshot.snapshot_id != 0u, "owner snapshot id missing");
    require(owner_snapshot.summary.observation_annotations.available, "owner snapshot observation annotation summary missing");
    require(owner_snapshot.summary.feature_annotations.available, "owner snapshot feature annotation summary missing");
    require(owner_snapshot.summary.dataset_attributes.available, "owner snapshot dataset attribute summary missing");
    require(owner_snapshot.execution_partitions.size() == 1u, "owner pack partition count mismatch");
    require(owner_snapshot.execution_partitions[0].execution_format == cs::dataset_execution_format_bucketed_blocked_ell,
            "owner pack partition format mismatch");
    require(owner_snapshot.execution_shards.size() == 1u, "owner execution shard count mismatch");
    require(owner_snapshot.execution_shards[0].owner_node_id == 7u, "owner shard owner node mismatch");
    require(owner_snapshot.execution_shards[0].owner_rank_id == 3u, "owner shard owner rank mismatch");
    require(owner_snapshot.runtime_service.runtime_generation.generation.canonical_generation == 11u,
            "owner runtime canonical generation mismatch");
    require(owner_snapshot.runtime_service.runtime_generation.generation.pack_generation == 13u,
            "owner runtime pack generation mismatch");

    const cse::client_snapshot_ref request_ref = cse::make_client_snapshot_ref(owner_snapshot);
    error.clear();
    require(cse::validate_client_snapshot_ref(owner_snapshot, request_ref, &error), error.c_str());
    cse::client_snapshot_ref stale_ref = request_ref;
    stale_ref.generation.pack_generation += 1u;
    error.clear();
    require(!cse::validate_client_snapshot_ref(owner_snapshot, stale_ref, &error), "stale request ref unexpectedly validated");
    require(error.find("pack_generation") != std::string::npos, "stale request ref error mismatch");

    cse::runtime_service_metadata staged_runtime;
    error.clear();
    require(cse::stage_append_only_runtime_service(owner_snapshot.runtime_service, &staged_runtime, &error), error.c_str());
    require(staged_runtime.runtime_generation.generation.canonical_generation == 17u, "staged canonical generation mismatch");
    require(staged_runtime.runtime_generation.generation.execution_plan_generation == 17u, "staged execution-plan generation mismatch");
    require(staged_runtime.runtime_generation.generation.pack_generation == 17u, "staged pack generation mismatch");
    require(staged_runtime.runtime_generation.staged_write_generation == 17u, "staged write generation mismatch");
    require(staged_runtime.runtime_generation.active_read_generation == owner_snapshot.runtime_service.runtime_generation.active_read_generation,
            "staged active read generation should not change before publish");

    cse::runtime_service_metadata published_runtime;
    error.clear();
    require(cse::publish_runtime_service_cutover(owner_snapshot.runtime_service, staged_runtime, &published_runtime, &error),
            error.c_str());
    require(published_runtime.runtime_generation.active_read_generation == 17u, "published active read generation mismatch");
    require(published_runtime.runtime_generation.staged_write_generation == 17u, "published staged write generation mismatch");
    require(published_runtime.runtime_generation.generation.service_epoch == 15u, "published service epoch mismatch");

    cse::pack_delivery_request delivery_request;
    delivery_request.request = request_ref;
    delivery_request.shard_id = 0u;
    cse::pack_delivery_descriptor delivery;
    error.clear();
    require(cse::describe_pack_delivery(owner_snapshot, delivery_request, &delivery, &error), error.c_str());
    require(delivery.snapshot_id == owner_snapshot.snapshot_id, "delivery snapshot id mismatch");
    require(delivery.owner_node_id == 7u && delivery.owner_rank_id == 3u, "delivery owner route mismatch");
    require(delivery.relative_pack_path == "packs/plan.12-pack.13-epoch.14/shard.0.cspack",
            "pack delivery path mismatch");

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
    require(decoded_snapshot.runtime_service.runtime_generation.generation.service_epoch
                == owner_snapshot.runtime_service.runtime_generation.generation.service_epoch,
            "decoded snapshot runtime service mismatch");
    require(decoded_snapshot.summary.observation_annotations.names == owner_snapshot.summary.observation_annotations.names,
            "decoded snapshot observation annotation summary mismatch");
    require(decoded_snapshot.summary.feature_annotations.names == owner_snapshot.summary.feature_annotations.names,
            "decoded snapshot feature annotation summary mismatch");
    require(decoded_snapshot.summary.dataset_attributes.keys == owner_snapshot.summary.dataset_attributes.keys,
            "decoded snapshot dataset attribute summary mismatch");

    cs::clear(&bucketed_part);
    cs::clear(&optimized_shard);
    cs::sparse::clear(&part);
    std::remove(derived_path.c_str());
    std::remove((derived_path + ".cache").c_str());
    std::error_code derived_ec;
    std::filesystem::remove_all(derived_cache_root, derived_ec);
    std::remove(path);
    return 0;
}
