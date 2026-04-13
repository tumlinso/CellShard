#include "../export/series_export.hh"
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

    cs::series_text_column_view view() const {
        cs::series_text_column_view out{};
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
    const cs::series_dataset_table_view datasets{
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
    const cs::series_provenance_view provenance{
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

    const std::uint64_t part_rows[] = { 2u };
    const std::uint64_t part_nnz[] = { 4u };
    const std::uint64_t part_aux[] = {
        (std::uint64_t) cs::sparse::pack_blocked_ell_aux(2u, 2ul)
    };
    const std::uint32_t part_axes[] = { 0u };
    const std::uint64_t part_row_offsets[] = { 0u, 2u };
    const std::uint32_t part_dataset_ids[] = { 0u };
    const std::uint32_t part_codec_ids[] = { 0u };
    const std::uint64_t shard_offsets[] = { 0u, 2u };
    cs::series_codec_descriptor codec{};
    codec.codec_id = 0u;
    codec.family = cs::series_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    const cs::series_layout_view layout{
        2u,
        4u,
        4u,
        1u,
        1u,
        part_rows,
        part_nnz,
        part_axes,
        part_aux,
        part_row_offsets,
        part_dataset_ids,
        part_codec_ids,
        shard_offsets,
        &codec,
        1u
    };

    const owned_text_column obs_batch_values = make_text_column({"batch0", "batch1"});
    const float obs_quality_values[] = { 1.25f, 2.5f };
    const cs::series_observation_metadata_column_view obs_columns[] = {
        { "batch", cs::series_observation_metadata_type_text, obs_batch_values.view(), nullptr, nullptr },
        { "quality", cs::series_observation_metadata_type_float32, {}, obs_quality_values, nullptr }
    };
    const cs::series_observation_metadata_view obs_metadata{
        2u,
        2u,
        obs_columns
    };

    fill_blocked_ell_part(&part);

    require(cs::create_series_blocked_ell_h5(path, &layout, &datasets, &provenance) != 0, "create_series_blocked_ell_h5 failed");
    require(cs::append_blocked_ell_part_h5(path, 0ul, &part) != 0, "append_blocked_ell_part_h5 failed");
    require(cs::append_series_observation_metadata_h5(path, &obs_metadata) != 0, "append_series_observation_metadata_h5 failed");

    cse::series_summary summary;
    std::string error;
    require(cse::load_series_summary(path, &summary, &error), error.c_str());
    require(summary.matrix_format == "blocked_ell", "summary matrix_format mismatch");
    require(summary.payload_layout == "shard_packed", "summary payload_layout mismatch");
    require(summary.rows == 2u && summary.cols == 4u && summary.nnz == 4u, "summary shape mismatch");
    require(summary.datasets.size() == 1u, "summary dataset count mismatch");
    require(summary.datasets[0].dataset_id == "dataset0", "summary dataset id mismatch");
    require(summary.partitions.size() == 1u && summary.partitions[0].aux == part_aux[0], "summary partition metadata mismatch");
    require(summary.shards.size() == 1u && summary.shards[0].row_end == 2u, "summary shard metadata mismatch");
    require(summary.obs_names.size() == 2u && summary.obs_names[1] == "cell_b", "summary obs names mismatch");
    require(summary.var_names.size() == 4u && summary.var_names[2] == "G2", "summary var names mismatch");

    cse::csr_matrix_export csr;
    error.clear();
    require(cse::load_series_as_csr(path, &csr, &error), error.c_str());
    require(csr.rows == 2u && csr.cols == 4u, "csr shape mismatch");
    require(csr.indptr == std::vector<std::int64_t>({0, 2, 4}), "csr indptr mismatch");
    require(csr.indices == std::vector<std::int64_t>({0, 2, 1, 3}), "csr indices mismatch");
    require(csr.data.size() == 4u, "csr value count mismatch");
    require(close_float(csr.data[0], 1.0f), "csr data[0] mismatch");
    require(close_float(csr.data[1], 2.0f), "csr data[1] mismatch");
    require(close_float(csr.data[2], 3.0f), "csr data[2] mismatch");
    require(close_float(csr.data[3], 4.0f), "csr data[3] mismatch");

    cse::anndata_export snapshot;
    error.clear();
    require(cse::load_series_for_anndata(path, &snapshot, &error), error.c_str());
    require(snapshot.obs_columns.size() == 2u, "anndata obs column count mismatch");
    require(snapshot.obs_columns[0].name == "batch", "anndata batch column missing");
    require(snapshot.obs_columns[0].text_values[0] == "batch0", "anndata batch values mismatch");
    require(snapshot.obs_columns[1].name == "quality", "anndata quality column missing");
    require(snapshot.obs_columns[1].float32_values.size() == 2u, "anndata quality values missing");
    require(close_float(snapshot.obs_columns[1].float32_values[1], 2.5f), "anndata quality value mismatch");
    require(snapshot.x.indptr == csr.indptr, "anndata csr indptr mismatch");
    require(snapshot.x.indices == csr.indices, "anndata csr indices mismatch");
    require(snapshot.x.data.size() == csr.data.size(), "anndata csr values mismatch");

    cs::sparse::clear(&part);
    std::remove(path);
    return 0;
}
