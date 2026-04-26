#include <CellShard/CellShard.hh>
#include <CellShard/io/cshard.hh>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace cs = ::cellshard;
namespace csc = ::cellshard::cshard;
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

bool close_float(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1.0e-3f;
}

owned_text_column make_text_column(const std::vector<const char *> &values) {
    owned_text_column out;
    std::uint32_t cursor = 0u;
    out.offsets.resize(values.size() + 1u, 0u);
    for (std::size_t i = 0u; i < values.size(); ++i) {
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

std::string temp_path(const char *suffix) {
    char path[] = "/tmp/cshard_v1_testXXXXXX";
    const int fd = ::mkstemps(path, 0);
    require(fd >= 0, "mkstemps failed");
    ::close(fd);
    std::remove(path);
    return std::string(path) + suffix;
}

csc::table_view make_table(std::uint64_t rows, const char *prefix) {
    csc::table_view table;
    csc::table_view::column index;
    table.rows = rows;
    index.name = "_index";
    index.type = csc::spec::table_column_text;
    for (std::uint64_t row = 0u; row < rows; ++row) index.text_values.push_back(std::string(prefix) + std::to_string(row));
    table.columns.push_back(std::move(index));
    return table;
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

std::string write_small_blocked_csh5() {
    const std::string path = temp_path(".csh5");
    cs::sparse::blocked_ell part;
    cs::bucketed_blocked_ell_partition bucketed_part;
    cs::bucketed_blocked_ell_shard optimized_shard;
    cs::sparse::init(&part);
    cs::init(&bucketed_part);
    cs::init(&optimized_shard);

    const owned_text_column dataset_ids = make_text_column({"dataset0"});
    const owned_text_column matrix_paths = make_text_column({"matrix.mtx"});
    const owned_text_column feature_paths = make_text_column({"features.tsv"});
    const owned_text_column barcode_paths = make_text_column({"barcodes.tsv"});
    const owned_text_column metadata_paths = make_text_column({"obs.tsv"});
    const std::uint32_t dataset_formats[] = {3u};
    const std::uint64_t dataset_row_begin[] = {0u};
    const std::uint64_t dataset_row_end[] = {2u};
    const std::uint64_t dataset_rows[] = {2u};
    const std::uint64_t dataset_cols[] = {4u};
    const std::uint64_t dataset_nnz[] = {4u};
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

    const owned_text_column global_barcodes = make_text_column({"cell0", "cell1"});
    const owned_text_column feature_ids = make_text_column({"gene0", "gene1", "gene2", "gene3"});
    const owned_text_column feature_names = make_text_column({"G0", "G1", "G2", "G3"});
    const owned_text_column feature_types = make_text_column({"gene", "gene", "gene", "gene"});
    const std::uint32_t cell_dataset_ids[] = {0u, 0u};
    const std::uint64_t cell_local_indices[] = {0u, 1u};
    const std::uint32_t feature_dataset_ids[] = {0u, 0u, 0u, 0u};
    const std::uint64_t feature_local_indices[] = {0u, 1u, 2u, 3u};
    const std::uint64_t dataset_feature_offsets[] = {0u, 4u};
    const std::uint32_t dataset_feature_to_global[] = {0u, 1u, 2u, 3u};
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

    const std::uint64_t partition_rows[] = {2u};
    const std::uint64_t partition_nnz[] = {4u};
    const std::uint64_t partition_aux[] = {(std::uint64_t) cs::sparse::pack_blocked_ell_aux(2u, 2ul)};
    const std::uint32_t partition_axes[] = {0u};
    const std::uint64_t partition_row_offsets[] = {0u, 2u};
    const std::uint32_t partition_dataset_ids[] = {0u};
    const std::uint32_t partition_codec_ids[] = {0u};
    const std::uint64_t shard_offsets[] = {0u, 2u};
    cs::dataset_codec_descriptor codec{};
    codec.codec_id = 0u;
    codec.family = cs::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
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

    fill_blocked_ell_part(&part);
    require(cs::create_dataset_blocked_ell_h5(path.c_str(), &layout, &datasets, &provenance) != 0,
            "create blocked .csh5 failed");
    require(cs::build_bucketed_blocked_ell_partition(&bucketed_part, &part, 1u, nullptr) != 0,
            "bucket blocked partition failed");
    bucketed_part.exec_to_canonical_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    bucketed_part.canonical_to_exec_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    require(bucketed_part.exec_to_canonical_cols != nullptr && bucketed_part.canonical_to_exec_cols != nullptr,
            "bucket col maps allocation failed");
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
            "optimized shard allocation failed");
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
    require(cs::append_bucketed_blocked_ell_shard_h5(path.c_str(), 0ul, &optimized_shard) != 0,
            "append optimized blocked shard failed");
    cs::clear(&optimized_shard);
    cs::sparse::clear(&part);
    return path;
}

void test_csr_fallback() {
    const std::string path = temp_path(".csr.cshard");
    cse::csr_matrix_export csr;
    std::string error;
    csr.rows = 2u;
    csr.cols = 4u;
    csr.indptr = {0, 2, 4};
    csr.indices = {0, 2, 1, 3};
    csr.data = {1.0f, 2.0f, 3.0f, 4.0f};
    require(csc::write_csr(path, csr, make_table(2u, "cell"), make_table(4u, "gene"), {}, &error), error.c_str());
    csc::cshard_file file = csc::cshard_file::open(path);
    require(file.describe().canonical_layout == "csr", "CSR fallback layout mismatch");
    const cse::csr_matrix_export rows = file.read_rows(1u, 1u);
    require(rows.indptr == std::vector<std::int64_t>({0, 2}), "CSR fallback indptr mismatch");
    require(rows.indices == std::vector<std::int64_t>({1, 3}), "CSR fallback indices mismatch");
    require(close_float(rows.data[0], 3.0f) && close_float(rows.data[1], 4.0f), "CSR fallback values mismatch");
}

void test_malformed_feature_hash() {
    const std::string path = temp_path(".bad.cshard");
    cse::csr_matrix_export csr;
    std::string error;
    csr.rows = 1u;
    csr.cols = 1u;
    csr.indptr = {0, 1};
    csr.indices = {0};
    csr.data = {1.0f};
    require(csc::write_csr(path, csr, make_table(1u, "cell"), make_table(1u, "gene"), {}, &error), error.c_str());
    {
        std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
        require((bool) out, "failed to reopen malformed cshard");
        cs::cshard::spec::header header{};
        out.read(reinterpret_cast<char *>(&header), sizeof(header));
        header.feature_order_hash = 0u;
        out.seekp(0);
        out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    }
    require(!csc::cshard_file::validate(path, &error), "zero feature hash should fail validation");
}

void test_convert_blocked_csh5() {
    const std::string h5_path = write_small_blocked_csh5();
    const std::string cshard_path = h5_path + ".cshard";
    std::string error;
    require(csc::convert_csh5_to_cshard(h5_path, cshard_path, {}, &error), error.c_str());
    csc::cshard_file file = csc::cshard_file::open(cshard_path);
    const csc::description desc = file.describe();
    require(desc.canonical_layout == "blocked_ell", "converted layout mismatch");
    require(desc.rows == 2u && desc.cols == 4u && desc.nnz == 4u, "converted shape mismatch");
    require(file.obs().head(1u).columns[0].text_values[0] == "cell0", "converted obs mismatch");
    require(file.var().head(1u).columns[0].text_values[0] == "gene0", "converted var mismatch");
    const cse::csr_matrix_export rows = file.read_rows(0u, 2u);
    require(rows.indptr == std::vector<std::int64_t>({0, 2, 4}), "converted indptr mismatch");
    require(rows.indices == std::vector<std::int64_t>({0, 2, 1, 3}), "converted indices mismatch");
    require(close_float(rows.data[0], 1.0f)
                && close_float(rows.data[1], 2.0f)
                && close_float(rows.data[2], 3.0f)
                && close_float(rows.data[3], 4.0f),
            "converted values mismatch");
}

} // namespace

int main() {
    test_csr_fallback();
    test_malformed_feature_hash();
    test_convert_blocked_csh5();
    return 0;
}
