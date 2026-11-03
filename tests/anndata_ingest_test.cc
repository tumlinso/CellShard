#include <CellShard/ingest/anndata.hh>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <exception>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

#undef assert
#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "check failed: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::abort(); \
        } \
    } while (0)

namespace anndata = cellshard::ingest::anndata_io;

static std::filesystem::path fixture(const std::filesystem::path& root, const char* name) {
    return root / name;
}

static bool has_column(const std::vector<std::string>& columns, const char* name) {
    for (const std::string& column : columns) if (column == name) return true;
    return false;
}

struct csr_probe {
    std::uint64_t entries = 0;
    unsigned row2_col3 = 0;
    bool row2_four = false;
    bool row2_five = false;
    bool row3_col2_nine = false;
};

static void inspect_csr_window(const anndata::sparse_chunk& chunk, void* context) {
    auto& probe = *static_cast<csr_probe*>(context);
    CHECK(chunk.csr);
    CHECK(chunk.col_begin == 0);
    CHECK(chunk.rows > 0);
    CHECK(chunk.offsets.size() == chunk.rows + 1);
    CHECK(chunk.values.size() == chunk.indices.size());
    CHECK(chunk.offsets.back() == chunk.values.size());

    for (std::size_t i = 0; i < chunk.rows; ++i) {
        const std::size_t row = chunk.row_begin + i;
        for (std::uint64_t k = chunk.offsets[i]; k < chunk.offsets[i + 1]; ++k) {
            const std::size_t col = chunk.col_begin + chunk.indices[k];
            ++probe.entries;
            if (row == 2 && col == 3) {
                ++probe.row2_col3;
                probe.row2_four |= chunk.values[k] == 4.0;
                probe.row2_five |= chunk.values[k] == 5.0;
            }
            if (row == 3 && col == 2 && chunk.values[k] == 9.0) probe.row3_col2_nine = true;
        }
    }
}

struct csc_probe {
    std::uint64_t entries = 0;
    unsigned row0_col0 = 0;
    bool row0_zero = false;
    bool row0_half = false;
    unsigned row24_col3 = 0;
    bool row24_three_point_five = false;
    bool row8_col1 = false;
};

static void inspect_csc_window(const anndata::sparse_chunk& chunk, void* context) {
    auto& probe = *static_cast<csc_probe*>(context);
    CHECK(!chunk.csr);
    CHECK(chunk.row_begin == 0);
    CHECK(chunk.rows == 32);
    CHECK(chunk.offsets.size() == chunk.cols + 1);
    CHECK(chunk.values.size() == chunk.indices.size());
    CHECK(chunk.offsets.back() == chunk.values.size());

    for (std::size_t j = 0; j < chunk.cols; ++j) {
        const std::size_t col = chunk.col_begin + j;
        for (std::uint64_t k = chunk.offsets[j]; k < chunk.offsets[j + 1]; ++k) {
            const std::size_t row = chunk.row_begin + chunk.indices[k];
            ++probe.entries;
            if (row == 0 && col == 0) {
                ++probe.row0_col0;
                probe.row0_zero |= chunk.values[k] == 0.0;
                probe.row0_half |= std::abs(chunk.values[k] - 0.5) < 1e-6;
            }
            if (row == 24 && col == 3) {
                ++probe.row24_col3;
                probe.row24_three_point_five |= std::abs(chunk.values[k] - 3.5) < 1e-6;
            }
            if (row == 8 && col == 1 && std::abs(chunk.values[k] - 1.08) < 1e-6) probe.row8_col1 = true;
        }
    }
}

struct small_csc_probe {
    std::uint64_t entries = 0;
    bool saw_seven = false;
    bool saw_eight = false;
};

static void inspect_counts_layer(const anndata::sparse_chunk& chunk, void* context) {
    auto& probe = *static_cast<small_csc_probe*>(context);
    CHECK(!chunk.csr);
    CHECK(chunk.row_begin == 0);
    CHECK(chunk.rows == 2);
    CHECK(chunk.values.size() == chunk.indices.size());
    for (double value : chunk.values) {
        ++probe.entries;
        probe.saw_seven |= value == 7.0;
        probe.saw_eight |= value == 8.0;
    }
}

struct raw_probe {
    std::uint64_t entries = 0;
    bool saw_twelve = false;
    bool saw_thirteen = false;
};

static void inspect_raw_window(const anndata::sparse_chunk& chunk, void* context) {
    auto& probe = *static_cast<raw_probe*>(context);
    CHECK(chunk.csr);
    CHECK(chunk.row_begin >= 1 && chunk.row_begin + chunk.rows <= 3);
    CHECK(chunk.col_begin == 0);
    for (std::size_t i = 0; i < chunk.rows; ++i) {
        for (std::uint64_t k = chunk.offsets[i]; k < chunk.offsets[i + 1]; ++k) {
            ++probe.entries;
            const std::size_t row = chunk.row_begin + i;
            const std::size_t col = chunk.indices[k];
            probe.saw_twelve |= row == 1 && col == 1 && chunk.values[k] == 12.0;
            probe.saw_thirteen |= row == 2 && col == 2 && chunk.values[k] == 13.0;
        }
    }
}

struct empty_probe {
    std::size_t chunks = 0;
};

static void inspect_empty(const anndata::sparse_chunk& chunk, void* context) {
    auto& probe = *static_cast<empty_probe*>(context);
    ++probe.chunks;
    CHECK(chunk.values.empty());
    CHECK(chunk.indices.empty());
}

static void test_csr_integer(const std::filesystem::path& root) {
    anndata::source source(fixture(root, "csr_integer.h5ad").string());
    const anndata::matrix_info info = source.info();
    CHECK(info.rows == 6 && info.cols == 5);
    CHECK(info.csr);
    CHECK(info.value_dtype == "int32");
    CHECK(source.observation_name(0) == "cell-0");
    CHECK(source.feature_name(4) == "gene-4");
    CHECK(source.feature_text("gene_id", 2) == std::optional<std::string>("ENSG000002"));
    CHECK(source.feature_text("feature_name", 2) == std::optional<std::string>("feature-name-2"));
    CHECK(source.feature_text("feature_type", 2) == std::optional<std::string>("Gene Expression"));

    const auto obs_columns = source.observation_columns();
    CHECK(has_column(obs_columns, "condition"));
    CHECK(has_column(obs_columns, "label"));
    CHECK(has_column(obs_columns, "batch"));
    CHECK(source.observation_text("condition", 2) == std::optional<std::string>("treated"));
    CHECK(!source.observation_text("label", 1).has_value());
    CHECK(source.observation_text("label", 2) == std::optional<std::string>(""));

    std::vector<std::uint64_t> row_nnz(info.rows);
    const anndata::read_stats counts = source.row_nnz(std::span<std::uint64_t>(row_nnz));
    CHECK((row_nnz == std::vector<std::uint64_t>{3, 0, 3, 1, 0, 1}));
    CHECK(counts.nnz == 8);

    csr_probe probe;
    const anndata::read_stats selected = source.visit_rows(1, 4, inspect_csr_window, &probe);
    CHECK(selected.nnz == 4);
    CHECK(probe.entries == 4);
    CHECK(probe.row2_col3 == 2);
    CHECK(probe.row2_four && probe.row2_five);
    CHECK(probe.row3_col2_nine);
}

static void test_csc_float(const std::filesystem::path& root) {
    constexpr std::size_t budget = 1024;
    anndata::source source(fixture(root, "csc_float.h5ad").string(), {}, budget);
    const anndata::matrix_info info = source.info();
    CHECK(info.rows == 64 && info.cols == 4);
    CHECK(!info.csr);
    CHECK(info.value_dtype == "float32");
    CHECK(source.observation_name(63) == "wide-cell-63");
    CHECK(source.feature_name(3) == "wide-gene-3");
    CHECK(source.observation_text("condition", 40) == std::optional<std::string>("right"));

    std::vector<std::uint64_t> row_nnz(info.rows);
    const anndata::read_stats counts = source.row_nnz(std::span<std::uint64_t>(row_nnz));
    CHECK(counts.nnz == 160);
    CHECK(counts.peak_buffer_bytes <= budget);
    for (std::size_t row = 0; row < info.rows; ++row) {
        const std::uint64_t expected =
            (row <= 38 ? 1u : 0u) + (row >= 8 && row <= 47 ? 1u : 0u) +
            (row >= 16 && row <= 55 ? 1u : 0u) + (row >= 24 && row <= 62 ? 1u : 0u) +
            (row == 0 ? 1u : 0u) + (row == 24 ? 1u : 0u);
        CHECK(row_nnz[row] == expected);
    }

    csc_probe probe;
    const anndata::read_stats selected = source.visit_rows(0, 32, inspect_csc_window, &probe);
    CHECK(selected.nnz == 82);
    CHECK(selected.peak_buffer_bytes <= budget);
    CHECK(selected.batches > 0 && selected.batches <= info.cols);
    CHECK(selected.batches < 32); // CSC work batches by source columns, not one batch per row.
    CHECK(probe.entries == 82);
    CHECK(probe.row0_col0 == 2 && probe.row0_zero && probe.row0_half);
    CHECK(probe.row24_col3 == 2 && probe.row24_three_point_five);
    CHECK(probe.row8_col1);
}

static void test_layer_and_raw(const std::filesystem::path& root) {
    const std::filesystem::path path = fixture(root, "layers_raw.h5ad");
    anndata::source counts(path.string(), {anndata::matrix_slot::layer, "counts"});
    const anndata::matrix_info layer_info = counts.info();
    CHECK(layer_info.rows == 4 && layer_info.cols == 4);
    CHECK(!layer_info.csr);
    CHECK(layer_info.value_dtype == "int32");
    CHECK(counts.observation_name(1) == "sample-b");
    CHECK(counts.feature_name(3) == "x3");
    CHECK(counts.feature_text("gene_id", 3) == std::optional<std::string>("X3"));
    std::vector<std::uint64_t> layer_nnz(layer_info.rows);
    CHECK(counts.row_nnz(std::span<std::uint64_t>(layer_nnz)).nnz == 4);
    CHECK((layer_nnz == std::vector<std::uint64_t>{1, 1, 1, 1}));
    small_csc_probe layer_probe;
    const anndata::read_stats layer_window = counts.visit_rows(0, 2, inspect_counts_layer, &layer_probe);
    CHECK(layer_window.nnz == 2);
    CHECK(layer_probe.entries == 2 && layer_probe.saw_seven && layer_probe.saw_eight);

    anndata::source raw(path.string(), {anndata::matrix_slot::raw, {}});
    const anndata::matrix_info raw_info = raw.info();
    CHECK(raw_info.rows == 4 && raw_info.cols == 3);
    CHECK(raw_info.csr);
    CHECK(raw_info.value_dtype == "int16");
    CHECK(raw.observation_name(0) == "sample-a");
    CHECK(raw.feature_name(0) == "raw-feature-a");
    CHECK(raw.feature_text("gene_id", 2) == std::optional<std::string>("R-C"));
    std::vector<std::uint64_t> raw_nnz(raw_info.rows);
    CHECK(raw.row_nnz(std::span<std::uint64_t>(raw_nnz)).nnz == 4);
    CHECK((raw_nnz == std::vector<std::uint64_t>{1, 1, 1, 1}));
    raw_probe probe;
    const anndata::read_stats raw_window = raw.visit_rows(1, 3, inspect_raw_window, &probe);
    CHECK(raw_window.nnz == 2);
    CHECK(probe.entries == 2 && probe.saw_twelve && probe.saw_thirteen);
}

static void test_empty_matrix(const std::filesystem::path& root) {
    anndata::source source(fixture(root, "empty.h5ad").string());
    const anndata::matrix_info info = source.info();
    CHECK(info.rows == 3 && info.cols == 2);
    CHECK(info.csr);
    CHECK(info.value_dtype == "int32");
    CHECK(source.observation_name(2) == "empty-cell-c");
    CHECK(source.feature_name(1) == "empty-gene-b");
    std::vector<std::uint64_t> row_nnz(info.rows, 99);
    const anndata::read_stats counts = source.row_nnz(std::span<std::uint64_t>(row_nnz));
    CHECK(counts.nnz == 0);
    CHECK((row_nnz == std::vector<std::uint64_t>{0, 0, 0}));
    empty_probe probe;
    const anndata::read_stats read = source.visit_rows(0, info.rows, inspect_empty, &probe);
    CHECK(read.nnz == 0);
    CHECK(read.peak_buffer_bytes == 0);
    CHECK(probe.chunks == 0);
}

struct dtype_probe {
    std::vector<double> values;
};

static void collect_dtype_value(const anndata::sparse_chunk& chunk, void* context) {
    auto& probe = *static_cast<dtype_probe*>(context);
    CHECK(chunk.csr);
    CHECK(chunk.row_begin == 0 && chunk.rows == 1);
    for (double value : chunk.values) probe.values.push_back(value);
}

static void test_dtype_conversions(const std::filesystem::path& root) {
    struct test_case {
        anndata::matrix_slot slot;
        const char* layer;
        const char* dtype;
        double first_value;
    };
    const std::array<test_case, 11> cases{{
        {anndata::matrix_slot::x, "", "bool", 1.0},
        {anndata::matrix_slot::layer, "int8", "int8", -3.0},
        {anndata::matrix_slot::layer, "uint8", "uint8", 3.0},
        {anndata::matrix_slot::layer, "int16", "int16", -3.0},
        {anndata::matrix_slot::layer, "uint16", "uint16", 3.0},
        {anndata::matrix_slot::layer, "int32", "int32", -3.0},
        {anndata::matrix_slot::layer, "uint32", "uint32", 3.0},
        {anndata::matrix_slot::layer, "int64", "int64", -3.0},
        {anndata::matrix_slot::layer, "uint64", "uint64", 3.0},
        {anndata::matrix_slot::layer, "float32", "float32", 1.25},
        {anndata::matrix_slot::layer, "float64", "float64", 1.25},
    }};
    const std::string path = fixture(root, "dtype_branches.h5ad").string();
    for (const test_case& item : cases) {
        anndata::source source(path, {item.slot, item.layer});
        const anndata::matrix_info info = source.info();
        CHECK(info.rows == 2 && info.cols == 2);
        CHECK(info.csr);
        CHECK(info.value_dtype == item.dtype);

        std::vector<std::uint64_t> row_nnz(info.rows);
        CHECK(source.row_nnz(std::span<std::uint64_t>(row_nnz)).nnz == 2);
        CHECK((row_nnz == std::vector<std::uint64_t>{1, 1}));
        dtype_probe probe;
        const anndata::read_stats selected = source.visit_rows(0, 1, collect_dtype_value, &probe);
        CHECK(selected.nnz == 1);
        CHECK(probe.values.size() == 1);
        CHECK(probe.values[0] == item.first_value);
    }
}

static void test_rejects(const std::filesystem::path& root, const char* name) {
    bool rejected = false;
    try {
        anndata::source source(fixture(root, name).string());
        (void)source.info();
    } catch (const std::exception&) {
        rejected = true;
    }
    CHECK(rejected);
}

int main(int argc, char** argv) {
    CHECK(argc == 2);
    const std::filesystem::path root(argv[1]);
    CHECK(std::filesystem::is_directory(root));

    test_csr_integer(root);
    test_csc_float(root);
    test_layer_and_raw(root);
    test_empty_matrix(root);
    test_dtype_conversions(root);
    test_rejects(root, "legacy_unversioned.h5ad");
    test_rejects(root, "modern_unversioned.h5ad");
    test_rejects(root, "unsupported_version.h5ad");
    test_rejects(root, "unsupported_encoding.h5ad");
    test_rejects(root, "dense.h5ad");
    return 0;
}
