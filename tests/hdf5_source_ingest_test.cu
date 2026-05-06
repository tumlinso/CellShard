#include <CellShard/ingest/dataset_ingest.cuh>

#include <hdf5.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#undef assert
#define assert(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "check failed: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::abort(); \
        } \
    } while (0)

namespace dataset = cellshard::ingest::dataset;
namespace tenx_h5 = cellshard::ingest::tenx_h5;
namespace h5ad = cellshard::ingest::h5ad;
namespace loom = cellshard::ingest::loom;
namespace mtx = cellshard::ingest::mtx;
namespace common = cellshard::ingest::common;
namespace sparse = cellshard::sparse;

static bool close_ok(herr_t rc) {
    return rc >= 0;
}

static void write_u64_dataset(hid_t parent, const char *name, const std::vector<std::uint64_t> &values) {
    hsize_t dims[1] = {(hsize_t) values.size()};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate2(parent, name, H5T_NATIVE_UINT64, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(space >= 0 && dset >= 0);
    if (!values.empty()) assert(H5Dwrite(dset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
    assert(close_ok(H5Dclose(dset)));
    assert(close_ok(H5Sclose(space)));
}

static void write_double_dataset(hid_t parent, const char *name, const std::vector<double> &values) {
    hsize_t dims[1] = {(hsize_t) values.size()};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate2(parent, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(space >= 0 && dset >= 0);
    if (!values.empty()) assert(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
    assert(close_ok(H5Dclose(dset)));
    assert(close_ok(H5Sclose(space)));
}

static void write_dense_dataset(hid_t parent,
                                const char *name,
                                hsize_t rows,
                                hsize_t cols,
                                const std::vector<double> &values) {
    hsize_t dims[2] = {rows, cols};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dset = H5Dcreate2(parent, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(space >= 0 && dset >= 0);
    assert(values.size() == (std::size_t) rows * (std::size_t) cols);
    if (!values.empty()) assert(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
    assert(close_ok(H5Dclose(dset)));
    assert(close_ok(H5Sclose(space)));
}

static void write_string_dataset(hid_t parent, const char *name, const std::vector<std::string> &values) {
    std::size_t width = 1u;
    for (const std::string &value : values) width = std::max(width, value.size() + 1u);
    std::vector<char> packed(values.size() * width, '\0');
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::memcpy(packed.data() + i * width, values[i].c_str(), values[i].size());
    }
    hsize_t dims[1] = {(hsize_t) values.size()};
    hid_t type = H5Tcopy(H5T_C_S1);
    hid_t space = H5Screate_simple(1, dims, nullptr);
    assert(type >= 0 && space >= 0);
    assert(H5Tset_size(type, width) >= 0);
    hid_t dset = H5Dcreate2(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    if (!values.empty()) assert(H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, packed.data()) >= 0);
    assert(close_ok(H5Dclose(dset)));
    assert(close_ok(H5Sclose(space)));
    assert(close_ok(H5Tclose(type)));
}

static void write_string_attr(hid_t parent, const char *name, const char *value) {
    hid_t type = H5Tcopy(H5T_C_S1);
    hid_t space = H5Screate(H5S_SCALAR);
    assert(type >= 0 && space >= 0);
    assert(H5Tset_size(type, std::strlen(value) + 1u) >= 0);
    hid_t attr = H5Acreate2(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    assert(attr >= 0);
    assert(H5Awrite(attr, type, value) >= 0);
    assert(close_ok(H5Aclose(attr)));
    assert(close_ok(H5Sclose(space)));
    assert(close_ok(H5Tclose(type)));
}

static void write_u64_vector_attr(hid_t parent, const char *name, const std::vector<std::uint64_t> &values) {
    hsize_t dims[1] = {(hsize_t) values.size()};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t attr = H5Acreate2(parent, name, H5T_NATIVE_UINT64, space, H5P_DEFAULT, H5P_DEFAULT);
    assert(space >= 0 && attr >= 0);
    if (!values.empty()) assert(H5Awrite(attr, H5T_NATIVE_UINT64, values.data()) >= 0);
    assert(close_ok(H5Aclose(attr)));
    assert(close_ok(H5Sclose(space)));
}

static void write_legacy_axis_dataset(hid_t parent,
                                      const char *name,
                                      const std::vector<std::string> &index,
                                      const std::vector<std::string> &gene_ids = {}) {
    std::size_t index_width = 1u;
    std::size_t gene_width = 1u;
    std::size_t row_width = 0u;
    for (const std::string &value : index) index_width = std::max(index_width, value.size() + 1u);
    for (const std::string &value : gene_ids) gene_width = std::max(gene_width, value.size() + 1u);

    hid_t index_type = H5Tcopy(H5T_C_S1);
    hid_t gene_type = H5Tcopy(H5T_C_S1);
    assert(index_type >= 0 && gene_type >= 0);
    assert(H5Tset_size(index_type, index_width) >= 0);
    assert(H5Tset_size(gene_type, gene_width) >= 0);
    row_width = index_width + (gene_ids.empty() ? 0u : gene_width);
    hid_t compound = H5Tcreate(H5T_COMPOUND, row_width);
    assert(compound >= 0);
    assert(H5Tinsert(compound, "index", 0u, index_type) >= 0);
    if (!gene_ids.empty()) assert(H5Tinsert(compound, "gene_ids", index_width, gene_type) >= 0);

    std::vector<char> rows(index.size() * row_width, '\0');
    for (std::size_t i = 0; i < index.size(); ++i) {
        std::memcpy(rows.data() + i * row_width, index[i].c_str(), index[i].size());
        if (!gene_ids.empty()) std::memcpy(rows.data() + i * row_width + index_width, gene_ids[i].c_str(), gene_ids[i].size());
    }
    hsize_t dims[1] = {(hsize_t) index.size()};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate2(parent, name, compound, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(space >= 0 && dset >= 0);
    if (!rows.empty()) assert(H5Dwrite(dset, compound, H5S_ALL, H5S_ALL, H5P_DEFAULT, rows.data()) >= 0);
    assert(close_ok(H5Dclose(dset)));
    assert(close_ok(H5Sclose(space)));
    assert(close_ok(H5Tclose(compound)));
    assert(close_ok(H5Tclose(gene_type)));
    assert(close_ok(H5Tclose(index_type)));
}

static std::string make_path(const char *name) {
    std::filesystem::path root = std::filesystem::temp_directory_path()
        / ("cellshard_hdf5_ingest_" + std::to_string((long long) getpid()));
    std::filesystem::create_directories(root);
    return (root / name).string();
}

static void create_tenx(const std::string &path, bool bad_index = false, bool missing_matrix = false) {
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    assert(file >= 0);
    if (!missing_matrix) {
        hid_t matrix = H5Gcreate2(file, "/matrix", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t features = H5Gcreate2(file, "/matrix/features", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(matrix >= 0 && features >= 0);
        write_double_dataset(matrix, "data", {1.0, 2.0, 3.0});
        write_u64_dataset(matrix, "indices", bad_index ? std::vector<std::uint64_t>{0u, 3u, 1u} : std::vector<std::uint64_t>{0u, 2u, 1u});
        write_u64_dataset(matrix, "indptr", {0u, 2u, 3u});
        write_u64_dataset(matrix, "shape", {3u, 2u});
        write_string_dataset(matrix, "barcodes", {"cellA", "cellB"});
        write_string_dataset(features, "id", {"gene0", "gene1", "gene2"});
        write_string_dataset(features, "name", {"G0", "G1", "G2"});
        write_string_dataset(features, "feature_type", {"Gene Expression", "Gene Expression", "Gene Expression"});
        assert(close_ok(H5Gclose(features)));
        assert(close_ok(H5Gclose(matrix)));
    }
    assert(close_ok(H5Fclose(file)));
}

static void create_legacy_h5ad(const std::string &path) {
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t x = H5Gcreate2(file, "/X", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(file >= 0 && x >= 0);
    write_string_attr(x, "h5sparse_format", "csr");
    write_u64_vector_attr(x, "h5sparse_shape", {2u, 3u});
    write_double_dataset(x, "data", {5.0, 2.0, 7.0});
    write_u64_dataset(x, "indices", {0u, 2u, 1u});
    write_u64_dataset(x, "indptr", {0u, 2u, 3u});
    write_legacy_axis_dataset(file, "/obs", {"cellA", "cellB"});
    write_legacy_axis_dataset(file, "/var", {"G0", "G1", "G2"}, {"gene0", "gene1", "gene2"});
    assert(close_ok(H5Gclose(x)));
    assert(close_ok(H5Fclose(file)));
}

static void create_loom(const std::string &path, bool processed = false, bool with_layer = false) {
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t row_attrs = H5Gcreate2(file, "/row_attrs", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t col_attrs = H5Gcreate2(file, "/col_attrs", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(file >= 0 && row_attrs >= 0 && col_attrs >= 0);
    if (with_layer) {
        write_dense_dataset(file, "matrix", 2u, 2u, {7.0, 0.0, 0.0, 8.0});
    } else {
        write_dense_dataset(file, "matrix", 3u, 2u, processed
            ? std::vector<double>{0.5, 0.0, 0.0, 1.0, 2.0, 0.0}
            : std::vector<double>{1.0, 0.0, 0.0, 4.0, 2.0, 5.0});
    }
    if (with_layer) {
        hid_t layers = H5Gcreate2(file, "/layers", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(layers >= 0);
        write_dense_dataset(layers, "counts", 2u, 2u, {7.0, 0.0, 0.0, 8.0});
        assert(close_ok(H5Gclose(layers)));
    }
    write_string_dataset(col_attrs, "CellID", {"cellA", "cellB"});
    write_string_dataset(row_attrs, "Accession", with_layer ? std::vector<std::string>{"gene0", "gene1"} : std::vector<std::string>{"gene0", "gene1", "gene2"});
    write_string_dataset(row_attrs, "Name", with_layer ? std::vector<std::string>{"G0", "G1"} : std::vector<std::string>{"G0", "G1", "G2"});
    assert(close_ok(H5Gclose(row_attrs)));
    assert(close_ok(H5Gclose(col_attrs)));
    assert(close_ok(H5Fclose(file)));
}

static void test_tenx_reader() {
    const std::string path = make_path("tenx.h5");
    create_tenx(path);

    mtx::header header;
    unsigned long *row_nnz = nullptr;
    std::string error;
    assert(tenx_h5::scan_row_nnz(path.c_str(), &header, &row_nnz, &error));
    assert(header.rows == 2ul);
    assert(header.cols == 3ul);
    assert(header.nnz_file == 3ul);
    assert(row_nnz[0] == 2ul && row_nnz[1] == 1ul);

    common::barcode_table barcodes;
    common::feature_table features;
    common::init(&barcodes);
    common::init(&features);
    assert(tenx_h5::load_barcodes(path.c_str(), &barcodes, &error));
    assert(tenx_h5::load_feature_table(path.c_str(), &features, &error));
    assert(common::count(&barcodes) == 2u);
    assert(std::string(common::get(&barcodes, 0u)) == "cellA");
    assert(common::count(&features) == 3u);
    assert(std::string(common::id(&features, 2u)) == "gene2");

    unsigned long row_offsets[3] = {0ul, 1ul, 2ul};
    unsigned long part_nnz[2] = {2ul, 1ul};
    cellshard::sharded<sparse::compressed> compressed;
    cellshard::init(&compressed);
    assert(tenx_h5::load_part_window_compressed(path.c_str(), &header, row_offsets, part_nnz, 2ul, 0ul, 2ul, &compressed, &error));
    assert(compressed.num_partitions == 2ul);
    assert(compressed.parts[0]->minorIdx[0] == 0u);
    assert(compressed.parts[0]->minorIdx[1] == 2u);
    assert(std::fabs(__half2float(compressed.parts[0]->val[1]) - 2.0f) < 0.01f);
    assert(compressed.parts[1]->minorIdx[0] == 1u);
    assert(std::fabs(__half2float(compressed.parts[1]->val[0]) - 3.0f) < 0.01f);

    cellshard::clear(&compressed);
    common::clear(&barcodes);
    common::clear(&features);
    std::free(row_nnz);

    const std::string bad_index = make_path("tenx_bad_index.h5");
    create_tenx(bad_index, true, false);
    assert(!tenx_h5::scan_row_nnz(bad_index.c_str(), &header, &row_nnz, &error));
    const std::string missing = make_path("tenx_missing.h5");
    create_tenx(missing, false, true);
    assert(!tenx_h5::scan_row_nnz(missing.c_str(), &header, &row_nnz, &error));
}

static void test_loom_reader() {
    const std::string path = make_path("loom.loom");
    create_loom(path);

    mtx::header header;
    unsigned long *row_nnz = nullptr;
    std::string error;
    assert(loom::scan_row_nnz(path.c_str(), "matrix", false, &header, &row_nnz, &error));
    assert(header.rows == 2ul);
    assert(header.cols == 3ul);
    assert(header.nnz_file == 4ul);
    assert(row_nnz[0] == 2ul && row_nnz[1] == 2ul);

    common::barcode_table barcodes;
    common::feature_table features;
    common::init(&barcodes);
    common::init(&features);
    assert(loom::load_barcodes(path.c_str(), "matrix", &barcodes, &error));
    assert(loom::load_feature_table(path.c_str(), "matrix", &features, &error));
    assert(std::string(common::get(&barcodes, 1u)) == "cellB");
    assert(std::string(common::name(&features, 2u)) == "G2");
    assert(std::string(common::type(&features, 0u)) == "gene");

    unsigned long row_offsets[2] = {0ul, 2ul};
    unsigned long part_nnz[1] = {4ul};
    cellshard::sharded<sparse::coo> coo;
    cellshard::init(&coo);
    assert(loom::load_part_window_coo(path.c_str(), "matrix", false, &header, row_offsets, part_nnz, 1ul, 0ul, 1ul, &coo, &error));
    assert(coo.num_partitions == 1ul);
    assert(sparse::at(coo.parts[0], 0u, 0u) != nullptr);
    assert(sparse::at(coo.parts[0], 1u, 1u) != nullptr);
    assert(sparse::at(coo.parts[0], 0u, 2u) != nullptr);
    assert(sparse::at(coo.parts[0], 1u, 2u) != nullptr);
    assert(std::fabs(__half2float(*sparse::at(coo.parts[0], 1u, 2u)) - 5.0f) < 0.01f);
    cellshard::clear(&coo);
    common::clear(&barcodes);
    common::clear(&features);
    std::free(row_nnz);

    const std::string layer_path = make_path("loom_layer.loom");
    create_loom(layer_path, false, true);
    assert(loom::scan_row_nnz(layer_path.c_str(), "layer:counts", false, &header, &row_nnz, &error));
    assert(header.rows == 2ul);
    assert(header.cols == 2ul);
    assert(header.nnz_file == 2ul);
    assert(row_nnz[0] == 1ul && row_nnz[1] == 1ul);
    std::free(row_nnz);

    const std::string processed_path = make_path("loom_processed.loom");
    create_loom(processed_path, true, false);
    assert(!loom::scan_row_nnz(processed_path.c_str(), "matrix", false, &header, &row_nnz, &error));
    assert(loom::scan_row_nnz(processed_path.c_str(), "matrix", true, &header, &row_nnz, &error));
    std::free(row_nnz);
}

static void test_legacy_h5ad_reader() {
    const std::string path = make_path("legacy_sparse.h5ad");
    create_legacy_h5ad(path);

    mtx::header header;
    unsigned long *row_nnz = nullptr;
    std::string error;
    assert(h5ad::scan_row_nnz(path.c_str(), "x", &header, &row_nnz, &error));
    assert(header.rows == 2ul);
    assert(header.cols == 3ul);
    assert(header.nnz_file == 3ul);
    assert(header.row_sorted == 1);
    assert(row_nnz[0] == 2ul && row_nnz[1] == 1ul);

    common::barcode_table barcodes;
    common::feature_table features;
    common::init(&barcodes);
    common::init(&features);
    assert(h5ad::load_barcodes(path.c_str(), &barcodes, &error));
    assert(common::count(&barcodes) == 2u);
    assert(std::string(common::get(&barcodes, 0u)) == "cellA");
    assert(h5ad::load_feature_table(path.c_str(), "x", &features, &error));
    assert(common::count(&features) == 3u);
    assert(std::string(common::id(&features, 2u)) == "G2");
    assert(std::string(common::name(&features, 1u)) == "G1");
    assert(std::string(common::type(&features, 0u)) == "gene");

    unsigned long row_offsets[3] = {0ul, 1ul, 2ul};
    unsigned long part_nnz[2] = {2ul, 1ul};
    cellshard::sharded<sparse::compressed> compressed;
    cellshard::init(&compressed);
    assert(h5ad::load_part_window_compressed(path.c_str(), "x", &header, row_offsets, part_nnz, 2ul, 0ul, 2ul, &compressed, &error));
    assert(compressed.num_partitions == 2ul);
    assert(compressed.parts[0]->minorIdx[0] == 0u);
    assert(compressed.parts[0]->minorIdx[1] == 2u);
    assert(std::fabs(__half2float(compressed.parts[1]->val[0]) - 7.0f) < 0.01f);
    cellshard::clear(&compressed);
    common::clear(&barcodes);
    common::clear(&features);
    std::free(row_nnz);
}

int main() {
    test_tenx_reader();
    test_legacy_h5ad_reader();
    test_loom_reader();
    return 0;
}
