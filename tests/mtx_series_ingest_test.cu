#include <CellShard/ingest/mtx_series.cuh>

#include <hdf5.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#undef assert
#define assert(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "check failed: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::abort(); \
        } \
    } while (0)

namespace fs = std::filesystem;
namespace mtx_series = ::cellshard::ingest::mtx_series;

static fs::path test_root() {
    fs::path root = fs::temp_directory_path() / ("cellshard_mtx_series_" + std::to_string((long long) getpid()));
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
}

static void write_text(const fs::path &path, const std::string &text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    assert((bool) out);
    out << text;
}

static std::uint64_t read_u64_attr(hid_t file, const char *name) {
    std::uint64_t value = 0;
    hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
    assert(attr >= 0);
    assert(H5Aread(attr, H5T_NATIVE_UINT64, &value) >= 0);
    assert(H5Aclose(attr) >= 0);
    return value;
}

static std::string read_string_attr(hid_t file, const char *name) {
    char value[128]{};
    hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
    hid_t type = H5Aget_type(attr);
    assert(attr >= 0 && type >= 0);
    assert(H5Aread(attr, type, value) >= 0);
    assert(H5Tclose(type) >= 0);
    assert(H5Aclose(attr) >= 0);
    return value;
}

static std::uint64_t read_group_extent(hid_t file, const char *name) {
    hid_t group = H5Gopen2(file, name, H5P_DEFAULT);
    assert(group >= 0);
    std::uint64_t value = read_u64_attr(group, "extent");
    assert(H5Gclose(group) >= 0);
    return value;
}

int main() {
    const fs::path root = test_root();
    write_text(root / "sample_1" / "matrix.mtx",
               "%%MatrixMarket matrix coordinate integer general\n"
               "3 2 4\n"
               "1 1 5\n"
               "3 1 2\n"
               "2 2 7\n"
               "3 2 1\n");
    write_text(root / "sample_1" / "barcodes.tsv", "cellA\ncellB\n");
    write_text(root / "sample_1" / "features.tsv", "gene0\tG0\tGene Expression\n"
                                                    "gene1\tG1\tGene Expression\n"
                                                    "gene2\tG2\tGene Expression\n");
    write_text(root / "sample_2" / "matrix.mtx",
               "%%MatrixMarket matrix coordinate integer general\n"
               "3 1 2\n"
               "1 1 4\n"
               "2 1 8\n");
    write_text(root / "sample_2" / "barcodes.tsv", "cellC\n");
    write_text(root / "sample_2" / "features.tsv", "gene0\tG0\tGene Expression\n"
                                                    "gene1\tG1\tGene Expression\n"
                                                    "gene2\tG2\tGene Expression\n");
    write_text(root / "cells.csv", "cell_id,batch\ncellA,b0\ncellB,b0\ncellC,b1\n");
    write_text(root / "genes.csv", "gene_id,gene_type\n"
                                   "gene0,protein_coding\n"
                                   "gene1,protein_coding\n");

    mtx_series::options opts;
    opts.root = root.string();
    opts.output_path = (root / "dataset.csh5").string();
    opts.cache_root = (root / "cache").string();
    opts.cell_metadata_path = (root / "cells.csv").string();
    opts.observation_columns = {"batch"};
    opts.feature_metadata_path = (root / "genes.csv").string();
    opts.feature_columns = {"gene_type"};
    opts.max_part_nnz = 3u;
    opts.cpu_workers = 2;
    opts.gpu_ids.push_back(0);

    mtx_series::stats stats;
    assert(mtx_series::convert_to_optimized_sliced_ell_csh5(opts, &stats));
    assert(stats.datasets == 2u);
    assert(stats.rows == 3u);
    assert(stats.cols == 3u);
    assert(stats.nnz == 6u);
    assert(stats.partitions == 3u);
    assert(stats.row_cache_misses == 2u);

    mtx_series::stats warm_stats;
    opts.prewarm_only = true;
    assert(mtx_series::convert_to_optimized_sliced_ell_csh5(opts, &warm_stats));
    assert(warm_stats.row_cache_hits == 2u);

    hid_t file = H5Fopen(opts.output_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(file >= 0);
    assert(read_u64_attr(file, "rows") == 3u);
    assert(read_u64_attr(file, "cols") == 3u);
    assert(read_u64_attr(file, "nnz") == 6u);
    assert(read_string_attr(file, "payload_layout") == "optimized_bucketed_sliced_ell");
    assert(read_group_extent(file, "observation_metadata") == 3u);
    assert(read_group_extent(file, "feature_metadata") == 3u);
    hid_t payload = H5Gopen2(file, "/payload/sliced_ell", H5P_DEFAULT);
    H5G_info_t info{};
    assert(payload >= 0);
    assert(H5Gget_info(payload, &info) >= 0);
    assert(info.nlinks == 3u);
    assert(H5Gclose(payload) >= 0);
    assert(H5Fclose(file) >= 0);

    fs::remove_all(root);
    return 0;
}
