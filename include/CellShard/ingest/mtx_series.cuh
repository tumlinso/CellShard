#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellshard {
namespace ingest {
namespace mtx_series {

enum class matrix_orientation : std::uint32_t {
    features_by_cells = 0u
};

struct options {
    std::string root;
    std::string manifest_path;
    std::string output_path;
    std::string tmp_output_path;
    std::string cache_root;
    std::string working_root;

    std::string source_subdir;
    std::string matrix_filename = "matrix.mtx";
    std::string barcode_filename = "barcodes.tsv";
    std::string feature_filename = "features.tsv";
    std::vector<std::string> dataset_ids;

    std::string cell_metadata_path;
    std::string cell_id_column = "cell_id";
    std::vector<std::string> observation_columns;
    std::string observation_prefilter_column;

    std::string feature_metadata_path;
    std::string feature_id_column = "gene_id";
    std::vector<std::string> feature_columns;

    matrix_orientation orientation = matrix_orientation::features_by_cells;
    std::uint64_t max_part_nnz = 1ull << 26u;
    std::uint32_t slice_rows = 64u;
    std::uint64_t target_shard_bytes = 1ull << 30u;
    std::size_t reader_bytes = (std::size_t) 8u << 20u;
    int cpu_workers = 0; // 0 means auto.
    std::vector<int> gpu_ids; // empty means all visible devices.
    bool prewarm_only = false;
    bool atomic_output = true;
    bool allow_missing_metadata = false;
};

struct stats {
    std::uint64_t datasets = 0u;
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t partitions = 0u;
    std::uint64_t shards = 0u;
    std::uint64_t row_cache_hits = 0u;
    std::uint64_t row_cache_misses = 0u;
    std::uint64_t canonical_sliced_bytes = 0u;
    std::uint64_t bucketed_sliced_bytes = 0u;
    double planning_seconds = 0.0;
    double metadata_seconds = 0.0;
    double payload_seconds = 0.0;
    double total_seconds = 0.0;
};

int convert_to_optimized_sliced_ell_csh5(const options &opts, stats *out_stats = nullptr);

} // namespace mtx_series
} // namespace ingest
} // namespace cellshard
