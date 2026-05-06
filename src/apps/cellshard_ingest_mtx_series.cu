#include <CellShard/ingest/mtx_series.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingest = ::cellshard::ingest::mtx_series;

namespace {

std::string require_option(int argc, char **argv, const char *name) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == name) return argv[i + 1];
    }
    throw std::runtime_error(std::string("missing required option ") + name);
}

std::vector<std::string> split_commas(const std::string &s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::vector<int> split_gpu_ids(const std::string &s) {
    std::vector<int> out;
    if (s == "auto" || s == "all") return out;
    for (const std::string &part : split_commas(s)) out.push_back(std::atoi(part.c_str()));
    return out;
}

std::uint64_t parse_u64(const std::string &s, const char *name) {
    char *end = nullptr;
    const unsigned long long value = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') throw std::runtime_error(std::string("invalid integer for ") + name);
    return (std::uint64_t) value;
}

int parse_workers(const std::string &s) {
    if (s == "auto") return 0;
    return (int) parse_u64(s, "--cpu-workers");
}

void usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  cellshard ingest mtx-series (--manifest path.tsv | --root dir) --out dataset.csh5 [options]\n"
                 "\n"
                 "options:\n"
                 "  --source-subdir NAME          optional per-dataset subdirectory under --root\n"
                 "  --cache-root DIR              row-count and runtime cache root\n"
                 "  --prewarm-only                populate row-count caches and exit\n"
                 "  --cpu-workers auto|N          default: auto\n"
                 "  --gpus auto|all|0,1           default: auto/all visible\n"
                 "  --dataset-ids a,b,c           limit root or manifest series\n"
                 "  --cell-metadata file.csv      observation metadata CSV\n"
                 "  --cell-id-column NAME         default: cell_id\n"
                 "  --obs-columns a,b,c           metadata columns to preserve\n"
                 "  --obs-prefilter-column NAME   enable subset metadata line prefiltering\n"
                 "  --feature-metadata file.csv   feature metadata CSV\n"
                 "  --feature-id-column NAME      default: gene_id\n"
                 "  --feature-columns a,b,c       feature metadata columns to preserve\n"
                 "  --max-part-nnz N              default: 67108864\n"
                 "  --slice-rows N                default: 64\n"
                 "  --target-shard-bytes N        default: 1073741824\n"
                 "  --tmp-out path                explicit temporary output path\n");
}

} // namespace

int cellshard_ingest_mtx_series_main(int argc, char **argv) {
    try {
        ingest::options opts;
        bool have_manifest = false, have_root = false;
        for (int i = 0; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--manifest" && i + 1 < argc) {
                opts.manifest_path = argv[++i];
                have_manifest = true;
            } else if (arg == "--root" && i + 1 < argc) {
                opts.root = argv[++i];
                have_root = true;
            } else if (arg == "--out" && i + 1 < argc) {
                opts.output_path = argv[++i];
            } else if (arg == "--tmp-out" && i + 1 < argc) {
                opts.tmp_output_path = argv[++i];
            } else if (arg == "--source-subdir" && i + 1 < argc) {
                opts.source_subdir = argv[++i];
            } else if (arg == "--matrix-filename" && i + 1 < argc) {
                opts.matrix_filename = argv[++i];
            } else if (arg == "--barcode-filename" && i + 1 < argc) {
                opts.barcode_filename = argv[++i];
            } else if (arg == "--feature-filename" && i + 1 < argc) {
                opts.feature_filename = argv[++i];
            } else if (arg == "--cache-root" && i + 1 < argc) {
                opts.cache_root = argv[++i];
            } else if (arg == "--dataset-ids" && i + 1 < argc) {
                opts.dataset_ids = split_commas(argv[++i]);
            } else if (arg == "--cell-metadata" && i + 1 < argc) {
                opts.cell_metadata_path = argv[++i];
            } else if (arg == "--cell-id-column" && i + 1 < argc) {
                opts.cell_id_column = argv[++i];
            } else if (arg == "--obs-columns" && i + 1 < argc) {
                opts.observation_columns = split_commas(argv[++i]);
            } else if (arg == "--obs-prefilter-column" && i + 1 < argc) {
                opts.observation_prefilter_column = argv[++i];
            } else if (arg == "--feature-metadata" && i + 1 < argc) {
                opts.feature_metadata_path = argv[++i];
            } else if (arg == "--feature-id-column" && i + 1 < argc) {
                opts.feature_id_column = argv[++i];
            } else if (arg == "--feature-columns" && i + 1 < argc) {
                opts.feature_columns = split_commas(argv[++i]);
            } else if (arg == "--max-part-nnz" && i + 1 < argc) {
                opts.max_part_nnz = parse_u64(argv[++i], "--max-part-nnz");
            } else if (arg == "--slice-rows" && i + 1 < argc) {
                opts.slice_rows = (std::uint32_t) parse_u64(argv[++i], "--slice-rows");
            } else if (arg == "--target-shard-bytes" && i + 1 < argc) {
                opts.target_shard_bytes = parse_u64(argv[++i], "--target-shard-bytes");
            } else if (arg == "--cpu-workers" && i + 1 < argc) {
                opts.cpu_workers = parse_workers(argv[++i]);
            } else if (arg == "--gpus" && i + 1 < argc) {
                opts.gpu_ids = split_gpu_ids(argv[++i]);
            } else if (arg == "--prewarm-only") {
                opts.prewarm_only = true;
            } else if (arg == "--allow-missing-metadata") {
                opts.allow_missing_metadata = true;
            } else if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            } else {
                throw std::runtime_error("unknown or incomplete ingest option: " + arg);
            }
        }
        if ((!have_manifest && !have_root) || (have_manifest && have_root)) {
            throw std::runtime_error("provide exactly one of --manifest or --root");
        }
        if (opts.output_path.empty() && !opts.prewarm_only) {
            (void) require_option(argc, argv, "--out");
        }
        if (!opts.cell_metadata_path.empty() && opts.observation_columns.empty()) {
            opts.observation_columns = {"day", "embryo_id", "experimental_batch", "major_trajectory", "celltype_update"};
        }
        if (!opts.feature_metadata_path.empty() && opts.feature_columns.empty()) {
            opts.feature_columns = {"gene_type", "gene_short_name", "chr"};
        }
        ingest::stats stats;
        if (!ingest::convert_to_optimized_sliced_ell_csh5(opts, &stats)) return 1;
        std::printf("datasets: %llu\n", (unsigned long long) stats.datasets);
        std::printf("shape: %llu x %llu\n", (unsigned long long) stats.rows, (unsigned long long) stats.cols);
        std::printf("nnz: %llu\n", (unsigned long long) stats.nnz);
        std::printf("partitions: %llu\n", (unsigned long long) stats.partitions);
        std::printf("row_cache_hits: %llu\n", (unsigned long long) stats.row_cache_hits);
        std::printf("row_cache_misses: %llu\n", (unsigned long long) stats.row_cache_misses);
        std::printf("total_seconds: %.6f\n", stats.total_seconds);
        return 0;
    } catch (const std::exception &exc) {
        std::fprintf(stderr, "cellshard ingest mtx-series: %s\n", exc.what());
        return 1;
    }
}
