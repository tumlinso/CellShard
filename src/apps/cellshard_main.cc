#include <CellShard/io/cshard.hh>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

namespace csc = ::cellshard::cshard;

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  cellshard cshard inspect <dataset.cshard>\n"
                 "  cellshard cshard validate <dataset.cshard>\n"
                 "  cellshard cshard read-rows <dataset.cshard> --start N --count N\n"
                 "  cellshard cshard convert <input.csh5> --out <output.cshard>\n");
}

bool parse_u64(const char *text, std::uint64_t *out) {
    char *end = nullptr;
    unsigned long long value = 0u;
    if (text == nullptr || out == nullptr) return false;
    value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    *out = (std::uint64_t) value;
    return true;
}

std::string require_option(int argc, char **argv, const char *name) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == name) return argv[i + 1];
    }
    throw std::runtime_error(std::string("missing required option ") + name);
}

int cshard_main(int argc, char **argv) {
    if (argc < 3) {
        usage();
        return 2;
    }
    const std::string command = argv[1];
    if (command == "inspect") {
        csc::cshard_file file = csc::cshard_file::open(argv[2]);
        const csc::description desc = file.describe();
        std::printf("path: %s\n", desc.path.c_str());
        std::printf("version: %u.%u\n", desc.version_major, desc.version_minor);
        std::printf("shape: %llu x %llu\n", (unsigned long long) desc.rows, (unsigned long long) desc.cols);
        std::printf("nnz: %llu\n", (unsigned long long) desc.nnz);
        std::printf("partitions: %llu\n", (unsigned long long) desc.partitions);
        std::printf("canonical_layout: %s\n", desc.canonical_layout.c_str());
        std::printf("feature_order_hash: %llu\n", (unsigned long long) desc.feature_order_hash);
        std::printf("pack_manifest: %s\n", desc.has_pack_manifest ? "yes" : "no");
        return 0;
    }
    if (command == "validate") {
        std::string error;
        if (!csc::cshard_file::validate(argv[2], &error)) {
            std::fprintf(stderr, "invalid: %s\n", error.c_str());
            return 1;
        }
        std::printf("valid\n");
        return 0;
    }
    if (command == "read-rows") {
        std::uint64_t start = 0u, count = 0u;
        const std::string start_text = require_option(argc - 2, argv + 2, "--start");
        const std::string count_text = require_option(argc - 2, argv + 2, "--count");
        if (!parse_u64(start_text.c_str(), &start) || !parse_u64(count_text.c_str(), &count)) {
            throw std::runtime_error("--start and --count must be unsigned integers");
        }
        csc::cshard_file file = csc::cshard_file::open(argv[2]);
        const auto csr = file.read_rows(start, count);
        std::printf("rows: %llu\n", (unsigned long long) csr.rows);
        std::printf("cols: %llu\n", (unsigned long long) csr.cols);
        std::printf("nnz: %zu\n", csr.data.size());
        std::printf("indptr:");
        for (std::int64_t value : csr.indptr) std::printf(" %lld", (long long) value);
        std::printf("\nindices:");
        for (std::int64_t value : csr.indices) std::printf(" %lld", (long long) value);
        std::printf("\ndata:");
        for (float value : csr.data) std::printf(" %.8g", (double) value);
        std::printf("\n");
        return 0;
    }
    if (command == "convert") {
        std::string error;
        const std::string out = require_option(argc - 2, argv + 2, "--out");
        if (!csc::convert_csh5_to_cshard(argv[2], out, {}, &error)) {
            std::fprintf(stderr, "convert failed: %s\n", error.c_str());
            return 1;
        }
        return 0;
    }
    usage();
    return 2;
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "cshard") return cshard_main(argc - 1, argv + 1);
        usage();
        return 2;
    } catch (const std::exception &exc) {
        std::fprintf(stderr, "cellshard: %s\n", exc.what());
        return 1;
    }
}
