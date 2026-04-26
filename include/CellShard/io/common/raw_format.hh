#pragma once

#include "../../core/types.cuh"

#include <cstdint>
#include <cstdio>

namespace cellshard {

enum disk_format : std::uint8_t {
    disk_format_none = 0,
    disk_format_dense = 1,
    disk_format_compressed = 2,
    disk_format_coo = 3,
    disk_format_dia = 4,
    disk_format_ell = 5,
    disk_format_blocked_ell = 6,
    disk_format_sliced_ell = 7,
    disk_format_quantized_blocked_ell = 8,

    disk_format_triplet = disk_format_coo,
    disk_format_diagonal = disk_format_dia,
};

// Minimal fixed header stored at the front of every raw matrix payload.
struct disk_header {
    disk_format format;
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;
};

inline int check_disk_format(disk_format expected, disk_format actual, const char *name) {
    if (expected == actual) return 1;
    std::fprintf(stderr,
                 "Error: expected format %u, got %u for %s\n",
                 (unsigned int) expected,
                 (unsigned int) actual,
                 name);
    return 0;
}

} // namespace cellshard
