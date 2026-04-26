#pragma once

#include "../common/generation.hh"

#include <cstdint>

namespace cellshard {

struct cspack_header {
    char     magic[8];          // "CSPACK\0"
    uint32_t version_major;
    uint32_t version_minor;
    uint32_t header_bytes;
    uint32_t endian;
    uint32_t alignment;
    uint32_t flags;

    uint64_t file_size;
    uint64_t metadata_count;
    uint64_t section_count;
    uint64_t shard_count;
    uint64_t partition_count;

    uint64_t metadata_offset;
    uint64_t section_dir_offset;
    uint64_t shard_dir_offset;
    uint64_t partition_dir_offset;
    uint64_t payload_offset;

    dataset_generation_ref generation;
};

} // namespace cellshard
