#pragma once

#include <cstdint>

namespace cellshard {

struct dataset_row_span {
    std::uint64_t row_begin;
    std::uint64_t row_end;
};

struct dataset_partition_ref {
    std::uint64_t shard_id;
    std::uint64_t partition_id;
};

struct dataset_partition_descriptor {
    dataset_partition_ref partition;
    dataset_row_span rows;
    std::uint32_t codec_id;
    std::uint32_t execution_format;
    std::uint64_t payload_bytes;
    std::uint64_t uncompressed_bytes;
};

} // namespace cellshard
