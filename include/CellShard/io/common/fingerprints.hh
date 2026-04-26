#pragma once

#include <cstdint>

namespace cellshard {

enum {
    dataset_fingerprint_algorithm_none = 0u,
    dataset_fingerprint_algorithm_xxh3_64 = 1u
};

struct dataset_feature_fingerprints {
    std::uint64_t feature_order_hash;
    std::uint64_t feature_id_hash;
    std::uint64_t feature_name_hash;
    std::uint64_t feature_type_hash;
    std::uint32_t algorithm;
    std::uint32_t reserved;
};

} // namespace cellshard
