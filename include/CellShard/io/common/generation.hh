#pragma once

#include <cstdint>

namespace cellshard {

struct dataset_generation_ref {
    std::uint64_t canonical_generation;
    std::uint64_t execution_plan_generation;
    std::uint64_t pack_generation;
    std::uint64_t service_epoch;
};

struct dataset_runtime_generation {
    dataset_generation_ref generation;
    std::uint64_t active_read_generation;
    std::uint64_t staged_write_generation;
};

} // namespace cellshard
