#pragma once

#include "CellShard/compiler/composition/superatom/candidate.hpp"

namespace cellshard::compiler::composition::superatom {

struct composition_observation {
    local_index candidate = 0;
    std::uint64_t frequency = 0;
    std::uint64_t reuse_count = 0;
};
struct composition_statistics {
    std::uint64_t frequency = 0;
    std::uint64_t reuse_count = 0;
    bool saturated = false;
};

inline std::uint64_t saturating_add(std::uint64_t left, std::uint64_t right,
                                    bool& saturated) noexcept {
    if (right > UINT64_MAX - left) { saturated = true; return UINT64_MAX; }
    return left + right;
}

inline bool aggregate_statistics(const composition_observation* observations,
                                 local_index observation_count,
                                 composition_statistics* output,
                                 local_index candidate_count) noexcept {
    if (observations == nullptr || output == nullptr) return false;
    for (local_index i = 0; i < candidate_count; ++i) output[i] = {};
    for (local_index i = 0; i < observation_count; ++i) {
        const auto& observation = observations[i];
        if (observation.candidate >= candidate_count) return false;
        auto& result = output[observation.candidate];
        result.frequency = saturating_add(result.frequency, observation.frequency, result.saturated);
        result.reuse_count = saturating_add(result.reuse_count, observation.reuse_count, result.saturated);
    }
    return true;
}

}  // namespace cellshard::compiler::composition::superatom
