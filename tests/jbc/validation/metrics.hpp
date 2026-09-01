#pragma once
#include "fixtures.hpp"
namespace cellshard::jbc::validation {
enum class mechanism : std::uint8_t { modular_support, repeated_composition, stable_structure, trajectory_prefix, multimodal_identity };
enum class phase : std::uint8_t { discovery, build, persist, reload, stage, execute, validate, mutate, collect, count };
inline constexpr std::uint16_t complete_phase_mask = (UINT16_C(1) << static_cast<unsigned>(phase::count)) - 1U;
struct metric_record {
    global_id fixture_id = 0;
    mechanism biological_mechanism = mechanism::modular_support;
    std::uint64_t elapsed_ns = 0;
    std::uint64_t bytes_moved = 0;
    std::uint64_t useful_interactions = 0;
    std::uint32_t launches = 0;
    std::uint16_t included_phases = 0;
    bool exact_output_match = false;
    bool matched_null = false;
};
inline bool complete_metric(const metric_record& record) noexcept {
    return record.fixture_id != 0 && record.included_phases == complete_phase_mask &&
           record.exact_output_match && record.useful_interactions != 0;
}
}  // namespace cellshard::jbc::validation
