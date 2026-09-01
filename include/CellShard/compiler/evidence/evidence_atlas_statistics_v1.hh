#pragma once

#include <CellShard/compiler/evidence/evidence_atlas_v1.hh>

#include <array>
#include <cstdint>
#include <limits>

namespace cellshard::compiler::evidence {

struct evidence_atlas_statistics_v1 {
    std::uint64_t record_count = 0;
    std::uint64_t total_observation_count = 0;
    std::uint64_t maximum_observation_count = 0;
    std::uint64_t negative_record_count = 0;
    std::array<std::uint64_t, 18> records_by_kind{};
    std::array<std::uint64_t, 9> records_by_family{};
};

enum class evidence_atlas_statistics_code_v1 : std::uint32_t {
    success = 0,
    invalid_atlas,
    observation_count_overflow,
    statistics_count_overflow,
    missing_output,
};

struct evidence_atlas_statistics_result_v1 {
    evidence_atlas_statistics_code_v1 code =
        evidence_atlas_statistics_code_v1::success;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == evidence_atlas_statistics_code_v1::success;
    }
};

[[nodiscard]] inline evidence_atlas_statistics_result_v1
validate_and_measure_evidence_atlas_v1(
    evidence_atlas_view_v1 atlas,
    std::uint64_t maximum_records,
    evidence_atlas_statistics_v1 *output) noexcept {
    if (output == nullptr)
        return {evidence_atlas_statistics_code_v1::missing_output};
    *output = {};
    const auto validation = evidence_atlas_requirements(
        {atlas.records, atlas.record_count, atlas.atlas_identity, atlas.atlas_generation},
        maximum_records);
    if (!validation.ok())
        return {evidence_atlas_statistics_code_v1::invalid_atlas,
                validation.index};
    for (std::uint64_t index = 0; index < atlas.record_count; ++index) {
        const auto &record = atlas.records[index];
        if (output->total_observation_count
            > std::numeric_limits<std::uint64_t>::max()
                - record.observation_count) {
            *output = {};
            return {evidence_atlas_statistics_code_v1::
                        observation_count_overflow,
                    index};
        }
        const auto kind_index = static_cast<std::uint32_t>(record.kind);
        const auto family_index = static_cast<std::uint32_t>(family_of(record.kind));
        if (output->records_by_kind[kind_index]
                == std::numeric_limits<std::uint64_t>::max()
            || output->records_by_family[family_index]
                == std::numeric_limits<std::uint64_t>::max()) {
            *output = {};
            return {evidence_atlas_statistics_code_v1::statistics_count_overflow,
                    index};
        }
        ++output->records_by_kind[kind_index];
        ++output->records_by_family[family_index];
        output->total_observation_count += record.observation_count;
        if (record.observation_count > output->maximum_observation_count)
            output->maximum_observation_count = record.observation_count;
        if (is_negative_evidence(record.kind)) ++output->negative_record_count;
    }
    output->record_count = atlas.record_count;
    return {evidence_atlas_statistics_code_v1::success, atlas.record_count};
}

// Mechanism statistics describe proposal evidence only and contain no exact
// coverage, contribution-owner, physical-view, placement, or execution fields.
[[nodiscard]] constexpr bool statistics_authorize_execution(
    const evidence_atlas_statistics_v1 &) noexcept { return false; }

} // namespace cellshard::compiler::evidence
