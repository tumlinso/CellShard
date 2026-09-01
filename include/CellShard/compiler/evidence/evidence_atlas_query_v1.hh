#pragma once

#include <CellShard/compiler/evidence/evidence_atlas_v1.hh>

#include <cstdint>

namespace cellshard::compiler::evidence {

struct evidence_query_v1 {
    evidence_identity_v1 subject_atom_identity{};
    evidence_identity_v1 source_identity{};
    evidence_kind kind = evidence_kind::invalid;
    evidence_family family = evidence_family::invalid;
    std::uint64_t minimum_observation_count = 0;
};

enum class evidence_query_code_v1 : std::uint32_t {
    success = 0,
    invalid_atlas,
    invalid_query,
    not_found,
    missing_output,
    insufficient_capacity,
};

struct evidence_query_result_v1 {
    evidence_query_code_v1 code = evidence_query_code_v1::success;
    std::uint64_t match_count = 0;
    const atom_evidence_record_v1 *record = nullptr;
    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == evidence_query_code_v1::success;
    }
};

[[nodiscard]] constexpr bool valid_evidence_query_v1(
    const evidence_query_v1 &query) noexcept {
    return (query.kind == evidence_kind::invalid || valid_evidence_kind(query.kind))
        && (query.family == evidence_family::invalid
            || (static_cast<std::uint32_t>(query.family) >= 1
                && static_cast<std::uint32_t>(query.family) <= 8));
}

[[nodiscard]] constexpr bool evidence_matches_query_v1(
    const atom_evidence_record_v1 &record,
    const evidence_query_v1 &query) noexcept {
    return (!valid_evidence_identity_v1(query.subject_atom_identity)
            || record.subject_atom_identity == query.subject_atom_identity)
        && (!valid_evidence_identity_v1(query.source_identity)
            || record.source_identity == query.source_identity)
        && (query.kind == evidence_kind::invalid || record.kind == query.kind)
        && (query.family == evidence_family::invalid
            || family_of(record.kind) == query.family)
        && record.observation_count >= query.minimum_observation_count;
}

[[nodiscard]] inline evidence_query_result_v1 evidence_filter_requirements_v1(
    evidence_atlas_view_v1 atlas,
    const evidence_query_v1 &query,
    std::uint64_t maximum_records) noexcept {
    const auto validation = evidence_atlas_requirements(
        {atlas.records, atlas.record_count, atlas.atlas_identity, atlas.atlas_generation},
        maximum_records);
    if (!validation.ok()) return {evidence_query_code_v1::invalid_atlas};
    if (!valid_evidence_query_v1(query)) return {evidence_query_code_v1::invalid_query};
    std::uint64_t count = 0;
    for (std::uint64_t index = 0; index < atlas.record_count; ++index)
        if (evidence_matches_query_v1(atlas.records[index], query)) ++count;
    return {evidence_query_code_v1::success, count};
}

[[nodiscard]] inline evidence_query_result_v1 filter_evidence_atlas_v1(
    evidence_atlas_view_v1 atlas,
    const evidence_query_v1 &query,
    const atom_evidence_record_v1 **output,
    std::uint64_t output_capacity,
    std::uint64_t maximum_records) noexcept {
    const auto requirement = evidence_filter_requirements_v1(atlas, query, maximum_records);
    if (!requirement.ok()) return requirement;
    if (requirement.match_count != 0 && output == nullptr)
        return {evidence_query_code_v1::missing_output, requirement.match_count};
    if (output_capacity < requirement.match_count)
        return {evidence_query_code_v1::insufficient_capacity, requirement.match_count};
    std::uint64_t count = 0;
    for (std::uint64_t index = 0; index < atlas.record_count; ++index)
        if (evidence_matches_query_v1(atlas.records[index], query))
            output[count++] = &atlas.records[index];
    return {evidence_query_code_v1::success, count};
}

[[nodiscard]] inline evidence_query_result_v1 find_evidence_v1(
    evidence_atlas_view_v1 atlas,
    evidence_identity_v1 identity,
    std::uint64_t maximum_records) noexcept {
    const auto validation = evidence_atlas_requirements(
        {atlas.records, atlas.record_count, atlas.atlas_identity, atlas.atlas_generation},
        maximum_records);
    if (!validation.ok()) return {evidence_query_code_v1::invalid_atlas};
    if (!valid_evidence_identity_v1(identity))
        return {evidence_query_code_v1::invalid_query};
    std::uint64_t first = 0;
    std::uint64_t last = atlas.record_count;
    while (first < last) {
        const auto middle = first + (last - first) / 2;
        if (evidence_identity_less_v1(atlas.records[middle].evidence_identity, identity))
            first = middle + 1;
        else
            last = middle;
    }
    if (first == atlas.record_count
        || !(atlas.records[first].evidence_identity == identity))
        return {evidence_query_code_v1::not_found};
    return {evidence_query_code_v1::success, 1, &atlas.records[first]};
}

} // namespace cellshard::compiler::evidence
