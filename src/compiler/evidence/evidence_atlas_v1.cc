#include <CellShard/compiler/evidence/evidence_atlas_v1.hh>

#include <limits>
#include <new>

namespace cellshard::compiler::evidence {

evidence_atlas_build_result_v1 evidence_atlas_requirements(
    evidence_atlas_source_v1 source,
    std::uint64_t maximum_records) noexcept {
    if (!valid_evidence_identity_v1(source.atlas_identity))
        return {evidence_atlas_build_code_v1::invalid_atlas_identity};
    if (source.atlas_generation == 0)
        return {evidence_atlas_build_code_v1::missing_atlas_generation};
    if (source.record_count == 0)
        return {evidence_atlas_build_code_v1::empty_source};
    if (source.records == nullptr)
        return {evidence_atlas_build_code_v1::missing_records};
    if (source.record_count > maximum_records)
        return {evidence_atlas_build_code_v1::record_limit_exceeded};
    constexpr auto record_size = sizeof(atom_evidence_record_v1);
    if (source.record_count > std::numeric_limits<std::uint64_t>::max() / record_size)
        return {evidence_atlas_build_code_v1::byte_overflow};
    for (std::uint64_t index = 0; index < source.record_count; ++index) {
        if (!validate_atom_evidence_record_v1(source.records[index]).valid())
            return {evidence_atlas_build_code_v1::invalid_record, index};
        if (index != 0 && !evidence_identity_less_v1(
                source.records[index - 1].evidence_identity,
                source.records[index].evidence_identity))
            return {evidence_atlas_build_code_v1::unordered_or_duplicate_record, index};
    }
    return {evidence_atlas_build_code_v1::ready, source.record_count,
            {source.record_count, source.record_count * record_size}};
}

evidence_atlas_build_result_v1 evidence_atlas_builder_v1::fill(
    evidence_atlas_source_v1 source,
    evidence_atlas_requirements_v1 capacity,
    std::uint64_t maximum_records) noexcept {
    reset();
    const auto requirement = evidence_atlas_requirements(source, maximum_records);
    if (!requirement.ok()) return requirement;
    if (capacity.record_capacity < requirement.requirements.record_capacity
        || capacity.record_bytes < requirement.requirements.record_bytes)
        return {evidence_atlas_build_code_v1::insufficient_capacity, 0,
                requirement.requirements};
    try {
        records_.assign(source.records, source.records + source.record_count);
    } catch (const std::bad_alloc &) {
        reset();
        return {evidence_atlas_build_code_v1::allocation_failure, 0,
                requirement.requirements};
    }
    view_.atlas_identity = source.atlas_identity;
    view_.atlas_generation = source.atlas_generation;
    rebind();
    return {evidence_atlas_build_code_v1::built, source.record_count,
            requirement.requirements};
}

void evidence_atlas_builder_v1::reset() noexcept {
    records_.clear();
    view_ = {};
}

void evidence_atlas_builder_v1::rebind() noexcept {
    view_.records = records_.empty() ? nullptr : records_.data();
    view_.record_count = records_.size();
}

} // namespace cellshard::compiler::evidence
