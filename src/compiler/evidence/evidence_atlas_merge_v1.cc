#include <CellShard/compiler/evidence/evidence_atlas_merge_v1.hh>

#include <new>
#include <vector>

namespace cellshard::compiler::evidence {

evidence_atlas_merge_result_v1 merge_evidence_atlases_v1(
    evidence_atlas_view_v1 left,
    evidence_atlas_view_v1 right,
    evidence_identity_v1 output_identity,
    std::uint64_t output_generation,
    std::uint64_t maximum_records,
    evidence_atlas_builder_v1 *output) noexcept {
    if (output == nullptr)
        return {evidence_atlas_merge_code_v1::build_failure};
    output->reset();
    const auto left_check = evidence_atlas_requirements(
        {left.records, left.record_count, left.atlas_identity, left.atlas_generation},
        maximum_records);
    if (!left_check.ok())
        return {evidence_atlas_merge_code_v1::invalid_left, left_check.index, left_check};
    const auto right_check = evidence_atlas_requirements(
        {right.records, right.record_count, right.atlas_identity, right.atlas_generation},
        maximum_records);
    if (!right_check.ok())
        return {evidence_atlas_merge_code_v1::invalid_right, right_check.index, right_check};
    if (left.record_count > maximum_records - right.record_count)
        return {evidence_atlas_merge_code_v1::output_limit_exceeded};

    std::vector<atom_evidence_record_v1> merged;
    try {
        merged.reserve(left.record_count + right.record_count);
        std::uint64_t li = 0;
        std::uint64_t ri = 0;
        while (li < left.record_count || ri < right.record_count) {
            if (li == left.record_count) merged.push_back(right.records[ri++]);
            else if (ri == right.record_count) merged.push_back(left.records[li++]);
            else if (evidence_identity_less_v1(left.records[li].evidence_identity,
                                               right.records[ri].evidence_identity))
                merged.push_back(left.records[li++]);
            else if (evidence_identity_less_v1(right.records[ri].evidence_identity,
                                               left.records[li].evidence_identity))
                merged.push_back(right.records[ri++]);
            else {
                if (!atom_evidence_record_equal_v1(left.records[li], right.records[ri]))
                    return {evidence_atlas_merge_code_v1::conflicting_duplicate,
                            merged.size()};
                merged.push_back(left.records[li]);
                ++li;
                ++ri;
            }
        }
    } catch (const std::bad_alloc &) {
        return {evidence_atlas_merge_code_v1::allocation_failure};
    }
    const evidence_atlas_source_v1 source{
        merged.data(), merged.size(), output_identity, output_generation};
    const auto requirements = evidence_atlas_requirements(source, maximum_records);
    if (!requirements.ok())
        return {evidence_atlas_merge_code_v1::build_failure,
                requirements.index, requirements};
    const auto build = output->fill(source, requirements.requirements, maximum_records);
    if (!build.ok())
        return {evidence_atlas_merge_code_v1::build_failure, build.index, build};
    return {evidence_atlas_merge_code_v1::merged, merged.size(), build};
}

} // namespace cellshard::compiler::evidence
