#pragma once

#include <CellShard/compiler/discovery/multimodal/identity_spine_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::multimodal {

inline constexpr std::uint64_t unmatched_subject_v1
    = std::numeric_limits<std::uint64_t>::max();

struct modality_entity_alignment_v1 {
    std::uint64_t modality_identity = 0;
    std::uint64_t observation_id = 0;
    std::uint64_t subject_id = unmatched_subject_v1;
    std::uint64_t entity_identity = 0;
};

struct modality_entity_alignment_summary_v1 {
    std::uint64_t matched_entity_count = 0;
    std::uint64_t modality_only_entity_count = 0;
};

enum class entity_alignment_code_v1 : std::uint32_t {
    summarized = 0,
    invalid_spine,
    missing_alignments,
    missing_summaries,
    insufficient_summary_capacity,
    unknown_modality,
    invalid_entity,
    subject_out_of_range,
    duplicate_observation,
    work_limit_exceeded,
};

struct entity_alignment_result_v1 {
    entity_alignment_code_v1 code = entity_alignment_code_v1::summarized;
    std::uint64_t alignment_index = 0;
    std::uint64_t work_items = 0;
    [[nodiscard]] constexpr bool summarized() const noexcept {
        return code == entity_alignment_code_v1::summarized;
    }
};

[[nodiscard]] inline entity_alignment_result_v1
summarize_entity_alignment_v1(
    multimodal_identity_spine_view_v1 spine,
    std::uint64_t subject_count,
    const modality_entity_alignment_v1 *alignments,
    std::uint64_t alignment_count,
    modality_entity_alignment_summary_v1 *summaries,
    std::uint64_t summary_capacity,
    std::uint64_t maximum_work_items) noexcept {
    if (!validate_multimodal_identity_spine_v1(spine).valid()
        || subject_count == 0)
        return {entity_alignment_code_v1::invalid_spine};
    if (alignment_count != 0 && alignments == nullptr)
        return {entity_alignment_code_v1::missing_alignments};
    if (summaries == nullptr)
        return {entity_alignment_code_v1::missing_summaries};
    if (summary_capacity < spine.modality_count)
        return {entity_alignment_code_v1::insufficient_summary_capacity};
    for (std::uint32_t index = 0; index < spine.modality_count; ++index)
        summaries[index] = {};
    entity_alignment_result_v1 result{};
    for (std::uint64_t index = 0; index < alignment_count; ++index) {
        const auto &alignment = alignments[index];
        std::uint32_t modality_index = spine.modality_count;
        for (std::uint32_t candidate = 0; candidate < spine.modality_count;
             ++candidate)
            if (spine.modalities[candidate].modality_identity
                == alignment.modality_identity) {
                modality_index = candidate;
                break;
            }
        if (modality_index == spine.modality_count)
            return {entity_alignment_code_v1::unknown_modality, index,
                    result.work_items};
        if (alignment.entity_identity == 0)
            return {entity_alignment_code_v1::invalid_entity, index,
                    result.work_items};
        if (alignment.subject_id != unmatched_subject_v1
            && alignment.subject_id >= subject_count)
            return {entity_alignment_code_v1::subject_out_of_range, index,
                    result.work_items};
        for (std::uint64_t previous = 0; previous < index; ++previous) {
            if (result.work_items == maximum_work_items)
                return {entity_alignment_code_v1::work_limit_exceeded, index,
                        result.work_items};
            ++result.work_items;
            if (alignments[previous].modality_identity
                    == alignment.modality_identity
                && alignments[previous].observation_id
                    == alignment.observation_id)
                return {entity_alignment_code_v1::duplicate_observation, index,
                        result.work_items};
        }
        if (alignment.subject_id == unmatched_subject_v1)
            ++summaries[modality_index].modality_only_entity_count;
        else
            ++summaries[modality_index].matched_entity_count;
    }
    result.alignment_index = alignment_count;
    return result;
}

static_assert(std::is_standard_layout<modality_entity_alignment_v1>::value);
static_assert(std::is_trivially_copyable<modality_entity_alignment_v1>::value);

} // namespace cellshard::compiler::discovery::multimodal
