#pragma once

#include <CellShard/compiler/discovery/multimodal/identity_spine_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::multimodal {

enum class modality_observation_status_v1 : std::uint8_t {
    observed = 0,
    not_assayed = 1,
    failed_quality_control = 2,
    below_detection = 3,
    unavailable = 4,
};

struct modality_missingness_view_v1 {
    const std::uint8_t *status = nullptr;
    std::uint64_t subject_count = 0;
    std::uint64_t subject_stride = 0;
    std::uint32_t modality_count = 0;
    std::uint32_t reserved = 0;
    std::uint64_t spine_identity = 0;
    std::uint64_t subject_axis_identity = 0;
    std::uint64_t subject_order_identity = 0;
    std::uint64_t missingness_generation = 0;
};

struct modality_missingness_summary_v1 {
    std::uint64_t observed_count = 0;
    std::uint64_t not_assayed_count = 0;
    std::uint64_t failed_quality_control_count = 0;
    std::uint64_t below_detection_count = 0;
    std::uint64_t unavailable_count = 0;
};

enum class modality_missingness_code_v1 : std::uint32_t {
    summarized = 0,
    identity_mismatch,
    invalid_shape,
    missing_status,
    missing_summary,
    insufficient_summary_capacity,
    invalid_status,
    work_limit_exceeded,
};

struct modality_missingness_result_v1 {
    modality_missingness_code_v1 code
        = modality_missingness_code_v1::summarized;
    std::uint64_t subject_index = 0;
    std::uint32_t modality_index = 0;
    std::uint32_t reserved = 0;
    std::uint64_t work_items = 0;
    [[nodiscard]] constexpr bool summarized() const noexcept {
        return code == modality_missingness_code_v1::summarized;
    }
};

[[nodiscard]] inline modality_missingness_result_v1
summarize_modality_missingness_v1(
    multimodal_identity_spine_view_v1 spine,
    modality_missingness_view_v1 missingness,
    modality_missingness_summary_v1 *summaries,
    std::uint64_t summary_capacity,
    std::uint64_t maximum_work_items) noexcept {
    if (!validate_multimodal_identity_spine_v1(spine).valid()
        || missingness.spine_identity != spine.spine_identity
        || missingness.subject_axis_identity != spine.subject_axis_identity
        || missingness.subject_order_identity != spine.subject_order_identity
        || missingness.missingness_generation == 0)
        return {modality_missingness_code_v1::identity_mismatch};
    if (missingness.subject_count == 0
        || missingness.modality_count != spine.modality_count
        || missingness.subject_stride < missingness.modality_count
        || missingness.subject_count
            > std::numeric_limits<std::uint64_t>::max()
                / missingness.subject_stride)
        return {modality_missingness_code_v1::invalid_shape};
    if (missingness.status == nullptr)
        return {modality_missingness_code_v1::missing_status};
    if (summaries == nullptr)
        return {modality_missingness_code_v1::missing_summary};
    if (summary_capacity < missingness.modality_count)
        return {modality_missingness_code_v1::insufficient_summary_capacity};
    for (std::uint32_t modality = 0; modality < missingness.modality_count;
         ++modality)
        summaries[modality] = {};
    modality_missingness_result_v1 result{};
    for (std::uint64_t subject = 0; subject < missingness.subject_count; ++subject) {
        for (std::uint32_t modality = 0;
             modality < missingness.modality_count; ++modality) {
            if (result.work_items == maximum_work_items)
                return {modality_missingness_code_v1::work_limit_exceeded,
                        subject, modality, 0, result.work_items};
            ++result.work_items;
            const auto status = static_cast<modality_observation_status_v1>(
                missingness.status[subject * missingness.subject_stride + modality]);
            auto &summary = summaries[modality];
            switch (status) {
            case modality_observation_status_v1::observed:
                ++summary.observed_count;
                break;
            case modality_observation_status_v1::not_assayed:
                ++summary.not_assayed_count;
                break;
            case modality_observation_status_v1::failed_quality_control:
                ++summary.failed_quality_control_count;
                break;
            case modality_observation_status_v1::below_detection:
                ++summary.below_detection_count;
                break;
            case modality_observation_status_v1::unavailable:
                ++summary.unavailable_count;
                break;
            default:
                return {modality_missingness_code_v1::invalid_status,
                        subject, modality, 0, result.work_items};
            }
        }
    }
    return result;
}

static_assert(std::is_standard_layout<modality_missingness_view_v1>::value);
static_assert(std::is_trivially_copyable<modality_missingness_view_v1>::value);
static_assert(std::is_standard_layout<modality_missingness_summary_v1>::value);
static_assert(std::is_trivially_copyable<modality_missingness_summary_v1>::value);

} // namespace cellshard::compiler::discovery::multimodal
