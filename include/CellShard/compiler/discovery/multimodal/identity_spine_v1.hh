#pragma once

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::multimodal {

enum class modality_kind_v1 : std::uint32_t {
    transcriptome = 1,
    chromatin = 2,
    protein = 3,
    spatial = 4,
    sequence = 5,
    custom = 0xffff'ffffu,
};

struct modality_identity_binding_v1 {
    std::uint64_t modality_identity = 0;
    std::uint64_t observation_axis_identity = 0;
    std::uint64_t observation_order_identity = 0;
    std::uint64_t feature_axis_identity = 0;
    std::uint64_t feature_order_identity = 0;
    std::uint64_t observation_to_subject_relation_identity = 0;
    std::uint64_t value_generation = 0;
    modality_kind_v1 kind = modality_kind_v1::custom;
    std::uint32_t reserved = 0;
};

struct multimodal_identity_spine_view_v1 {
    const modality_identity_binding_v1 *modalities = nullptr;
    std::uint32_t modality_count = 0;
    std::uint32_t reserved = 0;
    std::uint64_t spine_identity = 0;
    std::uint64_t cohort_identity = 0;
    std::uint64_t subject_axis_identity = 0;
    std::uint64_t subject_order_identity = 0;
    std::uint64_t structure_epoch = 0;
};

enum class identity_spine_code_v1 : std::uint32_t {
    valid = 0,
    invalid_spine_identity,
    insufficient_modalities,
    missing_modalities,
    invalid_modality,
    duplicate_modality_identity,
    direct_subject_order_mismatch,
    missing_subject_relation,
};

struct identity_spine_result_v1 {
    identity_spine_code_v1 code = identity_spine_code_v1::valid;
    std::uint32_t modality_index = 0;
    std::uint32_t conflicting_modality_index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == identity_spine_code_v1::valid;
    }
};

[[nodiscard]] inline identity_spine_result_v1
validate_multimodal_identity_spine_v1(
    multimodal_identity_spine_view_v1 spine) noexcept {
    if (spine.spine_identity == 0 || spine.cohort_identity == 0
        || spine.subject_axis_identity == 0
        || spine.subject_order_identity == 0 || spine.structure_epoch == 0)
        return {identity_spine_code_v1::invalid_spine_identity};
    if (spine.modality_count < 2)
        return {identity_spine_code_v1::insufficient_modalities};
    if (spine.modalities == nullptr)
        return {identity_spine_code_v1::missing_modalities};
    for (std::uint32_t index = 0; index < spine.modality_count; ++index) {
        const auto &modality = spine.modalities[index];
        if (modality.modality_identity == 0
            || modality.observation_axis_identity == 0
            || modality.observation_order_identity == 0
            || modality.feature_axis_identity == 0
            || modality.feature_order_identity == 0
            || modality.value_generation == 0)
            return {identity_spine_code_v1::invalid_modality, index};
        if (modality.observation_axis_identity == spine.subject_axis_identity) {
            if (modality.observation_order_identity
                    != spine.subject_order_identity
                || modality.observation_to_subject_relation_identity != 0)
                return {identity_spine_code_v1::direct_subject_order_mismatch,
                        index};
        } else if (modality.observation_to_subject_relation_identity == 0) {
            return {identity_spine_code_v1::missing_subject_relation, index};
        }
        for (std::uint32_t previous = 0; previous < index; ++previous)
            if (spine.modalities[previous].modality_identity
                == modality.modality_identity)
                return {identity_spine_code_v1::duplicate_modality_identity,
                        index, previous};
    }
    return {};
}

static_assert(std::is_standard_layout<modality_identity_binding_v1>::value);
static_assert(std::is_trivially_copyable<modality_identity_binding_v1>::value);
static_assert(std::is_standard_layout<multimodal_identity_spine_view_v1>::value);
static_assert(std::is_trivially_copyable<multimodal_identity_spine_view_v1>::value);

} // namespace cellshard::compiler::discovery::multimodal
