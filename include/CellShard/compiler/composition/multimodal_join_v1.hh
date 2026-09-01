#pragma once

#include <CellShard/compiler/composition/identity_spine_join_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

inline constexpr std::uint32_t max_multimodal_join_modalities_v1 = 16;

struct multimodal_join_entry_v1 {
    std::uint64_t logical_identity = 0;
    std::uint32_t local_indices[max_multimodal_join_modalities_v1]{};
};

struct multimodal_join_view_v1 {
    structure_id identity{};
    domain_id shared_identity_domain{};
    order_id shared_identity_order{};
    const identity_spine_view_v1 *modalities = nullptr;
    const multimodal_join_entry_v1 *entries = nullptr;
    std::uint32_t modality_count = 0;
    std::uint32_t entry_count = 0;
};

enum class multimodal_join_code_v1 : std::uint32_t {
    joined = 0,
    invalid_output_identity,
    invalid_shared_axis,
    invalid_modality_count,
    missing_modalities,
    invalid_modality,
    identity_count_mismatch,
    identity_mismatch,
    missing_storage,
    insufficient_capacity,
    missing_output,
};

struct multimodal_join_result_v1 {
    multimodal_join_code_v1 code = multimodal_join_code_v1::joined;
    std::uint32_t modality_index = 0;
    std::uint64_t identity = 0;
    [[nodiscard]] constexpr bool joined() const noexcept {
        return code == multimodal_join_code_v1::joined;
    }
};

[[nodiscard]] inline multimodal_join_result_v1 compose_multimodal_join_v1(
    structure_id output_identity,
    domain_id shared_identity_domain,
    order_id shared_identity_order,
    const identity_spine_view_v1 *modalities,
    std::uint32_t modality_count,
    multimodal_join_entry_v1 *storage,
    std::uint32_t capacity,
    multimodal_join_view_v1 *output) noexcept {
    if (!output_identity.valid()) {
        return {multimodal_join_code_v1::invalid_output_identity};
    }
    if (!shared_identity_domain.valid() || !shared_identity_order.valid()) {
        return {multimodal_join_code_v1::invalid_shared_axis};
    }
    if (modality_count < 2
        || modality_count > max_multimodal_join_modalities_v1) {
        return {multimodal_join_code_v1::invalid_modality_count};
    }
    if (modalities == nullptr) {
        return {multimodal_join_code_v1::missing_modalities};
    }
    for (std::uint32_t modality = 0; modality < modality_count; ++modality) {
        if (!validate_identity_spine_v1(modalities[modality]).joined()) {
            return {multimodal_join_code_v1::invalid_modality, modality};
        }
        if (modalities[modality].identity_count
            != modalities[0].identity_count) {
            return {multimodal_join_code_v1::identity_count_mismatch,
                    modality};
        }
    }
    if (modalities[0].identity_count != 0 && storage == nullptr) {
        return {multimodal_join_code_v1::missing_storage};
    }
    if (capacity < modalities[0].identity_count) {
        return {multimodal_join_code_v1::insufficient_capacity};
    }
    if (output == nullptr) return {multimodal_join_code_v1::missing_output};
    *output = {};
    for (std::uint32_t index = 0;
         index < modalities[0].identity_count;
         ++index) {
        const auto logical_identity = modalities[0].logical_identities[index];
        storage[index].logical_identity = logical_identity;
        for (std::uint32_t modality = 0;
             modality < modality_count;
             ++modality) {
            if (modalities[modality].logical_identities[index]
                != logical_identity) {
                return {multimodal_join_code_v1::identity_mismatch, modality,
                        logical_identity};
            }
            storage[index].local_indices[modality] = index;
        }
    }
    *output = {output_identity, shared_identity_domain, shared_identity_order,
               modalities, storage, modality_count,
               static_cast<std::uint32_t>(modalities[0].identity_count)};
    return {multimodal_join_code_v1::joined, modality_count,
            modalities[0].identity_count};
}

static_assert(std::is_trivially_copyable<multimodal_join_entry_v1>::value);
static_assert(std::is_trivially_copyable<multimodal_join_view_v1>::value);

} // namespace cellshard::compiler::composition
