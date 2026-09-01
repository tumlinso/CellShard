#pragma once

#include <CellShard/compiler/discovery/multimodal/identity_spine_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::multimodal {

enum class value_scalar_kind_v1 : std::uint32_t {
    unsigned_integer = 1,
    signed_integer = 2,
    floating_point = 3,
    boolean = 4,
};

enum class missing_value_policy_v1 : std::uint32_t {
    absent_is_zero = 1,
    absent_is_missing = 2,
    explicit_mask = 3,
};

struct modality_domain_overlay_v1 {
    std::uint64_t modality_identity = 0;
    std::uint64_t domain_identity = 0;
    std::uint64_t axis_identity = 0;
    std::uint64_t order_identity = 0;
    std::uint64_t geometry_identity = 0;
};

struct modality_value_overlay_v1 {
    std::uint64_t modality_identity = 0;
    std::uint64_t value_plane_identity = 0;
    std::uint64_t value_generation = 0;
    std::int64_t scale_numerator = 1;
    std::uint64_t scale_denominator = 1;
    value_scalar_kind_v1 scalar_kind = value_scalar_kind_v1::floating_point;
    missing_value_policy_v1 missing_policy
        = missing_value_policy_v1::absent_is_zero;
};

struct domain_value_overlays_view_v1 {
    const modality_domain_overlay_v1 *domains = nullptr;
    const modality_value_overlay_v1 *values = nullptr;
    std::uint32_t modality_count = 0;
    std::uint32_t reserved = 0;
    std::uint64_t spine_identity = 0;
    std::uint64_t structure_epoch = 0;
};

enum class domain_value_overlay_code_v1 : std::uint32_t {
    valid = 0,
    invalid_identity,
    count_mismatch,
    missing_overlays,
    modality_mismatch,
    invalid_domain,
    invalid_value,
    stale_value_generation,
};

struct domain_value_overlay_result_v1 {
    domain_value_overlay_code_v1 code = domain_value_overlay_code_v1::valid;
    std::uint32_t modality_index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == domain_value_overlay_code_v1::valid;
    }
};

[[nodiscard]] inline domain_value_overlay_result_v1
validate_domain_value_overlays_v1(
    multimodal_identity_spine_view_v1 spine,
    domain_value_overlays_view_v1 overlays) noexcept {
    if (!validate_multimodal_identity_spine_v1(spine).valid()
        || overlays.spine_identity != spine.spine_identity
        || overlays.structure_epoch != spine.structure_epoch)
        return {domain_value_overlay_code_v1::invalid_identity};
    if (overlays.modality_count != spine.modality_count)
        return {domain_value_overlay_code_v1::count_mismatch};
    if (overlays.domains == nullptr || overlays.values == nullptr)
        return {domain_value_overlay_code_v1::missing_overlays};
    for (std::uint32_t index = 0; index < overlays.modality_count; ++index) {
        const auto &binding = spine.modalities[index];
        const auto &domain = overlays.domains[index];
        const auto &value = overlays.values[index];
        if (domain.modality_identity != binding.modality_identity
            || value.modality_identity != binding.modality_identity)
            return {domain_value_overlay_code_v1::modality_mismatch, index};
        if (domain.domain_identity == 0 || domain.axis_identity == 0
            || domain.order_identity == 0 || domain.geometry_identity == 0
            || domain.axis_identity != binding.feature_axis_identity
            || domain.order_identity != binding.feature_order_identity)
            return {domain_value_overlay_code_v1::invalid_domain, index};
        if (value.value_plane_identity == 0 || value.value_generation == 0
            || value.scale_numerator == 0 || value.scale_denominator == 0)
            return {domain_value_overlay_code_v1::invalid_value, index};
        if (value.value_generation != binding.value_generation)
            return {domain_value_overlay_code_v1::stale_value_generation, index};
    }
    return {};
}

static_assert(std::is_standard_layout<modality_domain_overlay_v1>::value);
static_assert(std::is_trivially_copyable<modality_domain_overlay_v1>::value);
static_assert(std::is_standard_layout<modality_value_overlay_v1>::value);
static_assert(std::is_trivially_copyable<modality_value_overlay_v1>::value);

} // namespace cellshard::compiler::discovery::multimodal
