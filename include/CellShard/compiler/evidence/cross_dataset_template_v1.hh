#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::evidence {

struct dataset_template_binding_v1 {
    evidence_identity_v1 dataset_identity{};
    evidence_identity_v1 domain_identity{};
    evidence_identity_v1 order_identity{};
    evidence_identity_v1 selection_identity{};
    std::uint64_t dataset_generation = 0;
};

struct cross_dataset_template_view_v1 {
    const dataset_template_binding_v1 *bindings = nullptr;
    std::uint64_t binding_count = 0;
    std::uint64_t binding_capacity = 0;
    evidence_identity_v1 evidence_identity{};
    evidence_identity_v1 template_identity{};
    evidence_disposition_v1 disposition = evidence_disposition_v1::proposal_only;
    std::uint32_t reserved = 0;
};

enum class cross_dataset_template_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_identity,
    insufficient_bindings,
    missing_bindings,
    capacity_overflow,
    invalid_binding_identity,
    missing_dataset_generation,
    unordered_or_duplicate_dataset,
    non_proposal_disposition,
    nonzero_reserved,
};

struct cross_dataset_template_validation_v1 {
    cross_dataset_template_validation_code_v1 code =
        cross_dataset_template_validation_code_v1::valid;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == cross_dataset_template_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr cross_dataset_template_validation_v1
validate_cross_dataset_template_v1(
    cross_dataset_template_view_v1 view) noexcept {
    if (!valid_evidence_identity_v1(view.evidence_identity)
        || !valid_evidence_identity_v1(view.template_identity))
        return {cross_dataset_template_validation_code_v1::invalid_identity, 0};
    if (view.binding_count < 2)
        return {cross_dataset_template_validation_code_v1::insufficient_bindings, 0};
    if (view.bindings == nullptr)
        return {cross_dataset_template_validation_code_v1::missing_bindings, 0};
    if (view.binding_count > view.binding_capacity)
        return {cross_dataset_template_validation_code_v1::capacity_overflow, 0};
    for (std::uint64_t index = 0; index < view.binding_count; ++index) {
        const auto &binding = view.bindings[index];
        if (!valid_evidence_identity_v1(binding.dataset_identity)
            || !valid_evidence_identity_v1(binding.domain_identity)
            || !valid_evidence_identity_v1(binding.order_identity)
            || !valid_evidence_identity_v1(binding.selection_identity))
            return {cross_dataset_template_validation_code_v1::invalid_binding_identity, index};
        if (binding.dataset_generation == 0)
            return {cross_dataset_template_validation_code_v1::missing_dataset_generation, index};
        if (index != 0 && !evidence_identity_less_v1(
                view.bindings[index - 1].dataset_identity,
                binding.dataset_identity))
            return {cross_dataset_template_validation_code_v1::unordered_or_duplicate_dataset, index};
    }
    if (view.disposition != evidence_disposition_v1::proposal_only)
        return {cross_dataset_template_validation_code_v1::non_proposal_disposition, 0};
    if (view.reserved != 0)
        return {cross_dataset_template_validation_code_v1::nonzero_reserved, 0};
    return {cross_dataset_template_validation_code_v1::valid, view.binding_count};
}

[[nodiscard]] constexpr bool establishes_biological_identity(
    cross_dataset_template_view_v1) noexcept { return false; }

static_assert(std::is_standard_layout<dataset_template_binding_v1>::value);
static_assert(std::is_trivially_copyable<dataset_template_binding_v1>::value);
static_assert(offsetof(cross_dataset_template_view_v1, bindings) == 0);

} // namespace cellshard::compiler::evidence
