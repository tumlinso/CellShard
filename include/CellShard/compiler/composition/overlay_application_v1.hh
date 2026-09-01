#pragma once

#include <CellShard/compiler/composition/coverage_v1.hh>

namespace cellshard::compiler::composition {

enum class overlay_application_code_v1 : std::uint32_t {
    applied = 0,
    invalid_output_identity,
    invalid_base,
    invalid_additions,
    invalid_removals,
    axis_mismatch,
    removal_not_in_base,
    addition_already_in_base,
    count_overflow,
    missing_storage,
    insufficient_capacity,
    missing_output,
};

struct overlay_application_result_v1 {
    overlay_application_code_v1 code = overlay_application_code_v1::applied;
    std::uint64_t logical_identity = 0;
    [[nodiscard]] constexpr bool applied() const noexcept {
        return code == overlay_application_code_v1::applied;
    }
};

[[nodiscard]] inline overlay_application_result_v1 compose_overlay_application_v1(
    structure_id output_identity,
    const exact_coverage_view_v1 &base,
    const exact_coverage_view_v1 &additions,
    const exact_coverage_view_v1 &removals,
    mutable_coverage_storage_v1 storage,
    exact_coverage_view_v1 *output) noexcept {
    if (!output_identity.valid()) {
        return {overlay_application_code_v1::invalid_output_identity};
    }
    if (!validate_exact_coverage_v1(base).composed()) {
        return {overlay_application_code_v1::invalid_base};
    }
    if (!validate_exact_coverage_v1(additions).composed()) {
        return {overlay_application_code_v1::invalid_additions};
    }
    if (!validate_exact_coverage_v1(removals).composed()) {
        return {overlay_application_code_v1::invalid_removals};
    }
    if (base.domain != additions.domain || base.domain != removals.domain
        || base.order != additions.order || base.order != removals.order) {
        return {overlay_application_code_v1::axis_mismatch};
    }
    std::uint64_t base_index = 0;
    for (std::uint64_t removal = 0;
         removal < removals.logical_item_count;
         ++removal) {
        const auto identity = removals.logical_item_ids[removal];
        while (base_index < base.logical_item_count
               && base.logical_item_ids[base_index] < identity) {
            ++base_index;
        }
        if (base_index == base.logical_item_count
            || base.logical_item_ids[base_index] != identity) {
            return {overlay_application_code_v1::removal_not_in_base,
                    identity};
        }
    }
    base_index = 0;
    for (std::uint64_t addition = 0;
         addition < additions.logical_item_count;
         ++addition) {
        const auto identity = additions.logical_item_ids[addition];
        while (base_index < base.logical_item_count
               && base.logical_item_ids[base_index] < identity) {
            ++base_index;
        }
        if (base_index < base.logical_item_count
            && base.logical_item_ids[base_index] == identity) {
            return {overlay_application_code_v1::addition_already_in_base,
                    identity};
        }
    }
    if (base.logical_item_count
        > std::numeric_limits<std::uint64_t>::max()
              - additions.logical_item_count) {
        return {overlay_application_code_v1::count_overflow};
    }
    const auto maximum_count =
        base.logical_item_count + additions.logical_item_count;
    if (maximum_count != 0 && storage.logical_item_ids == nullptr) {
        return {overlay_application_code_v1::missing_storage};
    }
    if (storage.capacity < maximum_count) {
        return {overlay_application_code_v1::insufficient_capacity};
    }
    if (output == nullptr) return {overlay_application_code_v1::missing_output};
    *output = {};
    base_index = 0;
    std::uint64_t removal_index = 0;
    std::uint64_t addition_index = 0;
    std::uint64_t output_index = 0;
    while (base_index < base.logical_item_count
           || addition_index < additions.logical_item_count) {
        while (base_index < base.logical_item_count
               && removal_index < removals.logical_item_count
               && base.logical_item_ids[base_index]
                      == removals.logical_item_ids[removal_index]) {
            ++base_index;
            ++removal_index;
        }
        if (base_index == base.logical_item_count
            && addition_index == additions.logical_item_count) {
            break;
        }
        const bool take_base = base_index < base.logical_item_count
            && (addition_index == additions.logical_item_count
                || base.logical_item_ids[base_index]
                       < additions.logical_item_ids[addition_index]);
        if (take_base) {
            storage.logical_item_ids[output_index++] =
                base.logical_item_ids[base_index++];
        } else {
            storage.logical_item_ids[output_index++] =
                additions.logical_item_ids[addition_index++];
        }
    }
    *output = {output_identity, base.domain, base.order,
               storage.logical_item_ids, output_index};
    return {overlay_application_code_v1::applied, output_index};
}

} // namespace cellshard::compiler::composition
