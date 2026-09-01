#pragma once

#include <CellShard/compiler/composition/coverage_v1.hh>
#include <CellShard/compiler/composition/production_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

struct partial_result_tag {};
using partial_result_id = strong_id<partial_result_tag>;

struct partial_result_identity_v1 {
    partial_result_id identity{};
    std::uint64_t logical_result_identity = 0;
    operator_class_id combination_algebra{};
    scalar_encoding_id encoding{};
    std::uint64_t generation = 0;
    exact_coverage_view_v1 contribution_owners{};
};

struct partial_result_combination_v1 {
    composition_production_id production{};
    partial_result_id left{};
    partial_result_id right{};
    partial_result_identity_v1 combined{};
};

enum class partial_result_combination_code_v1 : std::uint32_t {
    combined = 0,
    invalid_production,
    invalid_output_identity,
    invalid_partial,
    result_identity_mismatch,
    algebra_mismatch,
    encoding_mismatch,
    stale_output_generation,
    contribution_composition_failed,
    missing_output,
};

struct partial_result_combination_result_v1 {
    partial_result_combination_code_v1 code =
        partial_result_combination_code_v1::combined;
    std::uint64_t subject = 0;
    [[nodiscard]] constexpr bool combined() const noexcept {
        return code == partial_result_combination_code_v1::combined;
    }
};

[[nodiscard]] inline bool valid_partial_result_identity_v1(
    const partial_result_identity_v1 &partial) noexcept {
    return partial.identity.valid() && partial.logical_result_identity != 0
        && partial.combination_algebra.valid() && partial.encoding.valid()
        && partial.generation != 0
        && validate_exact_coverage_v1(partial.contribution_owners).composed();
}

[[nodiscard]] inline partial_result_combination_result_v1
compose_partial_result_combination_v1(
    composition_production_id production,
    partial_result_id output_identity,
    std::uint64_t output_generation,
    structure_id output_coverage_identity,
    const partial_result_identity_v1 &left,
    const partial_result_identity_v1 &right,
    mutable_coverage_storage_v1 coverage_storage,
    partial_result_combination_v1 *output) noexcept {
    if (!production.valid()) {
        return {partial_result_combination_code_v1::invalid_production};
    }
    if (!output_identity.valid()) {
        return {partial_result_combination_code_v1::invalid_output_identity};
    }
    if (!valid_partial_result_identity_v1(left)
        || !valid_partial_result_identity_v1(right)) {
        return {partial_result_combination_code_v1::invalid_partial};
    }
    if (left.logical_result_identity != right.logical_result_identity) {
        return {partial_result_combination_code_v1::
                    result_identity_mismatch};
    }
    if (left.combination_algebra != right.combination_algebra) {
        return {partial_result_combination_code_v1::algebra_mismatch};
    }
    if (left.encoding != right.encoding) {
        return {partial_result_combination_code_v1::encoding_mismatch};
    }
    if (output_generation <= left.generation
        || output_generation <= right.generation) {
        return {partial_result_combination_code_v1::stale_output_generation};
    }
    exact_coverage_view_v1 combined_coverage{};
    const auto coverage_status = compose_disjoint_union_v1(
        output_coverage_identity, left.contribution_owners,
        right.contribution_owners, coverage_storage, &combined_coverage);
    if (!coverage_status.composed()) {
        return {partial_result_combination_code_v1::
                    contribution_composition_failed,
                static_cast<std::uint64_t>(coverage_status.code)};
    }
    if (output == nullptr) {
        return {partial_result_combination_code_v1::missing_output};
    }
    *output = {production, left.identity, right.identity,
               {output_identity, left.logical_result_identity,
                left.combination_algebra, left.encoding, output_generation,
                combined_coverage}};
    return {partial_result_combination_code_v1::combined,
            combined_coverage.logical_item_count};
}

static_assert(std::is_trivially_copyable<partial_result_identity_v1>::value);
static_assert(std::is_trivially_copyable<partial_result_combination_v1>::value);

} // namespace cellshard::compiler::composition
