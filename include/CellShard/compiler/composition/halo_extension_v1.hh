#pragma once

#include <CellShard/compiler/composition/coverage_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

enum class coverage_ownership_role_v1 : std::uint8_t {
    contribution_owner = 1,
    halo_read_only = 2,
};

struct owned_coverage_item_v1 {
    std::uint64_t logical_identity = 0;
    coverage_ownership_role_v1 role =
        coverage_ownership_role_v1::contribution_owner;
    std::uint8_t reserved[7]{};
};

struct halo_extension_view_v1 {
    structure_id identity{};
    domain_id domain{};
    order_id order{};
    const owned_coverage_item_v1 *items = nullptr;
    std::uint64_t item_count = 0;
    std::uint64_t owned_count = 0;
    std::uint64_t halo_count = 0;
};

enum class halo_extension_code_v1 : std::uint32_t {
    extended = 0,
    invalid_output_identity,
    invalid_owned_coverage,
    invalid_halo_coverage,
    domain_mismatch,
    order_mismatch,
    count_overflow,
    ownership_overlap,
    missing_storage,
    insufficient_capacity,
    missing_output,
};

struct halo_extension_result_v1 {
    halo_extension_code_v1 code = halo_extension_code_v1::extended;
    std::uint64_t logical_identity = 0;
    [[nodiscard]] constexpr bool extended() const noexcept {
        return code == halo_extension_code_v1::extended;
    }
};

[[nodiscard]] inline halo_extension_result_v1 compose_halo_extension_v1(
    structure_id output_identity,
    const exact_coverage_view_v1 &owned,
    const exact_coverage_view_v1 &halo,
    owned_coverage_item_v1 *storage,
    std::uint64_t capacity,
    halo_extension_view_v1 *output) noexcept {
    if (!output_identity.valid()) {
        return {halo_extension_code_v1::invalid_output_identity};
    }
    if (!validate_exact_coverage_v1(owned).composed()) {
        return {halo_extension_code_v1::invalid_owned_coverage};
    }
    if (!validate_exact_coverage_v1(halo).composed()) {
        return {halo_extension_code_v1::invalid_halo_coverage};
    }
    if (owned.domain != halo.domain) {
        return {halo_extension_code_v1::domain_mismatch};
    }
    if (owned.order != halo.order) {
        return {halo_extension_code_v1::order_mismatch};
    }
    if (owned.logical_item_count
        > std::numeric_limits<std::uint64_t>::max()
              - halo.logical_item_count) {
        return {halo_extension_code_v1::count_overflow};
    }
    const auto item_count = owned.logical_item_count + halo.logical_item_count;
    std::uint64_t owned_index = 0;
    std::uint64_t halo_index = 0;
    while (owned_index < owned.logical_item_count
           && halo_index < halo.logical_item_count) {
        const auto owned_id = owned.logical_item_ids[owned_index];
        const auto halo_id = halo.logical_item_ids[halo_index];
        if (owned_id == halo_id) {
            return {halo_extension_code_v1::ownership_overlap, owned_id};
        }
        if (owned_id < halo_id) ++owned_index;
        else ++halo_index;
    }
    if (item_count != 0 && storage == nullptr) {
        return {halo_extension_code_v1::missing_storage};
    }
    if (capacity < item_count) {
        return {halo_extension_code_v1::insufficient_capacity};
    }
    if (output == nullptr) return {halo_extension_code_v1::missing_output};
    *output = {};
    owned_index = 0;
    halo_index = 0;
    std::uint64_t output_index = 0;
    while (owned_index < owned.logical_item_count
           || halo_index < halo.logical_item_count) {
        const bool take_owned = halo_index == halo.logical_item_count
            || (owned_index < owned.logical_item_count
                && owned.logical_item_ids[owned_index]
                       < halo.logical_item_ids[halo_index]);
        if (take_owned) {
            storage[output_index++] = {
                owned.logical_item_ids[owned_index++],
                coverage_ownership_role_v1::contribution_owner, {}};
        } else {
            storage[output_index++] = {
                halo.logical_item_ids[halo_index++],
                coverage_ownership_role_v1::halo_read_only, {}};
        }
    }
    *output = {output_identity, owned.domain, owned.order, storage, item_count,
               owned.logical_item_count, halo.logical_item_count};
    return {halo_extension_code_v1::extended, item_count};
}

static_assert(std::is_trivially_copyable<owned_coverage_item_v1>::value);
static_assert(std::is_trivially_copyable<halo_extension_view_v1>::value);

} // namespace cellshard::compiler::composition
