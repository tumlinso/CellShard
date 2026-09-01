#pragma once

#include <CellShard/compiler/composition/physical_view_addition_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::composition {

struct packed_projection_extent_v1 {
    image_id projection{};
    std::uint64_t byte_offset = 0;
    std::uint64_t byte_count = 0;
    std::uint32_t required_alignment = 0;
    std::uint32_t reserved = 0;
};

struct projection_pack_composition_v1 {
    image_id pack_identity{};
    structure_id structure{};
    const packed_projection_extent_v1 *extents = nullptr;
    std::uint64_t packed_bytes = 0;
    std::uint32_t extent_count = 0;
    std::uint32_t reserved = 0;
};

enum class projection_packing_code_v1 : std::uint32_t {
    packed = 0,
    invalid_pack_identity,
    invalid_family,
    empty_family,
    invalid_view,
    structure_mismatch,
    arithmetic_overflow,
    missing_storage,
    insufficient_capacity,
    missing_output,
};

struct projection_packing_result_v1 {
    projection_packing_code_v1 code = projection_packing_code_v1::packed;
    std::uint32_t view_index = 0;
    [[nodiscard]] constexpr bool packed() const noexcept {
        return code == projection_packing_code_v1::packed;
    }
};

[[nodiscard]] inline projection_packing_result_v1
compose_projection_packing_v1(
    image_id pack_identity,
    const physical_view_family_v1 &family,
    packed_projection_extent_v1 *storage,
    std::uint32_t capacity,
    projection_pack_composition_v1 *output) noexcept {
    if (!pack_identity.valid()) {
        return {projection_packing_code_v1::invalid_pack_identity};
    }
    if (!family.structure.valid()
        || (family.view_count != 0 && family.views == nullptr)
        || family.view_count > max_composed_physical_views_v1) {
        return {projection_packing_code_v1::invalid_family};
    }
    if (family.view_count == 0) {
        return {projection_packing_code_v1::empty_family};
    }
    if (storage == nullptr) {
        return {projection_packing_code_v1::missing_storage};
    }
    if (capacity < family.view_count) {
        return {projection_packing_code_v1::insufficient_capacity};
    }
    if (output == nullptr) return {projection_packing_code_v1::missing_output};
    *output = {};
    std::uint64_t offset = 0;
    for (std::uint32_t index = 0; index < family.view_count; ++index) {
        const auto &view = family.views[index];
        if (!valid_physical_view_identity_v1(view)) {
            return {projection_packing_code_v1::invalid_view, index};
        }
        if (view.structure != family.structure) {
            return {projection_packing_code_v1::structure_mismatch, index};
        }
        const auto alignment = static_cast<std::uint64_t>(
            view.required_alignment);
        const auto mask = alignment - 1;
        if (offset > std::numeric_limits<std::uint64_t>::max() - mask) {
            return {projection_packing_code_v1::arithmetic_overflow, index};
        }
        const auto aligned_offset = (offset + mask) & ~mask;
        if (view.stored_bytes
            > std::numeric_limits<std::uint64_t>::max() - aligned_offset) {
            return {projection_packing_code_v1::arithmetic_overflow, index};
        }
        storage[index] = {view.identity, aligned_offset, view.stored_bytes,
                          view.required_alignment, 0};
        offset = aligned_offset + view.stored_bytes;
    }
    *output = {pack_identity, family.structure, storage, offset,
               family.view_count, 0};
    return {projection_packing_code_v1::packed, family.view_count};
}

static_assert(std::is_trivially_copyable<packed_projection_extent_v1>::value);
static_assert(std::is_trivially_copyable<projection_pack_composition_v1>::value);

} // namespace cellshard::compiler::composition
