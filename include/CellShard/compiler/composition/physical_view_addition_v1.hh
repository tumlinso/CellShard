#pragma once

#include <CellShard/artifact/image.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

inline constexpr std::uint32_t max_composed_physical_views_v1 = 64;

struct physical_view_identity_v1 {
    image_id identity{};
    structure_id structure{};
    geometry_id geometry{};
    operator_class_id operation{};
    scalar_encoding_id encoding{};
    order_id physical_order{};
    std::uint64_t stored_bytes = 0;
    std::uint64_t device_bytes = 0;
    std::uint32_t required_alignment = 0;
    std::uint32_t reserved = 0;
};

struct physical_view_family_v1 {
    structure_id structure{};
    const physical_view_identity_v1 *views = nullptr;
    std::uint32_t view_count = 0;
    std::uint32_t reserved = 0;
};

enum class physical_view_addition_code_v1 : std::uint32_t {
    added = 0,
    invalid_family_structure,
    excessive_view_count,
    missing_views,
    invalid_view,
    structure_mismatch,
    unordered_view_identity,
    duplicate_view_identity,
    missing_storage,
    insufficient_capacity,
    missing_output,
};

struct physical_view_addition_result_v1 {
    physical_view_addition_code_v1 code =
        physical_view_addition_code_v1::added;
    std::uint32_t view_index = 0;
    [[nodiscard]] constexpr bool added() const noexcept {
        return code == physical_view_addition_code_v1::added;
    }
};

[[nodiscard]] constexpr bool valid_physical_view_identity_v1(
    const physical_view_identity_v1 &view) noexcept {
    return view.identity.valid() && view.structure.valid()
        && view.geometry.valid() && view.operation.valid()
        && view.encoding.valid() && view.physical_order.valid()
        && view.stored_bytes != 0 && view.device_bytes != 0
        && valid_required_alignment(view.required_alignment)
        && view.reserved == 0;
}

[[nodiscard]] inline physical_view_addition_result_v1
compose_physical_view_addition_v1(
    const physical_view_family_v1 &family,
    const physical_view_identity_v1 &addition,
    physical_view_identity_v1 *storage,
    std::uint32_t capacity,
    physical_view_family_v1 *output) noexcept {
    if (!family.structure.valid()) {
        return {physical_view_addition_code_v1::invalid_family_structure};
    }
    if (family.view_count >= max_composed_physical_views_v1) {
        return {physical_view_addition_code_v1::excessive_view_count};
    }
    if (family.view_count != 0 && family.views == nullptr) {
        return {physical_view_addition_code_v1::missing_views};
    }
    for (std::uint32_t index = 0; index < family.view_count; ++index) {
        if (!valid_physical_view_identity_v1(family.views[index])) {
            return {physical_view_addition_code_v1::invalid_view, index};
        }
        if (family.views[index].structure != family.structure) {
            return {physical_view_addition_code_v1::structure_mismatch, index};
        }
        if (index != 0
            && family.views[index - 1].identity
                   >= family.views[index].identity) {
            return {physical_view_addition_code_v1::unordered_view_identity,
                    index};
        }
    }
    if (!valid_physical_view_identity_v1(addition)) {
        return {physical_view_addition_code_v1::invalid_view,
                family.view_count};
    }
    if (addition.structure != family.structure) {
        return {physical_view_addition_code_v1::structure_mismatch,
                family.view_count};
    }
    if (storage == nullptr) {
        return {physical_view_addition_code_v1::missing_storage};
    }
    if (capacity < family.view_count + 1u) {
        return {physical_view_addition_code_v1::insufficient_capacity};
    }
    if (output == nullptr) return {physical_view_addition_code_v1::missing_output};
    *output = {};
    std::uint32_t source = 0;
    std::uint32_t destination = 0;
    bool inserted = false;
    while (source < family.view_count) {
        if (!inserted && addition.identity < family.views[source].identity) {
            storage[destination++] = addition;
            inserted = true;
        } else if (family.views[source].identity == addition.identity) {
            return {physical_view_addition_code_v1::duplicate_view_identity,
                    source};
        }
        storage[destination++] = family.views[source++];
    }
    if (!inserted) storage[destination++] = addition;
    *output = {family.structure, storage, destination, 0};
    return {physical_view_addition_code_v1::added, destination};
}

static_assert(std::is_trivially_copyable<physical_view_identity_v1>::value);
static_assert(std::is_trivially_copyable<physical_view_family_v1>::value);

} // namespace cellshard::compiler::composition
