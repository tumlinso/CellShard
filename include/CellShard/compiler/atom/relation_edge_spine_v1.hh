#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t relation_edge_spine_contract_version_v1 = 1;

// Canonical edge IDs are global u64 values in deterministic ascending order.
// The array order is the canonical value/gradient/residual order shared by all
// physical views; an array position alone is never an edge identity.
struct relation_edge_spine_view_v1 {
    const std::uint64_t *global_edge_ids = nullptr;
    std::uint64_t edge_count = 0;
    atom_persistent_identity_v1 relation_identity{};
    std::uint64_t structure_epoch = 0;
};

enum class compact_edge_index_width_v1 : std::uint8_t {
    u8 = 1,
    u16 = 2,
    u32 = 4,
    u64 = 8,
};

// A physical support/value/forward/transpose/contraction/gradient view may use
// a different local order. Its byte-packed indices map each local edge back to
// a canonical spine ordinal. The mapping does not redefine edge identity.
struct relation_edge_local_map_view_v1 {
    const std::byte *canonical_ordinals = nullptr;
    std::uint64_t local_edge_count = 0;
    std::uint64_t canonical_ordinal_bytes = 0;
    compact_edge_index_width_v1 index_width = compact_edge_index_width_v1::u8;
    std::uint8_t reserved[7]{};
    atom_persistent_identity_v1 relation_identity{};
    std::uint64_t structure_epoch = 0;
};

enum class relation_edge_spine_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_relation_identity,
    missing_structure_epoch,
    empty_spine,
    missing_global_edge_ids,
    zero_global_edge_identity,
    unordered_or_duplicate_global_edge,
    empty_local_map,
    missing_local_map,
    invalid_index_width,
    index_bytes_overflow,
    invalid_index_bytes,
    nonzero_reserved,
    relation_mismatch,
    epoch_mismatch,
    width_too_small,
    missing_marks,
    insufficient_marks,
    local_index_out_of_range,
    duplicate_local_index,
};

struct relation_edge_spine_validation_v1 {
    relation_edge_spine_validation_code_v1 code =
        relation_edge_spine_validation_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == relation_edge_spine_validation_code_v1::valid;
    }
};

static_assert(offsetof(relation_edge_spine_view_v1, global_edge_ids) == 0,
              "edge spine views must remain pointer-first");
static_assert(offsetof(relation_edge_local_map_view_v1, canonical_ordinals) == 0,
              "local edge maps must remain pointer-first");
static_assert(std::is_standard_layout<relation_edge_spine_view_v1>::value);
static_assert(std::is_trivially_copyable<relation_edge_spine_view_v1>::value);
static_assert(std::is_standard_layout<relation_edge_local_map_view_v1>::value);
static_assert(
    std::is_trivially_copyable<relation_edge_local_map_view_v1>::value);

[[nodiscard]] constexpr bool valid_compact_edge_index_width_v1(
    compact_edge_index_width_v1 width) noexcept {
    const auto bytes = static_cast<std::uint8_t>(width);
    return bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8;
}

[[nodiscard]] constexpr std::uint64_t compact_edge_index_capacity_v1(
    compact_edge_index_width_v1 width) noexcept {
    switch (width) {
    case compact_edge_index_width_v1::u8:
        return UINT64_C(1) << 8;
    case compact_edge_index_width_v1::u16:
        return UINT64_C(1) << 16;
    case compact_edge_index_width_v1::u32:
        return UINT64_C(1) << 32;
    case compact_edge_index_width_v1::u64:
        return std::numeric_limits<std::uint64_t>::max();
    }
    return 0;
}

// Byte decoding is alignment-independent and explicitly little-endian. It is
// O(index_width), with index_width bounded by eight bytes.
[[nodiscard]] inline std::uint64_t read_compact_edge_index_v1(
    const std::byte *indices,
    std::uint64_t index,
    compact_edge_index_width_v1 width) noexcept {
    const auto bytes = static_cast<std::uint8_t>(width);
    const auto *source = indices + index * bytes;
    std::uint64_t value = 0;
    for (std::uint8_t byte = 0; byte < bytes; ++byte) {
        value |= static_cast<std::uint64_t>(source[byte]) << (byte * 8);
    }
    return value;
}

// O(edge_count) time, O(1) storage. Ascending global IDs make duplicates and
// the canonical order independently checkable without hidden allocation.
[[nodiscard]] constexpr relation_edge_spine_validation_v1
validate_relation_edge_spine_v1(
    relation_edge_spine_view_v1 spine) noexcept {
    if (!validate_atom_persistent_identity_v1(spine.relation_identity).valid()) {
        return {relation_edge_spine_validation_code_v1::
                    invalid_relation_identity,
                0};
    }
    if (spine.structure_epoch == 0) {
        return {relation_edge_spine_validation_code_v1::
                    missing_structure_epoch,
                0};
    }
    if (spine.edge_count == 0) {
        return {relation_edge_spine_validation_code_v1::empty_spine, 0};
    }
    if (spine.global_edge_ids == nullptr) {
        return {relation_edge_spine_validation_code_v1::
                    missing_global_edge_ids,
                0};
    }
    for (std::uint64_t index = 0; index < spine.edge_count; ++index) {
        if (spine.global_edge_ids[index] == 0) {
            return {relation_edge_spine_validation_code_v1::
                        zero_global_edge_identity,
                    index};
        }
        if (index != 0
            && spine.global_edge_ids[index - 1]
                >= spine.global_edge_ids[index]) {
            return {relation_edge_spine_validation_code_v1::
                        unordered_or_duplicate_global_edge,
                    index};
        }
    }
    return {relation_edge_spine_validation_code_v1::valid, spine.edge_count};
}

// O(spine.edge_count + local_edge_count) time and one caller-owned byte per
// canonical edge. The marks make arbitrary forward/transpose local orders
// independently checkable without sorting or hidden allocation.
[[nodiscard]] inline relation_edge_spine_validation_v1
validate_relation_edge_local_map_v1(
    relation_edge_spine_view_v1 spine,
    relation_edge_local_map_view_v1 mapping,
    std::uint8_t *marks,
    std::uint64_t mark_capacity) noexcept {
    const auto spine_result = validate_relation_edge_spine_v1(spine);
    if (!spine_result.valid()) {
        return spine_result;
    }
    if (mapping.local_edge_count == 0) {
        return {relation_edge_spine_validation_code_v1::empty_local_map, 0};
    }
    if (mapping.canonical_ordinals == nullptr) {
        return {relation_edge_spine_validation_code_v1::missing_local_map, 0};
    }
    if (!valid_compact_edge_index_width_v1(mapping.index_width)) {
        return {relation_edge_spine_validation_code_v1::invalid_index_width, 0};
    }
    const auto width_bytes = static_cast<std::uint8_t>(mapping.index_width);
    if (mapping.local_edge_count
        > std::numeric_limits<std::uint64_t>::max() / width_bytes) {
        return {relation_edge_spine_validation_code_v1::index_bytes_overflow,
                0};
    }
    if (mapping.canonical_ordinal_bytes
        != mapping.local_edge_count * width_bytes) {
        return {relation_edge_spine_validation_code_v1::invalid_index_bytes,
                mapping.canonical_ordinal_bytes};
    }
    for (const auto byte : mapping.reserved) {
        if (byte != 0) {
            return {relation_edge_spine_validation_code_v1::nonzero_reserved,
                    0};
        }
    }
    if (mapping.relation_identity != spine.relation_identity) {
        return {relation_edge_spine_validation_code_v1::relation_mismatch, 0};
    }
    if (mapping.structure_epoch != spine.structure_epoch) {
        return {relation_edge_spine_validation_code_v1::epoch_mismatch, 0};
    }
    if (spine.edge_count
        > compact_edge_index_capacity_v1(mapping.index_width)) {
        return {relation_edge_spine_validation_code_v1::width_too_small, 0};
    }
    if (marks == nullptr) {
        return {relation_edge_spine_validation_code_v1::missing_marks, 0};
    }
    if (mark_capacity < spine.edge_count) {
        return {relation_edge_spine_validation_code_v1::insufficient_marks,
                mark_capacity};
    }
    for (std::uint64_t index = 0; index < spine.edge_count; ++index) {
        marks[index] = 0;
    }
    for (std::uint64_t index = 0; index < mapping.local_edge_count; ++index) {
        const auto canonical = read_compact_edge_index_v1(
            mapping.canonical_ordinals, index, mapping.index_width);
        if (canonical >= spine.edge_count) {
            return {
                relation_edge_spine_validation_code_v1::local_index_out_of_range,
                index};
        }
        if (marks[canonical] != 0) {
            return {relation_edge_spine_validation_code_v1::duplicate_local_index,
                    index};
        }
        marks[canonical] = 1;
    }
    return {relation_edge_spine_validation_code_v1::valid,
            mapping.local_edge_count};
}

} // namespace cellshard::compiler::atom
