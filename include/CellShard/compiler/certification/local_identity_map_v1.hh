#pragma once

#include <CellShard/compiler/certification/atom_certification_v1.hh>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t local_identity_map_contract_version_v1 = 1;

struct local_identity_map_buffers_v1 {
    std::byte *canonical_to_local = nullptr;
    std::uint64_t canonical_to_local_bytes = 0;
    std::byte *local_to_canonical = nullptr;
    std::uint64_t local_to_canonical_bytes = 0;
    std::uint8_t *canonical_marks = nullptr;
    std::uint64_t canonical_mark_capacity = 0;
};

struct local_identity_map_view_v1 {
    const std::byte *canonical_to_local = nullptr;
    const std::byte *local_to_canonical = nullptr;
    std::uint64_t canonical_count = 0;
    std::uint64_t local_count = 0;
    std::uint64_t canonical_to_local_bytes = 0;
    std::uint64_t local_to_canonical_bytes = 0;
    certification_local_index_width_v1 index_width =
        certification_local_index_width_v1::u8;
    std::uint8_t reserved[7]{};
};

enum class local_identity_map_build_code_v1 : std::uint32_t {
    built = 0,
    empty_canonical,
    missing_canonical,
    empty_local,
    missing_local,
    invalid_maximum_width,
    unordered_or_duplicate_canonical,
    width_exceeded,
    byte_count_overflow,
    missing_output,
    insufficient_output,
    missing_marks,
    insufficient_marks,
    zero_local_identity,
    local_identity_not_canonical,
    duplicate_local_identity,
};

struct local_identity_map_build_result_v1 {
    local_identity_map_build_code_v1 code =
        local_identity_map_build_code_v1::built;
    local_identity_map_view_v1 map{};
    std::uint64_t index = 0;
    std::uint64_t required_canonical_bytes = 0;
    std::uint64_t required_local_bytes = 0;

    [[nodiscard]] constexpr bool built() const noexcept {
        return code == local_identity_map_build_code_v1::built;
    }
};

[[nodiscard]] constexpr std::uint64_t local_index_max_v1(
    certification_local_index_width_v1 width) noexcept {
    switch (width) {
    case certification_local_index_width_v1::u8:
        return UINT8_MAX;
    case certification_local_index_width_v1::u16:
        return UINT16_MAX;
    case certification_local_index_width_v1::u32:
        return UINT32_MAX;
    case certification_local_index_width_v1::u64:
        return UINT64_MAX;
    }
    return 0;
}

[[nodiscard]] constexpr certification_local_index_width_v1
minimum_local_index_width_v1(
    std::uint64_t canonical_count,
    std::uint64_t local_count) noexcept {
    const auto needed = canonical_count > local_count
        ? canonical_count
        : local_count;
    if (needed <= UINT8_MAX) {
        return certification_local_index_width_v1::u8;
    }
    if (needed <= UINT16_MAX) {
        return certification_local_index_width_v1::u16;
    }
    if (needed <= UINT32_MAX) {
        return certification_local_index_width_v1::u32;
    }
    return certification_local_index_width_v1::u64;
}

inline void write_local_index_v1(
    std::byte *destination,
    std::uint64_t index,
    certification_local_index_width_v1 width,
    std::uint64_t value) noexcept {
    const auto bytes = static_cast<std::uint8_t>(width);
    auto *output = destination + index * bytes;
    for (std::uint8_t byte = 0; byte < bytes; ++byte) {
        output[byte] = static_cast<std::byte>(value >> (byte * 8));
    }
}

[[nodiscard]] inline std::uint64_t read_local_index_v1(
    const std::byte *source,
    std::uint64_t index,
    certification_local_index_width_v1 width) noexcept {
    const auto bytes = static_cast<std::uint8_t>(width);
    const auto *input = source + index * bytes;
    std::uint64_t value = 0;
    for (std::uint8_t byte = 0; byte < bytes; ++byte) {
        value |= static_cast<std::uint64_t>(input[byte]) << (byte * 8);
    }
    return value;
}

// Builds both directions in O(C + L log C), with exactly C caller-owned mark
// bytes. Global identities remain u64; only the two local ordinal arrays use
// the smallest allowed compact width. The all-ones value is the absent sentinel.
[[nodiscard]] inline local_identity_map_build_result_v1
build_local_identity_maps_v1(
    const std::uint64_t *canonical_global_ids,
    std::uint64_t canonical_count,
    const std::uint64_t *local_global_ids,
    std::uint64_t local_count,
    certification_local_index_width_v1 maximum_width,
    local_identity_map_buffers_v1 buffers) noexcept {
    if (canonical_count == 0) {
        return {local_identity_map_build_code_v1::empty_canonical};
    }
    if (canonical_global_ids == nullptr) {
        return {local_identity_map_build_code_v1::missing_canonical};
    }
    if (local_count == 0) {
        return {local_identity_map_build_code_v1::empty_local};
    }
    if (local_global_ids == nullptr) {
        return {local_identity_map_build_code_v1::missing_local};
    }
    if (!valid_certification_local_index_width_v1(maximum_width)) {
        return {local_identity_map_build_code_v1::invalid_maximum_width};
    }
    for (std::uint64_t index = 0; index < canonical_count; ++index) {
        if (canonical_global_ids[index] == 0
            || (index != 0
                && canonical_global_ids[index - 1]
                    >= canonical_global_ids[index])) {
            return {local_identity_map_build_code_v1::
                        unordered_or_duplicate_canonical,
                    {},
                    index};
        }
    }
    const auto width = minimum_local_index_width_v1(canonical_count,
                                                     local_count);
    if (static_cast<std::uint8_t>(width)
        > static_cast<std::uint8_t>(maximum_width)) {
        return {local_identity_map_build_code_v1::width_exceeded};
    }
    const auto width_bytes = static_cast<std::uint8_t>(width);
    if (canonical_count > UINT64_MAX / width_bytes
        || local_count > UINT64_MAX / width_bytes) {
        return {local_identity_map_build_code_v1::byte_count_overflow};
    }
    const auto canonical_bytes = canonical_count * width_bytes;
    const auto local_bytes = local_count * width_bytes;
    if (buffers.canonical_to_local == nullptr
        || buffers.local_to_canonical == nullptr) {
        return {local_identity_map_build_code_v1::missing_output,
                {},
                0,
                canonical_bytes,
                local_bytes};
    }
    if (buffers.canonical_to_local_bytes < canonical_bytes
        || buffers.local_to_canonical_bytes < local_bytes) {
        return {local_identity_map_build_code_v1::insufficient_output,
                {},
                0,
                canonical_bytes,
                local_bytes};
    }
    if (buffers.canonical_marks == nullptr) {
        return {local_identity_map_build_code_v1::missing_marks};
    }
    if (buffers.canonical_mark_capacity < canonical_count) {
        return {local_identity_map_build_code_v1::insufficient_marks,
                {},
                buffers.canonical_mark_capacity};
    }
    const auto absent = local_index_max_v1(width);
    for (std::uint64_t index = 0; index < canonical_count; ++index) {
        buffers.canonical_marks[index] = 0;
        write_local_index_v1(
            buffers.canonical_to_local, index, width, absent);
    }
    for (std::uint64_t local_index = 0; local_index < local_count;
         ++local_index) {
        const auto identity = local_global_ids[local_index];
        if (identity == 0) {
            return {local_identity_map_build_code_v1::zero_local_identity,
                    {},
                    local_index};
        }
        std::uint64_t begin = 0;
        std::uint64_t end = canonical_count;
        while (begin < end) {
            const auto middle = begin + (end - begin) / 2;
            if (canonical_global_ids[middle] < identity) {
                begin = middle + 1;
            } else {
                end = middle;
            }
        }
        if (begin == canonical_count || canonical_global_ids[begin] != identity) {
            return {local_identity_map_build_code_v1::
                        local_identity_not_canonical,
                    {},
                    local_index};
        }
        if (buffers.canonical_marks[begin] != 0) {
            return {local_identity_map_build_code_v1::duplicate_local_identity,
                    {},
                    local_index};
        }
        buffers.canonical_marks[begin] = 1;
        write_local_index_v1(
            buffers.canonical_to_local, begin, width, local_index);
        write_local_index_v1(
            buffers.local_to_canonical, local_index, width, begin);
    }
    return {local_identity_map_build_code_v1::built,
            {buffers.canonical_to_local,
             buffers.local_to_canonical,
             canonical_count,
             local_count,
             canonical_bytes,
             local_bytes,
             width,
             {}},
            local_count,
            canonical_bytes,
            local_bytes};
}

} // namespace cellshard::compiler::certification
