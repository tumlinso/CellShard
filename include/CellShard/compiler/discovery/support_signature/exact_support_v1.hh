#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>
#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::support_signature {

struct destination_source_edge_v1 {
    std::uint64_t global_destination_id = 0;
    std::uint64_t global_source_id = 0;
};

struct exact_destination_support_view_v1 {
    const std::uint64_t *global_destination_ids = nullptr;
    const std::uint64_t *destination_offsets = nullptr;
    const std::uint64_t *global_source_ids = nullptr;
    std::uint64_t destination_count = 0;
    std::uint64_t edge_count = 0;
    atom::atom_persistent_identity_v1 relation_identity{};
    atom::atom_persistent_identity_v1 source_domain_identity{};
    atom::atom_persistent_identity_v1 destination_domain_identity{};
    std::uint64_t relation_generation = 0;
};

struct exact_destination_support_buffers_v1 {
    destination_source_edge_v1 *edge_scratch = nullptr;
    std::uint64_t scratch_capacity = 0;
    std::uint64_t *global_destination_ids = nullptr;
    std::uint64_t destination_capacity = 0;
    std::uint64_t *destination_offsets = nullptr;
    std::uint64_t offset_capacity = 0;
    std::uint64_t *global_source_ids = nullptr;
    std::uint64_t source_capacity = 0;
};

enum class exact_destination_support_code_v1 : std::uint32_t {
    built = 0,
    empty_edges,
    missing_edges,
    invalid_relation_identity,
    invalid_source_domain_identity,
    invalid_destination_domain_identity,
    missing_relation_generation,
    invalid_edge_identity,
    duplicate_edge,
    missing_output,
    insufficient_output,
};

struct exact_destination_support_result_v1 {
    exact_destination_support_code_v1 code =
        exact_destination_support_code_v1::built;
    exact_destination_support_view_v1 view{};
    std::uint64_t index = 0;
    std::uint64_t required_destinations = 0;
    [[nodiscard]] constexpr bool built() const noexcept {
        return code == exact_destination_support_code_v1::built;
    }
};

static_assert(std::is_standard_layout<destination_source_edge_v1>::value);
static_assert(std::is_trivially_copyable<destination_source_edge_v1>::value);
static_assert(offsetof(exact_destination_support_view_v1,
                       global_destination_ids) == 0);
static_assert(std::is_standard_layout<exact_destination_support_view_v1>::value);
static_assert(
    std::is_trivially_copyable<exact_destination_support_view_v1>::value);

[[nodiscard]] constexpr bool destination_source_edge_less_v1(
    destination_source_edge_v1 lhs,
    destination_source_edge_v1 rhs) noexcept {
    return lhs.global_destination_id < rhs.global_destination_id
        || (lhs.global_destination_id == rhs.global_destination_id
            && lhs.global_source_id < rhs.global_source_id);
}

// O(E log E) time and O(E) explicit caller-owned scratch. The result is the
// complete canonical source support for every observed destination, not a
// sketch, digest, storage ordinal, or execution certificate.
[[nodiscard]] inline exact_destination_support_result_v1
build_exact_destination_support_v1(
    const destination_source_edge_v1 *edges,
    std::uint64_t edge_count,
    atom::atom_persistent_identity_v1 relation_identity,
    atom::atom_persistent_identity_v1 source_domain_identity,
    atom::atom_persistent_identity_v1 destination_domain_identity,
    std::uint64_t relation_generation,
    exact_destination_support_buffers_v1 buffers) noexcept {
    if (edge_count == 0) {
        return {exact_destination_support_code_v1::empty_edges};
    }
    if (edges == nullptr) {
        return {exact_destination_support_code_v1::missing_edges};
    }
    if (!atom::validate_atom_persistent_identity_v1(relation_identity).valid()) {
        return {exact_destination_support_code_v1::invalid_relation_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             source_domain_identity).valid()) {
        return {exact_destination_support_code_v1::
                    invalid_source_domain_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             destination_domain_identity).valid()) {
        return {exact_destination_support_code_v1::
                    invalid_destination_domain_identity};
    }
    if (relation_generation == 0) {
        return {exact_destination_support_code_v1::
                    missing_relation_generation};
    }
    if (buffers.edge_scratch == nullptr
        || buffers.global_destination_ids == nullptr
        || buffers.destination_offsets == nullptr
        || buffers.global_source_ids == nullptr) {
        return {exact_destination_support_code_v1::missing_output};
    }
    if (buffers.scratch_capacity < edge_count
        || buffers.source_capacity < edge_count
        || buffers.destination_capacity < edge_count
        || buffers.offset_capacity < edge_count + 1) {
        return {exact_destination_support_code_v1::insufficient_output};
    }
    for (std::uint64_t index = 0; index < edge_count; ++index) {
        if (edges[index].global_destination_id == 0
            || edges[index].global_source_id == 0) {
            return {exact_destination_support_code_v1::invalid_edge_identity,
                    {}, index};
        }
        buffers.edge_scratch[index] = edges[index];
    }
    std::sort(buffers.edge_scratch, buffers.edge_scratch + edge_count,
              destination_source_edge_less_v1);
    for (std::uint64_t index = 1; index < edge_count; ++index) {
        const auto &previous = buffers.edge_scratch[index - 1];
        const auto &current = buffers.edge_scratch[index];
        if (previous.global_destination_id == current.global_destination_id
            && previous.global_source_id == current.global_source_id) {
            return {exact_destination_support_code_v1::duplicate_edge,
                    {}, index};
        }
    }
    std::uint64_t destination_count = 0;
    for (std::uint64_t index = 0; index < edge_count; ++index) {
        const auto &edge = buffers.edge_scratch[index];
        if (index == 0
            || edge.global_destination_id
                != buffers.edge_scratch[index - 1].global_destination_id) {
            buffers.global_destination_ids[destination_count] =
                edge.global_destination_id;
            buffers.destination_offsets[destination_count] = index;
            ++destination_count;
        }
        buffers.global_source_ids[index] = edge.global_source_id;
    }
    buffers.destination_offsets[destination_count] = edge_count;
    return {exact_destination_support_code_v1::built,
            {buffers.global_destination_ids, buffers.destination_offsets,
             buffers.global_source_ids, destination_count, edge_count,
             relation_identity, source_domain_identity,
             destination_domain_identity, relation_generation},
            edge_count, destination_count};
}

[[nodiscard]] constexpr bool authorizes_execution(
    exact_destination_support_view_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::support_signature
