#pragma once

#include <CellShard/compiler/composition/production_identity_v1.hh>
#include <CellShard/compiler/composition/relation_merge_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

inline constexpr std::uint32_t max_relation_bundle_relations_v1 = 32;

struct relation_bundle_tag {};
using relation_bundle_id = strong_id<relation_bundle_tag>;

struct relation_bundle_composition_v1 {
    relation_bundle_id identity{};
    composition_production_id production{};
    const typed_relation_view_v1 *relations = nullptr;
    std::uint32_t relation_count = 0;
    std::uint32_t reserved = 0;
    std::uint64_t total_logical_edges = 0;
};

enum class relation_bundle_code_v1 : std::uint32_t {
    composed = 0,
    invalid_bundle_identity,
    invalid_production,
    invalid_relation_count,
    missing_relations,
    invalid_relation,
    unordered_relation_identity,
    edge_count_overflow,
    missing_output,
};

struct relation_bundle_result_v1 {
    relation_bundle_code_v1 code = relation_bundle_code_v1::composed;
    std::uint32_t relation_index = 0;
    [[nodiscard]] constexpr bool composed() const noexcept {
        return code == relation_bundle_code_v1::composed;
    }
};

[[nodiscard]] inline relation_bundle_result_v1 compose_relation_bundle_v1(
    relation_bundle_id bundle_identity,
    composition_production_id production,
    const typed_relation_view_v1 *relations,
    std::uint32_t relation_count,
    relation_bundle_composition_v1 *output) noexcept {
    if (!bundle_identity.valid()) {
        return {relation_bundle_code_v1::invalid_bundle_identity};
    }
    if (!production.valid()) {
        return {relation_bundle_code_v1::invalid_production};
    }
    if (relation_count < 2
        || relation_count > max_relation_bundle_relations_v1) {
        return {relation_bundle_code_v1::invalid_relation_count};
    }
    if (relations == nullptr) {
        return {relation_bundle_code_v1::missing_relations};
    }
    std::uint64_t total_edges = 0;
    for (std::uint32_t index = 0; index < relation_count; ++index) {
        const auto &relation = relations[index];
        if (!validate_source_major_relation_v1(relation).composed()) {
            return {relation_bundle_code_v1::invalid_relation, index};
        }
        if (index != 0
            && relations[index - 1].identity >= relation.identity) {
            return {relation_bundle_code_v1::unordered_relation_identity,
                    index};
        }
        if (relation.edge_count
            > std::numeric_limits<std::uint64_t>::max() - total_edges) {
            return {relation_bundle_code_v1::edge_count_overflow, index};
        }
        total_edges += relation.edge_count;
    }
    if (output == nullptr) return {relation_bundle_code_v1::missing_output};
    *output = {bundle_identity, production, relations, relation_count, 0,
               total_edges};
    return {relation_bundle_code_v1::composed, relation_count};
}

static_assert(
    std::is_trivially_copyable<relation_bundle_composition_v1>::value);

} // namespace cellshard::compiler::composition
