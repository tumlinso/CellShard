#pragma once

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::co_support {

struct support_relation_view_v1 {
    const std::uint64_t *destination_offsets = nullptr;
    const std::uint32_t *source_ids = nullptr;
    const std::uint64_t *edge_weights = nullptr;
    const std::uint32_t *destination_strata = nullptr;
    std::uint64_t edge_count = 0;
    std::uint32_t source_count = 0;
    std::uint32_t destination_count = 0;
    std::uint64_t relation_identity = 0;
    std::uint64_t structure_epoch = 0;
};

enum class relation_statistics_code_v1 : std::uint32_t {
    computed = 0,
    invalid_identity,
    empty_shape,
    missing_offsets,
    missing_sources,
    invalid_offsets,
    source_out_of_range,
    unordered_or_duplicate_source,
    missing_prevalence,
    insufficient_prevalence_capacity,
    missing_degrees,
    insufficient_degree_capacity,
};

struct relation_statistics_result_v1 {
    relation_statistics_code_v1 code = relation_statistics_code_v1::computed;
    std::uint64_t edge_index = 0;
    std::uint32_t destination_index = 0;
    [[nodiscard]] constexpr bool computed() const noexcept {
        return code == relation_statistics_code_v1::computed;
    }
};

[[nodiscard]] inline relation_statistics_result_v1
compute_exact_relation_statistics_v1(
    support_relation_view_v1 relation,
    std::uint64_t *source_prevalence,
    std::uint64_t prevalence_capacity,
    std::uint64_t *destination_degree,
    std::uint64_t degree_capacity) noexcept {
    if (relation.relation_identity == 0 || relation.structure_epoch == 0)
        return {relation_statistics_code_v1::invalid_identity};
    if (relation.source_count == 0 || relation.destination_count == 0)
        return {relation_statistics_code_v1::empty_shape};
    if (relation.destination_offsets == nullptr)
        return {relation_statistics_code_v1::missing_offsets};
    if (relation.edge_count != 0 && relation.source_ids == nullptr)
        return {relation_statistics_code_v1::missing_sources};
    if (relation.destination_offsets[0] != 0
        || relation.destination_offsets[relation.destination_count]
            != relation.edge_count)
        return {relation_statistics_code_v1::invalid_offsets};
    if (source_prevalence == nullptr)
        return {relation_statistics_code_v1::missing_prevalence};
    if (prevalence_capacity < relation.source_count)
        return {relation_statistics_code_v1::insufficient_prevalence_capacity};
    if (destination_degree == nullptr)
        return {relation_statistics_code_v1::missing_degrees};
    if (degree_capacity < relation.destination_count)
        return {relation_statistics_code_v1::insufficient_degree_capacity};
    for (std::uint32_t source = 0; source < relation.source_count; ++source)
        source_prevalence[source] = 0;
    for (std::uint32_t destination = 0;
         destination < relation.destination_count; ++destination) {
        const auto begin = relation.destination_offsets[destination];
        const auto end = relation.destination_offsets[destination + 1];
        if (end < begin || end > relation.edge_count)
            return {relation_statistics_code_v1::invalid_offsets, begin, destination};
        destination_degree[destination] = end - begin;
        std::uint32_t previous = 0;
        for (std::uint64_t edge = begin; edge < end; ++edge) {
            const auto source = relation.source_ids[edge];
            if (source >= relation.source_count)
                return {relation_statistics_code_v1::source_out_of_range,
                        edge, destination};
            if (edge != begin && source <= previous)
                return {relation_statistics_code_v1::unordered_or_duplicate_source,
                        edge, destination};
            ++source_prevalence[source];
            previous = source;
        }
    }
    return {relation_statistics_code_v1::computed,
            relation.edge_count, relation.destination_count};
}

static_assert(std::is_standard_layout<support_relation_view_v1>::value);
static_assert(std::is_trivially_copyable<support_relation_view_v1>::value);

} // namespace cellshard::compiler::discovery::co_support
