#pragma once

#include <CellShard/compiler/discovery/co_support/raw_co_support_v1.hh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::co_support {

struct weighted_co_support_record_v1 {
    std::uint32_t source_a = 0;
    std::uint32_t source_b = 0;
    std::uint64_t sampled_destination_count = 0;
    std::uint64_t weighted_support = 0;
};

enum class weighted_co_support_code_v1 : std::uint32_t {
    aggregated = 0,
    invalid_relation,
    missing_pairs,
    missing_output,
    invalid_pair,
    pair_not_in_destination,
    insufficient_capacity,
    accumulation_overflow,
    work_limit_exceeded,
};

struct weighted_co_support_result_v1 {
    weighted_co_support_code_v1 code = weighted_co_support_code_v1::aggregated;
    std::uint64_t record_count = 0;
    std::uint64_t consumed_pair_count = 0;
    std::uint64_t work_items = 0;
    [[nodiscard]] constexpr bool aggregated() const noexcept {
        return code == weighted_co_support_code_v1::aggregated;
    }
};

[[nodiscard]] inline weighted_co_support_result_v1
aggregate_weighted_co_support_v1(
    support_relation_view_v1 relation,
    const sampled_source_pair_v1 *pairs,
    std::uint64_t pair_count,
    weighted_co_support_record_v1 *records,
    std::uint64_t record_capacity,
    std::uint64_t maximum_work_items) noexcept {
    if (relation.relation_identity == 0 || relation.structure_epoch == 0
        || relation.destination_offsets == nullptr
        || relation.edge_weights == nullptr
        || (relation.edge_count != 0 && relation.source_ids == nullptr)
        || relation.destination_offsets[0] != 0
        || relation.destination_offsets[relation.destination_count]
            != relation.edge_count)
        return {weighted_co_support_code_v1::invalid_relation};
    if (pair_count != 0 && pairs == nullptr)
        return {weighted_co_support_code_v1::missing_pairs};
    if (record_capacity != 0 && records == nullptr)
        return {weighted_co_support_code_v1::missing_output};

    weighted_co_support_result_v1 result{};
    for (std::uint64_t pair_index = 0; pair_index < pair_count; ++pair_index) {
        const auto &pair = pairs[pair_index];
        if (pair.source_a >= pair.source_b
            || pair.source_b >= relation.source_count
            || pair.destination_id >= relation.destination_count)
            return {weighted_co_support_code_v1::invalid_pair,
                    result.record_count, pair_index, result.work_items};
        const auto begin = relation.destination_offsets[pair.destination_id];
        const auto end = relation.destination_offsets[pair.destination_id + 1];
        if (end < begin || end > relation.edge_count)
            return {weighted_co_support_code_v1::invalid_relation,
                    result.record_count, pair_index, result.work_items};
        bool found_a = false;
        bool found_b = false;
        std::uint64_t weight_a = 0;
        std::uint64_t weight_b = 0;
        for (auto edge = begin; edge < end; ++edge) {
            if (result.work_items == maximum_work_items)
                return {weighted_co_support_code_v1::work_limit_exceeded,
                        result.record_count, pair_index, result.work_items};
            ++result.work_items;
            if (relation.source_ids[edge] == pair.source_a) {
                found_a = true;
                weight_a = relation.edge_weights[edge];
            } else if (relation.source_ids[edge] == pair.source_b) {
                found_b = true;
                weight_b = relation.edge_weights[edge];
            }
        }
        if (!found_a || !found_b)
            return {weighted_co_support_code_v1::pair_not_in_destination,
                    result.record_count, pair_index, result.work_items};
        const auto contribution = std::min(weight_a, weight_b);

        std::uint64_t position = 0;
        while (position < result.record_count
               && (records[position].source_a < pair.source_a
                   || (records[position].source_a == pair.source_a
                       && records[position].source_b < pair.source_b)))
            ++position;
        if (position < result.record_count
            && records[position].source_a == pair.source_a
            && records[position].source_b == pair.source_b) {
            if (records[position].sampled_destination_count
                    == std::numeric_limits<std::uint64_t>::max()
                || contribution > std::numeric_limits<std::uint64_t>::max()
                    - records[position].weighted_support)
                return {weighted_co_support_code_v1::accumulation_overflow,
                        result.record_count, pair_index, result.work_items};
            ++records[position].sampled_destination_count;
            records[position].weighted_support += contribution;
        } else {
            if (result.record_count == record_capacity)
                return {weighted_co_support_code_v1::insufficient_capacity,
                        result.record_count, pair_index, result.work_items};
            for (auto index = result.record_count; index > position; --index)
                records[index] = records[index - 1];
            records[position] = {pair.source_a, pair.source_b, 1, contribution};
            ++result.record_count;
        }
        ++result.consumed_pair_count;
    }
    return result;
}

static_assert(std::is_standard_layout<weighted_co_support_record_v1>::value);
static_assert(std::is_trivially_copyable<weighted_co_support_record_v1>::value);

} // namespace cellshard::compiler::discovery::co_support
