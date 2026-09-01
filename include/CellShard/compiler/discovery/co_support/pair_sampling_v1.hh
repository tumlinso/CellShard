#pragma once

#include <CellShard/compiler/discovery/co_support/relation_statistics_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::co_support {

struct sampled_source_pair_v1 {
    std::uint32_t source_a = 0;
    std::uint32_t source_b = 0;
    std::uint32_t destination_id = 0;
    std::uint32_t reserved = 0;
    std::uint64_t inclusion_numerator = 0;
    std::uint64_t inclusion_denominator = 1;
};

struct pair_sampling_config_v1 {
    std::uint64_t minimum_destination_degree = 2;
    std::uint64_t maximum_pairs_per_destination = 0;
    std::uint64_t maximum_work_items = 0;
};

enum class pair_sampling_code_v1 : std::uint32_t {
    sampled = 0,
    invalid_relation,
    invalid_config,
    missing_output,
    insufficient_capacity,
    pair_count_overflow,
    work_limit_exceeded,
};

struct pair_sampling_result_v1 {
    pair_sampling_code_v1 code = pair_sampling_code_v1::sampled;
    std::uint64_t pair_count = 0;
    std::uint64_t sampled_destination_count = 0;
    std::uint64_t work_items = 0;
    [[nodiscard]] constexpr bool sampled() const noexcept {
        return code == pair_sampling_code_v1::sampled;
    }
};

[[nodiscard]] inline bool pair_from_ordinal_v1(
    const std::uint32_t *sources,
    std::uint64_t degree,
    std::uint64_t ordinal,
    std::uint32_t *source_a,
    std::uint32_t *source_b,
    std::uint64_t *work_items,
    std::uint64_t maximum_work_items) noexcept {
    for (std::uint64_t left = 0; left + 1 < degree; ++left) {
        const auto row_count = degree - left - 1;
        if (*work_items == maximum_work_items) return false;
        ++*work_items;
        if (ordinal < row_count) {
            *source_a = sources[left];
            *source_b = sources[left + 1 + ordinal];
            return true;
        }
        ordinal -= row_count;
    }
    return false;
}

[[nodiscard]] inline pair_sampling_result_v1 sample_high_degree_pairs_v1(
    support_relation_view_v1 relation,
    pair_sampling_config_v1 config,
    sampled_source_pair_v1 *pairs,
    std::uint64_t pair_capacity) noexcept {
    if (relation.relation_identity == 0 || relation.structure_epoch == 0
        || relation.source_count == 0 || relation.destination_count == 0
        || relation.destination_offsets == nullptr
        || (relation.edge_count != 0 && relation.source_ids == nullptr))
        return {pair_sampling_code_v1::invalid_relation};
    if (config.minimum_destination_degree < 2
        || config.maximum_pairs_per_destination == 0
        || config.maximum_work_items == 0)
        return {pair_sampling_code_v1::invalid_config};
    if (pair_capacity != 0 && pairs == nullptr)
        return {pair_sampling_code_v1::missing_output};

    pair_sampling_result_v1 result{};
    for (std::uint32_t destination = 0;
         destination < relation.destination_count; ++destination) {
        const auto begin = relation.destination_offsets[destination];
        const auto end = relation.destination_offsets[destination + 1];
        if (end < begin || end > relation.edge_count)
            return {pair_sampling_code_v1::invalid_relation};
        const auto degree = end - begin;
        if (degree < config.minimum_destination_degree) continue;
        if (degree > 1
            && degree - 1 > std::numeric_limits<std::uint64_t>::max() / degree)
            return {pair_sampling_code_v1::pair_count_overflow};
        const auto total_pairs = degree * (degree - 1) / 2;
        const auto sample_count = total_pairs < config.maximum_pairs_per_destination
            ? total_pairs : config.maximum_pairs_per_destination;
        ++result.sampled_destination_count;
        for (std::uint64_t sample = 0; sample < sample_count; ++sample) {
            if (result.pair_count == pair_capacity) {
                result.code = pair_sampling_code_v1::insufficient_capacity;
                return result;
            }
            const auto quotient = total_pairs / sample_count;
            const auto remainder = total_pairs % sample_count;
            const auto ordinal = sample * quotient
                + (sample * remainder) / sample_count;
            std::uint32_t source_a = 0;
            std::uint32_t source_b = 0;
            if (!pair_from_ordinal_v1(relation.source_ids + begin, degree, ordinal,
                                      &source_a, &source_b, &result.work_items,
                                      config.maximum_work_items)) {
                result.code = pair_sampling_code_v1::work_limit_exceeded;
                return result;
            }
            pairs[result.pair_count++] = {source_a, source_b, destination, 0,
                                          sample_count, total_pairs};
        }
    }
    return result;
}

static_assert(std::is_standard_layout<sampled_source_pair_v1>::value);
static_assert(std::is_trivially_copyable<sampled_source_pair_v1>::value);

} // namespace cellshard::compiler::discovery::co_support
