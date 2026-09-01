#pragma once

#include <CellShard/compiler/discovery/co_support/pair_sampling_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::co_support {

struct raw_co_support_record_v1 {
    std::uint32_t source_a = 0;
    std::uint32_t source_b = 0;
    std::uint64_t sampled_destination_count = 0;
};

enum class raw_co_support_code_v1 : std::uint32_t {
    aggregated = 0,
    missing_pairs,
    missing_output,
    invalid_pair,
    insufficient_capacity,
    count_overflow,
    work_limit_exceeded,
};

struct raw_co_support_result_v1 {
    raw_co_support_code_v1 code = raw_co_support_code_v1::aggregated;
    std::uint64_t record_count = 0;
    std::uint64_t consumed_pair_count = 0;
    std::uint64_t work_items = 0;
    [[nodiscard]] constexpr bool aggregated() const noexcept {
        return code == raw_co_support_code_v1::aggregated;
    }
};

[[nodiscard]] inline raw_co_support_result_v1 aggregate_raw_co_support_v1(
    const sampled_source_pair_v1 *pairs,
    std::uint64_t pair_count,
    raw_co_support_record_v1 *records,
    std::uint64_t record_capacity,
    std::uint64_t maximum_work_items) noexcept {
    if (pair_count != 0 && pairs == nullptr)
        return {raw_co_support_code_v1::missing_pairs};
    if (record_capacity != 0 && records == nullptr)
        return {raw_co_support_code_v1::missing_output};
    raw_co_support_result_v1 result{};
    for (std::uint64_t pair_index = 0; pair_index < pair_count; ++pair_index) {
        const auto &pair = pairs[pair_index];
        if (pair.source_a >= pair.source_b || pair.inclusion_numerator == 0
            || pair.inclusion_denominator < pair.inclusion_numerator)
            return {raw_co_support_code_v1::invalid_pair, result.record_count,
                    pair_index, result.work_items};

        std::uint64_t position = 0;
        while (position < result.record_count) {
            if (result.work_items == maximum_work_items)
                return {raw_co_support_code_v1::work_limit_exceeded,
                        result.record_count, pair_index, result.work_items};
            ++result.work_items;
            const auto &record = records[position];
            if (record.source_a > pair.source_a
                || (record.source_a == pair.source_a
                    && record.source_b >= pair.source_b))
                break;
            ++position;
        }
        if (position < result.record_count
            && records[position].source_a == pair.source_a
            && records[position].source_b == pair.source_b) {
            if (records[position].sampled_destination_count
                == std::numeric_limits<std::uint64_t>::max())
                return {raw_co_support_code_v1::count_overflow,
                        result.record_count, pair_index, result.work_items};
            ++records[position].sampled_destination_count;
        } else {
            if (result.record_count == record_capacity)
                return {raw_co_support_code_v1::insufficient_capacity,
                        result.record_count, pair_index, result.work_items};
            for (auto index = result.record_count; index > position; --index)
                records[index] = records[index - 1];
            records[position] = {pair.source_a, pair.source_b, 1};
            ++result.record_count;
        }
        ++result.consumed_pair_count;
    }
    return result;
}

static_assert(std::is_standard_layout<raw_co_support_record_v1>::value);
static_assert(std::is_trivially_copyable<raw_co_support_record_v1>::value);

} // namespace cellshard::compiler::discovery::co_support
