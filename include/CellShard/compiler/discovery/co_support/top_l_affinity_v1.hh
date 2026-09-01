#pragma once

#include <CellShard/compiler/discovery/co_support/normalized_association_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::co_support {

struct source_affinity_edge_v1 {
    std::uint32_t source_id = 0;
    std::uint32_t neighbor_source_id = 0;
    std::uint32_t rank = 0;
    std::uint32_t reserved = 0;
    std::uint64_t score_numerator = 0;
    std::uint64_t score_denominator = 1;
};

enum class top_l_affinity_code_v1 : std::uint32_t {
    built = 0,
    invalid_shape,
    missing_associations,
    missing_output,
    insufficient_capacity,
    invalid_association,
    capacity_overflow,
};

struct top_l_affinity_result_v1 {
    top_l_affinity_code_v1 code = top_l_affinity_code_v1::built;
    std::uint64_t edge_count = 0;
    [[nodiscard]] constexpr bool built() const noexcept {
        return code == top_l_affinity_code_v1::built;
    }
};

[[nodiscard]] inline bool fraction_greater_v1(
    std::uint64_t left_numerator,
    std::uint64_t left_denominator,
    std::uint64_t right_numerator,
    std::uint64_t right_denominator) noexcept {
    bool reversed = false;
    while (true) {
        const auto left_quotient = left_numerator / left_denominator;
        const auto right_quotient = right_numerator / right_denominator;
        if (left_quotient != right_quotient)
            return reversed ? left_quotient < right_quotient
                            : left_quotient > right_quotient;
        const auto left_remainder = left_numerator % left_denominator;
        const auto right_remainder = right_numerator % right_denominator;
        if (left_remainder == 0 || right_remainder == 0) {
            if (left_remainder == right_remainder) return false;
            return reversed ? left_remainder == 0 : right_remainder == 0;
        }
        left_numerator = left_denominator;
        left_denominator = left_remainder;
        right_numerator = right_denominator;
        right_denominator = right_remainder;
        reversed = !reversed;
    }
}

[[nodiscard]] inline bool affinity_better_v1(
    const source_affinity_edge_v1 &left,
    const source_affinity_edge_v1 &right) noexcept {
    if (fraction_greater_v1(left.score_numerator, left.score_denominator,
                            right.score_numerator, right.score_denominator))
        return true;
    if (fraction_greater_v1(right.score_numerator, right.score_denominator,
                            left.score_numerator, left.score_denominator))
        return false;
    return left.neighbor_source_id < right.neighbor_source_id;
}

inline void retain_affinity_edge_v1(source_affinity_edge_v1 *bucket,
                                    std::uint32_t top_l,
                                    source_affinity_edge_v1 candidate) noexcept {
    const auto empty = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t count = 0;
    while (count < top_l && bucket[count].neighbor_source_id != empty) ++count;
    if (count < top_l) {
        bucket[count] = candidate;
        return;
    }
    std::uint32_t worst = 0;
    for (std::uint32_t index = 1; index < count; ++index)
        if (affinity_better_v1(bucket[worst], bucket[index])) worst = index;
    if (affinity_better_v1(candidate, bucket[worst])) bucket[worst] = candidate;
}

[[nodiscard]] inline top_l_affinity_result_v1 build_top_l_affinity_v1(
    const normalized_association_record_v1 *associations,
    std::uint64_t association_count,
    std::uint32_t source_count,
    std::uint32_t top_l,
    source_affinity_edge_v1 *output,
    std::uint64_t output_capacity) noexcept {
    if (source_count < 2 || top_l == 0)
        return {top_l_affinity_code_v1::invalid_shape};
    if (association_count != 0 && associations == nullptr)
        return {top_l_affinity_code_v1::missing_associations};
    if (source_count > std::numeric_limits<std::uint64_t>::max() / top_l)
        return {top_l_affinity_code_v1::capacity_overflow};
    const auto required = static_cast<std::uint64_t>(source_count) * top_l;
    if (output_capacity != 0 && output == nullptr)
        return {top_l_affinity_code_v1::missing_output};
    if (output_capacity < required)
        return {top_l_affinity_code_v1::insufficient_capacity};
    const auto empty = std::numeric_limits<std::uint32_t>::max();
    for (std::uint64_t index = 0; index < required; ++index) {
        output[index] = {};
        output[index].neighbor_source_id = empty;
    }
    for (std::uint64_t index = 0; index < association_count; ++index) {
        const auto &association = associations[index];
        if (association.source_a >= association.source_b
            || association.source_b >= source_count
            || association.raw_denominator == 0)
            return {top_l_affinity_code_v1::invalid_association};
        retain_affinity_edge_v1(output
                + static_cast<std::uint64_t>(association.source_a) * top_l,
            top_l, {association.source_a, association.source_b, 0, 0,
                    association.raw_numerator, association.raw_denominator});
        retain_affinity_edge_v1(output
                + static_cast<std::uint64_t>(association.source_b) * top_l,
            top_l, {association.source_b, association.source_a, 0, 0,
                    association.raw_numerator, association.raw_denominator});
    }
    std::uint64_t edge_count = 0;
    for (std::uint32_t source = 0; source < source_count; ++source) {
        auto *bucket = output + static_cast<std::uint64_t>(source) * top_l;
        std::uint32_t count = 0;
        while (count < top_l && bucket[count].neighbor_source_id != empty) ++count;
        for (std::uint32_t left = 0; left < count; ++left) {
            std::uint32_t best = left;
            for (std::uint32_t right = left + 1; right < count; ++right)
                if (affinity_better_v1(bucket[right], bucket[best])) best = right;
            const auto selected = bucket[best];
            bucket[best] = bucket[left];
            bucket[left] = selected;
            bucket[left].rank = left;
        }
        for (std::uint32_t index = 0; index < count; ++index)
            output[edge_count++] = bucket[index];
    }
    return {top_l_affinity_code_v1::built, edge_count};
}

static_assert(std::is_standard_layout<source_affinity_edge_v1>::value);
static_assert(std::is_trivially_copyable<source_affinity_edge_v1>::value);

} // namespace cellshard::compiler::discovery::co_support
