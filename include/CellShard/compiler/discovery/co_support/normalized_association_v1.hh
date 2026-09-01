#pragma once

#include <CellShard/compiler/discovery/co_support/weighted_co_support_v1.hh>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <type_traits>

namespace cellshard::compiler::discovery::co_support {

struct normalized_association_record_v1 {
    std::uint32_t source_a = 0;
    std::uint32_t source_b = 0;
    std::uint64_t raw_numerator = 0;
    std::uint64_t raw_denominator = 1;
    std::uint64_t weighted_numerator = 0;
    std::uint64_t weighted_denominator = 1;
};

enum class normalized_association_code_v1 : std::uint32_t {
    computed = 0,
    missing_records,
    missing_prevalence,
    missing_output,
    insufficient_capacity,
    invalid_record,
    source_out_of_range,
    zero_prevalence,
    support_exceeds_prevalence,
};

struct normalized_association_result_v1 {
    normalized_association_code_v1 code
        = normalized_association_code_v1::computed;
    std::uint64_t record_count = 0;
    [[nodiscard]] constexpr bool computed() const noexcept {
        return code == normalized_association_code_v1::computed;
    }
};

[[nodiscard]] inline normalized_association_result_v1
compute_normalized_association_v1(
    const weighted_co_support_record_v1 *records,
    std::uint64_t record_count,
    const std::uint64_t *source_prevalence,
    std::uint64_t prevalence_count,
    normalized_association_record_v1 *output,
    std::uint64_t output_capacity) noexcept {
    if (record_count != 0 && records == nullptr)
        return {normalized_association_code_v1::missing_records};
    if (prevalence_count != 0 && source_prevalence == nullptr)
        return {normalized_association_code_v1::missing_prevalence};
    if (output_capacity != 0 && output == nullptr)
        return {normalized_association_code_v1::missing_output};
    if (output_capacity < record_count)
        return {normalized_association_code_v1::insufficient_capacity};

    for (std::uint64_t index = 0; index < record_count; ++index) {
        const auto &record = records[index];
        if (record.source_a >= record.source_b
            || record.sampled_destination_count == 0)
            return {normalized_association_code_v1::invalid_record, index};
        if (record.source_b >= prevalence_count)
            return {normalized_association_code_v1::source_out_of_range, index};
        const auto shared_bound = std::min(source_prevalence[record.source_a],
                                           source_prevalence[record.source_b]);
        if (shared_bound == 0)
            return {normalized_association_code_v1::zero_prevalence, index};
        if (record.sampled_destination_count > shared_bound)
            return {normalized_association_code_v1::support_exceeds_prevalence,
                    index};
        const auto raw_divisor
            = std::gcd(record.sampled_destination_count, shared_bound);
        const auto weighted_divisor
            = std::gcd(record.weighted_support, shared_bound);
        output[index] = {
            record.source_a,
            record.source_b,
            record.sampled_destination_count / raw_divisor,
            shared_bound / raw_divisor,
            record.weighted_support / weighted_divisor,
            shared_bound / weighted_divisor,
        };
    }
    return {normalized_association_code_v1::computed, record_count};
}

static_assert(std::is_standard_layout<normalized_association_record_v1>::value);
static_assert(std::is_trivially_copyable<normalized_association_record_v1>::value);

} // namespace cellshard::compiler::discovery::co_support
