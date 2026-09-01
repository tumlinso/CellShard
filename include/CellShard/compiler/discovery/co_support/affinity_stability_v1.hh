#pragma once

#include <CellShard/compiler/discovery/co_support/top_l_affinity_v1.hh>

#include <cstdint>
#include <numeric>
#include <type_traits>

namespace cellshard::compiler::discovery::co_support {

struct affinity_observation_v1 {
    std::uint32_t resample_id = 0;
    std::uint32_t stratum_id = 0;
    std::uint32_t source_id = 0;
    std::uint32_t neighbor_source_id = 0;
};

struct affinity_stability_record_v1 {
    std::uint32_t source_id = 0;
    std::uint32_t neighbor_source_id = 0;
    std::uint32_t resample_presence_count = 0;
    std::uint32_t stratum_presence_count = 0;
    std::uint32_t resample_numerator = 0;
    std::uint32_t resample_denominator = 1;
    std::uint32_t stratum_numerator = 0;
    std::uint32_t stratum_denominator = 1;
};

enum class affinity_stability_code_v1 : std::uint32_t {
    computed = 0,
    invalid_shape,
    missing_observations,
    missing_output,
    invalid_observation,
    duplicate_observation,
    insufficient_capacity,
    work_limit_exceeded,
};

struct affinity_stability_result_v1 {
    affinity_stability_code_v1 code = affinity_stability_code_v1::computed;
    std::uint64_t record_count = 0;
    std::uint64_t consumed_observation_count = 0;
    std::uint64_t work_items = 0;
    [[nodiscard]] constexpr bool computed() const noexcept {
        return code == affinity_stability_code_v1::computed;
    }
};

[[nodiscard]] inline affinity_stability_result_v1
compute_affinity_stability_v1(
    const affinity_observation_v1 *observations,
    std::uint64_t observation_count,
    std::uint32_t source_count,
    std::uint32_t resample_count,
    std::uint32_t stratum_count,
    affinity_stability_record_v1 *output,
    std::uint64_t output_capacity,
    std::uint64_t maximum_work_items) noexcept {
    if (source_count < 2 || resample_count == 0 || stratum_count == 0)
        return {affinity_stability_code_v1::invalid_shape};
    if (observation_count != 0 && observations == nullptr)
        return {affinity_stability_code_v1::missing_observations};
    if (output_capacity != 0 && output == nullptr)
        return {affinity_stability_code_v1::missing_output};
    affinity_stability_result_v1 result{};
    for (std::uint64_t index = 0; index < observation_count; ++index) {
        const auto &observation = observations[index];
        if (observation.source_id >= source_count
            || observation.neighbor_source_id >= source_count
            || observation.source_id == observation.neighbor_source_id
            || observation.resample_id >= resample_count
            || observation.stratum_id >= stratum_count)
            return {affinity_stability_code_v1::invalid_observation,
                    result.record_count, index, result.work_items};
        bool new_resample = true;
        bool new_stratum = true;
        for (std::uint64_t previous = 0; previous < index; ++previous) {
            if (result.work_items == maximum_work_items)
                return {affinity_stability_code_v1::work_limit_exceeded,
                        result.record_count, index, result.work_items};
            ++result.work_items;
            const auto &seen = observations[previous];
            if (seen.source_id != observation.source_id
                || seen.neighbor_source_id != observation.neighbor_source_id)
                continue;
            if (seen.resample_id == observation.resample_id
                && seen.stratum_id == observation.stratum_id)
                return {affinity_stability_code_v1::duplicate_observation,
                        result.record_count, index, result.work_items};
            if (seen.resample_id == observation.resample_id) new_resample = false;
            if (seen.stratum_id == observation.stratum_id) new_stratum = false;
        }
        std::uint64_t position = 0;
        while (position < result.record_count
               && (output[position].source_id < observation.source_id
                   || (output[position].source_id == observation.source_id
                       && output[position].neighbor_source_id
                           < observation.neighbor_source_id)))
            ++position;
        if (position == result.record_count
            || output[position].source_id != observation.source_id
            || output[position].neighbor_source_id
                != observation.neighbor_source_id) {
            if (result.record_count == output_capacity)
                return {affinity_stability_code_v1::insufficient_capacity,
                        result.record_count, index, result.work_items};
            for (auto move = result.record_count; move > position; --move)
                output[move] = output[move - 1];
            output[position] = {observation.source_id,
                                observation.neighbor_source_id};
            ++result.record_count;
        }
        if (new_resample) ++output[position].resample_presence_count;
        if (new_stratum) ++output[position].stratum_presence_count;
        ++result.consumed_observation_count;
    }
    for (std::uint64_t index = 0; index < result.record_count; ++index) {
        auto &record = output[index];
        const auto resample_divisor
            = std::gcd(record.resample_presence_count, resample_count);
        const auto stratum_divisor
            = std::gcd(record.stratum_presence_count, stratum_count);
        record.resample_numerator
            = record.resample_presence_count / resample_divisor;
        record.resample_denominator = resample_count / resample_divisor;
        record.stratum_numerator = record.stratum_presence_count / stratum_divisor;
        record.stratum_denominator = stratum_count / stratum_divisor;
    }
    return result;
}

static_assert(std::is_standard_layout<affinity_observation_v1>::value);
static_assert(std::is_trivially_copyable<affinity_observation_v1>::value);
static_assert(std::is_standard_layout<affinity_stability_record_v1>::value);
static_assert(std::is_trivially_copyable<affinity_stability_record_v1>::value);

} // namespace cellshard::compiler::discovery::co_support
