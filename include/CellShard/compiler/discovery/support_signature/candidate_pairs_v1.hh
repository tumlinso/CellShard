#pragma once

#include <CellShard/compiler/discovery/support_signature/lsh_index_v1.hh>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::support_signature {

struct support_candidate_pair_v1 {
    std::uint32_t first_destination_index = 0;
    std::uint32_t second_destination_index = 0;
    std::uint32_t matching_band_count = 0;
    std::uint32_t reserved = 0;
};

struct support_candidate_pair_view_v1 {
    const support_candidate_pair_v1 *pairs = nullptr;
    std::uint64_t pair_count = 0;
    std::uint64_t destination_count = 0;
    std::uint32_t maximum_fan_out = 0;
    std::uint32_t reserved = 0;
    atom::atom_persistent_identity_v1 relation_identity{};
    std::uint64_t relation_generation = 0;
};

enum class support_candidate_pair_code_v1 : std::uint32_t {
    built = 0,
    invalid_index,
    invalid_fan_out,
    raw_pair_overflow,
    missing_output,
    insufficient_output,
    missing_fan_out_counts,
    insufficient_fan_out_counts,
    fan_out_exceeded,
};

struct support_candidate_pair_result_v1 {
    support_candidate_pair_code_v1 code = support_candidate_pair_code_v1::built;
    support_candidate_pair_view_v1 view{};
    std::uint64_t index = 0;
    std::uint64_t required_raw_pairs = 0;
    [[nodiscard]] constexpr bool built() const noexcept {
        return code == support_candidate_pair_code_v1::built;
    }
};

[[nodiscard]] constexpr bool valid_lsh_index_view_v1(
    deterministic_lsh_index_view_v1 index) noexcept {
    if (index.entries == nullptr || index.entry_count == 0
        || index.destination_count == 0 || index.band_count == 0
        || index.rows_per_band == 0 || index.maximum_bucket_size < 2
        || index.reserved != 0 || index.seed_namespace == 0
        || !atom::validate_atom_persistent_identity_v1(
                index.relation_identity).valid()
        || index.relation_generation == 0) {
        return false;
    }
    for (std::uint64_t entry = 0; entry < index.entry_count; ++entry) {
        if (index.entries[entry].band >= index.band_count
            || index.entries[entry].destination_index
                   >= index.destination_count
            || (entry != 0
                && !deterministic_lsh_entry_less_v1(
                    index.entries[entry - 1], index.entries[entry]))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] constexpr bool support_candidate_pair_less_v1(
    support_candidate_pair_v1 lhs,
    support_candidate_pair_v1 rhs) noexcept {
    return lhs.first_destination_index < rhs.first_destination_index
        || (lhs.first_destination_index == rhs.first_destination_index
            && lhs.second_destination_index < rhs.second_destination_index);
}

// O(raw_pairs log raw_pairs) time and explicit O(raw_pairs + destinations)
// caller storage. Duplicate pairs across bands are merged before the symmetric
// per-destination fan-out cap is enforced.
[[nodiscard]] inline support_candidate_pair_result_v1
build_deduplicated_candidate_pairs_v1(
    deterministic_lsh_index_view_v1 index,
    std::uint32_t maximum_fan_out,
    support_candidate_pair_v1 *pairs,
    std::uint64_t pair_capacity,
    std::uint32_t *fan_out_counts,
    std::uint64_t fan_out_capacity) noexcept {
    if (!valid_lsh_index_view_v1(index)) {
        return {support_candidate_pair_code_v1::invalid_index};
    }
    if (maximum_fan_out == 0) {
        return {support_candidate_pair_code_v1::invalid_fan_out};
    }
    std::uint64_t raw_count = 0;
    std::uint64_t bucket_begin = 0;
    while (bucket_begin < index.entry_count) {
        auto bucket_end = bucket_begin + 1;
        while (bucket_end < index.entry_count
               && index.entries[bucket_end].band
                      == index.entries[bucket_begin].band
               && index.entries[bucket_end].bucket_hash
                      == index.entries[bucket_begin].bucket_hash) {
            ++bucket_end;
        }
        const auto size = bucket_end - bucket_begin;
        if (size > 1) {
            if (size > std::numeric_limits<std::uint64_t>::max() / (size - 1)
                || raw_count > std::numeric_limits<std::uint64_t>::max()
                       - size * (size - 1) / 2) {
                return {support_candidate_pair_code_v1::raw_pair_overflow};
            }
            raw_count += size * (size - 1) / 2;
        }
        bucket_begin = bucket_end;
    }
    if (pairs == nullptr) {
        return {support_candidate_pair_code_v1::missing_output, {}, 0,
                raw_count};
    }
    if (pair_capacity < raw_count) {
        return {support_candidate_pair_code_v1::insufficient_output, {}, 0,
                raw_count};
    }
    std::uint64_t cursor = 0;
    bucket_begin = 0;
    while (bucket_begin < index.entry_count) {
        auto bucket_end = bucket_begin + 1;
        while (bucket_end < index.entry_count
               && index.entries[bucket_end].band
                      == index.entries[bucket_begin].band
               && index.entries[bucket_end].bucket_hash
                      == index.entries[bucket_begin].bucket_hash) {
            ++bucket_end;
        }
        for (auto first = bucket_begin; first < bucket_end; ++first) {
            for (auto second = first + 1; second < bucket_end; ++second) {
                auto lhs = index.entries[first].destination_index;
                auto rhs = index.entries[second].destination_index;
                if (rhs < lhs) std::swap(lhs, rhs);
                pairs[cursor++] = {lhs, rhs, 1, 0};
            }
        }
        bucket_begin = bucket_end;
    }
    std::sort(pairs, pairs + raw_count, support_candidate_pair_less_v1);
    std::uint64_t unique_count = 0;
    for (std::uint64_t raw = 0; raw < raw_count; ++raw) {
        if (unique_count != 0
            && pairs[unique_count - 1].first_destination_index
                   == pairs[raw].first_destination_index
            && pairs[unique_count - 1].second_destination_index
                   == pairs[raw].second_destination_index) {
            ++pairs[unique_count - 1].matching_band_count;
        } else {
            pairs[unique_count++] = pairs[raw];
        }
    }
    if (fan_out_counts == nullptr) {
        return {support_candidate_pair_code_v1::missing_fan_out_counts, {},
                0, raw_count};
    }
    if (fan_out_capacity < index.destination_count) {
        return {support_candidate_pair_code_v1::insufficient_fan_out_counts,
                {}, fan_out_capacity, raw_count};
    }
    for (std::uint64_t destination = 0;
         destination < index.destination_count;
         ++destination) {
        fan_out_counts[destination] = 0;
    }
    for (std::uint64_t pair = 0; pair < unique_count; ++pair) {
        const auto first = pairs[pair].first_destination_index;
        const auto second = pairs[pair].second_destination_index;
        if (++fan_out_counts[first] > maximum_fan_out
            || ++fan_out_counts[second] > maximum_fan_out) {
            return {support_candidate_pair_code_v1::fan_out_exceeded, {},
                    pair, raw_count};
        }
    }
    return {support_candidate_pair_code_v1::built,
            {pairs, unique_count, index.destination_count, maximum_fan_out, 0,
             index.relation_identity, index.relation_generation},
            unique_count, raw_count};
}

[[nodiscard]] constexpr bool authorizes_execution(
    support_candidate_pair_view_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::support_signature
