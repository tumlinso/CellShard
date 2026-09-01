#pragma once

#include <CellShard/compiler/discovery/support_signature/minhash_v1.hh>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::support_signature {

struct deterministic_lsh_entry_v1 {
    std::uint64_t bucket_hash = 0;
    std::uint32_t band = 0;
    std::uint32_t destination_index = 0;
};

struct deterministic_lsh_index_view_v1 {
    const deterministic_lsh_entry_v1 *entries = nullptr;
    std::uint64_t entry_count = 0;
    std::uint64_t destination_count = 0;
    std::uint32_t band_count = 0;
    std::uint32_t rows_per_band = 0;
    std::uint32_t maximum_bucket_size = 0;
    std::uint32_t reserved = 0;
    std::uint64_t seed_namespace = 0;
    atom::atom_persistent_identity_v1 relation_identity{};
    std::uint64_t relation_generation = 0;
};

enum class deterministic_lsh_code_v1 : std::uint32_t {
    built = 0,
    invalid_sketch,
    invalid_band_shape,
    invalid_bucket_bound,
    too_many_destinations,
    size_overflow,
    missing_output,
    insufficient_output,
    bucket_bound_exceeded,
};

struct deterministic_lsh_result_v1 {
    deterministic_lsh_code_v1 code = deterministic_lsh_code_v1::built;
    deterministic_lsh_index_view_v1 view{};
    std::uint64_t index = 0;
    std::uint64_t required_entries = 0;
    [[nodiscard]] constexpr bool built() const noexcept {
        return code == deterministic_lsh_code_v1::built;
    }
};

[[nodiscard]] constexpr bool deterministic_lsh_entry_less_v1(
    deterministic_lsh_entry_v1 lhs,
    deterministic_lsh_entry_v1 rhs) noexcept {
    if (lhs.band != rhs.band) return lhs.band < rhs.band;
    if (lhs.bucket_hash != rhs.bucket_hash)
        return lhs.bucket_hash < rhs.bucket_hash;
    return lhs.destination_index < rhs.destination_index;
}

[[nodiscard]] constexpr bool valid_minhash_view_v1(
    deterministic_minhash_view_v1 sketch) noexcept {
    return sketch.minima != nullptr && sketch.destination_count != 0
        && sketch.sketch_size != 0 && sketch.reserved == 0
        && sketch.seed_namespace != 0
        && atom::validate_atom_persistent_identity_v1(
               sketch.relation_identity).valid()
        && sketch.relation_generation != 0;
}

// O(D*B*R + D*B log(D*B)) time, O(D*B) caller-owned output. A hard bucket
// bound rejects adversarial collisions before candidate fan-out is possible.
[[nodiscard]] inline deterministic_lsh_result_v1 build_lsh_index_v1(
    deterministic_minhash_view_v1 sketch,
    std::uint32_t band_count,
    std::uint32_t rows_per_band,
    std::uint32_t maximum_bucket_size,
    deterministic_lsh_entry_v1 *output,
    std::uint64_t output_capacity) noexcept {
    if (!valid_minhash_view_v1(sketch)) {
        return {deterministic_lsh_code_v1::invalid_sketch};
    }
    if (band_count == 0 || rows_per_band == 0
        || static_cast<std::uint64_t>(band_count) * rows_per_band
               != sketch.sketch_size) {
        return {deterministic_lsh_code_v1::invalid_band_shape};
    }
    if (maximum_bucket_size < 2) {
        return {deterministic_lsh_code_v1::invalid_bucket_bound};
    }
    if (sketch.destination_count > UINT32_MAX) {
        return {deterministic_lsh_code_v1::too_many_destinations};
    }
    if (sketch.destination_count
        > std::numeric_limits<std::uint64_t>::max() / band_count) {
        return {deterministic_lsh_code_v1::size_overflow};
    }
    const auto required = sketch.destination_count * band_count;
    if (output == nullptr) {
        return {deterministic_lsh_code_v1::missing_output, {}, 0, required};
    }
    if (output_capacity < required) {
        return {deterministic_lsh_code_v1::insufficient_output, {}, 0,
                required};
    }
    for (std::uint64_t destination = 0;
         destination < sketch.destination_count;
         ++destination) {
        for (std::uint32_t band = 0; band < band_count; ++band) {
            auto bucket = stable_mix_u64_v1(
                sketch.seed_namespace ^ UINT64_C(0x4c53485f42414e44)
                ^ band);
            for (std::uint32_t row = 0; row < rows_per_band; ++row) {
                const auto offset = destination * sketch.sketch_size
                    + band * rows_per_band + row;
                bucket = stable_mix_u64_v1(bucket ^ sketch.minima[offset]);
            }
            output[destination * band_count + band] = {
                bucket, band, static_cast<std::uint32_t>(destination)};
        }
    }
    std::sort(output, output + required, deterministic_lsh_entry_less_v1);
    std::uint64_t bucket_begin = 0;
    while (bucket_begin < required) {
        auto bucket_end = bucket_begin + 1;
        while (bucket_end < required
               && output[bucket_end].band == output[bucket_begin].band
               && output[bucket_end].bucket_hash
                      == output[bucket_begin].bucket_hash) {
            ++bucket_end;
        }
        if (bucket_end - bucket_begin > maximum_bucket_size) {
            return {deterministic_lsh_code_v1::bucket_bound_exceeded, {},
                    bucket_begin, required};
        }
        bucket_begin = bucket_end;
    }
    return {deterministic_lsh_code_v1::built,
            {output, required, sketch.destination_count, band_count,
             rows_per_band, maximum_bucket_size, 0, sketch.seed_namespace,
             sketch.relation_identity, sketch.relation_generation},
            required, required};
}

[[nodiscard]] constexpr bool authorizes_execution(
    deterministic_lsh_index_view_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::support_signature
