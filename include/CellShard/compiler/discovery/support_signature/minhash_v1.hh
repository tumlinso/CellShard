#pragma once

#include <CellShard/compiler/discovery/support_signature/stable_hash_v1.hh>

#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::support_signature {

struct deterministic_minhash_view_v1 {
    const std::uint64_t *minima = nullptr;
    std::uint64_t destination_count = 0;
    std::uint32_t sketch_size = 0;
    std::uint32_t reserved = 0;
    std::uint64_t seed_namespace = 0;
    atom::atom_persistent_identity_v1 relation_identity{};
    std::uint64_t relation_generation = 0;
};

enum class deterministic_minhash_code_v1 : std::uint32_t {
    built = 0,
    invalid_support,
    invalid_sketch_size,
    missing_seed_namespace,
    size_overflow,
    missing_output,
    insufficient_output,
};

struct deterministic_minhash_result_v1 {
    deterministic_minhash_code_v1 code =
        deterministic_minhash_code_v1::built;
    deterministic_minhash_view_v1 view{};
    std::uint64_t required_minima = 0;
    [[nodiscard]] constexpr bool built() const noexcept {
        return code == deterministic_minhash_code_v1::built;
    }
};

// O(E*K) time and O(D*K) caller-owned output. Seeds derive only from the
// explicit stable seed namespace and sketch position; no process randomness,
// address, device, or iteration order enters the sketch.
[[nodiscard]] constexpr deterministic_minhash_result_v1 build_minhash_v1(
    exact_destination_support_view_v1 support,
    std::uint32_t sketch_size,
    std::uint32_t maximum_sketch_size,
    std::uint64_t seed_namespace,
    std::uint64_t *output,
    std::uint64_t output_capacity) noexcept {
    if (!validate_exact_destination_support_view_v1(support)) {
        return {deterministic_minhash_code_v1::invalid_support};
    }
    if (sketch_size == 0 || maximum_sketch_size == 0
        || sketch_size > maximum_sketch_size) {
        return {deterministic_minhash_code_v1::invalid_sketch_size};
    }
    if (seed_namespace == 0) {
        return {deterministic_minhash_code_v1::missing_seed_namespace};
    }
    if (support.destination_count
        > std::numeric_limits<std::uint64_t>::max() / sketch_size) {
        return {deterministic_minhash_code_v1::size_overflow};
    }
    const auto required = support.destination_count * sketch_size;
    if (output == nullptr) {
        return {deterministic_minhash_code_v1::missing_output, {}, required};
    }
    if (output_capacity < required) {
        return {deterministic_minhash_code_v1::insufficient_output, {},
                required};
    }
    for (std::uint64_t destination = 0;
         destination < support.destination_count;
         ++destination) {
        const auto begin = support.destination_offsets[destination];
        const auto end = support.destination_offsets[destination + 1];
        for (std::uint32_t sketch = 0; sketch < sketch_size; ++sketch) {
            const auto seed = stable_mix_u64_v1(
                seed_namespace ^ (UINT64_C(0x9e3779b97f4a7c15)
                                  * (sketch + 1)));
            auto minimum = std::numeric_limits<std::uint64_t>::max();
            for (auto index = begin; index < end; ++index) {
                const auto value = stable_mix_u64_v1(
                    support.global_source_ids[index] ^ seed);
                if (value < minimum) minimum = value;
            }
            output[destination * sketch_size + sketch] = minimum;
        }
    }
    return {deterministic_minhash_code_v1::built,
            {output, support.destination_count, sketch_size, 0,
             seed_namespace, support.relation_identity,
             support.relation_generation},
            required};
}

[[nodiscard]] constexpr bool authorizes_execution(
    deterministic_minhash_view_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::support_signature
