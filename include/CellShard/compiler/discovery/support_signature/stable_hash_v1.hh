#pragma once

#include <CellShard/compiler/discovery/support_signature/exact_support_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::support_signature {

struct stable_support_hash_v1 {
    std::uint64_t low = 0;
    std::uint64_t high = 0;
};

struct destination_support_hash_v1 {
    std::uint64_t global_destination_id = 0;
    std::uint64_t support_count = 0;
    stable_support_hash_v1 hash{};
};

enum class stable_support_hash_code_v1 : std::uint32_t {
    hashed = 0,
    invalid_view,
    missing_output,
    insufficient_output,
};

struct stable_support_hash_result_v1 {
    stable_support_hash_code_v1 code = stable_support_hash_code_v1::hashed;
    std::uint64_t hash_count = 0;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool hashed() const noexcept {
        return code == stable_support_hash_code_v1::hashed;
    }
};

static_assert(std::is_standard_layout<stable_support_hash_v1>::value);
static_assert(std::is_trivially_copyable<stable_support_hash_v1>::value);
static_assert(std::is_standard_layout<destination_support_hash_v1>::value);
static_assert(std::is_trivially_copyable<destination_support_hash_v1>::value);

[[nodiscard]] constexpr std::uint64_t stable_mix_u64_v1(
    std::uint64_t value) noexcept {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

[[nodiscard]] constexpr bool validate_exact_destination_support_view_v1(
    exact_destination_support_view_v1 view) noexcept {
    if (view.destination_count == 0 || view.edge_count == 0
        || view.global_destination_ids == nullptr
        || view.destination_offsets == nullptr
        || view.global_source_ids == nullptr
        || !atom::validate_atom_persistent_identity_v1(
                view.relation_identity).valid()
        || !atom::validate_atom_persistent_identity_v1(
                view.source_domain_identity).valid()
        || !atom::validate_atom_persistent_identity_v1(
                view.destination_domain_identity).valid()
        || view.relation_generation == 0
        || view.destination_offsets[0] != 0
        || view.destination_offsets[view.destination_count] != view.edge_count) {
        return false;
    }
    for (std::uint64_t destination = 0;
         destination < view.destination_count;
         ++destination) {
        if (view.global_destination_ids[destination] == 0
            || (destination != 0
                && view.global_destination_ids[destination - 1]
                       >= view.global_destination_ids[destination])) {
            return false;
        }
        const auto begin = view.destination_offsets[destination];
        const auto end = view.destination_offsets[destination + 1];
        if (begin >= end || end > view.edge_count) return false;
        for (auto index = begin; index < end; ++index) {
            if (view.global_source_ids[index] == 0
                || (index != begin
                    && view.global_source_ids[index - 1]
                           >= view.global_source_ids[index])) {
                return false;
            }
        }
    }
    return true;
}

namespace detail {

constexpr void hash_fold_v1(
    std::uint64_t value,
    stable_support_hash_v1 *hash) noexcept {
    hash->low = stable_mix_u64_v1(hash->low ^ value);
    hash->high = stable_mix_u64_v1(
        hash->high + value + UINT64_C(0x517cc1b727220a95));
}

} // namespace detail

// This stable hash accelerates candidate lookup only. Exact support words and
// their generation remain required for equality and collision resolution.
[[nodiscard]] constexpr stable_support_hash_result_v1
hash_exact_destination_support_v1(
    exact_destination_support_view_v1 view,
    destination_support_hash_v1 *output,
    std::uint64_t output_capacity) noexcept {
    if (!validate_exact_destination_support_view_v1(view)) {
        return {stable_support_hash_code_v1::invalid_view};
    }
    if (output == nullptr) {
        return {stable_support_hash_code_v1::missing_output};
    }
    if (output_capacity < view.destination_count) {
        return {stable_support_hash_code_v1::insufficient_output,
                view.destination_count};
    }
    for (std::uint64_t destination = 0;
         destination < view.destination_count;
         ++destination) {
        stable_support_hash_v1 hash{
            UINT64_C(0x243f6a8885a308d3), UINT64_C(0x13198a2e03707344)};
        detail::hash_fold_v1(view.relation_identity.producer_namespace, &hash);
        detail::hash_fold_v1(view.relation_identity.local_identity, &hash);
        detail::hash_fold_v1(view.source_domain_identity.producer_namespace,
                             &hash);
        detail::hash_fold_v1(view.source_domain_identity.local_identity, &hash);
        detail::hash_fold_v1(
            view.destination_domain_identity.producer_namespace, &hash);
        detail::hash_fold_v1(
            view.destination_domain_identity.local_identity, &hash);
        detail::hash_fold_v1(view.relation_generation, &hash);
        detail::hash_fold_v1(view.global_destination_ids[destination], &hash);
        const auto begin = view.destination_offsets[destination];
        const auto end = view.destination_offsets[destination + 1];
        detail::hash_fold_v1(end - begin, &hash);
        for (auto index = begin; index < end; ++index) {
            detail::hash_fold_v1(view.global_source_ids[index], &hash);
        }
        output[destination] = {
            view.global_destination_ids[destination], end - begin, hash};
    }
    return {stable_support_hash_code_v1::hashed, view.destination_count,
            view.edge_count};
}

} // namespace cellshard::compiler::discovery::support_signature
