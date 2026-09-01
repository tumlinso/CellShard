#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <cstdint>
#include <type_traits>
namespace cellshard::artifact::atom_store {
struct basis_record_v1 {
    std::uint64_t basis_identity = 0;
    std::uint64_t member_offset = 0;
    std::uint64_t member_count = 0;
    std::int64_t utility_numerator = 0;
    std::uint64_t utility_denominator = 1;
    std::uint64_t budget_identity = 0;
    content_digest_v1 evidence{};
};
struct superatom_record_v1 {
    semantic_identity_v1 semantic{};
    std::uint64_t member_offset = 0;
    std::uint64_t member_count = 0;
    std::uint64_t reconstruction_offset = 0;
    std::uint64_t reconstruction_count = 0;
    content_digest_v1 exact_reconstruction{};
};
[[nodiscard]] constexpr bool valid_basis_record_v1(const basis_record_v1 &r) noexcept {
    return r.basis_identity != 0 && r.member_count != 0 && r.utility_denominator != 0
        && r.budget_identity != 0 && valid_content_digest_v1(r.evidence);
}
[[nodiscard]] constexpr bool valid_superatom_record_v1(const superatom_record_v1 &r) noexcept {
    return r.semantic.valid() && r.member_count >= 2 && r.reconstruction_count != 0
        && valid_content_digest_v1(r.exact_reconstruction);
}
static_assert(std::is_trivially_copyable<basis_record_v1>::value);
static_assert(std::is_trivially_copyable<superatom_record_v1>::value);
} // namespace cellshard::artifact::atom_store
