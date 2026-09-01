#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <cstdint>
#include <type_traits>
namespace cellshard::artifact::atom_store {
struct composition_record_v1 {
    semantic_identity_v1 result{};
    action_identity_v1 action{};
    std::uint64_t operand_offset = 0;
    std::uint64_t operand_count = 0;
    std::uint64_t derivation_generation = 0;
};
struct grammar_record_v1 {
    std::uint64_t grammar_identity = 0;
    std::uint64_t production_identity = 0;
    std::uint64_t symbol_offset = 0;
    std::uint64_t symbol_count = 0;
    semantic_identity_v1 result{};
    content_digest_v1 evidence{};
    std::uint32_t induced = 0;
    std::uint32_t certified = 0;
};
[[nodiscard]] constexpr bool valid_composition_record_v1(const composition_record_v1 &r) noexcept {
    return r.result.valid() && r.action.valid() && r.operand_count >= 2
        && r.derivation_generation != 0;
}
[[nodiscard]] constexpr bool valid_grammar_record_v1(const grammar_record_v1 &r) noexcept {
    return r.grammar_identity != 0 && r.production_identity != 0 && r.symbol_count != 0
        && r.result.valid() && valid_content_digest_v1(r.evidence)
        && r.induced <= 1 && r.certified == 1;
}
static_assert(std::is_trivially_copyable<composition_record_v1>::value);
static_assert(std::is_trivially_copyable<grammar_record_v1>::value);
} // namespace cellshard::artifact::atom_store
