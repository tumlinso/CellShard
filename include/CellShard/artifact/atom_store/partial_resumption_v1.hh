#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <cstdint>
#include <type_traits>
namespace cellshard::artifact::atom_store {
struct partial_record_v1 {
    semantic_identity_v1 atom{};
    std::uint64_t partial_algebra_identity = 0;
    std::uint64_t contribution_begin = 0;
    std::uint64_t contribution_count = 0;
    std::uint64_t contribution_owner = 0;
    content_digest_v1 reconstruction_evidence{};
};
struct lowering_resumption_record_v1 {
    action_identity_v1 action{};
    materialization_identity_v1 materialization{};
    std::uint64_t completed_step_count = 0;
    std::uint64_t resume_cursor_offset = 0;
    std::uint64_t resume_cursor_bytes = 0;
    content_digest_v1 dependency_closure{};
};
[[nodiscard]] constexpr bool valid_partial_record_v1(const partial_record_v1 &r) noexcept {
    return r.atom.valid() && r.partial_algebra_identity != 0 && r.contribution_count != 0
        && r.contribution_owner != 0 && valid_content_digest_v1(r.reconstruction_evidence);
}
[[nodiscard]] constexpr bool valid_lowering_resumption_record_v1(
    const lowering_resumption_record_v1 &r) noexcept {
    return r.action.valid() && r.materialization.valid() && r.completed_step_count != 0
        && r.resume_cursor_bytes != 0 && valid_content_digest_v1(r.dependency_closure);
}
static_assert(std::is_trivially_copyable<partial_record_v1>::value);
static_assert(std::is_trivially_copyable<lowering_resumption_record_v1>::value);
} // namespace cellshard::artifact::atom_store
