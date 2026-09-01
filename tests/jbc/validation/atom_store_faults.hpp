#pragma once
#include <array>
#include <cstdint>
namespace cellshard::jbc::validation {
enum class store_fault : std::uint8_t { none, truncated_record, bad_checksum, crash_before_commit, crash_after_commit, interrupted_consolidation, gc_with_live_pin };
enum class recovery_outcome : std::uint8_t { accept, reject_corrupt_tail, retain_old_head, adopt_committed_head, resume_consolidation, retain_pinned_atom };
struct fault_case { store_fault fault; recovery_outcome expected; bool live_generation_preserved; };
inline constexpr std::array<fault_case, 7> atom_store_fault_matrix{{
    {store_fault::none, recovery_outcome::accept, true},
    {store_fault::truncated_record, recovery_outcome::reject_corrupt_tail, true},
    {store_fault::bad_checksum, recovery_outcome::reject_corrupt_tail, true},
    {store_fault::crash_before_commit, recovery_outcome::retain_old_head, true},
    {store_fault::crash_after_commit, recovery_outcome::adopt_committed_head, true},
    {store_fault::interrupted_consolidation, recovery_outcome::resume_consolidation, true},
    {store_fault::gc_with_live_pin, recovery_outcome::retain_pinned_atom, true}}};
inline bool complete_fault_matrix() noexcept {
    std::uint32_t seen = 0;
    for (const auto& item : atom_store_fault_matrix) {
        seen |= UINT32_C(1) << static_cast<unsigned>(item.fault);
        if (!item.live_generation_preserved) return false;
    }
    return seen == ((UINT32_C(1) << atom_store_fault_matrix.size()) - 1U);
}
}  // namespace cellshard::jbc::validation
