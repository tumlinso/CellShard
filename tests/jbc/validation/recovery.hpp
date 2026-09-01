#pragma once
#include "fixtures.hpp"
#include <array>
namespace cellshard::jbc::validation {
struct recovery_evidence {
    global_id object_id = 0;
    global_id committed_generation_before = 0;
    global_id intended_generation = 0;
    global_id recovered_generation = 0;
    std::array<std::uint64_t, 4> committed_digest_before{};
    std::array<std::uint64_t, 4> recovered_digest{};
    bool commit_record_durable = false;
    bool replay_idempotent = false;
    bool partial_generation_visible = false;
};
inline bool valid_recovery(const recovery_evidence& evidence) noexcept {
    if (evidence.object_id == 0 || evidence.committed_generation_before == 0 ||
        evidence.intended_generation <= evidence.committed_generation_before ||
        !evidence.replay_idempotent || evidence.partial_generation_visible) return false;
    if (evidence.commit_record_durable)
        return evidence.recovered_generation == evidence.intended_generation;
    return evidence.recovered_generation == evidence.committed_generation_before &&
           evidence.recovered_digest == evidence.committed_digest_before;
}
}  // namespace cellshard::jbc::validation
