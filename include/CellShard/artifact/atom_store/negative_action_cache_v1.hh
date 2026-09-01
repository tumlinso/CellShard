#pragma once

#include <CellShard/artifact/atom_store/identity_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::artifact::atom_store {

enum class negative_action_reason_v1 : std::uint32_t {
    unsupported_capability = 1,
    invalid_dependency_closure = 2,
    exact_certification_failed = 3,
};

struct negative_action_cache_entry_v1 {
    action_identity_v1 action{};
    content_digest_v1 source_content{};
    std::uint64_t structure_epoch = 0;
    std::uint64_t evidence_generation = 0;
    std::uint64_t valid_through_generation = 0;
    negative_action_reason_v1 reason{};
    std::uint32_t reserved = 0;
};

[[nodiscard]] constexpr bool valid_negative_action_cache_entry_v1(
    const negative_action_cache_entry_v1 &entry) noexcept {
    bool digest_nonzero = false;
    for (auto byte : entry.source_content.bytes)
        digest_nonzero = digest_nonzero || byte != std::byte{0};
    const bool known_reason = entry.reason == negative_action_reason_v1::unsupported_capability
        || entry.reason == negative_action_reason_v1::invalid_dependency_closure
        || entry.reason == negative_action_reason_v1::exact_certification_failed;
    return entry.action.valid() && valid_content_digest_v1(entry.source_content)
        && digest_nonzero && entry.structure_epoch != 0 && entry.evidence_generation != 0
        && entry.valid_through_generation >= entry.evidence_generation && known_reason;
}

class negative_action_cache_v1 {
public:
    constexpr negative_action_cache_v1(negative_action_cache_entry_v1 *entries,
                                       std::size_t capacity) noexcept
        : entries_(entries), capacity_(capacity) {}

    [[nodiscard]] constexpr const negative_action_cache_entry_v1 *find(
        action_identity_v1 action, const content_digest_v1 &source,
        std::uint64_t structure_epoch, std::uint64_t evidence_generation) const noexcept {
        for (std::size_t i = 0; i < size_; ++i) {
            const auto &entry = entries_[i];
            if (entry.action == action && entry.structure_epoch == structure_epoch
                && same_digest(entry.source_content, source)
                && evidence_generation >= entry.evidence_generation
                && evidence_generation <= entry.valid_through_generation) return &entry;
        }
        return nullptr;
    }

    [[nodiscard]] constexpr bool insert(const negative_action_cache_entry_v1 &entry) noexcept {
        if (entries_ == nullptr || size_ == capacity_
            || !valid_negative_action_cache_entry_v1(entry)) return false;
        entries_[size_++] = entry;
        return true;
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }

private:
    [[nodiscard]] static constexpr bool same_digest(const content_digest_v1 &a,
                                                    const content_digest_v1 &b) noexcept {
        if (a.algorithm != b.algorithm || a.digest_bytes != b.digest_bytes) return false;
        for (std::size_t i = 0; i < a.bytes.size(); ++i)
            if (a.bytes[i] != b.bytes[i]) return false;
        return true;
    }
    negative_action_cache_entry_v1 *entries_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t size_ = 0;
};

static_assert(std::is_trivially_copyable<negative_action_cache_entry_v1>::value);

} // namespace cellshard::artifact::atom_store
