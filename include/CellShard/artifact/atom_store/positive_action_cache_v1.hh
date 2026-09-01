#pragma once

#include <CellShard/artifact/atom_store/identity_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::artifact::atom_store {

struct positive_action_cache_entry_v1 {
    action_identity_v1 action{};
    content_digest_v1 source_content{};
    materialization_identity_v1 materialization{};
    content_digest_v1 output_content{};
    std::uint64_t structure_epoch = 0;
    std::uint64_t evidence_generation = 0;
    std::uint32_t certified = 0;
    std::uint32_t reserved = 0;
};

[[nodiscard]] constexpr bool action_cache_digest_nonzero_v1(
    const content_digest_v1 &digest) noexcept {
    for (auto byte : digest.bytes) if (byte != std::byte{0}) return true;
    return false;
}

[[nodiscard]] constexpr bool valid_positive_action_cache_entry_v1(
    const positive_action_cache_entry_v1 &entry) noexcept {
    return entry.action.valid() && entry.materialization.valid()
        && valid_content_digest_v1(entry.source_content)
        && valid_content_digest_v1(entry.output_content)
        && action_cache_digest_nonzero_v1(entry.source_content)
        && action_cache_digest_nonzero_v1(entry.output_content)
        && entry.structure_epoch != 0 && entry.evidence_generation != 0
        && entry.certified == 1;
}

enum class positive_action_cache_insert_result_v1 : std::uint32_t {
    inserted,
    already_present,
    invalid_entry,
    conflicting_result,
    capacity_exhausted,
};

class positive_action_cache_v1 {
public:
    constexpr positive_action_cache_v1(positive_action_cache_entry_v1 *entries,
                                       std::size_t capacity) noexcept
        : entries_(entries), capacity_(capacity) {}

    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr std::size_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] constexpr const positive_action_cache_entry_v1 *find(
        action_identity_v1 action, const content_digest_v1 &source_content,
        std::uint64_t structure_epoch) const noexcept {
        for (std::size_t index = 0; index < size_; ++index) {
            const auto &entry = entries_[index];
            if (entry.action == action && entry.structure_epoch == structure_epoch
                && digest_equal(entry.source_content, source_content)) return &entry;
        }
        return nullptr;
    }

    [[nodiscard]] constexpr positive_action_cache_insert_result_v1 insert(
        const positive_action_cache_entry_v1 &candidate) noexcept {
        if (!valid_positive_action_cache_entry_v1(candidate) || entries_ == nullptr)
            return positive_action_cache_insert_result_v1::invalid_entry;
        const auto *existing = find(candidate.action, candidate.source_content,
                                    candidate.structure_epoch);
        if (existing != nullptr) {
            if (existing->materialization == candidate.materialization
                && digest_equal(existing->output_content, candidate.output_content)
                && existing->evidence_generation == candidate.evidence_generation)
                return positive_action_cache_insert_result_v1::already_present;
            return positive_action_cache_insert_result_v1::conflicting_result;
        }
        if (size_ == capacity_)
            return positive_action_cache_insert_result_v1::capacity_exhausted;
        entries_[size_++] = candidate;
        return positive_action_cache_insert_result_v1::inserted;
    }

private:
    [[nodiscard]] static constexpr bool digest_equal(
        const content_digest_v1 &left, const content_digest_v1 &right) noexcept {
        if (left.algorithm != right.algorithm || left.digest_bytes != right.digest_bytes)
            return false;
        for (std::size_t index = 0; index < left.bytes.size(); ++index) {
            if (left.bytes[index] != right.bytes[index]) return false;
        }
        return true;
    }

    positive_action_cache_entry_v1 *entries_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t size_ = 0;
};

static_assert(std::is_trivially_copyable<positive_action_cache_entry_v1>::value);

} // namespace cellshard::artifact::atom_store
