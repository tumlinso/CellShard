#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace cellshard::artifact::atom_store {
enum class lineage_operation_v1 : std::uint32_t { compose = 1, lower = 2, resume = 3 };
struct composition_lineage_record_v1 {
    semantic_identity_v1 result{};
    action_identity_v1 action{};
    std::uint64_t parent_offset = 0;
    std::uint64_t parent_count = 0;
    std::uint64_t lineage_generation = 0;
    lineage_operation_v1 operation{};
    std::uint32_t reserved = 0;
};
struct provenance_record_v1 {
    semantic_identity_v1 subject{};
    content_digest_v1 source_content{};
    content_digest_v1 evidence_content{};
    std::uint64_t provider_identity = 0;
    std::uint64_t evidence_generation = 0;
    std::uint64_t source_epoch = 0;
};
[[nodiscard]] constexpr bool lineage_digest_nonzero_v1(const content_digest_v1 &d) noexcept {
    for (auto b : d.bytes) if (b != std::byte{0}) return true;
    return false;
}
[[nodiscard]] constexpr bool valid_composition_lineage_record_v1(const composition_lineage_record_v1 &r) noexcept {
    return r.result.valid() && r.action.valid() && r.parent_count != 0
        && r.lineage_generation != 0 && (r.operation == lineage_operation_v1::compose
        || r.operation == lineage_operation_v1::lower || r.operation == lineage_operation_v1::resume);
}
[[nodiscard]] constexpr bool valid_provenance_record_v1(const provenance_record_v1 &r) noexcept {
    return r.subject.valid() && valid_content_digest_v1(r.source_content)
        && valid_content_digest_v1(r.evidence_content) && lineage_digest_nonzero_v1(r.source_content)
        && lineage_digest_nonzero_v1(r.evidence_content) && r.provider_identity != 0
        && r.evidence_generation != 0 && r.source_epoch != 0;
}
static_assert(std::is_trivially_copyable<composition_lineage_record_v1>::value);
static_assert(std::is_trivially_copyable<provenance_record_v1>::value);
}
