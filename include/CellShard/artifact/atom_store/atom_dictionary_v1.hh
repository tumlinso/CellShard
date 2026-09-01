#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <cstdint>
#include <type_traits>
namespace cellshard::artifact::atom_store {
enum class atom_kind_v1 : std::uint32_t {
    relation = 1, bundle = 2, trajectory = 3, multimodal = 4,
    partial = 5, provider_defined = 0xffff'ffffu,
};
struct atom_dictionary_record_v1 {
    semantic_identity_v1 semantic{};
    content_digest_v1 exact_coverage{};
    std::uint64_t domain_binding_offset = 0;
    std::uint64_t domain_binding_count = 0;
    std::uint64_t dependency_offset = 0;
    std::uint64_t dependency_count = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t payload_count = 0;
    std::uint64_t structure_epoch = 0;
    atom_kind_v1 kind = atom_kind_v1::provider_defined;
    std::uint32_t certified = 0;
};
[[nodiscard]] constexpr bool digest_has_content_v1(
    const content_digest_v1 &digest) noexcept {
    for (auto byte : digest.bytes) if (byte != std::byte{0}) return true;
    return false;
}
[[nodiscard]] constexpr bool valid_atom_dictionary_record_v1(
    const atom_dictionary_record_v1 &record) noexcept {
    return record.semantic.valid() && valid_content_digest_v1(record.exact_coverage)
        && digest_has_content_v1(record.exact_coverage)
        && record.domain_binding_count != 0 && record.payload_count != 0
        && record.structure_epoch != 0 && record.certified == 1;
}
static_assert(std::is_trivially_copyable<atom_dictionary_record_v1>::value);
} // namespace cellshard::artifact::atom_store
