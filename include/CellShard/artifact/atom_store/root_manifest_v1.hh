#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <cstdint>
#include <type_traits>
namespace cellshard::artifact::atom_store {
struct root_generation_manifest_v1 {
    semantic_identity_v1 store_identity{};
    std::uint64_t generation = 0;
    std::uint64_t structure_epoch = 0;
    content_digest_v1 root_content{};
    content_digest_v1 parent_root_content{};
    std::uint64_t atom_count = 0;
    std::uint64_t dependency_count = 0;
    std::uint64_t materialization_count = 0;
    std::uint64_t replica_count = 0;
};
[[nodiscard]] constexpr bool digest_is_zero_v1(const content_digest_v1 &digest) noexcept {
    for (auto byte : digest.bytes) if (byte != std::byte{0}) return false;
    return true;
}
[[nodiscard]] constexpr bool valid_root_generation_manifest_v1(
    const root_generation_manifest_v1 &manifest) noexcept {
    if (!manifest.store_identity.valid() || manifest.generation == 0
        || manifest.structure_epoch == 0 || !valid_content_digest_v1(manifest.root_content)
        || digest_is_zero_v1(manifest.root_content)) return false;
    if (manifest.generation == 1) return digest_is_zero_v1(manifest.parent_root_content);
    return valid_content_digest_v1(manifest.parent_root_content)
        && !digest_is_zero_v1(manifest.parent_root_content)
        && manifest.parent_root_content.bytes != manifest.root_content.bytes;
}
static_assert(std::is_trivially_copyable<root_generation_manifest_v1>::value);
} // namespace cellshard::artifact::atom_store
