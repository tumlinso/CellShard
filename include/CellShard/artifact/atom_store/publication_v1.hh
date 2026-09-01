#pragma once
#include <CellShard/artifact/atom_store/root_manifest_v1.hh>
#include <cstddef>
#include <cstdint>
namespace cellshard::artifact::atom_store {
enum class publication_status_v1 : std::uint32_t { success, invalid_generation, stage_failed, object_sync_failed, root_conflict, root_sync_failed };
struct publication_backend_v1 {
    void *context = nullptr;
    bool (*stage_immutable)(void *, const content_digest_v1 &, const std::byte *, std::size_t) = nullptr;
    bool (*sync_immutable)(void *, const content_digest_v1 &) = nullptr;
    bool (*compare_exchange_root)(void *, const content_digest_v1 &, const content_digest_v1 &) = nullptr;
    bool (*sync_root)(void *) = nullptr;
};
[[nodiscard]] publication_status_v1 publish_root_generation_v1(const root_generation_manifest_v1 &current, const root_generation_manifest_v1 &next, const std::byte *image, std::size_t image_bytes, const publication_backend_v1 &backend) noexcept;
}
