#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <cstddef>
#include <cstdint>
namespace cellshard::artifact::atom_store {
enum class compatibility_family_v1 : std::uint32_t { csh5=1, cspack=2, cpexec01=3, cpexec02=4 };
struct compatibility_import_v1 { semantic_identity_v1 atom{}; action_identity_v1 import_action{}; compatibility_family_v1 family{}; std::uint32_t reserved=0; content_digest_v1 source_content{}; std::uint64_t source_bytes=0; };
enum class compatibility_import_status_v1 : std::uint32_t { success, invalid_input, unrecognized, csh5_confirmation_required };
[[nodiscard]] compatibility_import_status_v1 inspect_compatibility_import_v1(const std::byte *source,std::size_t bytes,bool csh5_magic_confirmed,semantic_identity_v1 atom,action_identity_v1 action,compatibility_import_v1 *out) noexcept;
}
