#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <cstddef>
#include <cstdint>
namespace cellshard::artifact::atom_store {
inline constexpr std::uint64_t atom_link_missing_v1=UINT64_MAX;
enum class atom_link_status_v1 : std::uint32_t { success, invalid_input, unsorted_dictionary, cuda_failure };
[[nodiscard]] atom_link_status_v1 link_atoms_cpu_reference_v1(const semantic_identity_v1 *dictionary,std::size_t dictionary_count,const semantic_identity_v1 *queries,std::size_t query_count,std::uint64_t *indices) noexcept;
// All pointers are device-resident. The launch is allocation-free and async on
// the caller stream; dictionary validation remains a preparation-time CPU step.
[[nodiscard]] atom_link_status_v1 link_atoms_gpu_v1(const semantic_identity_v1 *dictionary,std::size_t dictionary_count,const semantic_identity_v1 *queries,std::size_t query_count,std::uint64_t *indices,void *stream) noexcept;
}
