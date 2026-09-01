#pragma once
#include <CellShard/artifact/atom_store/cpu_block_codec_v1.hh>
namespace cellshard::artifact::atom_store {
struct atom_block_view_v1 { const std::byte *data=nullptr; std::size_t bytes=0; };
struct atom_compression_experiment_v1 { std::size_t raw_bytes=0; std::size_t monolithic_bytes=0; std::size_t atom_aware_bytes=0; std::size_t atoms_using_rle=0; bool atom_aware_wins=false; };
[[nodiscard]] codec_status_v1 run_atom_compression_experiment_v1(const atom_block_view_v1 *atoms,std::size_t atom_count,std::size_t metadata_bytes_per_atom,std::byte *scratch,std::size_t scratch_bytes,atom_compression_experiment_v1 *out) noexcept;
}
