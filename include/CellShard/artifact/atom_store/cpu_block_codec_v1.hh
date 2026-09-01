#pragma once
#include <CellShard/artifact/atom_store/codec_registry_v1.hh>
namespace cellshard::artifact::atom_store {
inline constexpr std::uint64_t byte_rle_codec_identity_v1=3;
struct cpu_block_choice_v1 { std::uint64_t codec_identity=raw_codec_identity_v1; std::size_t encoded_bytes=0; };
[[nodiscard]] codec_provider_v1 byte_rle_codec_provider_v1() noexcept;
[[nodiscard]] codec_status_v1 select_cpu_block_codec_v1(const std::byte *input,std::size_t input_bytes,std::byte *candidate_buffer,std::size_t candidate_capacity,cpu_block_choice_v1 *choice) noexcept;
}
