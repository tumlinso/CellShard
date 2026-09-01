#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace cellshard::artifact::atom_store {

inline constexpr std::array<std::byte, 8> file_magic_v1{
    std::byte{'C'}, std::byte{'S'}, std::byte{'A'}, std::byte{'T'},
    std::byte{'O'}, std::byte{'M'}, std::byte{'0'}, std::byte{'1'}};
inline constexpr std::string_view family_name_v1 = "CSATOM v1";
inline constexpr std::string_view file_suffix_v1 = ".csatom";
inline constexpr std::uint32_t schema_version_v1 = 1;
inline constexpr std::uint32_t endian_marker_v1 = UINT32_C(0x01020304);

} // namespace cellshard::artifact::atom_store
