#pragma once

#include <array>
#include <cstdint>

namespace cellshard::jbc::validation {

using global_id = std::uint64_t;
struct fixture {
    global_id fixture_id;
    const char* name;
    std::uint32_t rows;
    std::uint32_t columns;
    const std::uint32_t* row_offsets;
    const std::uint32_t* column_indices;
    std::uint32_t nonzeros;
    bool biological;
};

inline constexpr std::array<std::uint32_t, 5> module_offsets{{0, 2, 4, 6, 8}};
inline constexpr std::array<std::uint32_t, 8> module_columns{{0, 1, 0, 1, 4, 5, 4, 5}};
inline constexpr std::array<std::uint32_t, 5> chain_offsets{{0, 1, 3, 5, 6}};
inline constexpr std::array<std::uint32_t, 6> chain_columns{{0, 0, 1, 1, 2, 2}};
inline constexpr std::array<std::uint32_t, 5> synthetic_offsets{{0, 2, 4, 6, 8}};
inline constexpr std::array<std::uint32_t, 8> synthetic_columns{{0, 3, 1, 4, 2, 5, 0, 5}};

inline constexpr std::array<fixture, 3> corpus{{
    {UINT64_C(0x100000001), "modular_regulation", 4, 6, module_offsets.data(), module_columns.data(), 8, true},
    {UINT64_C(0x100000002), "trajectory_chain", 4, 3, chain_offsets.data(), chain_columns.data(), 6, true},
    {UINT64_C(0x100000003), "uniform_synthetic", 4, 6, synthetic_offsets.data(), synthetic_columns.data(), 8, false}}};

inline bool valid_fixture(const fixture& item) noexcept {
    if (item.fixture_id == 0 || item.rows == 0 || item.columns == 0 ||
        item.row_offsets == nullptr || (item.nonzeros != 0 && item.column_indices == nullptr)) return false;
    if (item.row_offsets[0] != 0 || item.row_offsets[item.rows] != item.nonzeros) return false;
    for (std::uint32_t row = 0; row < item.rows; ++row) {
        if (item.row_offsets[row] > item.row_offsets[row + 1]) return false;
        for (std::uint32_t i = item.row_offsets[row]; i < item.row_offsets[row + 1]; ++i)
            if (item.column_indices[i] >= item.columns) return false;
    }
    return true;
}

}  // namespace cellshard::jbc::validation
