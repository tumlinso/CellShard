#pragma once
#include "fixtures.hpp"
namespace cellshard::jbc::validation {
inline constexpr std::uint32_t max_null_swaps = 4096;
inline std::uint32_t edge_row(const fixture& item, std::uint32_t edge) noexcept {
    std::uint32_t low = 0, high = item.rows;
    while (low < high) { const auto middle = low + (high - low) / 2;
        if (item.row_offsets[middle + 1] <= edge) low = middle + 1; else high = middle; }
    return low;
}
inline bool row_contains(const fixture& item, const std::uint32_t* columns,
                         std::uint32_t row, std::uint32_t column,
                         std::uint32_t except) noexcept {
    for (std::uint32_t i = item.row_offsets[row]; i < item.row_offsets[row + 1]; ++i)
        if (i != except && columns[i] == column) return true;
    return false;
}
inline std::uint32_t matched_degree_null(const fixture& item, std::uint64_t seed,
                                         std::uint32_t requested_swaps,
                                         std::uint32_t* output) noexcept {
    if (!valid_fixture(item) || output == nullptr || requested_swaps > max_null_swaps) return 0;
    for (std::uint32_t i = 0; i < item.nonzeros; ++i) output[i] = item.column_indices[i];
    if (item.nonzeros < 2) return 0;
    std::uint32_t applied = 0;
    for (std::uint32_t attempt = 0; attempt < requested_swaps; ++attempt) {
        seed = seed * UINT64_C(6364136223846793005) + 1;
        const auto first = static_cast<std::uint32_t>((seed >> 32U) % item.nonzeros);
        seed = seed * UINT64_C(6364136223846793005) + 1;
        const auto second = static_cast<std::uint32_t>((seed >> 32U) % item.nonzeros);
        const auto first_row = edge_row(item, first), second_row = edge_row(item, second);
        if (first == second || first_row == second_row || output[first] == output[second] ||
            row_contains(item, output, first_row, output[second], first) ||
            row_contains(item, output, second_row, output[first], second)) continue;
        const auto temporary = output[first]; output[first] = output[second]; output[second] = temporary; ++applied;
    }
    return applied;
}
}  // namespace cellshard::jbc::validation
