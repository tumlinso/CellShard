#pragma once

#include <cstddef>

namespace cellshard {

inline constexpr unsigned int blocked_ell_block_size_candidates[] = {4u, 8u, 16u, 32u};
inline constexpr unsigned int blocked_ell_block_size_candidate_count =
    (unsigned int) (sizeof(blocked_ell_block_size_candidates) / sizeof(blocked_ell_block_size_candidates[0]));

} // namespace cellshard
