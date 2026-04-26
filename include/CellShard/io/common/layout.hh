#pragma once

namespace cellshard {

enum {
    dataset_matrix_family_none = 0u,
    dataset_matrix_family_blocked_ell = 1u,
    dataset_matrix_family_optimized_blocked_ell = 2u,
    dataset_matrix_family_sliced_ell = 3u,
    dataset_matrix_family_quantized_blocked_ell = 4u
};

enum {
    dataset_execution_format_unknown = 0u,
    dataset_execution_format_compressed = 1u,
    dataset_execution_format_blocked_ell = 2u,
    dataset_execution_format_mixed = 3u,
    dataset_execution_format_bucketed_blocked_ell = 4u,
    dataset_execution_format_sliced_ell = 5u,
    dataset_execution_format_bucketed_sliced_ell = 6u,
    dataset_execution_format_quantized_blocked_ell = 7u
};

} // namespace cellshard
