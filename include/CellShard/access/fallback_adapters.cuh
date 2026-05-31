#pragma once

#include "adapter.cuh"
#include "../formats/compressed.cuh"
#include "../formats/dense.cuh"

#include <cstring>

namespace cellshard {
namespace access {

struct dense_fallback_binding {
    const cellshard::dense *archive;
    cellshard::dense *pack;
    std::uint32_t assay_id;
};

struct compressed_fallback_binding {
    const cellshard::sparse::compressed *archive;
    cellshard::sparse::compressed *pack;
    std::uint32_t assay_id;
};

namespace detail {

inline bool selection_is_all(const cell_selection_view &selection) {
    return selection.span_count == 0u && selection.index_count == 0u;
}

inline cell_span full_span(std::uint64_t count) {
    return cell_span{0u, count};
}

inline std::uint64_t selected_feature_count(const feature_selection_view &features, std::uint64_t fallback_all_count) {
    return selected_count(features, fallback_all_count);
}

inline const types::storage_value_t *dense_value_ptr(const cellshard::dense *matrix, std::uint64_t row, std::uint64_t col) {
    return cellshard::at(matrix, static_cast<types::dim_t>(row), static_cast<types::idx_t>(col));
}

inline std::size_t dense_copy_feature_spans(
    const cellshard::dense *matrix,
    std::uint64_t row,
    const feature_selection_view &features,
    types::storage_value_t *out) {
    std::size_t written = 0u;
    if (selection_is_all(features)) {
        if (matrix->order == cellshard::dense_row_major) {
            const types::storage_value_t *src = dense_value_ptr(matrix, row, 0u);
            const std::size_t count = static_cast<std::size_t>(matrix->cols);
            std::memcpy(out, src, count * sizeof(types::storage_value_t));
            return count;
        }
        for (std::uint64_t col = 0u; col < matrix->cols; ++col) {
            const types::storage_value_t *src = dense_value_ptr(matrix, row, col);
            out[written++] = src != nullptr ? *src : types::storage_value_t{};
        }
        return written;
    }
    if (features.index_count != 0u) {
        for (std::uint64_t i = 0u; i < features.index_count; ++i) {
            const types::storage_value_t *src = dense_value_ptr(matrix, row, features.indices[i]);
            out[written++] = src != nullptr ? *src : types::storage_value_t{};
        }
        return written;
    }
    for (std::uint32_t span_i = 0u; span_i < features.span_count; ++span_i) {
        const cell_span span = features.spans[span_i];
        if (matrix->order == cellshard::dense_row_major) {
            const types::storage_value_t *src = dense_value_ptr(matrix, row, span.begin);
            const std::size_t count = static_cast<std::size_t>(span.size());
            std::memcpy(out + written, src, count * sizeof(types::storage_value_t));
            written += count;
        } else {
            for (std::uint64_t col = span.begin; col < span.end; ++col) {
                const types::storage_value_t *src = dense_value_ptr(matrix, row, col);
                out[written++] = src != nullptr ? *src : types::storage_value_t{};
            }
        }
    }
    return written;
}

inline std::uint64_t compressed_major_count(const cellshard::sparse::compressed *matrix) {
    return static_cast<std::uint64_t>(cellshard::sparse::major_dim(matrix));
}

} // namespace detail

template<>
struct archive_adapter<dense_fallback_binding> {
    static archive_descriptor describe(const adapter_view<dense_fallback_binding> &view) {
        const cellshard::dense *matrix = view.binding != nullptr ? view.binding->archive : nullptr;
        if (matrix == nullptr) return archive_descriptor{};
        return archive_descriptor{
            matrix->rows,
            matrix->cols,
            static_cast<std::uint64_t>(matrix->rows) * static_cast<std::uint64_t>(matrix->cols),
            view.binding->assay_id,
            types::value_code<types::storage_value_t>::code,
            disk_format_dense,
            dataset_execution_format_dense,
            capability_contiguous_payload | capability_cell_span_copy | capability_feature_span_copy | capability_debug_at
        };
    }

    static fallback_payload_view contiguous_payload(const adapter_view<dense_fallback_binding> &view) {
        const cellshard::dense *matrix = view.binding != nullptr ? view.binding->archive : nullptr;
        if (matrix == nullptr) return fallback_payload_view{};
        return fallback_payload_view{
            byte_span{matrix, sizeof(*matrix), 0u},
            byte_span{},
            byte_span{},
            byte_span{matrix->val, cellshard::payload_bytes(matrix), types::value_code<types::storage_value_t>::code},
            byte_span{}
        };
    }

    static copy_preflight preflight_cell_spans(
        const adapter_view<dense_fallback_binding> &view,
        const cell_selection_view &cells,
        const feature_selection_view &features) {
        const archive_descriptor desc = describe(view);
        const std::uint64_t cell_count = selected_count(cells, desc.cell_count);
        const std::uint64_t feature_count = detail::selected_feature_count(features, desc.feature_count);
        return copy_preflight{
            cell_count,
            feature_count,
            cell_count * feature_count,
            static_cast<std::size_t>(cell_count * feature_count * sizeof(types::storage_value_t)),
            0u
        };
    }

    static pack_build_result copy_cell_spans(
        const adapter_view<dense_fallback_binding> &view,
        const cell_span_copy_request &request) {
        const cellshard::dense *matrix = view.binding != nullptr ? view.binding->archive : nullptr;
        if (matrix == nullptr) return pack_build_result{};
        const copy_preflight need = preflight_cell_spans(view, request.cells, request.features);
        if (request.output.data == nullptr || request.output.bytes < need.output_bytes) return pack_build_result{};

        types::storage_value_t *out = static_cast<types::storage_value_t *>(request.output.data);
        std::uint64_t out_row = 0u;
        if (request.cells.index_count != 0u) {
            for (std::uint64_t i = 0u; i < request.cells.index_count; ++i, ++out_row) {
                detail::dense_copy_feature_spans(matrix, request.cells.indices[i], request.features, out + out_row * need.feature_count);
            }
        } else {
            const cell_span full = detail::full_span(matrix->rows);
            const cell_span *spans = request.cells.span_count != 0u ? request.cells.spans : &full;
            const std::uint32_t span_count = request.cells.span_count != 0u ? request.cells.span_count : 1u;
            for (std::uint32_t span_i = 0u; span_i < span_count; ++span_i) {
                for (std::uint64_t row = spans[span_i].begin; row < spans[span_i].end; ++row, ++out_row) {
                    detail::dense_copy_feature_spans(matrix, row, request.features, out + out_row * need.feature_count);
                }
            }
        }
        return pack_build_result{need.cell_count, need.feature_count, need.nnz, need.output_bytes, dataset_execution_format_dense, 1u};
    }
};

template<>
struct pack_adapter<dense_fallback_binding> {
    static pack_descriptor describe(const adapter_view<dense_fallback_binding> &view) {
        const cellshard::dense *matrix = view.binding != nullptr
            ? (view.binding->pack != nullptr ? view.binding->pack : view.binding->archive)
            : nullptr;
        if (matrix == nullptr) return pack_descriptor{};
        return pack_descriptor{
            matrix->rows,
            matrix->cols,
            static_cast<std::uint64_t>(matrix->rows) * static_cast<std::uint64_t>(matrix->cols),
            types::value_code<types::storage_value_t>::code,
            dataset_execution_format_dense,
            cellshard::payload_bytes(matrix),
            capability_contiguous_payload | capability_pack_read | capability_pack_write
        };
    }
};

template<>
struct archive_adapter<compressed_fallback_binding> {
    static archive_descriptor describe(const adapter_view<compressed_fallback_binding> &view) {
        const cellshard::sparse::compressed *matrix = view.binding != nullptr ? view.binding->archive : nullptr;
        if (matrix == nullptr) return archive_descriptor{};
        return archive_descriptor{
            matrix->rows,
            matrix->cols,
            matrix->nnz,
            view.binding->assay_id,
            types::value_code<types::storage_value_t>::code,
            disk_format_compressed,
            dataset_execution_format_compressed,
            capability_contiguous_payload | capability_cell_span_copy | capability_feature_span_copy | capability_pinned_host_staging | capability_debug_at
        };
    }

    static fallback_payload_view contiguous_payload(const adapter_view<compressed_fallback_binding> &view) {
        const cellshard::sparse::compressed *matrix = view.binding != nullptr ? view.binding->archive : nullptr;
        if (matrix == nullptr) return fallback_payload_view{};
        return fallback_payload_view{
            byte_span{matrix, sizeof(*matrix), 0u},
            byte_span{matrix->majorPtr, static_cast<std::size_t>(detail::compressed_major_count(matrix) + 1u) * sizeof(types::ptr_t), types::value_code<types::ptr_t>::code},
            byte_span{matrix->minorIdx, static_cast<std::size_t>(matrix->nnz) * sizeof(types::idx_t), types::value_code<types::idx_t>::code},
            byte_span{matrix->val, static_cast<std::size_t>(matrix->nnz) * sizeof(types::storage_value_t), types::value_code<types::storage_value_t>::code},
            byte_span{}
        };
    }

    static copy_preflight preflight_cell_spans(
        const adapter_view<compressed_fallback_binding> &view,
        const cell_selection_view &cells,
        const feature_selection_view &features) {
        const archive_descriptor desc = describe(view);
        const std::uint64_t cell_count = selected_count(cells, desc.cell_count);
        const std::uint64_t feature_count = detail::selected_feature_count(features, desc.feature_count);
        return copy_preflight{cell_count, feature_count, desc.nnz, 0u, 0u};
    }
};

template<>
struct pack_adapter<compressed_fallback_binding> {
    static pack_descriptor describe(const adapter_view<compressed_fallback_binding> &view) {
        const cellshard::sparse::compressed *matrix = view.binding != nullptr
            ? (view.binding->pack != nullptr ? view.binding->pack : view.binding->archive)
            : nullptr;
        if (matrix == nullptr) return pack_descriptor{};
        return pack_descriptor{
            matrix->rows,
            matrix->cols,
            matrix->nnz,
            types::value_code<types::storage_value_t>::code,
            dataset_execution_format_compressed,
            cellshard::sparse::bytes(matrix),
            capability_contiguous_payload | capability_pack_read | capability_pack_write
        };
    }
};

template<class Policy>
struct archive_to_pack<dense_fallback_binding, dense_fallback_binding, Policy> {
    static pack_build_result build(
        const adapter_view<dense_fallback_binding> &archive,
        const adapter_view<dense_fallback_binding> &,
        const pack_build_request &request) {
        return archive_adapter<dense_fallback_binding>::copy_cell_spans(
            archive,
            cell_span_copy_request{request.cells, request.features, request.output, request.scratch});
    }
};

} // namespace access
} // namespace cellshard
