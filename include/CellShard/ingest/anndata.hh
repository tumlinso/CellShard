#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cellshard::ingest::anndata_io {

enum class matrix_slot { x, layer, raw };
struct matrix_selection {
    matrix_slot slot = matrix_slot::x;
    std::string layer;
};
struct matrix_info {
    std::size_t rows, cols;
    bool csr;
    std::string value_dtype;
};
struct read_stats {
    std::uint64_t nnz = 0;
    std::size_t batches = 0, peak_buffer_bytes = 0;
    bool count_like = true;
};

// A synchronous borrowed chunk in the source's CSR/CSC order. Indices and
// offsets are explicitly widened to uint64; numerical values convert to double.
// Offsets start at zero. Minor indices are local to this rectangular chunk.
// The arrays remain valid only during the visitor call; consume/copy explicitly.
// Empty rectangles have no stored entries and need not produce a callback.
struct sparse_chunk {
    std::size_t row_begin, col_begin, rows, cols;
    bool csr;
    std::span<const double> values;
    std::span<const std::uint64_t> indices, offsets;
};
using chunk_visitor = void (*)(const sparse_chunk&, void*);

// Owns a read-only backed AnnData session. Native module and storage types stay
// in the compiled implementation. Reader payload buffers and anndata scratch
// are each bounded by working_bytes; output/metadata and HDF5 internals are
// separate. Oversized indivisible sparse segments fail without a full read.
class source {
public:
    explicit source(std::string_view path, matrix_selection matrix = {},
                    std::size_t working_bytes = 8u << 20u);
    ~source();
    source(source&&) noexcept;
    source& operator=(source&&) noexcept;
    source(const source&) = delete;
    source& operator=(const source&) = delete;
    matrix_info info() const;
    read_stats row_nnz(std::span<std::uint64_t> counts) const;
    read_stats visit_rows(std::size_t first, std::size_t last,
                          chunk_visitor visitor, void* context) const;
    // Canonical stored obs/selected-var index labels; raw uses its own var axis.
    std::string observation_name(std::size_t row) const;
    std::string feature_name(std::size_t col) const;
    std::vector<std::string> observation_columns() const;
    std::vector<std::string> feature_columns() const;
    // Metadata is explicitly formatted as text; null remains distinct from "".
    std::optional<std::string> observation_text(std::string_view column, std::size_t row) const;
    std::optional<std::string> feature_text(std::string_view column, std::size_t col) const;
private:
    struct impl;
    std::unique_ptr<impl> p_;
};

} // namespace cellshard::ingest::anndata_io
