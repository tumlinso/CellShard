#include <CellShard/io/csh5/api.cuh>

#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cs = cellshard;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void set_row(cs::sparse::sliced_ell *m,
             std::uint32_t row,
             const std::vector<std::pair<std::uint32_t, float>> &values) {
    const std::uint32_t slice = cs::sparse::find_slice(m, row);
    const std::uint32_t row_begin = m->slice_row_offsets[slice];
    const std::uint32_t width = m->slice_widths[slice];
    const std::size_t base = cs::sparse::slice_slot_base(m, slice)
        + (std::size_t) (row - row_begin) * (std::size_t) width;
    require(values.size() <= width, "row values exceed Sliced-ELL width");
    for (std::size_t i = 0u; i < values.size(); ++i) {
        m->col_idx[base + i] = values[i].first;
        m->val[base + i] = __float2half(values[i].second);
    }
}

std::vector<std::pair<std::uint32_t, float>> canonical_row(const cs::sparse::sliced_ell *m,
                                                           std::uint32_t row) {
    const std::uint32_t slice = cs::sparse::find_slice(m, row);
    const std::uint32_t row_begin = m->slice_row_offsets[slice];
    const std::uint32_t width = m->slice_widths[slice];
    const std::size_t base = cs::sparse::slice_slot_base(m, slice)
        + (std::size_t) (row - row_begin) * (std::size_t) width;
    std::vector<std::pair<std::uint32_t, float>> out;
    for (std::uint32_t slot = 0u; slot < width; ++slot) {
        const std::uint32_t col = m->col_idx[base + slot];
        if (col == cs::sparse::sliced_ell_invalid_col) continue;
        out.push_back({col, __half2float(m->val[base + slot])});
    }
    return out;
}

std::vector<std::pair<std::uint32_t, float>> execution_row(const cs::bucketed_sliced_ell_partition *m,
                                                           std::uint32_t exec_row) {
    std::uint32_t segment = 0u;
    while (segment + 1u < m->segment_count && exec_row >= m->segment_row_offsets[segment + 1u]) ++segment;
    const cs::sparse::sliced_ell *part = m->segments + segment;
    const std::uint32_t local_row = exec_row - m->segment_row_offsets[segment];
    const std::uint32_t width = part->slice_count != 0u ? part->slice_widths[0] : 0u;
    const std::size_t base = (std::size_t) local_row * (std::size_t) width;
    std::vector<std::pair<std::uint32_t, float>> out;
    for (std::uint32_t slot = 0u; slot < width; ++slot) {
        const std::uint32_t col = part->col_idx[base + slot];
        if (col == cs::sparse::sliced_ell_invalid_col) continue;
        out.push_back({col, __half2float(part->val[base + slot])});
    }
    return out;
}

void require_same_row(const std::vector<std::pair<std::uint32_t, float>> &lhs,
                      const std::vector<std::pair<std::uint32_t, float>> &rhs,
                      const char *message) {
    require(lhs.size() == rhs.size(), message);
    for (std::size_t i = 0u; i < lhs.size(); ++i) {
        require(lhs[i].first == rhs[i].first, message);
        require(std::fabs(lhs[i].second - rhs[i].second) < 0.001f, message);
    }
}

void test_bucketed_sliced_execution_build() {
    const std::uint32_t offsets[] = {0u, 2u, 4u, 6u};
    const std::uint32_t widths[] = {4u, 2u, 5u};
    cs::sparse::sliced_ell src;
    cs::bucketed_sliced_ell_partition bucketed;
    std::uint64_t bucketed_bytes = 0u;

    cs::sparse::init(&src, 6u, 8u, 9u);
    cs::init(&bucketed);
    require(cs::sparse::allocate(&src, 3u, offsets, widths) != 0, "source Sliced-ELL allocation failed");
    set_row(&src, 0u, {{0u, 1.0f}, {3u, 2.0f}});
    set_row(&src, 1u, {{1u, 3.0f}});
    set_row(&src, 2u, {{2u, 4.0f}, {5u, 5.0f}});
    set_row(&src, 3u, {});
    set_row(&src, 4u, {{0u, 6.0f}, {1u, 7.0f}, {7u, 8.0f}});
    set_row(&src, 5u, {{6u, 9.0f}});

    require(cs::build_bucketed_sliced_ell_partition(&bucketed, &src, 8u, &bucketed_bytes) != 0,
            "optimized Sliced-ELL partition build failed");
    require(bucketed.rows == src.rows && bucketed.cols == src.cols && bucketed.nnz == src.nnz,
            "bucketed Sliced-ELL shape mismatch");
    require(bucketed.segment_count >= 1u, "bucketed Sliced-ELL has no segments");
    require(bucketed_bytes != 0u, "bucketed Sliced-ELL byte estimate missing");
    for (std::uint32_t row = 0u; row < src.rows; ++row) {
        const std::uint32_t exec_row = bucketed.canonical_to_exec_rows[row];
        require(exec_row < src.rows, "canonical_to_exec row out of range");
        require(bucketed.exec_to_canonical_rows[exec_row] == row, "row-order inverse mismatch");
        require_same_row(canonical_row(&src, row), execution_row(&bucketed, exec_row), "row payload mismatch");
    }

    cs::clear(&bucketed);
    cs::sparse::clear(&src);
}

} // namespace

int main() {
    try {
        test_bucketed_sliced_execution_build();
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
