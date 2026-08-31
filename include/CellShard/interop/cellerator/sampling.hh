#pragma once

#include <CellShard/export/dataset_export.hh>

#include <Cellerator/compute/sampling.hh>
#include <Cellerator/compute/sampling_materialization.hh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace cellshard::interop::cellerator {

// Optional storage-to-compute adapter. CellShard owns the dataset path and
// observation-name load; Cellerator owns the deterministic sampling contract.
inline bool build_sample_plan(
    const char *path,
    const ::cellerator::compute::sampling::sample_spec &spec,
    ::cellerator::compute::sampling::sample_plan *out,
    std::string *error = nullptr) {
    namespace sampling = ::cellerator::compute::sampling;
    exporting::dataset_summary summary;
    sampling::cell_identity_view identities;
    std::vector<const char *> stable_ids;
    if (!exporting::load_dataset_summary(path, &summary, error)) return false;
    if (summary.obs_names.size() == summary.rows) {
        stable_ids.reserve(summary.obs_names.size());
        for (const std::string &value : summary.obs_names) {
            stable_ids.push_back(value.c_str());
        }
        identities.kind = sampling::cell_identity_kind::stable_item_id;
        identities.stable_cell_ids = stable_ids.empty() ? nullptr : stable_ids.data();
        identities.count = (std::uint64_t) stable_ids.size();
    }
    return sampling::build_sample_plan(summary.rows, spec, identities, out, error);
}

// Path-backed selected-row loading remains CellShard-owned. Only the resulting
// canonical CSR arrays cross into Cellerator's in-memory structural adapter.
inline bool materialize_sampled_csr_structure(
    const char *path,
    const ::cellerator::compute::sampling::sample_plan &plan,
    ::cellerator::compute::sampling::owned_sampled_csr_structure *out,
    std::string *error = nullptr) {
    namespace sampling = ::cellerator::compute::sampling;
    namespace types = ::cellerator::types;
    exporting::dataset_summary summary;
    exporting::csr_matrix_export csr;
    std::vector<std::uint64_t> rows = plan.global_row_indices;
    std::vector<types::ptr_t> row_ptr;
    std::vector<types::idx_t> column_indices;
    if (path == nullptr || *path == '\0' || out == nullptr) {
        if (error != nullptr) *error = "CellShard sampled materialization requires a path and output";
        return false;
    }
    std::sort(rows.begin(), rows.end());
    if (!exporting::load_dataset_summary(path, &summary, error)
        || !exporting::load_dataset_rows_as_csr(
            path, rows.empty() ? nullptr : rows.data(), rows.size(), &csr, error)) {
        return false;
    }
    if (csr.rows != rows.size() || csr.cols != summary.cols
        || csr.indptr.size() != rows.size() + 1u
        || csr.indices.size() != csr.data.size()) {
        if (error != nullptr) *error = "CellShard selected-row CSR arrays are inconsistent";
        return false;
    }
    row_ptr.reserve(csr.indptr.size());
    for (std::int64_t pointer : csr.indptr) {
        if (pointer < 0
            || (std::uint64_t) pointer > (std::uint64_t) std::numeric_limits<types::ptr_t>::max()) {
            if (error != nullptr) *error = "CellShard selected-row pointer exceeds Cellerator limits";
            return false;
        }
        row_ptr.push_back((types::ptr_t) pointer);
    }
    column_indices.reserve(csr.indices.size());
    for (std::int64_t column : csr.indices) {
        if (column < 0
            || (std::uint64_t) column > (std::uint64_t) std::numeric_limits<types::idx_t>::max()) {
            if (error != nullptr) *error = "CellShard selected-row index exceeds Cellerator limits";
            return false;
        }
        column_indices.push_back((types::idx_t) column);
    }
    const sampling::selected_csr_structure_view selected{
        summary.rows,
        (std::uint64_t) rows.size(),
        summary.cols,
        (std::uint64_t) column_indices.size(),
        row_ptr.data(),
        column_indices.empty() ? nullptr : column_indices.data(),
        rows.empty() ? nullptr : rows.data()
    };
    return sampling::materialize_selected_csr_structure(selected, plan, out, error);
}

} // namespace cellshard::interop::cellerator
