#include "../internal/common.hh"

#include <algorithm>

namespace cellshard::exporting {

namespace {

using namespace detail;

template<typename Part>
using append_row_fn = void (*)(const Part *, std::uint32_t, std::vector<std::int64_t> *, std::vector<float> *);

template<typename Part>
void release_fetched_partitions(cellshard::sharded<Part> *view, const std::vector<unsigned long> &partition_ids) {
    if (view == nullptr) return;
    for (unsigned long partition_id : partition_ids) (void) cellshard::drop_partition(view, partition_id);
}

template<typename Part, append_row_fn<Part> append_row>
bool load_dataset_as_csr_impl(const char *path,
                              const char *format_label,
                              csr_matrix_export *out,
                              std::string *error) {
    cellshard::sharded<Part> view;
    cellshard::shard_storage storage;
    unsigned long global_row = 0ul;
    cellshard::init(&view);
    cellshard::init(&storage);

    const auto fail = [&](const std::string &message) {
        clear_loaded_view(&view, &storage);
        set_error(error, message);
        return false;
    };

    if (!cellshard::load_header(path, &view, &storage)) {
        return fail(std::string("failed to load ") + format_label + " dataset header");
    }

    out->rows = view.rows;
    out->cols = view.cols;
    out->indptr.assign((std::size_t) view.rows + 1u, 0);
    out->indices.reserve((std::size_t) view.nnz);
    out->data.reserve((std::size_t) view.nnz);

    for (unsigned long partition_id = 0; partition_id < view.num_partitions; ++partition_id) {
        const Part *part = nullptr;
        if (!cellshard::fetch_partition(&view, &storage, partition_id)) {
            return fail(std::string("failed to materialize ") + format_label + " dataset partition");
        }
        part = view.parts[partition_id];
        if (part == nullptr) {
            return fail(std::string(format_label) + " dataset partition is null after fetch");
        }
        for (std::uint32_t row = 0u; row < part->rows; ++row) {
            append_row(part, row, &out->indices, &out->data);
            ++global_row;
            out->indptr[global_row] = (std::int64_t) out->data.size();
        }
        if (!cellshard::drop_partition(&view, partition_id)) {
            return fail(std::string("failed to release ") + format_label + " dataset partition");
        }
    }

    clear_loaded_view(&view, &storage);
    return true;
}

template<typename Part, append_row_fn<Part> append_row>
bool load_dataset_rows_as_csr_impl(const char *path,
                                   const char *format_label,
                                   const std::uint64_t *row_indices,
                                   std::size_t row_count,
                                   csr_matrix_export *out,
                                   std::string *error) {
    cellshard::sharded<Part> view;
    cellshard::shard_storage storage;
    std::vector<unsigned long> fetched_partitions;
    cellshard::init(&view);
    cellshard::init(&storage);

    const auto fail = [&](const std::string &message) {
        release_fetched_partitions(&view, fetched_partitions);
        clear_loaded_view(&view, &storage);
        set_error(error, message);
        return false;
    };

    if (!cellshard::load_header(path, &view, &storage)) {
        return fail(std::string("failed to load ") + format_label + " dataset header");
    }

    out->rows = (std::uint64_t) row_count;
    out->cols = view.cols;
    out->indptr.assign(row_count + 1u, 0);
    fetched_partitions.reserve(std::min<std::size_t>(row_count, (std::size_t) view.num_partitions));

    for (std::size_t i = 0; i < row_count; ++i) {
        const std::uint64_t global_row = row_indices[i];
        const Part *part = nullptr;
        unsigned long partition_id = 0ul;
        unsigned long partition_row_begin = 0ul;
        std::uint64_t local_row = 0u;

        if (global_row >= view.rows) return fail("row index is out of range");

        partition_id = cellshard::find_partition(&view, (unsigned long) global_row);
        if (partition_id >= view.num_partitions) {
            return fail("failed to resolve row index to a partition");
        }
        if (!cellshard::partition_loaded(&view, partition_id)) {
            if (!cellshard::fetch_partition(&view, &storage, partition_id)) {
                return fail(std::string("failed to fetch ") + format_label + " partition for row subset");
            }
            fetched_partitions.push_back(partition_id);
        }

        part = view.parts[partition_id];
        if (part == nullptr) {
            return fail(std::string(format_label) + " partition is null after row subset fetch");
        }

        partition_row_begin = cellshard::first_row_in_partition(&view, partition_id);
        local_row = global_row - (std::uint64_t) partition_row_begin;
        if (local_row >= part->rows) {
            return fail(std::string("resolved local row is outside the fetched ") + format_label + " partition");
        }

        append_row(part, (std::uint32_t) local_row, &out->indices, &out->data);
        out->indptr[i + 1u] = (std::int64_t) out->data.size();
    }

    for (unsigned long partition_id : fetched_partitions) {
        if (!cellshard::drop_partition(&view, partition_id)) {
            return fail(std::string("failed to release ") + format_label + " partition after row subset export");
        }
    }

    clear_loaded_view(&view, &storage);
    return true;
}

} // namespace

bool load_dataset_as_csr(const char *path, csr_matrix_export *out, std::string *error) {
    std::string matrix_format;

    if (out == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "dataset path is empty");
        return false;
    }
    *out = csr_matrix_export{};
    warn_cpu_materialize_once("CSR materialization");

    if (!load_matrix_format(path, &matrix_format, error)) return false;

    if (matrix_format == "blocked_ell") {
        return load_dataset_as_csr_impl<cellshard::sparse::blocked_ell, append_blocked_ell_row>(
            path, "blocked-ELL", out, error);
    }
    if (matrix_format == "sliced_ell") {
        return load_dataset_as_csr_impl<cellshard::sparse::sliced_ell, append_sliced_ell_row>(
            path, "sliced-ELL", out, error);
    }

    set_error(error, "unsupported matrix_format for CSR export: " + matrix_format);
    return false;
}

bool load_dataset_rows_as_csr(const char *path,
                              const std::uint64_t *row_indices,
                              std::size_t row_count,
                              csr_matrix_export *out,
                              std::string *error) {
    std::string matrix_format;

    if (out == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "dataset path is empty");
        return false;
    }
    if (row_count != 0u && row_indices == nullptr) {
        set_error(error, "row_indices is null");
        return false;
    }
    *out = csr_matrix_export{};
    warn_cpu_materialize_once("row-subset CSR materialization");

    if (!load_matrix_format(path, &matrix_format, error)) return false;

    if (matrix_format == "blocked_ell") {
        return load_dataset_rows_as_csr_impl<cellshard::sparse::blocked_ell, append_blocked_ell_row>(
            path, "blocked-ELL", row_indices, row_count, out, error);
    }
    if (matrix_format == "sliced_ell") {
        return load_dataset_rows_as_csr_impl<cellshard::sparse::sliced_ell, append_sliced_ell_row>(
            path, "sliced-ELL", row_indices, row_count, out, error);
    }

    set_error(error, "unsupported matrix_format for row-subset CSR export: " + matrix_format);
    return false;
}

bool load_dataset_for_anndata(const char *path, anndata_export *out, std::string *error) {
    if (out == nullptr) {
        set_error(error, "anndata export output is null");
        return false;
    }
    *out = anndata_export{};
    if (!load_dataset_summary(path, &out->summary, error)) return false;
    if (!(::cellshard::exporting::load_observation_metadata)(path, &out->obs_columns, error)) return false;
    if (!(::cellshard::exporting::load_feature_metadata)(path, &out->var_columns, error)) return false;
    if (!(::cellshard::exporting::load_dataset_attributes)(path, &out->uns, error)) return false;
    if (!load_dataset_as_csr(path, &out->x, error)) return false;
    return true;
}

} // namespace cellshard::exporting
