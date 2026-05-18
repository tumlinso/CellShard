#include <CellShard/export/dataset_export.hh>

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace cse = ::cellshard::exporting;

namespace {

constexpr double default_abs_tolerance = 1.0e-3;
constexpr double default_rel_tolerance = 1.0e-5;

struct csr_matrix_view {
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::vector<std::int64_t> indptr;
    std::vector<std::int64_t> indices;
    std::vector<float> data;
};

struct value_summary {
    std::uint64_t compared = 0u;
    std::uint64_t mismatched_columns = 0u;
    std::uint64_t mismatched_values = 0u;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double mean_abs = 0.0;
    double total_abs = 0.0;
    std::uint64_t worst_row = 0u;
    std::int64_t worst_col = -1;
    float ref_value = 0.0f;
    float test_value = 0.0f;
};

struct axis_summary {
    std::uint64_t compared = 0u;
    std::uint64_t mismatched = 0u;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double mean_abs = 0.0;
    double total_abs = 0.0;
    std::uint64_t worst_index = 0u;
    double ref_value = 0.0;
    double test_value = 0.0;
};

struct validation_report {
    std::string h5ad_path;
    std::string csh5_path;
    double abs_tolerance = default_abs_tolerance;
    double rel_tolerance = default_rel_tolerance;
    std::uint64_t ref_rows = 0u;
    std::uint64_t ref_cols = 0u;
    std::uint64_t ref_nnz = 0u;
    std::uint64_t test_rows = 0u;
    std::uint64_t test_cols = 0u;
    std::uint64_t test_nnz = 0u;
    std::uint64_t indptr_mismatches = 0u;
    value_summary values;
    axis_summary rows;
    axis_summary cols;

    bool passed() const {
        return ref_rows == test_rows
            && ref_cols == test_cols
            && ref_nnz == test_nnz
            && indptr_mismatches == 0u
            && values.mismatched_columns == 0u
            && values.mismatched_values == 0u
            && rows.mismatched == 0u
            && cols.mismatched == 0u;
    }
};

struct options {
    std::string h5ad_path = "data/test/reference/pbmc3k_raw.h5ad";
    std::string csh5_path = "data/test/reference/pbmc3k_raw.csh5";
    double abs_tolerance = default_abs_tolerance;
    double rel_tolerance = default_rel_tolerance;
};

bool close_if_valid(hid_t id, herr_t (*close_fn)(hid_t)) {
    return id >= 0 ? close_fn(id) >= 0 : true;
}

bool dataset_extent_1d(hid_t parent, const char *name, hsize_t *extent, std::string *error) {
    hid_t dataset = (hid_t) -1;
    hid_t space = (hid_t) -1;
    int rank = 0;
    hsize_t dims[2] = {0u, 0u};
    bool ok = false;

    dataset = H5Dopen2(parent, name, H5P_DEFAULT);
    if (dataset < 0) {
        if (error) *error = std::string("failed to open dataset ") + name;
        return false;
    }
    space = H5Dget_space(dataset);
    if (space < 0) {
        if (error) *error = std::string("failed to inspect dataset ") + name;
        close_if_valid(dataset, H5Dclose);
        return false;
    }
    rank = H5Sget_simple_extent_ndims(space);
    if (rank != 1 || H5Sget_simple_extent_dims(space, dims, nullptr) < 0) {
        if (error) *error = std::string("dataset is not one-dimensional: ") + name;
        close_if_valid(space, H5Sclose);
        close_if_valid(dataset, H5Dclose);
        return false;
    }
    *extent = dims[0];
    ok = true;
    close_if_valid(space, H5Sclose);
    close_if_valid(dataset, H5Dclose);
    return ok;
}

template<typename T>
bool read_numeric_vector(hid_t parent, const char *name, hid_t mem_type, std::vector<T> *out, std::string *error) {
    hid_t dataset = (hid_t) -1;
    hid_t space = (hid_t) -1;
    hsize_t dims[1] = {0u};
    bool ok = false;

    if (out == nullptr) return false;
    dataset = H5Dopen2(parent, name, H5P_DEFAULT);
    if (dataset < 0) {
        if (error) *error = std::string("failed to open dataset ") + name;
        return false;
    }
    space = H5Dget_space(dataset);
    if (space < 0 || H5Sget_simple_extent_ndims(space) != 1 || H5Sget_simple_extent_dims(space, dims, nullptr) < 0) {
        if (error) *error = std::string("failed to inspect one-dimensional dataset ") + name;
        close_if_valid(space, H5Sclose);
        close_if_valid(dataset, H5Dclose);
        return false;
    }
    out->assign((std::size_t) dims[0], T{});
    ok = out->empty() || H5Dread(dataset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out->data()) >= 0;
    if (!ok && error) *error = std::string("failed to read dataset ") + name;
    close_if_valid(space, H5Sclose);
    close_if_valid(dataset, H5Dclose);
    return ok;
}

bool load_h5ad_x_as_csr(const std::string &path, csr_matrix_view *out, std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t x = (hid_t) -1;
    hsize_t obs_rows = 0u, var_cols = 0u;
    bool ok = false;

    if (out == nullptr) return false;
    *out = csr_matrix_view{};
    file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        if (error) *error = "failed to open AnnData fixture: " + path;
        return false;
    }
    x = H5Gopen2(file, "/X", H5P_DEFAULT);
    if (x < 0) {
        if (error) *error = "AnnData fixture does not contain sparse /X group";
        close_if_valid(file, H5Fclose);
        return false;
    }

    ok = dataset_extent_1d(file, "/obs", &obs_rows, error)
        && dataset_extent_1d(file, "/var", &var_cols, error)
        && read_numeric_vector(x, "indptr", H5T_NATIVE_INT64, &out->indptr, error)
        && read_numeric_vector(x, "indices", H5T_NATIVE_INT64, &out->indices, error)
        && read_numeric_vector(x, "data", H5T_NATIVE_FLOAT, &out->data, error);
    close_if_valid(x, H5Gclose);
    close_if_valid(file, H5Fclose);
    if (!ok) return false;

    out->rows = (std::uint64_t) obs_rows;
    out->cols = (std::uint64_t) var_cols;
    if (out->indptr.size() != (std::size_t) out->rows + 1u) {
        if (error) *error = "AnnData /X/indptr length does not match obs rows";
        return false;
    }
    if (out->indices.size() != out->data.size()) {
        if (error) *error = "AnnData /X indices/data lengths differ";
        return false;
    }
    if (out->indptr.back() != (std::int64_t) out->data.size()) {
        if (error) *error = "AnnData /X/indptr terminal value does not match nnz";
        return false;
    }
    return true;
}

bool load_csh5_as_csr(const std::string &path, csr_matrix_view *out, std::string *error) {
    cse::csr_matrix_export csr;
    if (out == nullptr) return false;
    *out = csr_matrix_view{};
    if (!cse::load_dataset_as_csr(path.c_str(), &csr, error)) return false;
    out->rows = csr.rows;
    out->cols = csr.cols;
    out->indptr = std::move(csr.indptr);
    out->indices = std::move(csr.indices);
    out->data = std::move(csr.data);
    return true;
}

bool validate_csr_structure(const csr_matrix_view &m, const char *label, std::string *error) {
    if (m.indptr.size() != (std::size_t) m.rows + 1u) {
        if (error) *error = std::string(label) + " indptr length does not match row count";
        return false;
    }
    if (m.indices.size() != m.data.size()) {
        if (error) *error = std::string(label) + " indices/data lengths differ";
        return false;
    }
    if (m.indptr.empty() || m.indptr.front() != 0 || m.indptr.back() != (std::int64_t) m.data.size()) {
        if (error) *error = std::string(label) + " indptr endpoints are invalid";
        return false;
    }
    for (std::size_t i = 1u; i < m.indptr.size(); ++i) {
        if (m.indptr[i] < m.indptr[i - 1u]) {
            if (error) *error = std::string(label) + " indptr is not monotonic";
            return false;
        }
    }
    for (std::size_t i = 0u; i < m.indices.size(); ++i) {
        if (m.indices[i] < 0 || (std::uint64_t) m.indices[i] >= m.cols) {
            if (error) *error = std::string(label) + " column index out of range";
            return false;
        }
    }
    return true;
}

bool close_enough(double ref, double test, double abs_tolerance, double rel_tolerance) {
    const double abs_delta = std::fabs(test - ref);
    const double scale = std::max(std::fabs(ref), std::fabs(test));
    return abs_delta <= abs_tolerance || abs_delta <= rel_tolerance * scale;
}

double relative_delta(double ref, double test) {
    const double denom = std::max(std::fabs(ref), std::fabs(test));
    return denom == 0.0 ? 0.0 : std::fabs(test - ref) / denom;
}

std::vector<double> row_sums(const csr_matrix_view &m) {
    std::vector<double> sums((std::size_t) m.rows, 0.0);
    for (std::uint64_t row = 0u; row < m.rows; ++row) {
        const std::int64_t begin = m.indptr[(std::size_t) row];
        const std::int64_t end = m.indptr[(std::size_t) row + 1u];
        for (std::int64_t idx = begin; idx < end; ++idx) sums[(std::size_t) row] += (double) m.data[(std::size_t) idx];
    }
    return sums;
}

std::vector<double> column_sums(const csr_matrix_view &m) {
    std::vector<double> sums((std::size_t) m.cols, 0.0);
    for (std::uint64_t row = 0u; row < m.rows; ++row) {
        const std::int64_t begin = m.indptr[(std::size_t) row];
        const std::int64_t end = m.indptr[(std::size_t) row + 1u];
        for (std::int64_t idx = begin; idx < end; ++idx) {
            const std::int64_t col = m.indices[(std::size_t) idx];
            if (col >= 0 && (std::uint64_t) col < m.cols) sums[(std::size_t) col] += (double) m.data[(std::size_t) idx];
        }
    }
    return sums;
}

axis_summary summarize_axis(const std::vector<double> &ref,
                            const std::vector<double> &test,
                            double abs_tolerance,
                            double rel_tolerance) {
    axis_summary out;
    out.compared = (std::uint64_t) std::min(ref.size(), test.size());
    for (std::size_t i = 0u; i < (std::size_t) out.compared; ++i) {
        const double abs_delta = std::fabs(test[i] - ref[i]);
        const double rel_delta = relative_delta(ref[i], test[i]);
        out.total_abs += abs_delta;
        if (abs_delta > out.max_abs) {
            out.max_abs = abs_delta;
            out.max_rel = rel_delta;
            out.worst_index = (std::uint64_t) i;
            out.ref_value = ref[i];
            out.test_value = test[i];
        }
        if (!close_enough(ref[i], test[i], abs_tolerance, rel_tolerance)) ++out.mismatched;
    }
    out.mean_abs = out.compared == 0u ? 0.0 : out.total_abs / (double) out.compared;
    return out;
}

value_summary summarize_values(const csr_matrix_view &ref,
                               const csr_matrix_view &test,
                               double abs_tolerance,
                               double rel_tolerance) {
    value_summary out;
    const std::uint64_t rows = std::min(ref.rows, test.rows);
    for (std::uint64_t row = 0u; row < rows; ++row) {
        const std::int64_t ref_begin = ref.indptr[(std::size_t) row];
        const std::int64_t ref_end = ref.indptr[(std::size_t) row + 1u];
        const std::int64_t test_begin = test.indptr[(std::size_t) row];
        const std::int64_t test_end = test.indptr[(std::size_t) row + 1u];
        const std::int64_t count = std::min(ref_end - ref_begin, test_end - test_begin);
        for (std::int64_t offset = 0; offset < count; ++offset) {
            const std::size_t ref_idx = (std::size_t) (ref_begin + offset);
            const std::size_t test_idx = (std::size_t) (test_begin + offset);
            const float ref_value = ref.data[ref_idx];
            const float test_value = test.data[test_idx];
            const double abs_delta = std::fabs((double) test_value - (double) ref_value);
            const double rel_delta = relative_delta((double) ref_value, (double) test_value);
            ++out.compared;
            out.total_abs += abs_delta;
            if (ref.indices[ref_idx] != test.indices[test_idx]) ++out.mismatched_columns;
            if (abs_delta > out.max_abs) {
                out.max_abs = abs_delta;
                out.max_rel = rel_delta;
                out.worst_row = row;
                out.worst_col = ref.indices[ref_idx];
                out.ref_value = ref_value;
                out.test_value = test_value;
            }
            if (!close_enough((double) ref_value, (double) test_value, abs_tolerance, rel_tolerance)) ++out.mismatched_values;
        }
    }
    out.mean_abs = out.compared == 0u ? 0.0 : out.total_abs / (double) out.compared;
    return out;
}

std::uint64_t count_indptr_mismatches(const csr_matrix_view &ref, const csr_matrix_view &test) {
    const std::size_t count = std::min(ref.indptr.size(), test.indptr.size());
    std::uint64_t mismatches = ref.indptr.size() == test.indptr.size() ? 0u : 1u;
    for (std::size_t i = 0u; i < count; ++i) {
        if (ref.indptr[i] != test.indptr[i]) ++mismatches;
    }
    return mismatches;
}

validation_report compare(const options &opts, const csr_matrix_view &ref, const csr_matrix_view &test) {
    validation_report report;
    report.h5ad_path = opts.h5ad_path;
    report.csh5_path = opts.csh5_path;
    report.abs_tolerance = opts.abs_tolerance;
    report.rel_tolerance = opts.rel_tolerance;
    report.ref_rows = ref.rows;
    report.ref_cols = ref.cols;
    report.ref_nnz = (std::uint64_t) ref.data.size();
    report.test_rows = test.rows;
    report.test_cols = test.cols;
    report.test_nnz = (std::uint64_t) test.data.size();
    report.indptr_mismatches = count_indptr_mismatches(ref, test);
    report.values = summarize_values(ref, test, opts.abs_tolerance, opts.rel_tolerance);
    report.rows = summarize_axis(row_sums(ref), row_sums(test), opts.abs_tolerance, opts.rel_tolerance);
    report.cols = summarize_axis(column_sums(ref), column_sums(test), opts.abs_tolerance, opts.rel_tolerance);
    return report;
}

void print_axis(const char *label, const axis_summary &summary) {
    std::printf("%s compared: %llu\n", label, (unsigned long long) summary.compared);
    std::printf("%s mismatches: %llu\n", label, (unsigned long long) summary.mismatched);
    std::printf("%s max_abs: %.9g\n", label, summary.max_abs);
    std::printf("%s max_rel: %.9g\n", label, summary.max_rel);
    std::printf("%s mean_abs: %.9g\n", label, summary.mean_abs);
    std::printf("%s total_abs: %.9g\n", label, summary.total_abs);
    if (summary.max_abs == 0.0) {
        std::printf("%s worst: none\n", label);
    } else {
        std::printf("%s worst: index=%llu ref=%.9g csh5=%.9g\n",
                    label,
                    (unsigned long long) summary.worst_index,
                    summary.ref_value,
                    summary.test_value);
    }
}

void print_report(const validation_report &report) {
    std::printf("PBMC3K H5AD to CSH5 ingest validation\n");
    std::printf("h5ad: %s\n", report.h5ad_path.c_str());
    std::printf("csh5: %s\n", report.csh5_path.c_str());
    std::printf("tolerance: abs<=%.9g or rel<=%.9g\n", report.abs_tolerance, report.rel_tolerance);
    std::printf("shape h5ad: %llu x %llu nnz=%llu\n",
                (unsigned long long) report.ref_rows,
                (unsigned long long) report.ref_cols,
                (unsigned long long) report.ref_nnz);
    std::printf("shape csh5: %llu x %llu nnz=%llu\n",
                (unsigned long long) report.test_rows,
                (unsigned long long) report.test_cols,
                (unsigned long long) report.test_nnz);
    std::printf("indptr mismatches: %llu\n", (unsigned long long) report.indptr_mismatches);
    std::printf("value compared: %llu\n", (unsigned long long) report.values.compared);
    std::printf("value column mismatches: %llu\n", (unsigned long long) report.values.mismatched_columns);
    std::printf("value mismatches: %llu\n", (unsigned long long) report.values.mismatched_values);
    std::printf("value max_abs: %.9g\n", report.values.max_abs);
    std::printf("value max_rel: %.9g\n", report.values.max_rel);
    std::printf("value mean_abs: %.9g\n", report.values.mean_abs);
    std::printf("value total_abs: %.9g\n", report.values.total_abs);
    if (report.values.max_abs == 0.0) {
        std::printf("value worst: none\n");
    } else {
        std::printf("value worst: row=%llu col=%lld ref=%.9g csh5=%.9g\n",
                    (unsigned long long) report.values.worst_row,
                    (long long) report.values.worst_col,
                    (double) report.values.ref_value,
                    (double) report.values.test_value);
    }
    print_axis("row_sum", report.rows);
    print_axis("col_sum", report.cols);
    std::printf("status: %s\n", report.passed() ? "PASS" : "FAIL");
}

bool parse_double_arg(const char *text, double *out) {
    char *end = nullptr;
    if (text == nullptr || out == nullptr) return false;
    const double value = std::strtod(text, &end);
    if (end == text || (end != nullptr && *end != '\0')) return false;
    *out = value;
    return true;
}

bool parse_args(int argc, char **argv, options *out) {
    if (out == nullptr) return false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--h5ad") == 0 && i + 1 < argc) {
            out->h5ad_path = argv[++i];
        } else if (std::strcmp(argv[i], "--csh5") == 0 && i + 1 < argc) {
            out->csh5_path = argv[++i];
        } else if (std::strcmp(argv[i], "--abs-tol") == 0 && i + 1 < argc) {
            if (!parse_double_arg(argv[++i], &out->abs_tolerance)) return false;
        } else if (std::strcmp(argv[i], "--rel-tol") == 0 && i + 1 < argc) {
            if (!parse_double_arg(argv[++i], &out->rel_tolerance)) return false;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--h5ad PATH] [--csh5 PATH] [--abs-tol VALUE] [--rel-tol VALUE]\n", argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    options opts;
    csr_matrix_view h5ad, csh5;
    std::string error;

    if (!parse_args(argc, argv, &opts)) return 2;
    if (!load_h5ad_x_as_csr(opts.h5ad_path, &h5ad, &error)) {
        std::fprintf(stderr, "failed to load h5ad fixture: %s\n", error.c_str());
        return 2;
    }
    if (!load_csh5_as_csr(opts.csh5_path, &csh5, &error)) {
        std::fprintf(stderr, "failed to load csh5 fixture: %s\n", error.c_str());
        return 2;
    }
    if (!validate_csr_structure(h5ad, "h5ad", &error) || !validate_csr_structure(csh5, "csh5", &error)) {
        std::fprintf(stderr, "invalid CSR input: %s\n", error.c_str());
        return 2;
    }

    const validation_report report = compare(opts, h5ad, csh5);
    print_report(report);
    return report.passed() ? 0 : 1;
}
