#include "h5ad_writer.hh"

#include "../src/sharded/series_h5.cuh"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace cellshard::exporting {

namespace {

inline void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) *error = message;
}

template<typename T>
py::array_t<T> copy_1d_array(const std::vector<T> &values) {
    py::array_t<T> out((py::ssize_t) values.size());
    if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(T));
    return out;
}

inline std::vector<std::string> default_names(const char *prefix, std::size_t count) {
    std::vector<std::string> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) out.push_back(std::string(prefix) + std::to_string(i));
    return out;
}

inline py::object build_obs_dataframe(const anndata_export &input, py::module_ &pandas) {
    py::dict columns;
    const std::vector<std::string> obs_index = input.summary.obs_names.empty()
        ? default_names("cell_", (std::size_t) input.summary.rows)
        : input.summary.obs_names;

    for (const observation_metadata_column &column : input.obs_columns) {
        if (column.type == cellshard::series_observation_metadata_type_text) {
            columns[py::str(column.name)] = py::cast(column.text_values);
        } else if (column.type == cellshard::series_observation_metadata_type_float32) {
            columns[py::str(column.name)] = copy_1d_array(column.float32_values);
        } else if (column.type == cellshard::series_observation_metadata_type_uint8) {
            columns[py::str(column.name)] = copy_1d_array(column.uint8_values);
        }
    }

    return pandas.attr("DataFrame")(columns, py::arg("index") = py::cast(obs_index));
}

inline py::object build_var_dataframe(const anndata_export &input, py::module_ &pandas) {
    py::dict columns;
    std::vector<std::string> var_index = input.summary.var_ids;
    if (var_index.empty()) var_index = input.summary.var_names;
    if (var_index.empty()) var_index = default_names("feature_", (std::size_t) input.summary.cols);

    if (!input.summary.var_names.empty()) columns[py::str("feature_name")] = py::cast(input.summary.var_names);
    if (!input.summary.var_types.empty()) columns[py::str("feature_type")] = py::cast(input.summary.var_types);
    if (!input.summary.var_ids.empty() && input.summary.var_ids != var_index) {
        columns[py::str("feature_id")] = py::cast(input.summary.var_ids);
    }

    return pandas.attr("DataFrame")(columns, py::arg("index") = py::cast(var_index));
}

inline py::dict build_uns_dict(const anndata_export &input) {
    py::dict cellshard_dict;
    py::list datasets;
    py::list codecs;

    cellshard_dict[py::str("matrix_format")] = py::str(input.summary.matrix_format);
    cellshard_dict[py::str("payload_layout")] = py::str(input.summary.payload_layout);
    cellshard_dict[py::str("rows")] = py::int_(input.summary.rows);
    cellshard_dict[py::str("cols")] = py::int_(input.summary.cols);
    cellshard_dict[py::str("nnz")] = py::int_(input.summary.nnz);
    cellshard_dict[py::str("num_partitions")] = py::int_(input.summary.num_partitions);
    cellshard_dict[py::str("num_shards")] = py::int_(input.summary.num_shards);
    cellshard_dict[py::str("num_datasets")] = py::int_(input.summary.num_datasets);

    for (const series_dataset_summary &dataset : input.summary.datasets) {
        py::dict entry;
        entry[py::str("dataset_id")] = py::str(dataset.dataset_id);
        entry[py::str("matrix_path")] = py::str(dataset.matrix_path);
        entry[py::str("feature_path")] = py::str(dataset.feature_path);
        entry[py::str("barcode_path")] = py::str(dataset.barcode_path);
        entry[py::str("metadata_path")] = py::str(dataset.metadata_path);
        entry[py::str("format")] = py::int_(dataset.format);
        entry[py::str("row_begin")] = py::int_(dataset.row_begin);
        entry[py::str("row_end")] = py::int_(dataset.row_end);
        entry[py::str("rows")] = py::int_(dataset.rows);
        entry[py::str("cols")] = py::int_(dataset.cols);
        entry[py::str("nnz")] = py::int_(dataset.nnz);
        datasets.append(entry);
    }

    for (const series_codec_summary &codec : input.summary.codecs) {
        py::dict entry;
        entry[py::str("codec_id")] = py::int_(codec.codec_id);
        entry[py::str("family")] = py::int_(codec.family);
        entry[py::str("value_code")] = py::int_(codec.value_code);
        entry[py::str("scale_value_code")] = py::int_(codec.scale_value_code);
        entry[py::str("bits")] = py::int_(codec.bits);
        entry[py::str("flags")] = py::int_(codec.flags);
        codecs.append(entry);
    }

    cellshard_dict[py::str("datasets")] = datasets;
    cellshard_dict[py::str("codecs")] = codecs;

    py::dict uns;
    uns[py::str("cellshard")] = cellshard_dict;
    return uns;
}

inline bool write_h5ad_impl(const anndata_export &input, const char *path, std::string *error) {
    std::unique_ptr<py::scoped_interpreter> owned_interpreter;

    try {
        if (!Py_IsInitialized()) owned_interpreter = std::make_unique<py::scoped_interpreter>();
        py::gil_scoped_acquire acquire;
        py::module_ scipy_sparse = py::module_::import("scipy.sparse");
        py::module_ pandas = py::module_::import("pandas");
        py::module_ anndata = py::module_::import("anndata");

        py::array_t<float> data = copy_1d_array(input.x.data);
        py::array_t<std::int64_t> indices = copy_1d_array(input.x.indices);
        py::array_t<std::int64_t> indptr = copy_1d_array(input.x.indptr);

        py::object matrix = scipy_sparse.attr("csr_matrix")(
            py::make_tuple(
                py::make_tuple(data, indices, indptr),
                py::make_tuple((py::ssize_t) input.x.rows, (py::ssize_t) input.x.cols)
            )
        );

        py::object obs = build_obs_dataframe(input, pandas);
        py::object var = build_var_dataframe(input, pandas);
        py::dict uns = build_uns_dict(input);
        py::object adata = anndata.attr("AnnData")(
            py::arg("X") = matrix,
            py::arg("obs") = obs,
            py::arg("var") = var,
            py::arg("uns") = uns
        );
        adata.attr("write_h5ad")(py::str(path));
        return true;
    } catch (const py::error_already_set &exc) {
        set_error(error, exc.what());
        return false;
    } catch (const std::exception &exc) {
        set_error(error, exc.what());
        return false;
    }
}

} // namespace

bool write_h5ad_with_python(const anndata_export &input, const char *path, std::string *error) {
    if (path == nullptr || *path == '\0') {
        set_error(error, "h5ad output path is empty");
        return false;
    }
    return write_h5ad_impl(input, path, error);
}

bool write_series_file_to_h5ad_with_python(const char *series_path, const char *path, std::string *error) {
    anndata_export snapshot;
    if (!load_series_for_anndata(series_path, &snapshot, error)) return false;
    return write_h5ad_with_python(snapshot, path, error);
}

} // namespace cellshard::exporting
