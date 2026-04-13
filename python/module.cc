#include "../export/h5ad_writer.hh"
#include "../export/series_export.hh"
#include "../src/formats/compressed.cuh"
#include "../src/formats/blocked_ell.cuh"
#include "../src/sharded/sharded.cuh"
#include "../src/sharded/sharded_host.cuh"
#include "../src/sharded/shard_paths.cuh"
#include "../src/sharded/series_h5.cuh"
#include "../src/sharded/disk.cuh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace cse = cellshard::exporting;

namespace {

inline void warn_cpu_materialize_once(const char *label) {
#if !CELLSHARD_ENABLE_CUDA
    static bool warned = false;
    if (!warned) {
        std::fprintf(stderr,
                     "Warning: CellShard was built without CUDA; %s is running in a slow host-only inspect/materialize mode.\n",
                     label != nullptr ? label : "this operation");
        warned = true;
    }
#else
    (void) label;
#endif
}

template<typename T>
py::array_t<T> copy_1d_array(const std::vector<T> &values) {
    py::array_t<T> out((py::ssize_t) values.size());
    if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(T));
    return out;
}

py::array_t<float> copy_half_values(const __half *values, std::size_t count) {
    py::array_t<float> out((py::ssize_t) count);
    float *dst = out.mutable_data();
    for (std::size_t i = 0; i < count; ++i) dst[i] = __half2float(values[i]);
    return out;
}

struct series_file_handle {
    std::string path;

    cse::series_summary summary() const {
        cse::series_summary out;
        std::string error;
        if (!cse::load_series_summary(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    cse::csr_matrix_export materialize_csr() const {
        cse::csr_matrix_export out;
        std::string error;
        if (!cse::load_series_as_csr(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    py::dict materialize_partition(unsigned long partition_id) const {
        cse::series_summary info;
        std::string error;
        if (!cse::load_series_summary(path.c_str(), &info, &error)) throw std::runtime_error(error);
        if (partition_id >= info.partitions.size()) throw std::out_of_range("partition_id is out of range");
        warn_cpu_materialize_once("partition materialization");

        if (info.matrix_format == "compressed") {
            cellshard::sharded<cellshard::sparse::compressed> view;
            cellshard::shard_storage storage;
            py::dict out;
            cellshard::init(&view);
            cellshard::init(&storage);
            if (!cellshard::load_header(path.c_str(), &view, &storage)) throw std::runtime_error("failed to load compressed series header");
            if (!cellshard::fetch_partition(&view, &storage, partition_id)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                throw std::runtime_error("failed to fetch compressed partition");
            }
            const cellshard::sparse::compressed *partition = view.parts[partition_id];
            std::vector<std::uint32_t> major_ptr((std::size_t) partition->rows + 1u, 0u);
            std::vector<std::uint32_t> minor_idx((std::size_t) partition->nnz, 0u);
            for (std::uint32_t i = 0u; i <= partition->rows; ++i) major_ptr[i] = partition->majorPtr[i];
            for (std::uint32_t i = 0u; i < partition->nnz; ++i) minor_idx[i] = partition->minorIdx[i];
            out[py::str("format")] = py::str("compressed");
            out[py::str("rows")] = py::int_(partition->rows);
            out[py::str("cols")] = py::int_(partition->cols);
            out[py::str("nnz")] = py::int_(partition->nnz);
            out[py::str("axis")] = py::int_(partition->axis);
            out[py::str("major_ptr")] = copy_1d_array(major_ptr);
            out[py::str("minor_idx")] = copy_1d_array(minor_idx);
            out[py::str("values")] = copy_half_values(partition->val, partition->nnz);
            cellshard::clear(&storage);
            cellshard::clear(&view);
            return out;
        }

        if (info.matrix_format == "blocked_ell") {
            cellshard::sharded<cellshard::sparse::blocked_ell> view;
            cellshard::shard_storage storage;
            py::dict out;
            cellshard::init(&view);
            cellshard::init(&storage);
            if (!cellshard::load_header(path.c_str(), &view, &storage)) throw std::runtime_error("failed to load blocked-ELL series header");
            if (!cellshard::fetch_partition(&view, &storage, partition_id)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                throw std::runtime_error("failed to fetch blocked-ELL partition");
            }
            const cellshard::sparse::blocked_ell *partition = view.parts[partition_id];
            const std::size_t row_blocks = cellshard::sparse::row_block_count(partition);
            const std::size_t width = cellshard::sparse::ell_width_blocks(partition);
            py::array_t<std::uint32_t> block_idx({(py::ssize_t) row_blocks, (py::ssize_t) width});
            py::array_t<float> values({(py::ssize_t) partition->rows, (py::ssize_t) partition->ell_cols});
            auto *block_dst = block_idx.mutable_data();
            auto *value_dst = values.mutable_data();
            for (std::size_t i = 0; i < row_blocks * width; ++i) block_dst[i] = partition->blockColIdx[i];
            for (std::size_t i = 0; i < (std::size_t) partition->rows * (std::size_t) partition->ell_cols; ++i) {
                value_dst[i] = __half2float(partition->val[i]);
            }
            out[py::str("format")] = py::str("blocked_ell");
            out[py::str("rows")] = py::int_(partition->rows);
            out[py::str("cols")] = py::int_(partition->cols);
            out[py::str("nnz")] = py::int_(partition->nnz);
            out[py::str("block_size")] = py::int_(partition->block_size);
            out[py::str("ell_cols")] = py::int_(partition->ell_cols);
            out[py::str("block_col_idx")] = block_idx;
            out[py::str("values")] = values;
            cellshard::clear(&storage);
            cellshard::clear(&view);
            return out;
        }

        throw std::runtime_error("unsupported matrix_format in series file");
    }

    void write_h5ad(const std::string &output_path) const {
        std::string error;
        if (!cse::write_series_file_to_h5ad_with_python(path.c_str(), output_path.c_str(), &error)) {
            throw std::runtime_error(error);
        }
    }
};

} // namespace

PYBIND11_MODULE(_cellshard, m) {
    m.doc() = "Optional CellShard Python bindings for metadata inspection and H5AD export.";

    py::class_<cse::series_dataset_summary>(m, "SeriesDatasetSummary")
        .def_readonly("dataset_id", &cse::series_dataset_summary::dataset_id)
        .def_readonly("matrix_path", &cse::series_dataset_summary::matrix_path)
        .def_readonly("feature_path", &cse::series_dataset_summary::feature_path)
        .def_readonly("barcode_path", &cse::series_dataset_summary::barcode_path)
        .def_readonly("metadata_path", &cse::series_dataset_summary::metadata_path)
        .def_readonly("format", &cse::series_dataset_summary::format)
        .def_readonly("row_begin", &cse::series_dataset_summary::row_begin)
        .def_readonly("row_end", &cse::series_dataset_summary::row_end)
        .def_readonly("rows", &cse::series_dataset_summary::rows)
        .def_readonly("cols", &cse::series_dataset_summary::cols)
        .def_readonly("nnz", &cse::series_dataset_summary::nnz);

    py::class_<cse::series_codec_summary>(m, "SeriesCodecSummary")
        .def_readonly("codec_id", &cse::series_codec_summary::codec_id)
        .def_readonly("family", &cse::series_codec_summary::family)
        .def_readonly("value_code", &cse::series_codec_summary::value_code)
        .def_readonly("scale_value_code", &cse::series_codec_summary::scale_value_code)
        .def_readonly("bits", &cse::series_codec_summary::bits)
        .def_readonly("flags", &cse::series_codec_summary::flags);

    py::class_<cse::series_partition_summary>(m, "SeriesPartitionSummary")
        .def_readonly("partition_id", &cse::series_partition_summary::partition_id)
        .def_readonly("row_begin", &cse::series_partition_summary::row_begin)
        .def_readonly("row_end", &cse::series_partition_summary::row_end)
        .def_readonly("rows", &cse::series_partition_summary::rows)
        .def_readonly("nnz", &cse::series_partition_summary::nnz)
        .def_readonly("aux", &cse::series_partition_summary::aux)
        .def_readonly("dataset_id", &cse::series_partition_summary::dataset_id)
        .def_readonly("axis", &cse::series_partition_summary::axis)
        .def_readonly("codec_id", &cse::series_partition_summary::codec_id);

    py::class_<cse::series_shard_summary>(m, "SeriesShardSummary")
        .def_readonly("shard_id", &cse::series_shard_summary::shard_id)
        .def_readonly("partition_begin", &cse::series_shard_summary::partition_begin)
        .def_readonly("partition_end", &cse::series_shard_summary::partition_end)
        .def_readonly("row_begin", &cse::series_shard_summary::row_begin)
        .def_readonly("row_end", &cse::series_shard_summary::row_end);

    py::class_<cse::series_summary>(m, "SeriesSummary")
        .def_readonly("path", &cse::series_summary::path)
        .def_readonly("matrix_format", &cse::series_summary::matrix_format)
        .def_readonly("payload_layout", &cse::series_summary::payload_layout)
        .def_readonly("rows", &cse::series_summary::rows)
        .def_readonly("cols", &cse::series_summary::cols)
        .def_readonly("nnz", &cse::series_summary::nnz)
        .def_readonly("num_partitions", &cse::series_summary::num_partitions)
        .def_readonly("num_shards", &cse::series_summary::num_shards)
        .def_readonly("num_datasets", &cse::series_summary::num_datasets)
        .def_readonly("datasets", &cse::series_summary::datasets)
        .def_readonly("partitions", &cse::series_summary::partitions)
        .def_readonly("shards", &cse::series_summary::shards)
        .def_readonly("codecs", &cse::series_summary::codecs)
        .def_readonly("obs_names", &cse::series_summary::obs_names)
        .def_readonly("var_ids", &cse::series_summary::var_ids)
        .def_readonly("var_names", &cse::series_summary::var_names)
        .def_readonly("var_types", &cse::series_summary::var_types);

    py::class_<cse::csr_matrix_export>(m, "CsrMatrixExport")
        .def_readonly("rows", &cse::csr_matrix_export::rows)
        .def_readonly("cols", &cse::csr_matrix_export::cols)
        .def("indptr_array", [](const cse::csr_matrix_export &self) { return copy_1d_array(self.indptr); })
        .def("indices_array", [](const cse::csr_matrix_export &self) { return copy_1d_array(self.indices); })
        .def("data_array", [](const cse::csr_matrix_export &self) { return copy_1d_array(self.data); });

    py::class_<series_file_handle>(m, "SeriesFile")
        .def(py::init([](const std::string &path) { return series_file_handle{path}; }))
        .def_readonly("path", &series_file_handle::path)
        .def("summary", &series_file_handle::summary)
        .def("materialize_csr", &series_file_handle::materialize_csr)
        .def("materialize_partition", &series_file_handle::materialize_partition)
        .def("write_h5ad", &series_file_handle::write_h5ad);

    m.def("open_series", [](const std::string &path) { return series_file_handle{path}; });
    m.def("load_series_summary", [](const std::string &path) {
        cse::series_summary out;
        std::string error;
        if (!cse::load_series_summary(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    });
    m.def("load_series_as_csr", [](const std::string &path) {
        cse::csr_matrix_export out;
        std::string error;
        if (!cse::load_series_as_csr(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    });
    m.def("write_h5ad", [](const std::string &series_path, const std::string &output_path) {
        std::string error;
        if (!cse::write_series_file_to_h5ad_with_python(series_path.c_str(), output_path.c_str(), &error)) {
            throw std::runtime_error(error);
        }
    });
}
