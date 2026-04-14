#include "../export/h5ad_writer.hh"
#include "../export/dataset_export.hh"
#include "../src/formats/compressed.cuh"
#include "../src/formats/blocked_ell.cuh"
#include "../src/sharded/sharded.cuh"
#include "../src/sharded/sharded_host.cuh"
#include "../src/sharded/shard_paths.cuh"
#include "../src/disk/csh5.cuh"
#include "../src/sharded/disk.cuh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <cstring>
#include <memory>
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

py::object build_scipy_csr(const cse::csr_matrix_export &self) {
    py::module_ scipy_sparse = py::module_::import("scipy.sparse");
    py::array_t<float> data = copy_1d_array(self.data);
    py::array_t<std::int64_t> indices = copy_1d_array(self.indices);
    py::array_t<std::int64_t> indptr = copy_1d_array(self.indptr);
    return scipy_sparse.attr("csr_matrix")(
        py::make_tuple(
            py::make_tuple(data, indices, indptr),
            py::make_tuple((py::ssize_t) self.rows, (py::ssize_t) self.cols)
        )
    );
}

py::object build_torch_sparse_csr(const cse::csr_matrix_export &self) {
    py::module_ torch = py::module_::import("torch");
    py::object crow = torch.attr("from_numpy")(copy_1d_array(self.indptr)).attr("to")(torch.attr("int64")).attr("clone")();
    py::object col = torch.attr("from_numpy")(copy_1d_array(self.indices)).attr("to")(torch.attr("int64")).attr("clone")();
    py::object values = torch.attr("from_numpy")(copy_1d_array(self.data)).attr("to")(torch.attr("float32")).attr("clone")();
    return torch.attr("sparse_csr_tensor")(
        crow,
        col,
        values,
        py::arg("size") = py::make_tuple((py::ssize_t) self.rows, (py::ssize_t) self.cols)
    );
}

std::vector<std::uint8_t> bytes_to_vector(const py::bytes &bytes) {
    const std::string payload = bytes;
    return std::vector<std::uint8_t>(payload.begin(), payload.end());
}

py::bytes serialize_snapshot_bytes(const cse::global_metadata_snapshot &snapshot) {
    std::vector<std::uint8_t> payload;
    std::string error;
    if (!cse::serialize_global_metadata_snapshot(snapshot, &payload, &error)) throw std::runtime_error(error);
    return py::bytes(reinterpret_cast<const char *>(payload.data()), payload.size());
}

cse::global_metadata_snapshot deserialize_snapshot_bytes(const py::bytes &bytes) {
    const std::vector<std::uint8_t> payload = bytes_to_vector(bytes);
    cse::global_metadata_snapshot snapshot;
    std::string error;
    if (!cse::deserialize_global_metadata_snapshot(payload.data(), payload.size(), &snapshot, &error)) {
        throw std::runtime_error(error);
    }
    return snapshot;
}

inline std::uint64_t parse_row_index_item(py::handle item) {
    const std::int64_t value = py::cast<std::int64_t>(item);
    if (value < 0) throw py::value_error("row indices must be non-negative");
    return (std::uint64_t) value;
}

std::vector<std::uint64_t> normalize_row_indices(py::handle handle) {
    py::object obj = py::reinterpret_borrow<py::object>(handle);

    if (py::hasattr(obj, "detach") && py::hasattr(obj, "cpu") && py::hasattr(obj, "tolist")) {
        py::object values = obj.attr("detach")().attr("cpu")().attr("reshape")(py::int_(-1)).attr("tolist")();
        return normalize_row_indices(values);
    }

    if (py::isinstance<py::array>(obj)) {
        py::object values = obj.attr("reshape")(py::int_(-1)).attr("tolist")();
        return normalize_row_indices(values);
    }

    if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj) && !py::isinstance<py::bytes>(obj)) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
        std::vector<std::uint64_t> out;
        out.reserve((std::size_t) seq.size());
        for (py::handle item : seq) out.push_back(parse_row_index_item(item));
        return out;
    }

    throw py::type_error("row indices must be a Python sequence, NumPy array, or torch tensor");
}

struct dataset_file_handle {
    std::string path;

    cse::dataset_summary summary() const {
        cse::dataset_summary out;
        std::string error;
        if (!cse::load_dataset_summary(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    cse::global_metadata_snapshot snapshot() const {
        cse::global_metadata_snapshot out;
        std::string error;
        if (!cse::load_dataset_global_metadata_snapshot(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    cse::csr_matrix_export materialize_csr() const {
        cse::csr_matrix_export out;
        std::string error;
        if (!cse::load_dataset_as_csr(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    cse::csr_matrix_export materialize_rows_csr(py::handle row_indices) const {
        const std::vector<std::uint64_t> rows = normalize_row_indices(row_indices);
        cse::csr_matrix_export out;
        std::string error;
        if (!cse::load_dataset_rows_as_csr(path.c_str(), rows.data(), rows.size(), &out, &error)) {
            throw std::runtime_error(error);
        }
        return out;
    }

    py::dict materialize_partition(unsigned long partition_id) const {
        cse::dataset_summary info;
        std::string error;
        if (!cse::load_dataset_summary(path.c_str(), &info, &error)) throw std::runtime_error(error);
        if (partition_id >= info.partitions.size()) throw std::out_of_range("partition_id is out of range");
        warn_cpu_materialize_once("partition materialization");

        if (info.matrix_format == "compressed") {
            cellshard::sharded<cellshard::sparse::compressed> view;
            cellshard::shard_storage storage;
            py::dict out;
            cellshard::init(&view);
            cellshard::init(&storage);
            if (!cellshard::load_header(path.c_str(), &view, &storage)) throw std::runtime_error("failed to load compressed dataset header");
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
            if (!cellshard::load_header(path.c_str(), &view, &storage)) throw std::runtime_error("failed to load blocked-ELL dataset header");
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

        throw std::runtime_error("unsupported matrix_format in dataset file");
    }

    void write_h5ad(const std::string &output_path) const {
        std::string error;
        if (!cse::write_dataset_file_to_h5ad_with_python(path.c_str(), output_path.c_str(), &error)) {
            throw std::runtime_error(error);
        }
    }
};

struct dataset_owner_handle;

struct dataset_client_handle {
    std::shared_ptr<dataset_owner_handle> owner;
    cse::global_metadata_snapshot snapshot;
    cse::client_snapshot_ref request_ref;

    dataset_client_handle(const py::bytes &snapshot_bytes, std::shared_ptr<dataset_owner_handle> owner_in);

    py::bytes serialized_snapshot() const {
        return serialize_snapshot_bytes(snapshot);
    }
};

struct dataset_owner_handle {
    std::string path;

    cse::global_metadata_snapshot snapshot() const {
        cse::global_metadata_snapshot out;
        std::string error;
        if (!cse::load_dataset_global_metadata_snapshot(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    py::bytes serialized_snapshot() const {
        return serialize_snapshot_bytes(snapshot());
    }

    cse::client_snapshot_ref current_request_ref() const {
        return cse::make_client_snapshot_ref(snapshot());
    }

    void validate_request(const cse::client_snapshot_ref &request) const {
        std::string error;
        const cse::global_metadata_snapshot current = snapshot();
        if (!cse::validate_client_snapshot_ref(current, request, &error)) throw std::runtime_error(error);
    }

    cse::csr_matrix_export materialize_csr(const cse::client_snapshot_ref &request) const {
        cse::csr_matrix_export out;
        std::string error;
        const cse::global_metadata_snapshot current = snapshot();
        if (!cse::validate_client_snapshot_ref(current, request, &error)) throw std::runtime_error(error);
        if (!cse::load_dataset_as_csr(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    cse::csr_matrix_export materialize_rows_csr(const cse::client_snapshot_ref &request,
                                                const std::vector<std::uint64_t> &row_indices) const {
        cse::csr_matrix_export out;
        std::string error;
        const cse::global_metadata_snapshot current = snapshot();
        if (!cse::validate_client_snapshot_ref(current, request, &error)) throw std::runtime_error(error);
        if (!cse::load_dataset_rows_as_csr(path.c_str(), row_indices.data(), row_indices.size(), &out, &error)) {
            throw std::runtime_error(error);
        }
        return out;
    }

    dataset_client_handle bootstrap_client() const {
        return dataset_client_handle(serialized_snapshot(), std::make_shared<dataset_owner_handle>(*this));
    }
};

dataset_client_handle::dataset_client_handle(const py::bytes &snapshot_bytes,
                                             std::shared_ptr<dataset_owner_handle> owner_in)
    : owner(std::move(owner_in)),
      snapshot(deserialize_snapshot_bytes(snapshot_bytes)),
      request_ref(cse::make_client_snapshot_ref(snapshot)) {
    if (!owner) throw std::invalid_argument("DatasetClient requires a non-null DatasetOwner");
    owner->validate_request(request_ref);
}

} // namespace

PYBIND11_MODULE(_cellshard, m) {
    m.doc() = "Optional CellShard Python bindings for owner/client inspection and CSR retrieval.";

    py::class_<cse::source_dataset_summary>(m, "SourceDatasetSummary")
        .def_readonly("dataset_id", &cse::source_dataset_summary::dataset_id)
        .def_readonly("matrix_path", &cse::source_dataset_summary::matrix_path)
        .def_readonly("feature_path", &cse::source_dataset_summary::feature_path)
        .def_readonly("barcode_path", &cse::source_dataset_summary::barcode_path)
        .def_readonly("metadata_path", &cse::source_dataset_summary::metadata_path)
        .def_readonly("format", &cse::source_dataset_summary::format)
        .def_readonly("row_begin", &cse::source_dataset_summary::row_begin)
        .def_readonly("row_end", &cse::source_dataset_summary::row_end)
        .def_readonly("rows", &cse::source_dataset_summary::rows)
        .def_readonly("cols", &cse::source_dataset_summary::cols)
        .def_readonly("nnz", &cse::source_dataset_summary::nnz);

    py::class_<cse::dataset_codec_summary>(m, "DatasetCodecSummary")
        .def_readonly("codec_id", &cse::dataset_codec_summary::codec_id)
        .def_readonly("family", &cse::dataset_codec_summary::family)
        .def_readonly("value_code", &cse::dataset_codec_summary::value_code)
        .def_readonly("scale_value_code", &cse::dataset_codec_summary::scale_value_code)
        .def_readonly("bits", &cse::dataset_codec_summary::bits)
        .def_readonly("flags", &cse::dataset_codec_summary::flags);

    py::class_<cse::dataset_partition_summary>(m, "DatasetPartitionSummary")
        .def_readonly("partition_id", &cse::dataset_partition_summary::partition_id)
        .def_readonly("row_begin", &cse::dataset_partition_summary::row_begin)
        .def_readonly("row_end", &cse::dataset_partition_summary::row_end)
        .def_readonly("rows", &cse::dataset_partition_summary::rows)
        .def_readonly("nnz", &cse::dataset_partition_summary::nnz)
        .def_readonly("aux", &cse::dataset_partition_summary::aux)
        .def_readonly("dataset_id", &cse::dataset_partition_summary::dataset_id)
        .def_readonly("axis", &cse::dataset_partition_summary::axis)
        .def_readonly("codec_id", &cse::dataset_partition_summary::codec_id);

    py::class_<cse::dataset_shard_summary>(m, "DatasetShardSummary")
        .def_readonly("shard_id", &cse::dataset_shard_summary::shard_id)
        .def_readonly("partition_begin", &cse::dataset_shard_summary::partition_begin)
        .def_readonly("partition_end", &cse::dataset_shard_summary::partition_end)
        .def_readonly("row_begin", &cse::dataset_shard_summary::row_begin)
        .def_readonly("row_end", &cse::dataset_shard_summary::row_end);

    py::class_<cse::dataset_summary>(m, "DatasetSummary")
        .def_readonly("path", &cse::dataset_summary::path)
        .def_readonly("matrix_format", &cse::dataset_summary::matrix_format)
        .def_readonly("payload_layout", &cse::dataset_summary::payload_layout)
        .def_readonly("rows", &cse::dataset_summary::rows)
        .def_readonly("cols", &cse::dataset_summary::cols)
        .def_readonly("nnz", &cse::dataset_summary::nnz)
        .def_readonly("num_partitions", &cse::dataset_summary::num_partitions)
        .def_readonly("num_shards", &cse::dataset_summary::num_shards)
        .def_readonly("num_datasets", &cse::dataset_summary::num_datasets)
        .def_readonly("datasets", &cse::dataset_summary::datasets)
        .def_readonly("partitions", &cse::dataset_summary::partitions)
        .def_readonly("shards", &cse::dataset_summary::shards)
        .def_readonly("codecs", &cse::dataset_summary::codecs)
        .def_readonly("obs_names", &cse::dataset_summary::obs_names)
        .def_readonly("var_ids", &cse::dataset_summary::var_ids)
        .def_readonly("var_names", &cse::dataset_summary::var_names)
        .def_readonly("var_types", &cse::dataset_summary::var_types);

    py::class_<cse::observation_metadata_column>(m, "ObservationMetadataColumn")
        .def_readonly("name", &cse::observation_metadata_column::name)
        .def_readonly("type", &cse::observation_metadata_column::type)
        .def_readonly("text_values", &cse::observation_metadata_column::text_values)
        .def_readonly("float32_values", &cse::observation_metadata_column::float32_values)
        .def_readonly("uint8_values", &cse::observation_metadata_column::uint8_values);

    py::class_<cse::embedded_metadata_table>(m, "EmbeddedMetadataTable")
        .def_readonly("dataset_index", &cse::embedded_metadata_table::dataset_index)
        .def_readonly("row_begin", &cse::embedded_metadata_table::row_begin)
        .def_readonly("row_end", &cse::embedded_metadata_table::row_end)
        .def_readonly("rows", &cse::embedded_metadata_table::rows)
        .def_readonly("cols", &cse::embedded_metadata_table::cols)
        .def_readonly("column_names", &cse::embedded_metadata_table::column_names)
        .def_readonly("field_values", &cse::embedded_metadata_table::field_values)
        .def_readonly("row_offsets", &cse::embedded_metadata_table::row_offsets);

    py::class_<cse::execution_partition_metadata>(m, "ExecutionPartitionMetadata")
        .def_readonly("partition_id", &cse::execution_partition_metadata::partition_id)
        .def_readonly("row_begin", &cse::execution_partition_metadata::row_begin)
        .def_readonly("row_end", &cse::execution_partition_metadata::row_end)
        .def_readonly("rows", &cse::execution_partition_metadata::rows)
        .def_readonly("nnz", &cse::execution_partition_metadata::nnz)
        .def_readonly("aux", &cse::execution_partition_metadata::aux)
        .def_readonly("dataset_id", &cse::execution_partition_metadata::dataset_id)
        .def_readonly("axis", &cse::execution_partition_metadata::axis)
        .def_readonly("codec_id", &cse::execution_partition_metadata::codec_id)
        .def_readonly("execution_format", &cse::execution_partition_metadata::execution_format)
        .def_readonly("blocked_ell_block_size", &cse::execution_partition_metadata::blocked_ell_block_size)
        .def_readonly("blocked_ell_bucket_count", &cse::execution_partition_metadata::blocked_ell_bucket_count)
        .def_readonly("blocked_ell_fill_ratio", &cse::execution_partition_metadata::blocked_ell_fill_ratio)
        .def_readonly("execution_bytes", &cse::execution_partition_metadata::execution_bytes)
        .def_readonly("blocked_ell_bytes", &cse::execution_partition_metadata::blocked_ell_bytes)
        .def_readonly("bucketed_blocked_ell_bytes", &cse::execution_partition_metadata::bucketed_blocked_ell_bytes);

    py::class_<cse::execution_shard_metadata>(m, "ExecutionShardMetadata")
        .def_readonly("shard_id", &cse::execution_shard_metadata::shard_id)
        .def_readonly("partition_begin", &cse::execution_shard_metadata::partition_begin)
        .def_readonly("partition_end", &cse::execution_shard_metadata::partition_end)
        .def_readonly("row_begin", &cse::execution_shard_metadata::row_begin)
        .def_readonly("row_end", &cse::execution_shard_metadata::row_end)
        .def_readonly("execution_format", &cse::execution_shard_metadata::execution_format)
        .def_readonly("blocked_ell_block_size", &cse::execution_shard_metadata::blocked_ell_block_size)
        .def_readonly("bucketed_partition_count", &cse::execution_shard_metadata::bucketed_partition_count)
        .def_readonly("bucketed_segment_count", &cse::execution_shard_metadata::bucketed_segment_count)
        .def_readonly("blocked_ell_fill_ratio", &cse::execution_shard_metadata::blocked_ell_fill_ratio)
        .def_readonly("execution_bytes", &cse::execution_shard_metadata::execution_bytes)
        .def_readonly("bucketed_blocked_ell_bytes", &cse::execution_shard_metadata::bucketed_blocked_ell_bytes)
        .def_readonly("preferred_pair", &cse::execution_shard_metadata::preferred_pair)
        .def_readonly("owner_node_id", &cse::execution_shard_metadata::owner_node_id)
        .def_readonly("owner_rank_id", &cse::execution_shard_metadata::owner_rank_id);

    py::class_<cse::runtime_service_metadata>(m, "RuntimeServiceMetadata")
        .def_readonly("service_mode", &cse::runtime_service_metadata::service_mode)
        .def_readonly("live_write_mode", &cse::runtime_service_metadata::live_write_mode)
        .def_readonly("prefer_pack_delivery", &cse::runtime_service_metadata::prefer_pack_delivery)
        .def_readonly("remote_pack_delivery", &cse::runtime_service_metadata::remote_pack_delivery)
        .def_readonly("single_reader_coordinator", &cse::runtime_service_metadata::single_reader_coordinator)
        .def_readonly("maintenance_lock_blocks_overwrite", &cse::runtime_service_metadata::maintenance_lock_blocks_overwrite)
        .def_readonly("canonical_generation", &cse::runtime_service_metadata::canonical_generation)
        .def_readonly("execution_plan_generation", &cse::runtime_service_metadata::execution_plan_generation)
        .def_readonly("pack_generation", &cse::runtime_service_metadata::pack_generation)
        .def_readonly("service_epoch", &cse::runtime_service_metadata::service_epoch)
        .def_readonly("active_read_generation", &cse::runtime_service_metadata::active_read_generation)
        .def_readonly("staged_write_generation", &cse::runtime_service_metadata::staged_write_generation);

    py::class_<cse::client_snapshot_ref>(m, "ClientSnapshotRef")
        .def_readonly("snapshot_id", &cse::client_snapshot_ref::snapshot_id)
        .def_readonly("canonical_generation", &cse::client_snapshot_ref::canonical_generation)
        .def_readonly("execution_plan_generation", &cse::client_snapshot_ref::execution_plan_generation)
        .def_readonly("pack_generation", &cse::client_snapshot_ref::pack_generation)
        .def_readonly("service_epoch", &cse::client_snapshot_ref::service_epoch);

    py::class_<cse::global_metadata_snapshot>(m, "GlobalMetadataSnapshot")
        .def_readonly("snapshot_id", &cse::global_metadata_snapshot::snapshot_id)
        .def_readonly("summary", &cse::global_metadata_snapshot::summary)
        .def_readonly("embedded_metadata", &cse::global_metadata_snapshot::embedded_metadata)
        .def_readonly("observation_metadata_rows", &cse::global_metadata_snapshot::observation_metadata_rows)
        .def_readonly("observation_metadata", &cse::global_metadata_snapshot::observation_metadata)
        .def_readonly("execution_partitions", &cse::global_metadata_snapshot::execution_partitions)
        .def_readonly("execution_shards", &cse::global_metadata_snapshot::execution_shards)
        .def_readonly("runtime_service", &cse::global_metadata_snapshot::runtime_service)
        .def("serialized_bytes", [](const cse::global_metadata_snapshot &self) { return serialize_snapshot_bytes(self); });

    py::class_<cse::csr_matrix_export>(m, "CsrMatrixExport")
        .def_readonly("rows", &cse::csr_matrix_export::rows)
        .def_readonly("cols", &cse::csr_matrix_export::cols)
        .def("indptr_array", [](const cse::csr_matrix_export &self) { return copy_1d_array(self.indptr); })
        .def("indices_array", [](const cse::csr_matrix_export &self) { return copy_1d_array(self.indices); })
        .def("data_array", [](const cse::csr_matrix_export &self) { return copy_1d_array(self.data); })
        .def("to_scipy_csr", [](const cse::csr_matrix_export &self) { return build_scipy_csr(self); })
        .def("to_torch_sparse_csr", [](const cse::csr_matrix_export &self) { return build_torch_sparse_csr(self); });

    py::class_<dataset_file_handle>(m, "DatasetFile")
        .def(py::init([](const std::string &path) { return dataset_file_handle{path}; }))
        .def_readonly("path", &dataset_file_handle::path)
        .def("summary", &dataset_file_handle::summary)
        .def("snapshot", &dataset_file_handle::snapshot)
        .def("materialize_csr", &dataset_file_handle::materialize_csr)
        .def("materialize_rows_csr", &dataset_file_handle::materialize_rows_csr)
        .def("materialize_partition", &dataset_file_handle::materialize_partition)
        .def("write_h5ad", &dataset_file_handle::write_h5ad);

    py::class_<dataset_owner_handle, std::shared_ptr<dataset_owner_handle>>(m, "DatasetOwner")
        .def(py::init([](const std::string &path) {
            return std::make_shared<dataset_owner_handle>(dataset_owner_handle{path});
        }))
        .def_readonly("path", &dataset_owner_handle::path)
        .def("snapshot", &dataset_owner_handle::snapshot)
        .def("serialized_snapshot", &dataset_owner_handle::serialized_snapshot)
        .def("current_request_ref", &dataset_owner_handle::current_request_ref)
        .def("validate_request", &dataset_owner_handle::validate_request)
        .def("materialize_csr", &dataset_owner_handle::materialize_csr)
        .def("materialize_rows_csr", [](const dataset_owner_handle &self,
                                        const cse::client_snapshot_ref &request,
                                        py::handle row_indices) {
            return self.materialize_rows_csr(request, normalize_row_indices(row_indices));
        })
        .def("bootstrap_client", &dataset_owner_handle::bootstrap_client);

    py::class_<dataset_client_handle>(m, "DatasetClient")
        .def(py::init([](const py::bytes &snapshot_bytes, const std::shared_ptr<dataset_owner_handle> &owner) {
            return dataset_client_handle(snapshot_bytes, owner);
        }))
        .def_property_readonly("snapshot", [](const dataset_client_handle &self) { return self.snapshot; })
        .def_property_readonly("request_ref", [](const dataset_client_handle &self) { return self.request_ref; })
        .def("serialized_snapshot", &dataset_client_handle::serialized_snapshot)
        .def("materialize_csr", [](const dataset_client_handle &self) {
            return self.owner->materialize_csr(self.request_ref);
        })
        .def("materialize_rows_csr", [](const dataset_client_handle &self, py::handle row_indices) {
            return self.owner->materialize_rows_csr(self.request_ref, normalize_row_indices(row_indices));
        })
        .def("materialize_scipy_csr", [](const dataset_client_handle &self) {
            return build_scipy_csr(self.owner->materialize_csr(self.request_ref));
        })
        .def("materialize_rows_scipy_csr", [](const dataset_client_handle &self, py::handle row_indices) {
            return build_scipy_csr(self.owner->materialize_rows_csr(self.request_ref, normalize_row_indices(row_indices)));
        })
        .def("materialize_torch_sparse_csr", [](const dataset_client_handle &self) {
            return build_torch_sparse_csr(self.owner->materialize_csr(self.request_ref));
        })
        .def("materialize_rows_torch_sparse_csr", [](const dataset_client_handle &self, py::handle row_indices) {
            return build_torch_sparse_csr(self.owner->materialize_rows_csr(self.request_ref, normalize_row_indices(row_indices)));
        });

    m.def("open_dataset", [](const std::string &path) { return dataset_file_handle{path}; });
    m.def("open_dataset_owner", [](const std::string &path) {
        return std::make_shared<dataset_owner_handle>(dataset_owner_handle{path});
    });
    m.def("bootstrap_dataset_client", [](const std::shared_ptr<dataset_owner_handle> &owner) {
        if (!owner) throw std::invalid_argument("bootstrap_dataset_client requires a non-null DatasetOwner");
        return owner->bootstrap_client();
    });
    m.def("load_dataset_summary", [](const std::string &path) {
        cse::dataset_summary out;
        std::string error;
        if (!cse::load_dataset_summary(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    });
    m.def("load_dataset_as_csr", [](const std::string &path) {
        cse::csr_matrix_export out;
        std::string error;
        if (!cse::load_dataset_as_csr(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    });
    m.def("load_dataset_rows_as_csr", [](const std::string &path, py::handle row_indices) {
        const std::vector<std::uint64_t> rows = normalize_row_indices(row_indices);
        cse::csr_matrix_export out;
        std::string error;
        if (!cse::load_dataset_rows_as_csr(path.c_str(), rows.data(), rows.size(), &out, &error)) {
            throw std::runtime_error(error);
        }
        return out;
    });
    m.def("load_dataset_global_metadata_snapshot", [](const std::string &path) {
        cse::global_metadata_snapshot out;
        std::string error;
        if (!cse::load_dataset_global_metadata_snapshot(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    });
    m.def("serialize_global_metadata_snapshot", [](const cse::global_metadata_snapshot &snapshot) {
        return serialize_snapshot_bytes(snapshot);
    });
    m.def("deserialize_global_metadata_snapshot", [](const py::bytes &payload) {
        return deserialize_snapshot_bytes(payload);
    });
    m.def("make_client_snapshot_ref", [](const cse::global_metadata_snapshot &snapshot) {
        return cse::make_client_snapshot_ref(snapshot);
    });
    m.def("validate_client_snapshot_ref", [](const cse::global_metadata_snapshot &owner_snapshot,
                                             const cse::client_snapshot_ref &request) {
        std::string error;
        if (!cse::validate_client_snapshot_ref(owner_snapshot, request, &error)) throw std::runtime_error(error);
        return true;
    });
    m.def("write_h5ad", [](const std::string &dataset_path, const std::string &output_path) {
        std::string error;
        if (!cse::write_dataset_file_to_h5ad_with_python(dataset_path.c_str(), output_path.c_str(), &error)) {
            throw std::runtime_error(error);
        }
    });
}
