#pragma once

#include "../../include/CellShard/export/dataset_export.hh"
#include "../../export/h5ad_writer.hh"
#include "../../include/CellShard/formats/blocked_ell.cuh"
#include "../../include/CellShard/formats/compressed.cuh"
#include "../../include/CellShard/io/csh5/api.cuh"
#include "../../include/CellShard/runtime/host/sharded_host.cuh"
#include "../../include/CellShard/runtime/storage/shard_storage.cuh"
#include "../../include/CellShard/runtime/layout/sharded.cuh"
#include "../../include/CellShard/runtime/storage/disk.cuh"

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

namespace cellshard::python_bindings {

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
inline py::array_t<T> copy_1d_array(const std::vector<T> &values) {
    py::array_t<T> out((py::ssize_t) values.size());
    if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(T));
    return out;
}

inline py::array_t<float> copy_half_values(const __half *values, std::size_t count) {
    py::array_t<float> out((py::ssize_t) count);
    float *dst = out.mutable_data();
    for (std::size_t i = 0; i < count; ++i) dst[i] = __half2float(values[i]);
    return out;
}

inline py::object build_scipy_csr(const cse::csr_matrix_export &self) {
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

inline py::object build_torch_sparse_csr(const cse::csr_matrix_export &self) {
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

inline std::vector<std::uint8_t> bytes_to_vector(const py::bytes &bytes) {
    const std::string payload = bytes;
    return std::vector<std::uint8_t>(payload.begin(), payload.end());
}

inline py::bytes serialize_snapshot_bytes(const cse::global_metadata_snapshot &snapshot) {
    std::vector<std::uint8_t> payload;
    std::string error;
    if (!cse::serialize_global_metadata_snapshot(snapshot, &payload, &error)) throw std::runtime_error(error);
    return py::bytes(reinterpret_cast<const char *>(payload.data()), payload.size());
}

inline cse::global_metadata_snapshot deserialize_snapshot_bytes(const py::bytes &bytes) {
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

inline std::vector<std::uint64_t> normalize_row_indices(py::handle handle) {
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

    std::vector<cse::observation_metadata_column> observation_metadata() const {
        std::vector<cse::observation_metadata_column> out;
        std::string error;
        if (!cse::load_observation_metadata(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    std::vector<cse::annotation_column> feature_metadata() const {
        std::vector<cse::annotation_column> out;
        std::string error;
        if (!cse::load_feature_metadata(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    std::vector<cse::dataset_attribute> dataset_attributes() const {
        std::vector<cse::dataset_attribute> out;
        std::string error;
        if (!cse::load_dataset_attributes(path.c_str(), &out, &error)) throw std::runtime_error(error);
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

        throw std::runtime_error("unsupported matrix_format for partition materialization; supported native format is blocked_ell");
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

    cse::global_metadata_snapshot snapshot_view() const {
        cse::global_metadata_snapshot out;
        std::string error;
        if (!cse::load_dataset_global_metadata_snapshot(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    inline void validate_request_or_throw(const cse::client_snapshot_ref &request) const {
        std::string error;
        const cse::global_metadata_snapshot current = snapshot_view();
        if (!cse::validate_client_snapshot_ref(current, request, &error)) throw std::runtime_error(error);
    }

    cse::global_metadata_snapshot snapshot() const {
        return snapshot_view();
    }

    std::vector<cse::observation_metadata_column> observation_metadata(const cse::client_snapshot_ref &request) const {
        std::vector<cse::observation_metadata_column> out;
        std::string error;
        validate_request_or_throw(request);
        if (!cse::load_observation_metadata(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    std::vector<cse::annotation_column> feature_metadata(const cse::client_snapshot_ref &request) const {
        std::vector<cse::annotation_column> out;
        std::string error;
        validate_request_or_throw(request);
        if (!cse::load_feature_metadata(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    std::vector<cse::dataset_attribute> dataset_attributes(const cse::client_snapshot_ref &request) const {
        std::vector<cse::dataset_attribute> out;
        std::string error;
        validate_request_or_throw(request);
        if (!cse::load_dataset_attributes(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    py::bytes serialized_snapshot() const {
        return serialize_snapshot_bytes(snapshot_view());
    }

    cse::client_snapshot_ref current_request_ref() const {
        return cse::make_client_snapshot_ref(snapshot_view());
    }

    cse::runtime_service_metadata stage_append_runtime_service() const {
        cse::runtime_service_metadata staged;
        std::string error;
        const cse::global_metadata_snapshot current = snapshot_view();
        if (!cse::stage_append_only_runtime_service(current.runtime_service, &staged, &error)) {
            throw std::runtime_error(error);
        }
        return staged;
    }

    cse::runtime_service_metadata publish_runtime_service_cutover(const cse::runtime_service_metadata &staged) const {
        cse::runtime_service_metadata published;
        std::string error;
        const cse::global_metadata_snapshot current = snapshot_view();
        if (!cse::publish_runtime_service_cutover(current.runtime_service, staged, &published, &error)) {
            throw std::runtime_error(error);
        }
        return published;
    }

    void validate_request(const cse::client_snapshot_ref &request) const {
        validate_request_or_throw(request);
    }

    cse::pack_delivery_descriptor describe_pack_delivery(const cse::pack_delivery_request &request) const {
        cse::pack_delivery_descriptor out;
        std::string error;
        const cse::global_metadata_snapshot current = snapshot_view();
        if (!cse::describe_pack_delivery(current, request, &out, &error)) throw std::runtime_error(error);
        return out;
    }

    cse::csr_matrix_export materialize_csr(const cse::client_snapshot_ref &request) const {
        cse::csr_matrix_export out;
        std::string error;
        validate_request_or_throw(request);
        if (!cse::load_dataset_as_csr(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    cse::csr_matrix_export materialize_rows_csr(const cse::client_snapshot_ref &request,
                                                const std::vector<std::uint64_t> &row_indices) const {
        cse::csr_matrix_export out;
        std::string error;
        validate_request_or_throw(request);
        if (!cse::load_dataset_rows_as_csr(path.c_str(), row_indices.data(), row_indices.size(), &out, &error)) {
            throw std::runtime_error(error);
        }
        return out;
    }

    dataset_client_handle bootstrap_client() const {
        return dataset_client_handle(serialized_snapshot(), std::make_shared<dataset_owner_handle>(*this));
    }
};

inline dataset_client_handle::dataset_client_handle(const py::bytes &snapshot_bytes,
                                                    std::shared_ptr<dataset_owner_handle> owner_in)
    : owner(std::move(owner_in)),
      snapshot(deserialize_snapshot_bytes(snapshot_bytes)),
      request_ref(cse::make_client_snapshot_ref(snapshot)) {
    if (!owner) throw std::invalid_argument("DatasetClient requires a non-null DatasetOwner");
    owner->validate_request(request_ref);
}

void bind_dataset_export_types(py::module_ &m);
void bind_dataset_handles(py::module_ &m);

} // namespace cellshard::python_bindings
