#pragma once

#include "../../include/CellShard/export/dataset_export.hh"
#include "../../export/h5ad_writer.hh"
#include "../../include/CellShard/formats/blocked_ell.cuh"
#include "../../include/CellShard/formats/compressed.cuh"
#include "../../include/CellShard/io/cshard.hh"
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
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

inline std::vector<std::uint16_t> copy_storage_bits(const real::storage_t *values, std::size_t count) {
    static_assert(sizeof(real::storage_t) == sizeof(std::uint16_t),
                  "Python native CellShard storage export currently expects 16-bit storage_t");
    std::vector<std::uint16_t> out(count);
    if (count != 0u && values != nullptr) std::memcpy(out.data(), values, count * sizeof(std::uint16_t));
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

template<typename MatrixT>
struct scoped_sharded_load {
    cellshard::sharded<MatrixT> view;
    cellshard::shard_storage storage;

    scoped_sharded_load() {
        cellshard::init(&view);
        cellshard::init(&storage);
    }

    ~scoped_sharded_load() {
        cellshard::clear(&storage);
        cellshard::clear(&view);
    }

    scoped_sharded_load(const scoped_sharded_load &) = delete;
    scoped_sharded_load &operator=(const scoped_sharded_load &) = delete;
};

struct blocked_ell_partition_export {
    std::uint32_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint32_t nnz = 0u;
    std::uint32_t block_size = 0u;
    std::uint32_t ell_cols = 0u;
    std::uint32_t row_block_count = 0u;
    std::uint32_t ell_width_blocks = 0u;
    std::vector<std::uint32_t> block_col_idx;
    std::vector<std::uint16_t> values_storage;

    py::array_t<std::uint32_t> block_col_idx_array() const {
        py::array_t<std::uint32_t> out({(py::ssize_t) row_block_count, (py::ssize_t) ell_width_blocks});
        if (!block_col_idx.empty()) std::memcpy(out.mutable_data(), block_col_idx.data(), block_col_idx.size() * sizeof(std::uint32_t));
        return out;
    }

    py::array_t<std::uint16_t> values_storage_array() const {
        py::array_t<std::uint16_t> out({(py::ssize_t) rows, (py::ssize_t) ell_cols});
        if (!values_storage.empty()) std::memcpy(out.mutable_data(), values_storage.data(), values_storage.size() * sizeof(std::uint16_t));
        return out;
    }

    py::array_t<float> values_float32() const {
        py::array_t<float> out({(py::ssize_t) rows, (py::ssize_t) ell_cols});
        float *dst = out.mutable_data();
        for (std::size_t i = 0; i < values_storage.size(); ++i) {
            real::storage_t stored;
            std::memcpy(&stored, values_storage.data() + i, sizeof(stored));
            dst[i] = __half2float(stored);
        }
        return out;
    }

    py::dict as_dict() const {
        py::dict out;
        out[py::str("format")] = py::str("blocked_ell");
        out[py::str("rows")] = py::int_(rows);
        out[py::str("cols")] = py::int_(cols);
        out[py::str("nnz")] = py::int_(nnz);
        out[py::str("block_size")] = py::int_(block_size);
        out[py::str("ell_cols")] = py::int_(ell_cols);
        out[py::str("row_block_count")] = py::int_(row_block_count);
        out[py::str("ell_width_blocks")] = py::int_(ell_width_blocks);
        out[py::str("block_col_idx")] = block_col_idx_array();
        out[py::str("values_storage")] = values_storage_array();
        return out;
    }
};

struct sliced_ell_partition_export {
    std::uint32_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint32_t nnz = 0u;
    std::uint32_t slice_count = 0u;
    std::uint32_t total_slots = 0u;
    std::vector<std::uint32_t> slice_row_offsets;
    std::vector<std::uint32_t> slice_widths;
    std::vector<std::uint32_t> col_idx;
    std::vector<std::uint16_t> values_storage;

    py::array_t<std::uint32_t> slice_row_offsets_array() const { return copy_1d_array(slice_row_offsets); }
    py::array_t<std::uint32_t> slice_widths_array() const { return copy_1d_array(slice_widths); }
    py::array_t<std::uint32_t> col_idx_array() const { return copy_1d_array(col_idx); }
    py::array_t<std::uint16_t> values_storage_array() const { return copy_1d_array(values_storage); }

    py::array_t<float> values_float32() const {
        py::array_t<float> out((py::ssize_t) values_storage.size());
        float *dst = out.mutable_data();
        for (std::size_t i = 0; i < values_storage.size(); ++i) {
            real::storage_t stored;
            std::memcpy(&stored, values_storage.data() + i, sizeof(stored));
            dst[i] = __half2float(stored);
        }
        return out;
    }

    py::dict as_dict() const {
        py::dict out;
        out[py::str("format")] = py::str("sliced_ell");
        out[py::str("rows")] = py::int_(rows);
        out[py::str("cols")] = py::int_(cols);
        out[py::str("nnz")] = py::int_(nnz);
        out[py::str("slice_count")] = py::int_(slice_count);
        out[py::str("total_slots")] = py::int_(total_slots);
        out[py::str("slice_row_offsets")] = slice_row_offsets_array();
        out[py::str("slice_widths")] = slice_widths_array();
        out[py::str("col_idx")] = col_idx_array();
        out[py::str("values_storage")] = values_storage_array();
        return out;
    }
};

inline blocked_ell_partition_export make_blocked_ell_partition_export(const cellshard::sparse::blocked_ell *partition) {
    if (partition == nullptr) throw std::runtime_error("fetched blocked-ELL partition is null");
    blocked_ell_partition_export out;
    out.rows = partition->rows;
    out.cols = partition->cols;
    out.nnz = partition->nnz;
    out.block_size = partition->block_size;
    out.ell_cols = partition->ell_cols;
    out.row_block_count = cellshard::sparse::row_block_count(partition);
    out.ell_width_blocks = cellshard::sparse::ell_width_blocks(partition);
    const std::size_t idx_count = (std::size_t) out.row_block_count * (std::size_t) out.ell_width_blocks;
    const std::size_t value_count = (std::size_t) out.rows * (std::size_t) out.ell_cols;
    out.block_col_idx.resize(idx_count);
    if (idx_count != 0u && partition->blockColIdx != nullptr) {
        std::memcpy(out.block_col_idx.data(), partition->blockColIdx, idx_count * sizeof(std::uint32_t));
    }
    out.values_storage = copy_storage_bits(partition->val, value_count);
    return out;
}

inline sliced_ell_partition_export make_sliced_ell_partition_export(const cellshard::sparse::sliced_ell *partition) {
    if (partition == nullptr) throw std::runtime_error("fetched sliced-ELL partition is null");
    sliced_ell_partition_export out;
    out.rows = partition->rows;
    out.cols = partition->cols;
    out.nnz = partition->nnz;
    out.slice_count = partition->slice_count;
    out.total_slots = cellshard::sparse::total_slots(partition);
    out.slice_row_offsets.resize((std::size_t) out.slice_count + 1u);
    out.slice_widths.resize(out.slice_count);
    out.col_idx.resize(out.total_slots);
    if (!out.slice_row_offsets.empty() && partition->slice_row_offsets != nullptr) {
        std::memcpy(out.slice_row_offsets.data(), partition->slice_row_offsets, out.slice_row_offsets.size() * sizeof(std::uint32_t));
    }
    if (!out.slice_widths.empty() && partition->slice_widths != nullptr) {
        std::memcpy(out.slice_widths.data(), partition->slice_widths, out.slice_widths.size() * sizeof(std::uint32_t));
    }
    if (!out.col_idx.empty() && partition->col_idx != nullptr) {
        std::memcpy(out.col_idx.data(), partition->col_idx, out.col_idx.size() * sizeof(std::uint32_t));
    }
    out.values_storage = copy_storage_bits(partition->val, out.total_slots);
    return out;
}

inline blocked_ell_partition_export fetch_blocked_ell_partition_export(const std::string &path,
                                                                       unsigned long partition_id) {
    scoped_sharded_load<cellshard::sparse::blocked_ell> loaded;
    if (!cellshard::load_header(path.c_str(), &loaded.view, &loaded.storage)) {
        throw std::runtime_error("failed to load blocked-ELL dataset header");
    }
    if (partition_id >= loaded.view.num_partitions) throw std::out_of_range("partition_id is out of range");
    if (!cellshard::fetch_dataset_blocked_ell_h5_partition(&loaded.view, &loaded.storage, partition_id)) {
        throw std::runtime_error("failed to fetch blocked-ELL partition");
    }
    return make_blocked_ell_partition_export(loaded.view.parts[partition_id]);
}

inline sliced_ell_partition_export fetch_sliced_ell_partition_export(const std::string &path,
                                                                     unsigned long partition_id) {
    scoped_sharded_load<cellshard::sparse::sliced_ell> loaded;
    if (!cellshard::load_header(path.c_str(), &loaded.view, &loaded.storage)) {
        throw std::runtime_error("failed to load sliced-ELL dataset header");
    }
    if (partition_id >= loaded.view.num_partitions) throw std::out_of_range("partition_id is out of range");
    if (!cellshard::fetch_dataset_sliced_ell_h5_partition(&loaded.view, &loaded.storage, partition_id)) {
        throw std::runtime_error("failed to fetch sliced-ELL partition");
    }
    return make_sliced_ell_partition_export(loaded.view.parts[partition_id]);
}

inline py::object fetch_native_partition_object(const std::string &path,
                                                const cse::dataset_summary &info,
                                                unsigned long partition_id) {
    if (partition_id >= info.partitions.size()) throw std::out_of_range("partition_id is out of range");
    warn_cpu_materialize_once("native partition fetch");
    if (info.matrix_format == "blocked_ell") return py::cast(fetch_blocked_ell_partition_export(path, partition_id));
    if (info.matrix_format == "sliced_ell") return py::cast(fetch_sliced_ell_partition_export(path, partition_id));
    throw std::runtime_error(
        "unsupported native matrix layout '" + info.matrix_format
        + "' for partition fetch; use format=\"csr\" only if this dataset supports CSR export");
}

struct native_matrix_view {
    std::string path;
    cse::dataset_summary summary;

    explicit native_matrix_view(std::string path_in)
        : path(std::move(path_in)) {
        std::string error;
        if (!cse::load_dataset_summary(path.c_str(), &summary, &error)) throw std::runtime_error(error);
    }

    py::tuple shape() const {
        return py::make_tuple((py::ssize_t) summary.rows, (py::ssize_t) summary.cols);
    }

    py::object partition(unsigned long partition_id) const {
        return fetch_native_partition_object(path, summary, partition_id);
    }

    py::list shard(unsigned long shard_id) const {
        if (shard_id >= summary.shards.size()) throw std::out_of_range("shard_id is out of range");
        py::list out;
        const cse::dataset_shard_summary &shard_info = summary.shards[shard_id];
        for (std::uint64_t part = shard_info.partition_begin; part < shard_info.partition_end; ++part) {
            out.append(partition((unsigned long) part));
        }
        return out;
    }

    cse::csr_matrix_export to_csr() const {
        cse::csr_matrix_export out;
        std::string error;
        if (!cse::load_dataset_as_csr(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    }

    py::object to_scipy_csr() const {
        return build_scipy_csr(to_csr());
    }

    py::object to_torch_sparse_csr() const {
        return build_torch_sparse_csr(to_csr());
    }
};

struct native_row_selection {
    std::string path;
    std::vector<std::uint64_t> row_indices;
    cse::dataset_summary summary;

    native_row_selection(std::string path_in, std::vector<std::uint64_t> rows_in)
        : path(std::move(path_in)),
          row_indices(std::move(rows_in)) {
        std::string error;
        if (!cse::load_dataset_summary(path.c_str(), &summary, &error)) throw std::runtime_error(error);
    }

    py::tuple shape() const {
        return py::make_tuple((py::ssize_t) row_indices.size(), (py::ssize_t) summary.cols);
    }

    cse::csr_matrix_export to_csr() const {
        cse::csr_matrix_export out;
        std::string error;
        if (!cse::load_dataset_rows_as_csr(path.c_str(), row_indices.data(), row_indices.size(), &out, &error)) {
            throw std::runtime_error(error);
        }
        return out;
    }

    py::object to_scipy_csr() const {
        return build_scipy_csr(to_csr());
    }

    py::object to_torch_sparse_csr() const {
        return build_torch_sparse_csr(to_csr());
    }
};

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

    py::object materialize_partition(unsigned long partition_id) const {
        cse::dataset_summary info;
        std::string error;
        if (!cse::load_dataset_summary(path.c_str(), &info, &error)) throw std::runtime_error(error);
        return fetch_native_partition_object(path, info, partition_id);
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
