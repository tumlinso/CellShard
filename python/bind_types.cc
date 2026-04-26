#include "internal/module_support.hh"

namespace cellshard::python_bindings {

void bind_dataset_export_types(py::module_ &m) {
    py::class_<cse::source_dataset_summary>(m, "SourceDatasetSummary")
        .def_readonly("dataset_id", &cse::source_dataset_summary::dataset_id)
        .def_readonly("matrix_path", &cse::source_dataset_summary::matrix_path)
        .def_readonly("feature_path", &cse::source_dataset_summary::feature_path)
        .def_readonly("barcode_path", &cse::source_dataset_summary::barcode_path)
        .def_readonly("metadata_path", &cse::source_dataset_summary::metadata_path)
        .def_readonly("format", &cse::source_dataset_summary::format)
        .def_property_readonly("row_begin", [](const cse::source_dataset_summary &self) { return self.row_span.row_begin; })
        .def_property_readonly("row_end", [](const cse::source_dataset_summary &self) { return self.row_span.row_end; })
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
        .def_property_readonly("row_begin", [](const cse::dataset_partition_summary &self) { return self.row_span.row_begin; })
        .def_property_readonly("row_end", [](const cse::dataset_partition_summary &self) { return self.row_span.row_end; })
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
        .def_property_readonly("row_begin", [](const cse::dataset_shard_summary &self) { return self.row_span.row_begin; })
        .def_property_readonly("row_end", [](const cse::dataset_shard_summary &self) { return self.row_span.row_end; });

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
        .def_readonly("var_types", &cse::dataset_summary::var_types)
        .def_readonly("observation_annotations", &cse::dataset_summary::observation_annotations)
        .def_readonly("feature_annotations", &cse::dataset_summary::feature_annotations)
        .def_readonly("dataset_attributes", &cse::dataset_summary::dataset_attributes);

    py::class_<cse::annotation_summary>(m, "AnnotationSummary")
        .def_readonly("available", &cse::annotation_summary::available)
        .def_readonly("extent", &cse::annotation_summary::extent)
        .def_readonly("names", &cse::annotation_summary::names)
        .def_readonly("types", &cse::annotation_summary::types);

    py::class_<cse::dataset_attribute>(m, "DatasetAttribute")
        .def_readonly("key", &cse::dataset_attribute::key)
        .def_readonly("value", &cse::dataset_attribute::value);

    py::class_<cse::dataset_attribute_summary>(m, "DatasetAttributeSummary")
        .def_readonly("available", &cse::dataset_attribute_summary::available)
        .def_readonly("keys", &cse::dataset_attribute_summary::keys);

    py::class_<cse::observation_metadata_column>(m, "ObservationMetadataColumn")
        .def_readonly("name", &cse::observation_metadata_column::name)
        .def_readonly("type", &cse::observation_metadata_column::type)
        .def_readonly("text_values", &cse::observation_metadata_column::text_values)
        .def_readonly("float32_values", &cse::observation_metadata_column::float32_values)
        .def_readonly("uint8_values", &cse::observation_metadata_column::uint8_values);

    py::class_<cse::embedded_metadata_table>(m, "EmbeddedMetadataTable")
        .def_readonly("dataset_index", &cse::embedded_metadata_table::dataset_index)
        .def_property_readonly("row_begin", [](const cse::embedded_metadata_table &self) { return self.row_span.row_begin; })
        .def_property_readonly("row_end", [](const cse::embedded_metadata_table &self) { return self.row_span.row_end; })
        .def_readonly("rows", &cse::embedded_metadata_table::rows)
        .def_readonly("cols", &cse::embedded_metadata_table::cols)
        .def_readonly("column_names", &cse::embedded_metadata_table::column_names)
        .def_readonly("field_values", &cse::embedded_metadata_table::field_values)
        .def_readonly("row_offsets", &cse::embedded_metadata_table::row_offsets);

    py::class_<cse::execution_partition_metadata>(m, "ExecutionPartitionMetadata")
        .def_readonly("partition_id", &cse::execution_partition_metadata::partition_id)
        .def_property_readonly("row_begin", [](const cse::execution_partition_metadata &self) { return self.row_span.row_begin; })
        .def_property_readonly("row_end", [](const cse::execution_partition_metadata &self) { return self.row_span.row_end; })
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
        .def_property_readonly("row_begin", [](const cse::execution_shard_metadata &self) { return self.row_span.row_begin; })
        .def_property_readonly("row_end", [](const cse::execution_shard_metadata &self) { return self.row_span.row_end; })
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
        .def_property_readonly("canonical_generation", [](const cse::runtime_service_metadata &self) {
            return self.runtime_generation.generation.canonical_generation;
        })
        .def_property_readonly("execution_plan_generation", [](const cse::runtime_service_metadata &self) {
            return self.runtime_generation.generation.execution_plan_generation;
        })
        .def_property_readonly("pack_generation", [](const cse::runtime_service_metadata &self) {
            return self.runtime_generation.generation.pack_generation;
        })
        .def_property_readonly("service_epoch", [](const cse::runtime_service_metadata &self) {
            return self.runtime_generation.generation.service_epoch;
        })
        .def_property_readonly("active_read_generation", [](const cse::runtime_service_metadata &self) {
            return self.runtime_generation.active_read_generation;
        })
        .def_property_readonly("staged_write_generation", [](const cse::runtime_service_metadata &self) {
            return self.runtime_generation.staged_write_generation;
        });

    py::class_<cse::client_snapshot_ref>(m, "ClientSnapshotRef")
        .def_readonly("snapshot_id", &cse::client_snapshot_ref::snapshot_id)
        .def_property_readonly("canonical_generation", [](const cse::client_snapshot_ref &self) {
            return self.generation.canonical_generation;
        })
        .def_property_readonly("execution_plan_generation", [](const cse::client_snapshot_ref &self) {
            return self.generation.execution_plan_generation;
        })
        .def_property_readonly("pack_generation", [](const cse::client_snapshot_ref &self) {
            return self.generation.pack_generation;
        })
        .def_property_readonly("service_epoch", [](const cse::client_snapshot_ref &self) {
            return self.generation.service_epoch;
        });

    py::class_<cse::pack_delivery_request>(m, "PackDeliveryRequest")
        .def(py::init<>())
        .def_readwrite("request", &cse::pack_delivery_request::request)
        .def_readwrite("shard_id", &cse::pack_delivery_request::shard_id);

    py::class_<cse::pack_delivery_descriptor>(m, "PackDeliveryDescriptor")
        .def_readonly("snapshot_id", &cse::pack_delivery_descriptor::snapshot_id)
        .def_readonly("shard_id", &cse::pack_delivery_descriptor::shard_id)
        .def_property_readonly("canonical_generation", [](const cse::pack_delivery_descriptor &self) {
            return self.generation.canonical_generation;
        })
        .def_property_readonly("execution_plan_generation", [](const cse::pack_delivery_descriptor &self) {
            return self.generation.execution_plan_generation;
        })
        .def_property_readonly("pack_generation", [](const cse::pack_delivery_descriptor &self) {
            return self.generation.pack_generation;
        })
        .def_property_readonly("service_epoch", [](const cse::pack_delivery_descriptor &self) {
            return self.generation.service_epoch;
        })
        .def_readonly("owner_node_id", &cse::pack_delivery_descriptor::owner_node_id)
        .def_readonly("owner_rank_id", &cse::pack_delivery_descriptor::owner_rank_id)
        .def_readonly("execution_format", &cse::pack_delivery_descriptor::execution_format)
        .def_readonly("relative_pack_path", &cse::pack_delivery_descriptor::relative_pack_path);

    py::class_<cse::global_metadata_snapshot>(m, "GlobalMetadataSnapshot")
        .def_readonly("snapshot_id", &cse::global_metadata_snapshot::snapshot_id)
        .def_readonly("summary", &cse::global_metadata_snapshot::summary)
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

    py::class_<cellshard::cshard::table_view::column>(m, "CshardTableColumn")
        .def_readonly("name", &cellshard::cshard::table_view::column::name)
        .def_readonly("type", &cellshard::cshard::table_view::column::type)
        .def_readonly("text_values", &cellshard::cshard::table_view::column::text_values)
        .def_readonly("float32_values", &cellshard::cshard::table_view::column::float32_values)
        .def_readonly("uint8_values", &cellshard::cshard::table_view::column::uint8_values);

    py::class_<cellshard::cshard::table_view>(m, "CshardTable")
        .def_readonly("rows", &cellshard::cshard::table_view::rows)
        .def_readonly("columns", &cellshard::cshard::table_view::columns)
        .def("head", &cellshard::cshard::table_view::head);

    py::class_<cellshard::cshard::description>(m, "CshardDescription")
        .def_readonly("path", &cellshard::cshard::description::path)
        .def_readonly("version_major", &cellshard::cshard::description::version_major)
        .def_readonly("version_minor", &cellshard::cshard::description::version_minor)
        .def_readonly("rows", &cellshard::cshard::description::rows)
        .def_readonly("cols", &cellshard::cshard::description::cols)
        .def_readonly("nnz", &cellshard::cshard::description::nnz)
        .def_readonly("partitions", &cellshard::cshard::description::partitions)
        .def_readonly("feature_order_hash", &cellshard::cshard::description::feature_order_hash)
        .def_readonly("canonical_layout", &cellshard::cshard::description::canonical_layout)
        .def_readonly("has_pack_manifest", &cellshard::cshard::description::has_pack_manifest);

    py::class_<cellshard::cshard::cshard_file>(m, "CshardFile")
        .def_static("open", &cellshard::cshard::cshard_file::open)
        .def_property_readonly("path", [](const cellshard::cshard::cshard_file &self) { return self.path(); })
        .def("describe", &cellshard::cshard::cshard_file::describe)
        .def("read_rows", &cellshard::cshard::cshard_file::read_rows)
        .def("obs", &cellshard::cshard::cshard_file::obs)
        .def("var", &cellshard::cshard::cshard_file::var)
        .def("has_pack_manifest", &cellshard::cshard::cshard_file::has_pack_manifest);

    m.def("load_dataset_summary", [](const std::string &path) {
        cse::dataset_summary out;
        std::string error;
        if (!cse::load_dataset_summary(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    });
    m.def("load_observation_metadata", [](const std::string &path) {
        std::vector<cse::observation_metadata_column> out;
        std::string error;
        if (!cse::load_observation_metadata(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    });
    m.def("load_feature_metadata", [](const std::string &path) {
        std::vector<cse::annotation_column> out;
        std::string error;
        if (!cse::load_feature_metadata(path.c_str(), &out, &error)) throw std::runtime_error(error);
        return out;
    });
    m.def("load_dataset_attributes", [](const std::string &path) {
        std::vector<cse::dataset_attribute> out;
        std::string error;
        if (!cse::load_dataset_attributes(path.c_str(), &out, &error)) throw std::runtime_error(error);
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
    m.def("open_cshard", [](const std::string &path) {
        return cellshard::cshard::cshard_file::open(path);
    });
    m.def("validate_cshard", [](const std::string &path) {
        std::string error;
        if (!cellshard::cshard::cshard_file::validate(path, &error)) throw std::runtime_error(error);
        return true;
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
    m.def("stage_append_only_runtime_service", [](const cse::runtime_service_metadata &current) {
        cse::runtime_service_metadata staged;
        std::string error;
        if (!cse::stage_append_only_runtime_service(current, &staged, &error)) throw std::runtime_error(error);
        return staged;
    });
    m.def("publish_runtime_service_cutover", [](const cse::runtime_service_metadata &current,
                                                const cse::runtime_service_metadata &staged) {
        cse::runtime_service_metadata published;
        std::string error;
        if (!cse::publish_runtime_service_cutover(current, staged, &published, &error)) throw std::runtime_error(error);
        return published;
    });
    m.def("describe_pack_delivery", [](const cse::global_metadata_snapshot &owner_snapshot,
                                       const cse::pack_delivery_request &request) {
        cse::pack_delivery_descriptor out;
        std::string error;
        if (!cse::describe_pack_delivery(owner_snapshot, request, &out, &error)) throw std::runtime_error(error);
        return out;
    });
}

} // namespace cellshard::python_bindings
