#include "internal/module_support.hh"

namespace cellshard::python_bindings {

void bind_dataset_handles(py::module_ &m) {
    py::class_<dataset_file_handle>(m, "DatasetFile")
        .def(py::init([](const std::string &path) { return dataset_file_handle{path}; }))
        .def_readonly("path", &dataset_file_handle::path)
        .def("summary", &dataset_file_handle::summary)
        .def("snapshot", &dataset_file_handle::snapshot)
        .def("observation_metadata", &dataset_file_handle::observation_metadata)
        .def("feature_metadata", &dataset_file_handle::feature_metadata)
        .def("dataset_attributes", &dataset_file_handle::dataset_attributes)
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
        .def("stage_append_runtime_service", &dataset_owner_handle::stage_append_runtime_service)
        .def("publish_runtime_service_cutover", &dataset_owner_handle::publish_runtime_service_cutover)
        .def("validate_request", &dataset_owner_handle::validate_request)
        .def("observation_metadata", &dataset_owner_handle::observation_metadata)
        .def("feature_metadata", &dataset_owner_handle::feature_metadata)
        .def("dataset_attributes", &dataset_owner_handle::dataset_attributes)
        .def("describe_pack_delivery", &dataset_owner_handle::describe_pack_delivery)
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
        .def("observation_metadata", [](const dataset_client_handle &self) {
            return self.owner->observation_metadata(self.request_ref);
        })
        .def("feature_metadata", [](const dataset_client_handle &self) {
            return self.owner->feature_metadata(self.request_ref);
        })
        .def("dataset_attributes", [](const dataset_client_handle &self) {
            return self.owner->dataset_attributes(self.request_ref);
        })
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
    m.def("write_h5ad", [](const std::string &dataset_path, const std::string &output_path) {
        std::string error;
        if (!cse::write_dataset_file_to_h5ad_with_python(dataset_path.c_str(), output_path.c_str(), &error)) {
            throw std::runtime_error(error);
        }
    });
}

} // namespace cellshard::python_bindings
