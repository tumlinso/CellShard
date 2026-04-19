#include "internal/module_support.hh"

namespace cellshard::python_bindings {
void bind_dataset_export_types(py::module_ &m);
void bind_dataset_handles(py::module_ &m);
} // namespace cellshard::python_bindings

PYBIND11_MODULE(_cellshard, m) {
    m.doc() = "Optional CellShard Python bindings for owner/client inspection and CSR retrieval.";
    ::cellshard::python_bindings::bind_dataset_export_types(m);
    ::cellshard::python_bindings::bind_dataset_handles(m);
}
