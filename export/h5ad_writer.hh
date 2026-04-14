#pragma once

#include "dataset_export.hh"

#include <string>

namespace cellshard::exporting {

bool write_h5ad_with_python(const anndata_export &input,
                            const char *path,
                            std::string *error = nullptr);

bool write_dataset_file_to_h5ad_with_python(const char *dataset_path,
                                           const char *path,
                                           std::string *error = nullptr);

} // namespace cellshard::exporting
