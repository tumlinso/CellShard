#pragma once

#include "series_export.hh"

#include <string>

namespace cellshard::exporting {

bool write_h5ad_with_python(const anndata_export &input,
                            const char *path,
                            std::string *error = nullptr);

bool write_series_file_to_h5ad_with_python(const char *series_path,
                                           const char *path,
                                           std::string *error = nullptr);

} // namespace cellshard::exporting
