#pragma once

inline bool append_metadata_string(common::metadata_table *table, const std::string &value) {
    return common::append(&table->field_values, value.c_str(), value.size()) != 0;
}

inline bool load_observation_columns(const char *path,
                                     std::vector<observation_column> *columns,
                                     unsigned long *rows_out,
                                     std::string *error) {
    return load_dataframe_columns(path, "/obs", columns, rows_out, error);
}

inline bool load_dataframe_columns(const char *path,
                                   const char *group_path,
                                   std::vector<observation_column> *columns,
                                   unsigned long *rows_out,
                                   std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t axis = (hid_t) -1;
    common::text_column axis_index;
    std::vector<std::string> names;
    unsigned long rows = 0ul;
    bool ok = false;

    if (columns == nullptr || group_path == nullptr) return false;
    columns->clear();
    common::init(&axis_index);
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open h5ad file");
        goto done;
    }
    if (!load_dataframe_index(file, group_path, &axis_index, error)) goto done;
    rows = (unsigned long) axis_index.count;
    axis = H5Gopen2(file, group_path, H5P_DEFAULT);
    if (axis < 0) {
        set_error(error, "failed to open h5ad dataframe group");
        goto done;
    }
    if (!list_child_names(axis, &names)) {
        set_error(error, "failed to enumerate h5ad dataframe columns");
        goto done;
    }
    for (const std::string &name : names) {
        observation_column column;
        hid_t dset = (hid_t) -1;
        hid_t sub = (hid_t) -1;
        hid_t type = (hid_t) -1;
        hsize_t length = 0u;
        std::string encoding;

        if (!name.empty() && name[0] == '_') continue;
        init(&column);
        column.name = name;

        if (load_dataframe_text_column(axis, name, rows, &column.text_values)) {
            column.type = observation_column_text;
            columns->push_back(column);
            continue;
        }

        dset = open_optional_dataset(axis, name.c_str());
        if (dset >= 0) {
            if (!dataset_length(dset, &length) || (unsigned long) length != rows) {
                H5Dclose(dset);
                continue;
            }
            type = H5Dget_type(dset);
            if (type >= 0) {
                const H5T_class_t klass = H5Tget_class(type);
                if (klass == H5T_FLOAT || klass == H5T_INTEGER) {
                    H5Dclose(dset);
                    dset = (hid_t) -1;
                    if (klass == H5T_INTEGER) {
                        std::vector<std::uint8_t> values_u8;
                        std::vector<long long> values_i64;
                        if (read_dataset_i64_range(axis, name.c_str(), 0u, (std::uint64_t) rows, &values_i64)) {
                            bool is_bool = true;
                            for (long long value : values_i64) {
                                if (value != 0 && value != 1) {
                                    is_bool = false;
                                    break;
                                }
                            }
                            if (is_bool) {
                                column.type = observation_column_uint8;
                                column.uint8_values.resize(values_i64.size(), (std::uint8_t) 0u);
                                for (std::size_t i = 0; i < values_i64.size(); ++i) {
                                    column.uint8_values[i] = (std::uint8_t) (values_i64[i] != 0 ? 1u : 0u);
                                }
                                columns->push_back(column);
                                if (type >= 0) H5Tclose(type);
                                continue;
                            }
                        }
                    }
                    column.type = observation_column_float32;
                    if (!read_dataset_float32(axis, name.c_str(), &column.float32_values)) {
                        if (type >= 0) H5Tclose(type);
                        clear(&column);
                        continue;
                    }
                    columns->push_back(column);
                    if (type >= 0) H5Tclose(type);
                    continue;
                }
            }
        }

        if (type >= 0) H5Tclose(type);
        if (dset >= 0) H5Dclose(dset);
        sub = open_optional_group(axis, name.c_str());
        if (sub >= 0) {
            if (read_attr_string(sub, "encoding-type", &encoding) && encoding == "categorical"
                && load_categorical_text_column(sub, rows, &column.text_values)) {
                column.type = observation_column_text;
                columns->push_back(column);
            }
            H5Gclose(sub);
        }
        clear(&column);
    }
    if (rows_out != nullptr) *rows_out = rows;
    ok = true;

done:
    common::clear(&axis_index);
    if (axis >= 0) H5Gclose(axis);
    if (file >= 0) H5Fclose(file);
    if (!ok) {
        for (observation_column &column : *columns) clear(&column);
        columns->clear();
    }
    return ok;
}

inline bool build_metadata_table_from_observation_columns(const std::vector<observation_column> &columns,
                                                          unsigned long rows,
                                                          common::metadata_table *table) {
    if (table == nullptr) return false;
    common::clear(table);
    common::init(table);
    if (columns.empty()) return true;

    if (!common::reserve_rows(table, (unsigned int) rows)) return false;
    for (const observation_column &column : columns) {
        if (!common::append(&table->column_names, column.name.c_str(), column.name.size())) return false;
    }
    table->num_cols = (unsigned int) columns.size();
    table->row_offsets[0] = 0u;

    for (unsigned long row = 0; row < rows; ++row) {
        for (const observation_column &column : columns) {
            if (column.type == observation_column_text) {
                const char *value = row < (unsigned long) column.text_values.count
                    ? common::get(&column.text_values, (unsigned int) row)
                    : "";
                if (!common::append(&table->field_values, value != nullptr ? value : "", std::strlen(value != nullptr ? value : ""))) return false;
            } else if (column.type == observation_column_float32) {
                char buffer[64];
                const float value = row < column.float32_values.size()
                    ? column.float32_values[(std::size_t) row]
                    : std::numeric_limits<float>::quiet_NaN();
                if (std::isfinite(value)) std::snprintf(buffer, sizeof(buffer), "%.9g", value);
                else buffer[0] = '\0';
                if (!common::append(&table->field_values, buffer, std::strlen(buffer))) return false;
            } else if (column.type == observation_column_uint8) {
                const char buffer[2] = { row < column.uint8_values.size() && column.uint8_values[(std::size_t) row] != 0 ? '1' : '0', '\0' };
                if (!common::append(&table->field_values, buffer, 1u)) return false;
            } else {
                if (!common::append(&table->field_values, "", 0u)) return false;
            }
        }
        ++table->num_rows;
        table->row_offsets[table->num_rows] = table->field_values.count;
    }
    return true;
}

inline bool load_feature_columns(const char *path,
                                 const char *matrix_source,
                                 std::vector<observation_column> *columns,
                                 unsigned long *rows_out,
                                 std::string *error) {
    selected_matrix_info info;
    if (!probe_selected_matrix(path, matrix_source, &info, error)) return false;
    return load_dataframe_columns(path, info.var_path.c_str(), columns, rows_out, error);
}

inline bool choose_feature_text_column(hid_t file,
                                       const std::string &var_path,
                                       unsigned long expected_rows,
                                       const char *const *candidates,
                                       std::size_t candidate_count,
                                       common::text_column *out) {
    hid_t group = (hid_t) -1;
    bool ok = false;

    if (out == nullptr) return false;
    common::clear(out);
    common::init(out);
    group = H5Gopen2(file, var_path.c_str(), H5P_DEFAULT);
    if (group < 0) goto done;
    for (std::size_t i = 0; i < candidate_count; ++i) {
        if (load_dataframe_text_column(group, candidates[i], expected_rows, out)) {
            ok = true;
            break;
        }
    }

done:
    if (group >= 0) H5Gclose(group);
    if (!ok) {
        common::clear(out);
        common::init(out);
    }
    return ok;
}

inline bool load_feature_table(const char *path,
                               const char *matrix_source,
                               common::feature_table *out,
                               std::string *error) {
    hid_t file = (hid_t) -1;
    selected_matrix_info info;
    common::text_column ids;
    common::text_column names;
    common::text_column types;
    bool ok = false;
    static constexpr const char *name_candidates[] = {
        "gene_short_name", "feature_name", "gene_name", "gene_symbol", "symbol", "name"
    };
    static constexpr const char *type_candidates[] = {
        "gene_type", "feature_type", "feature_types", "gene_biotype", "biotype", "type"
    };

    if (out == nullptr) return false;
    common::init(&ids);
    common::init(&names);
    common::init(&types);
    common::clear(out);
    common::init(out);
    if (!probe_selected_matrix(path, matrix_source, &info, error)) goto done;

    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open h5ad file");
        goto done;
    }
    if (!load_dataframe_index(file, info.var_path, &ids, error)) goto done;
    if ((unsigned long) ids.count != info.cols) {
        set_error(error, "h5ad var index length does not match selected matrix columns");
        goto done;
    }
    if (!choose_feature_text_column(file, info.var_path, info.cols, name_candidates, sizeof(name_candidates) / sizeof(name_candidates[0]), &names)) {
        for (unsigned int i = 0; i < ids.count; ++i) {
            const char *value = common::get(&ids, i);
            if (!common::append(&names, value != nullptr ? value : "", std::strlen(value != nullptr ? value : ""))) goto done;
        }
    }
    if (!choose_feature_text_column(file, info.var_path, info.cols, type_candidates, sizeof(type_candidates) / sizeof(type_candidates[0]), &types)) {
        for (unsigned int i = 0; i < ids.count; ++i) {
            if (!common::append(&types, "gene", 4u)) goto done;
        }
    }
    if (names.count != ids.count || types.count != ids.count) goto done;
    out->ids = ids;
    out->names = names;
    out->types = types;
    common::init(&ids);
    common::init(&names);
    common::init(&types);
    ok = true;

done:
    common::clear(&ids);
    common::clear(&names);
    common::clear(&types);
    if (file >= 0) H5Fclose(file);
    if (!ok) {
        common::clear(out);
        common::init(out);
    }
    return ok;
}

inline bool load_barcodes(const char *path, common::barcode_table *out, std::string *error) {
    hid_t file = (hid_t) -1;
    bool ok = false;

    if (out == nullptr) return false;
    common::clear(out);
    common::init(out);
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open h5ad file");
        goto done;
    }
    ok = load_dataframe_index(file, "/obs", &out->values, error);

done:
    if (file >= 0) H5Fclose(file);
    if (!ok) {
        common::clear(out);
        common::init(out);
    }
    return ok;
}
