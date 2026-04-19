#include "execution_internal.hh"

inline int validate_common_layout_for_create(const char *filename,
                                             const dataset_layout_view *layout,
                                             std::uint64_t idx_limit) {
    if (filename == 0 || layout == 0) return 0;
    if (layout->partition_rows == 0 || layout->partition_nnz == 0 || layout->partition_aux == 0
        || layout->partition_row_offsets == 0 || layout->partition_dataset_ids == 0
        || layout->partition_codec_ids == 0 || layout->shard_offsets == 0) {
        return 0;
    }
    if (layout->cols > idx_limit) {
        std::fprintf(stderr,
                     "cellshard: dataset column count exceeds the current u32 execution limit while writing %s (cols=%llu, limit=%llu)\n",
                     filename,
                     (unsigned long long) layout->cols,
                     (unsigned long long) idx_limit);
        return 0;
    }
    return 1;
}

inline int write_common_dataset_file_scaffold(hid_t *file,
                                              hid_t *matrix,
                                              hid_t *dsets,
                                              hid_t *prov,
                                              hid_t *codecs,
                                              hid_t *payload_root,
                                              const char *filename,
                                              const dataset_layout_view *layout,
                                              const dataset_dataset_table_view *datasets,
                                              const char *matrix_format,
                                              const char *payload_layout) {
    if (file == 0 || matrix == 0 || dsets == 0 || prov == 0 || codecs == 0 || payload_root == 0
        || filename == 0 || layout == 0 || matrix_format == 0 || payload_layout == 0) {
        return 0;
    }

    *file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (*file < 0) return 0;
    if (!write_attr_string(*file, "cellshard_magic", dataset_magic)
        || !write_attr_u32(*file, "schema_version", dataset_h5_schema_version)
        || !write_attr_string(*file, "matrix_format", matrix_format)
        || !write_attr_string(*file, "payload_layout", payload_layout)
        || !write_attr_u64(*file, "rows", layout->rows)
        || !write_attr_u64(*file, "cols", layout->cols)
        || !write_attr_u64(*file, "nnz", layout->nnz)
        || !write_attr_u64(*file, "num_partitions", layout->num_partitions)
        || !write_attr_u64(*file, "num_shards", layout->num_shards)
        || !write_attr_u64(*file, "num_codecs", layout->num_codecs)
        || !write_attr_u64(*file, "num_datasets", datasets != 0 ? datasets->count : 0u)) {
        return 0;
    }

    *matrix = create_group(*file, matrix_group);
    *dsets = create_group(*file, datasets_group);
    *prov = create_group(*file, provenance_group);
    *codecs = create_group(*file, codecs_group);
    *payload_root = create_group(*file, payload_group);
    return *matrix >= 0 && *dsets >= 0 && *prov >= 0 && *codecs >= 0 && *payload_root >= 0;
}

inline int write_common_matrix_tables(hid_t matrix,
                                      const dataset_layout_view *layout,
                                      const std::uint64_t *partition_aux,
                                      const std::uint32_t *partition_axes) {
    if (matrix < 0 || layout == 0) return 0;
    return write_dataset_1d(matrix, "partition_rows", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, layout->partition_rows)
        && write_dataset_1d(matrix, "partition_nnz", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, layout->partition_nnz)
        && write_dataset_1d(matrix, "partition_axes", H5T_NATIVE_UINT32, (hsize_t) layout->num_partitions, partition_axes)
        && write_dataset_1d(matrix, "partition_aux", H5T_NATIVE_UINT64, (hsize_t) layout->num_partitions, partition_aux)
        && write_dataset_1d(matrix,
                            "partition_row_offsets",
                            H5T_NATIVE_UINT64,
                            (hsize_t) layout->num_partitions + 1u,
                            layout->partition_row_offsets)
        && write_dataset_1d(matrix,
                            "partition_dataset_ids",
                            H5T_NATIVE_UINT32,
                            (hsize_t) layout->num_partitions,
                            layout->partition_dataset_ids)
        && write_dataset_1d(matrix,
                            "partition_codec_ids",
                            H5T_NATIVE_UINT32,
                            (hsize_t) layout->num_partitions,
                            layout->partition_codec_ids)
        && write_dataset_1d(matrix,
                            "shard_offsets",
                            H5T_NATIVE_UINT64,
                            (hsize_t) layout->num_shards + 1u,
                            layout->shard_offsets);
}

inline int write_dataset_table_group(hid_t dsets, const dataset_dataset_table_view *datasets) {
    if (dsets < 0) return 0;
    if (datasets == 0) return 1;
    return write_text_column(dsets, "dataset_ids", &datasets->dataset_ids)
        && write_text_column(dsets, "matrix_paths", &datasets->matrix_paths)
        && write_text_column(dsets, "feature_paths", &datasets->feature_paths)
        && write_text_column(dsets, "barcode_paths", &datasets->barcode_paths)
        && write_text_column(dsets, "metadata_paths", &datasets->metadata_paths)
        && write_dataset_1d(dsets, "formats", H5T_NATIVE_UINT32, (hsize_t) datasets->count, datasets->formats)
        && write_dataset_1d(dsets, "row_begin", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->row_begin)
        && write_dataset_1d(dsets, "row_end", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->row_end)
        && write_dataset_1d(dsets, "rows", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->rows)
        && write_dataset_1d(dsets, "cols", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->cols)
        && write_dataset_1d(dsets, "nnz", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->nnz);
}

inline int write_provenance_group(hid_t prov,
                                  const dataset_layout_view *layout,
                                  const dataset_dataset_table_view *datasets,
                                  const dataset_provenance_view *provenance) {
    if (prov < 0 || layout == 0) return 0;
    if (provenance == 0) return 1;
    if (!write_text_column(prov, "global_barcodes", &provenance->global_barcodes)
        || !write_dataset_1d(prov, "cell_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->rows, provenance->cell_dataset_ids)
        || !write_dataset_1d(prov, "cell_local_indices", H5T_NATIVE_UINT64, (hsize_t) layout->rows, provenance->cell_local_indices)
        || !write_text_column(prov, "feature_ids", &provenance->feature_ids)
        || !write_text_column(prov, "feature_names", &provenance->feature_names)
        || !write_text_column(prov, "feature_types", &provenance->feature_types)
        || !write_dataset_1d(prov, "feature_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->cols, provenance->feature_dataset_ids)
        || !write_dataset_1d(prov, "feature_local_indices", H5T_NATIVE_UINT64, (hsize_t) layout->cols, provenance->feature_local_indices)) {
        return 0;
    }
    if (datasets == 0) return 1;
    return write_dataset_1d(prov,
                            "dataset_feature_offsets",
                            H5T_NATIVE_UINT64,
                            (hsize_t) datasets->count + 1u,
                            provenance->dataset_feature_offsets)
        && write_dataset_1d(prov,
                            "dataset_feature_to_global",
                            H5T_NATIVE_UINT32,
                            (hsize_t) provenance->dataset_feature_offsets[datasets->count],
                            provenance->dataset_feature_to_global);
}

inline int write_codec_table_group(hid_t codecs, const dataset_layout_view *layout) {
    if (codecs < 0 || layout == 0) return 0;
    if (layout->num_codecs == 0u) return 1;

    std::vector<std::uint32_t> codec_id((std::size_t) layout->num_codecs, 0u);
    std::vector<std::uint32_t> family((std::size_t) layout->num_codecs, 0u);
    std::vector<std::uint32_t> value_code((std::size_t) layout->num_codecs, 0u);
    std::vector<std::uint32_t> scale_value_code((std::size_t) layout->num_codecs, 0u);
    std::vector<std::uint32_t> bits((std::size_t) layout->num_codecs, 0u);
    std::vector<std::uint32_t> flags((std::size_t) layout->num_codecs, 0u);

    for (std::uint32_t i = 0u; i < layout->num_codecs; ++i) {
        codec_id[(std::size_t) i] = layout->codecs[i].codec_id;
        family[(std::size_t) i] = layout->codecs[i].family;
        value_code[(std::size_t) i] = layout->codecs[i].value_code;
        scale_value_code[(std::size_t) i] = layout->codecs[i].scale_value_code;
        bits[(std::size_t) i] = layout->codecs[i].bits;
        flags[(std::size_t) i] = layout->codecs[i].flags;
    }

    return write_dataset_1d(codecs, "codec_id", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, codec_id.data())
        && write_dataset_1d(codecs, "family", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, family.data())
        && write_dataset_1d(codecs, "value_code", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, value_code.data())
        && write_dataset_1d(codecs, "scale_value_code", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, scale_value_code.data())
        && write_dataset_1d(codecs, "bits", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, bits.data())
        && write_dataset_1d(codecs, "flags", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, flags.data());
}

inline int create_dataset_blocked_ell_h5_impl(const char *filename,
                                              const dataset_layout_view *layout,
                                              const dataset_dataset_table_view *datasets,
                                              const dataset_provenance_view *provenance,
                                              int optimized_payload_only) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t dsets = (hid_t) -1;
    hid_t prov = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    hid_t payload_root = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    std::uint64_t total_block_idx = 0u;
    std::uint64_t total_values = 0u;
    const std::uint64_t dim_limit = local_dim_limit();
    const std::uint64_t nnz_limit = local_nnz_limit();
    const std::uint64_t idx_limit = local_index_limit();
    std::vector<std::uint64_t> partition_aux;
    std::vector<std::uint32_t> partition_axes;
    std::vector<std::uint64_t> partition_block_idx_offsets;
    std::vector<std::uint64_t> partition_value_offsets;
    std::vector<std::uint64_t> shard_block_idx_offsets;
    std::vector<std::uint64_t> shard_value_offsets;
    int ok = 0;

    if (!validate_common_layout_for_create(filename, layout, idx_limit)) return 0;

    partition_aux.assign((std::size_t) layout->num_partitions, 0u);
    partition_axes.assign((std::size_t) layout->num_partitions, 0u);
    if (!optimized_payload_only) {
        partition_block_idx_offsets.assign((std::size_t) layout->num_partitions, 0u);
        partition_value_offsets.assign((std::size_t) layout->num_partitions, 0u);
        shard_block_idx_offsets.assign((std::size_t) layout->num_shards + 1u, 0u);
        shard_value_offsets.assign((std::size_t) layout->num_shards + 1u, 0u);
    }

    for (std::uint32_t i = 0u; i < layout->num_partitions; ++i) {
        const std::uint64_t part_block_idx =
            (std::uint64_t) blocked_ell_part_block_index_count(layout->partition_rows[i], layout->partition_aux[i]);
        const std::uint64_t part_values =
            (std::uint64_t) blocked_ell_part_value_count(layout->partition_rows[i], layout->partition_aux[i]);
        if (layout->partition_rows[i] > dim_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "rows", layout->partition_rows[i], dim_limit);
            goto done;
        }
        if (layout->partition_nnz[i] > nnz_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "nnz", layout->partition_nnz[i], nnz_limit);
            goto done;
        }
        if (part_block_idx > idx_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "block_col_idx_count", part_block_idx, idx_limit);
            goto done;
        }
        if (part_values > nnz_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "value_count", part_values, nnz_limit);
            goto done;
        }
        partition_aux[(std::size_t) i] = layout->partition_aux[i];
        partition_axes[(std::size_t) i] = layout->partition_axes != 0 ? layout->partition_axes[i] : 0u;
        if (!optimized_payload_only) {
            partition_block_idx_offsets[(std::size_t) i] = total_block_idx;
            partition_value_offsets[(std::size_t) i] = total_values;
            total_block_idx += part_block_idx;
            total_values += part_values;
        }
    }

    if (!write_common_dataset_file_scaffold(&file,
                                            &matrix,
                                            &dsets,
                                            &prov,
                                            &codecs,
                                            &payload_root,
                                            filename,
                                            layout,
                                            datasets,
                                            "blocked_ell",
                                            optimized_payload_only ? payload_layout_optimized_blocked_ell
                                                                   : payload_layout_shard_packed)) {
        goto done;
    }
    payload = (!optimized_payload_only && payload_root >= 0) ? create_group(payload_root, "blocked_ell") : (hid_t) -1;
    if (!optimized_payload_only && payload < 0) goto done;

    if (!write_common_matrix_tables(matrix, layout, partition_aux.data(), partition_axes.data())
        || !write_dataset_table_group(dsets, datasets)
        || !write_provenance_group(prov, layout, datasets, provenance)
        || !write_codec_table_group(codecs, layout)) {
        goto done;
    }

    if (!optimized_payload_only && layout->num_shards != 0u) {
        unsigned long part_begin = 0ul;
        for (std::uint32_t shard_i = 0u; shard_i < layout->num_shards; ++shard_i) {
            const std::uint64_t row_begin = layout->shard_offsets[shard_i];
            const std::uint64_t row_end = layout->shard_offsets[shard_i + 1u];
            unsigned long part_end = part_begin;
            std::uint64_t shard_nnz = 0u;
            while (part_begin < layout->num_partitions && layout->partition_row_offsets[part_begin] < row_begin) ++part_begin;
            part_end = part_begin;
            while (part_end < layout->num_partitions && layout->partition_row_offsets[part_end + 1u] <= row_end) {
                shard_nnz += layout->partition_nnz[part_end];
                ++part_end;
            }
            shard_block_idx_offsets[(std::size_t) shard_i] =
                part_begin < layout->num_partitions ? partition_block_idx_offsets[(std::size_t) part_begin] : total_block_idx;
            shard_value_offsets[(std::size_t) shard_i] =
                part_begin < layout->num_partitions ? partition_value_offsets[(std::size_t) part_begin] : total_values;
            if (part_end == layout->num_partitions) {
                shard_block_idx_offsets[(std::size_t) shard_i + 1u] = total_block_idx;
                shard_value_offsets[(std::size_t) shard_i + 1u] = total_values;
            } else {
                shard_block_idx_offsets[(std::size_t) shard_i + 1u] = partition_block_idx_offsets[(std::size_t) part_end];
                shard_value_offsets[(std::size_t) shard_i + 1u] = partition_value_offsets[(std::size_t) part_end];
            }
            if (row_end - row_begin > dim_limit) warn_dataset_u32_limit(filename, "shard", shard_i, "rows", row_end - row_begin, dim_limit);
            if (shard_nnz > nnz_limit) warn_dataset_u32_limit(filename, "shard", shard_i, "nnz", shard_nnz, nnz_limit);
            if (shard_block_idx_offsets[(std::size_t) shard_i + 1u] - shard_block_idx_offsets[(std::size_t) shard_i] > idx_limit) {
                warn_dataset_u32_limit(filename,
                                       "shard",
                                       shard_i,
                                       "block_col_idx_count",
                                       shard_block_idx_offsets[(std::size_t) shard_i + 1u]
                                           - shard_block_idx_offsets[(std::size_t) shard_i],
                                       idx_limit);
            }
            if (shard_value_offsets[(std::size_t) shard_i + 1u] - shard_value_offsets[(std::size_t) shard_i] > nnz_limit) {
                warn_dataset_u32_limit(filename,
                                       "shard",
                                       shard_i,
                                       "value_count",
                                       shard_value_offsets[(std::size_t) shard_i + 1u]
                                           - shard_value_offsets[(std::size_t) shard_i],
                                       nnz_limit);
            }
            part_begin = part_end;
        }
    }

    if (!optimized_payload_only) {
        if (!write_dataset_1d(payload,
                              "partition_block_idx_offsets",
                              H5T_NATIVE_UINT64,
                              (hsize_t) layout->num_partitions,
                              partition_block_idx_offsets.data())
            || !write_dataset_1d(payload,
                                 "partition_value_offsets",
                                 H5T_NATIVE_UINT64,
                                 (hsize_t) layout->num_partitions,
                                 partition_value_offsets.data())
            || !write_dataset_1d(payload,
                                 "shard_block_idx_offsets",
                                 H5T_NATIVE_UINT64,
                                 (hsize_t) layout->num_shards + 1u,
                                 shard_block_idx_offsets.data())
            || !write_dataset_1d(payload,
                                 "shard_value_offsets",
                                 H5T_NATIVE_UINT64,
                                 (hsize_t) layout->num_shards + 1u,
                                 shard_value_offsets.data())
            || !write_dataset_1d(payload, "block_col_idx", H5T_NATIVE_UINT32, (hsize_t) total_block_idx, 0)
            || !write_dataset_1d(payload, "values", H5T_NATIVE_UINT16, (hsize_t) total_values, 0)) {
            goto done;
        }
    }

    ok = 1;

done:
    if (payload >= 0) H5Gclose(payload);
    if (payload_root >= 0) H5Gclose(payload_root);
    if (codecs >= 0) H5Gclose(codecs);
    if (prov >= 0) H5Gclose(prov);
    if (dsets >= 0) H5Gclose(dsets);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int create_dataset_blocked_ell_h5(const char *filename,
                                  const dataset_layout_view *layout,
                                  const dataset_dataset_table_view *datasets,
                                  const dataset_provenance_view *provenance) {
    return create_dataset_blocked_ell_h5_impl(filename, layout, datasets, provenance, 0);
}

int create_dataset_optimized_blocked_ell_h5(const char *filename,
                                            const dataset_layout_view *layout,
                                            const dataset_dataset_table_view *datasets,
                                            const dataset_provenance_view *provenance) {
    return create_dataset_blocked_ell_h5_impl(filename, layout, datasets, provenance, 1);
}

int create_dataset_quantized_blocked_ell_h5(const char *filename,
                                            const dataset_layout_view *layout,
                                            const dataset_dataset_table_view *datasets,
                                            const dataset_provenance_view *provenance) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t dsets = (hid_t) -1;
    hid_t prov = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    hid_t payload_root = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    const std::uint64_t dim_limit = local_dim_limit();
    const std::uint64_t nnz_limit = local_nnz_limit();
    const std::uint64_t idx_limit = local_index_limit();
    std::vector<std::uint64_t> partition_aux;
    std::vector<std::uint32_t> partition_axes;
    int ok = 0;

    if (!validate_common_layout_for_create(filename, layout, idx_limit)) return 0;

    partition_aux.assign((std::size_t) layout->num_partitions, 0u);
    partition_axes.assign((std::size_t) layout->num_partitions, 0u);
    for (std::uint32_t i = 0u; i < layout->num_partitions; ++i) {
        const std::uint32_t block_size = sparse::unpack_quantized_blocked_ell_block_size((unsigned long) layout->partition_aux[i]);
        const std::uint32_t bits = sparse::unpack_quantized_blocked_ell_bits((unsigned long) layout->partition_aux[i]);
        const std::uint32_t ell_cols = sparse::unpack_quantized_blocked_ell_cols((unsigned long) layout->partition_aux[i]);
        const std::uint64_t row_blocks = block_size == 0u ? 0u : (layout->partition_rows[i] + block_size - 1u) / block_size;
        const std::uint64_t ell_width = block_size == 0u ? 0u : ell_cols / block_size;
        const std::uint64_t block_idx_count = row_blocks * ell_width;
        const std::uint64_t packed_value_bytes =
            (std::uint64_t) layout->partition_rows[i] * (std::uint64_t) sparse::quantized_blocked_ell_aligned_row_bytes(bits, ell_cols);
        if (layout->partition_rows[i] > dim_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "rows", layout->partition_rows[i], dim_limit);
            goto done;
        }
        if (layout->partition_nnz[i] > nnz_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "nnz", layout->partition_nnz[i], nnz_limit);
            goto done;
        }
        if (block_idx_count > idx_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "block_col_idx_count", block_idx_count, idx_limit);
            goto done;
        }
        if (packed_value_bytes > nnz_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "packed_value_bytes", packed_value_bytes, nnz_limit);
            goto done;
        }
        partition_aux[(std::size_t) i] = layout->partition_aux[i];
        partition_axes[(std::size_t) i] = layout->partition_axes != 0 ? layout->partition_axes[i] : 0u;
    }

    if (!write_common_dataset_file_scaffold(&file,
                                            &matrix,
                                            &dsets,
                                            &prov,
                                            &codecs,
                                            &payload_root,
                                            filename,
                                            layout,
                                            datasets,
                                            "quantized_blocked_ell",
                                            payload_layout_shard_packed)) {
        goto done;
    }
    payload = payload_root >= 0 ? create_group(payload_root, "quantized_blocked_ell") : (hid_t) -1;
    if (payload < 0) goto done;

    if (!write_common_matrix_tables(matrix, layout, partition_aux.data(), partition_axes.data())
        || !write_dataset_table_group(dsets, datasets)
        || !write_provenance_group(prov, layout, datasets, provenance)
        || !write_codec_table_group(codecs, layout)) {
        goto done;
    }

    ok = 1;

done:
    if (payload >= 0) H5Gclose(payload);
    if (payload_root >= 0) H5Gclose(payload_root);
    if (codecs >= 0) H5Gclose(codecs);
    if (prov >= 0) H5Gclose(prov);
    if (dsets >= 0) H5Gclose(dsets);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int create_dataset_sliced_ell_h5(const char *filename,
                                 const dataset_layout_view *layout,
                                 const dataset_dataset_table_view *datasets,
                                 const dataset_provenance_view *provenance) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t dsets = (hid_t) -1;
    hid_t prov = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    hid_t payload_root = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    const std::uint64_t dim_limit = local_dim_limit();
    const std::uint64_t nnz_limit = local_nnz_limit();
    const std::uint64_t idx_limit = local_index_limit();
    std::vector<std::uint64_t> partition_aux;
    std::vector<std::uint32_t> partition_axes;
    int ok = 0;

    if (!validate_common_layout_for_create(filename, layout, idx_limit)) return 0;

    partition_aux.assign((std::size_t) layout->num_partitions, 0u);
    partition_axes.assign((std::size_t) layout->num_partitions, 0u);
    for (std::uint32_t i = 0u; i < layout->num_partitions; ++i) {
        const std::uint64_t total_slots = sparse::unpack_sliced_ell_total_slots((unsigned long) layout->partition_aux[i]);
        if (layout->partition_rows[i] > dim_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "rows", layout->partition_rows[i], dim_limit);
            goto done;
        }
        if (layout->partition_nnz[i] > nnz_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "nnz", layout->partition_nnz[i], nnz_limit);
            goto done;
        }
        if (total_slots > idx_limit) {
            ok = fail_dataset_u32_limit(filename, "part", i, "slot_count", total_slots, idx_limit);
            goto done;
        }
        partition_aux[(std::size_t) i] = layout->partition_aux[i];
        partition_axes[(std::size_t) i] = layout->partition_axes != 0 ? layout->partition_axes[i] : 0u;
    }

    if (!write_common_dataset_file_scaffold(&file,
                                            &matrix,
                                            &dsets,
                                            &prov,
                                            &codecs,
                                            &payload_root,
                                            filename,
                                            layout,
                                            datasets,
                                            "sliced_ell",
                                            payload_layout_optimized_sliced_ell)) {
        goto done;
    }
    payload = payload_root >= 0 ? create_group(payload_root, "sliced_ell") : (hid_t) -1;
    if (payload < 0) goto done;

    if (!write_common_matrix_tables(matrix, layout, partition_aux.data(), partition_axes.data())
        || !write_dataset_table_group(dsets, datasets)
        || !write_provenance_group(prov, layout, datasets, provenance)
        || !write_codec_table_group(codecs, layout)) {
        goto done;
    }

    ok = 1;

done:
    if (payload >= 0) H5Gclose(payload);
    if (payload_root >= 0) H5Gclose(payload_root);
    if (codecs >= 0) H5Gclose(codecs);
    if (prov >= 0) H5Gclose(prov);
    if (dsets >= 0) H5Gclose(dsets);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

} // namespace cellshard
