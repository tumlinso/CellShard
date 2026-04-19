#include "base_internal.hh"

int append_dataset_embedded_metadata_h5(const char *filename,
                                        const dataset_embedded_metadata_view *metadata) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    int ok = 0;

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    root = create_group(file, embedded_metadata_group);
    if (root < 0) goto done;

    if (!write_attr_u32(root, "count", metadata != 0 ? metadata->count : 0u)) goto done;
    if (metadata == 0 || metadata->count == 0u) {
        ok = 1;
        goto done;
    }

    if (!write_dataset_1d(root, "dataset_indices", H5T_NATIVE_UINT32, (hsize_t) metadata->count, metadata->dataset_indices)) goto done;
    if (!write_dataset_1d(root, "global_row_begin", H5T_NATIVE_UINT64, (hsize_t) metadata->count, metadata->global_row_begin)) goto done;
    if (!write_dataset_1d(root, "global_row_end", H5T_NATIVE_UINT64, (hsize_t) metadata->count, metadata->global_row_end)) goto done;

    for (std::uint32_t i = 0; i < metadata->count; ++i) {
        char name[64];
        hid_t table = (hid_t) -1;
        const dataset_metadata_table_view *view = metadata->tables + i;
        if (std::snprintf(name, sizeof(name), "table_%u", i) <= 0) goto done;
        table = create_group(root, name);
        if (table < 0) goto done;
        if (!write_attr_u32(table, "rows", view->rows) || !write_attr_u32(table, "cols", view->cols)) {
            H5Gclose(table);
            goto done;
        }
        if (!write_text_column(table, "column_names", &view->column_names)
            || !write_text_column(table, "field_values", &view->field_values)
            || !write_dataset_1d(table, "row_offsets", H5T_NATIVE_UINT32, (hsize_t) view->rows + 1u, view->row_offsets)) {
            H5Gclose(table);
            goto done;
        }
        H5Gclose(table);
    }

    ok = 1;

done:
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

inline int append_dataset_annotation_group_h5(hid_t file,
                                              const char *group_path,
                                              std::uint64_t extent,
                                              std::uint32_t cols,
                                              const dataset_observation_metadata_column_view *columns) {
    hid_t root = (hid_t) -1;
    int ok = 0;

    if (file < 0 || group_path == 0) return 0;
    root = create_group(file, group_path);
    if (root < 0) goto done;

    if (!write_attr_u64(root, "extent", extent)
        || !write_attr_u32(root, "cols", cols)) goto done;
    if (cols == 0u) {
        ok = 1;
        goto done;
    }

    for (std::uint32_t i = 0; i < cols; ++i) {
        char name[64];
        hid_t column = (hid_t) -1;
        const dataset_observation_metadata_column_view *view = columns + i;
        const hsize_t values = (hsize_t) extent;

        if (view == 0 || view->name == 0) goto done;
        if (std::snprintf(name, sizeof(name), "column_%u", i) <= 0) goto done;
        column = create_group(root, name);
        if (column < 0) goto done;
        if (!write_attr_string(column, "name", view->name)
            || !write_attr_u32(column, "type", view->type)) {
            H5Gclose(column);
            goto done;
        }

        if (view->type == dataset_observation_metadata_type_text) {
            if (view->text_values.count != extent
                || !write_text_column(column, "values", &view->text_values)) {
                H5Gclose(column);
                goto done;
            }
        } else if (view->type == dataset_observation_metadata_type_float32) {
            if ((values != 0u && view->float32_values == 0)
                || !write_dataset_1d(column, "values", H5T_NATIVE_FLOAT, values, view->float32_values)) {
                H5Gclose(column);
                goto done;
            }
        } else if (view->type == dataset_observation_metadata_type_uint8) {
            if ((values != 0u && view->uint8_values == 0)
                || !write_dataset_1d(column, "values", H5T_NATIVE_UINT8, values, view->uint8_values)) {
                H5Gclose(column);
                goto done;
            }
        } else {
            H5Gclose(column);
            goto done;
        }

        H5Gclose(column);
    }

    ok = 1;

done:
    if (root >= 0) H5Gclose(root);
    return ok;
}

int append_dataset_observation_annotations_h5(const char *filename,
                                              const dataset_annotation_view *metadata) {
    hid_t file = (hid_t) -1;
    int ok = 0;

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    ok = append_dataset_annotation_group_h5(file,
                                            observation_metadata_group,
                                            metadata != 0 ? metadata->extent : 0u,
                                            metadata != 0 ? metadata->cols : 0u,
                                            metadata != 0 ? metadata->columns : nullptr);

done:
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_observation_metadata_h5(const char *filename,
                                           const dataset_observation_metadata_view *metadata) {
    const dataset_annotation_view adapted = {
        metadata != 0 ? metadata->rows : 0u,
        metadata != 0 ? metadata->cols : 0u,
        metadata != 0 ? metadata->columns : nullptr
    };
    return append_dataset_observation_annotations_h5(filename, metadata != 0 ? &adapted : nullptr);
}

int append_dataset_feature_metadata_h5(const char *filename,
                                       const dataset_feature_metadata_view *metadata) {
    hid_t file = (hid_t) -1;
    int ok = 0;

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    ok = append_dataset_annotation_group_h5(file,
                                            feature_metadata_group,
                                            metadata != 0 ? metadata->cols : 0u,
                                            metadata != 0 ? metadata->annotation_count : 0u,
                                            metadata != 0 ? metadata->annotations : nullptr);

done:
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_user_attributes_h5(const char *filename,
                                      const dataset_user_attribute_view *attributes) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    int ok = 0;

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    root = create_group(file, user_attributes_group);
    if (root < 0) goto done;
    if (!write_attr_u32(root, "count", attributes != 0 ? attributes->count : 0u)) goto done;
    if (attributes == 0 || attributes->count == 0u) {
        ok = 1;
        goto done;
    }
    if (attributes->keys.count != attributes->count
        || attributes->values.count != attributes->count
        || !write_text_column(root, "keys", &attributes->keys)
        || !write_text_column(root, "values", &attributes->values)) goto done;
    ok = 1;

done:
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_browse_cache_h5(const char *filename,
                                   const dataset_browse_cache_view *browse) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    int ok = 0;
    const hsize_t selected = (hsize_t) (browse != 0 ? browse->selected_feature_count : 0u);

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    root = create_group(file, browse_group);
    if (root < 0) goto done;

    if (!write_attr_u32(root, "selected_feature_count", browse != 0 ? browse->selected_feature_count : 0u)) goto done;
    if (!write_attr_u32(root, "dataset_count", browse != 0 ? browse->dataset_count : 0u)) goto done;
    if (!write_attr_u32(root, "shard_count", browse != 0 ? browse->shard_count : 0u)) goto done;
    if (!write_attr_u32(root, "partition_count", browse != 0 ? browse->partition_count : 0u)) goto done;
    if (!write_attr_u32(root, "sample_rows_per_partition", browse != 0 ? browse->sample_rows_per_partition : 0u)) goto done;

    if (browse == 0 || browse->selected_feature_count == 0u) {
        ok = 1;
        goto done;
    }

    if (!write_dataset_1d(root, "selected_feature_indices", H5T_NATIVE_UINT32, selected, browse->selected_feature_indices)) goto done;
    if (!write_dataset_1d(root, "gene_sum", H5T_NATIVE_FLOAT, selected, browse->gene_sum)) goto done;
    if (!write_dataset_1d(root, "gene_detected", H5T_NATIVE_FLOAT, selected, browse->gene_detected)) goto done;
    if (!write_dataset_1d(root, "gene_sq_sum", H5T_NATIVE_FLOAT, selected, browse->gene_sq_sum)) goto done;

    if (browse->dataset_count != 0u
        && !write_dataset_1d(root,
                             "dataset_feature_mean",
                             H5T_NATIVE_FLOAT,
                             (hsize_t) browse->dataset_count * selected,
                             browse->dataset_feature_mean)) goto done;

    if (browse->shard_count != 0u
        && !write_dataset_1d(root,
                             "shard_feature_mean",
                             H5T_NATIVE_FLOAT,
                             (hsize_t) browse->shard_count * selected,
                             browse->shard_feature_mean)) goto done;

    if (browse->partition_count != 0u) {
        const hsize_t row_count = (hsize_t) browse->partition_count * (hsize_t) browse->sample_rows_per_partition;
        const hsize_t value_count = row_count * selected;
        if (!write_dataset_1d(root,
                              "partition_sample_row_offsets",
                              H5T_NATIVE_UINT32,
                              (hsize_t) browse->partition_count + 1u,
                              browse->partition_sample_row_offsets)) goto done;
        if (!write_dataset_1d(root,
                              "partition_sample_global_rows",
                              H5T_NATIVE_UINT64,
                              row_count,
                              browse->partition_sample_global_rows)) goto done;
        if (!write_dataset_1d(root,
                              "partition_sample_values",
                              H5T_NATIVE_FLOAT,
                              value_count,
                              browse->partition_sample_values)) goto done;
    }

    ok = 1;

done:
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_preprocess_h5(const char *filename,
                                 const dataset_preprocess_view *preprocess) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    hid_t cell_qc = (hid_t) -1;
    hid_t gene_qc = (hid_t) -1;
    int ok = 0;
    const char *assay = (preprocess != 0 && preprocess->assay != 0) ? preprocess->assay : "";
    const char *matrix_orientation = (preprocess != 0 && preprocess->matrix_orientation != 0) ? preprocess->matrix_orientation : "";
    const char *matrix_state = (preprocess != 0 && preprocess->matrix_state != 0) ? preprocess->matrix_state : "";
    const char *pipeline_scope = (preprocess != 0 && preprocess->pipeline_scope != 0) ? preprocess->pipeline_scope : "";
    const char *raw_matrix_name = (preprocess != 0 && preprocess->raw_matrix_name != 0) ? preprocess->raw_matrix_name : "";
    const char *active_matrix_name = (preprocess != 0 && preprocess->active_matrix_name != 0) ? preprocess->active_matrix_name : "";
    const char *feature_namespace = (preprocess != 0 && preprocess->feature_namespace != 0) ? preprocess->feature_namespace : "";
    const char *mito_prefix = (preprocess != 0 && preprocess->mito_prefix != 0) ? preprocess->mito_prefix : "";
    const hsize_t rows = (hsize_t) (preprocess != 0 ? preprocess->rows : 0u);
    const hsize_t cols = (hsize_t) (preprocess != 0 ? preprocess->cols : 0u);

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    root = create_group(file, preprocess_group);
    if (root < 0) goto done;

    if (!write_attr_string(root, "assay", assay)
        || !write_attr_string(root, "matrix_orientation", matrix_orientation)
        || !write_attr_string(root, "matrix_state", matrix_state)
        || !write_attr_string(root, "pipeline_scope", pipeline_scope)
        || !write_attr_string(root, "raw_matrix_name", raw_matrix_name)
        || !write_attr_string(root, "active_matrix_name", active_matrix_name)
        || !write_attr_string(root, "feature_namespace", feature_namespace)
        || !write_attr_string(root, "mito_prefix", mito_prefix)
        || !write_attr_u32(root, "raw_counts_available", preprocess != 0 ? preprocess->raw_counts_available : 0u)
        || !write_attr_u32(root, "processed_matrix_available", preprocess != 0 ? preprocess->processed_matrix_available : 0u)
        || !write_attr_u32(root, "normalized_log1p_metrics", preprocess != 0 ? preprocess->normalized_log1p_metrics : 0u)
        || !write_attr_u32(root, "hvg_available", preprocess != 0 ? preprocess->hvg_available : 0u)
        || !write_attr_u32(root, "mark_mito_from_feature_names", preprocess != 0 ? preprocess->mark_mito_from_feature_names : 0u)
        || !write_attr_u64(root, "rows", preprocess != 0 ? preprocess->rows : 0u)
        || !write_attr_u32(root, "cols", preprocess != 0 ? preprocess->cols : 0u)
        || !write_attr_u64(root, "nnz", preprocess != 0 ? preprocess->nnz : 0u)
        || !write_attr_u32(root, "partitions_processed", preprocess != 0 ? preprocess->partitions_processed : 0u)
        || !write_attr_u32(root, "mito_feature_count", preprocess != 0 ? preprocess->mito_feature_count : 0u)
        || !write_attr_f32(root, "target_sum", preprocess != 0 ? preprocess->target_sum : 0.0f)
        || !write_attr_f32(root, "min_counts", preprocess != 0 ? preprocess->min_counts : 0.0f)
        || !write_attr_u32(root, "min_genes", preprocess != 0 ? preprocess->min_genes : 0u)
        || !write_attr_f32(root, "max_mito_fraction", preprocess != 0 ? preprocess->max_mito_fraction : 0.0f)
        || !write_attr_f32(root, "min_gene_sum", preprocess != 0 ? preprocess->min_gene_sum : 0.0f)
        || !write_attr_f32(root, "min_detected_cells", preprocess != 0 ? preprocess->min_detected_cells : 0.0f)
        || !write_attr_f32(root, "min_variance", preprocess != 0 ? preprocess->min_variance : 0.0f)
        || !write_attr_f64(root, "kept_cells", preprocess != 0 ? preprocess->kept_cells : 0.0)
        || !write_attr_u32(root, "kept_genes", preprocess != 0 ? preprocess->kept_genes : 0u)
        || !write_attr_f64(root, "gene_sum_checksum", preprocess != 0 ? preprocess->gene_sum_checksum : 0.0)) {
        goto done;
    }

    cell_qc = create_group(root, "cell_qc");
    gene_qc = create_group(root, "gene_qc");
    if (cell_qc < 0 || gene_qc < 0) goto done;

    if (!write_attr_u64(cell_qc, "rows", preprocess != 0 ? preprocess->rows : 0u)
        || !write_attr_u32(gene_qc, "cols", preprocess != 0 ? preprocess->cols : 0u)) {
        goto done;
    }

    if (rows != 0u) {
        if ((preprocess == 0)
            || !write_dataset_1d(cell_qc, "total_counts", H5T_NATIVE_FLOAT, rows, preprocess->cell_total_counts)
            || !write_dataset_1d(cell_qc, "mito_counts", H5T_NATIVE_FLOAT, rows, preprocess->cell_mito_counts)
            || !write_dataset_1d(cell_qc, "max_counts", H5T_NATIVE_FLOAT, rows, preprocess->cell_max_counts)
            || !write_dataset_1d(cell_qc, "detected_genes", H5T_NATIVE_UINT32, rows, preprocess->cell_detected_genes)
            || !write_dataset_1d(cell_qc, "keep", H5T_NATIVE_UINT8, rows, preprocess->cell_keep)) {
            goto done;
        }
    }

    if (cols != 0u) {
        if ((preprocess == 0)
            || !write_dataset_1d(gene_qc, "sum", H5T_NATIVE_FLOAT, cols, preprocess->gene_sum)
            || !write_dataset_1d(gene_qc, "sq_sum", H5T_NATIVE_FLOAT, cols, preprocess->gene_sq_sum)
            || !write_dataset_1d(gene_qc, "detected_cells", H5T_NATIVE_FLOAT, cols, preprocess->gene_detected_cells)
            || !write_dataset_1d(gene_qc, "keep", H5T_NATIVE_UINT8, cols, preprocess->gene_keep)
            || !write_dataset_1d(gene_qc, "flags", H5T_NATIVE_UINT8, cols, preprocess->gene_flags)) {
            goto done;
        }
    }

    ok = 1;

done:
    if (gene_qc >= 0) H5Gclose(gene_qc);
    if (cell_qc >= 0) H5Gclose(cell_qc);
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_execution_h5(const char *filename,
                                const dataset_execution_view *execution) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    int ok = 0;

    if (filename == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    root = create_group(file, execution_group);
    if (root < 0) goto done;

    if (!write_attr_u32(root, "partition_count", execution != 0 ? execution->partition_count : 0u)) goto done;
    if (!write_attr_u32(root, "shard_count", execution != 0 ? execution->shard_count : 0u)) goto done;
    if (!write_attr_u32(root, "preferred_base_format", execution != 0 ? execution->preferred_base_format : dataset_execution_format_unknown)) goto done;

    if (execution == 0) {
        ok = 1;
        goto done;
    }

    if (execution->partition_count != 0u) {
        if (!write_dataset_1d(root,
                              "partition_execution_formats",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->partition_count,
                              execution->partition_execution_formats)) goto done;
        if (!write_dataset_1d(root,
                              "partition_blocked_ell_block_sizes",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->partition_count,
                              execution->partition_blocked_ell_block_sizes)) goto done;
        if (!write_dataset_1d(root,
                              "partition_blocked_ell_bucket_counts",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->partition_count,
                              execution->partition_blocked_ell_bucket_counts)) goto done;
        if (!write_dataset_1d(root,
                              "partition_blocked_ell_fill_ratios",
                              H5T_NATIVE_FLOAT,
                              (hsize_t) execution->partition_count,
                              execution->partition_blocked_ell_fill_ratios)) goto done;
        if (!write_dataset_1d(root,
                              "partition_execution_bytes",
                              H5T_NATIVE_UINT64,
                              (hsize_t) execution->partition_count,
                              execution->partition_execution_bytes)) goto done;
        if (!write_dataset_1d(root,
                              "partition_blocked_ell_bytes",
                              H5T_NATIVE_UINT64,
                              (hsize_t) execution->partition_count,
                              execution->partition_blocked_ell_bytes)) goto done;
        if (!write_dataset_1d(root,
                              "partition_bucketed_blocked_ell_bytes",
                              H5T_NATIVE_UINT64,
                              (hsize_t) execution->partition_count,
                              execution->partition_bucketed_blocked_ell_bytes)) goto done;
        if (execution->partition_sliced_ell_slice_counts != 0
            && !write_dataset_1d(root,
                                 "partition_sliced_ell_slice_counts",
                                 H5T_NATIVE_UINT32,
                                 (hsize_t) execution->partition_count,
                                 execution->partition_sliced_ell_slice_counts)) goto done;
        if (execution->partition_sliced_ell_slice_rows != 0
            && !write_dataset_1d(root,
                                 "partition_sliced_ell_slice_rows",
                                 H5T_NATIVE_UINT32,
                                 (hsize_t) execution->partition_count,
                                 execution->partition_sliced_ell_slice_rows)) goto done;
        if (execution->partition_sliced_ell_bytes != 0
            && !write_dataset_1d(root,
                                 "partition_sliced_ell_bytes",
                                 H5T_NATIVE_UINT64,
                                 (hsize_t) execution->partition_count,
                                 execution->partition_sliced_ell_bytes)) goto done;
        if (execution->partition_bucketed_sliced_ell_bytes != 0
            && !write_dataset_1d(root,
                                 "partition_bucketed_sliced_ell_bytes",
                                 H5T_NATIVE_UINT64,
                                 (hsize_t) execution->partition_count,
                                 execution->partition_bucketed_sliced_ell_bytes)) goto done;
    }

    if (execution->shard_count != 0u) {
        if (!write_dataset_1d(root,
                              "shard_execution_formats",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->shard_count,
                              execution->shard_execution_formats)) goto done;
        if (!write_dataset_1d(root,
                              "shard_blocked_ell_block_sizes",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->shard_count,
                              execution->shard_blocked_ell_block_sizes)) goto done;
        if (!write_dataset_1d(root,
                              "shard_bucketed_partition_counts",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->shard_count,
                              execution->shard_bucketed_partition_counts)) goto done;
        if (!write_dataset_1d(root,
                              "shard_bucketed_segment_counts",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->shard_count,
                              execution->shard_bucketed_segment_counts)) goto done;
        if (!write_dataset_1d(root,
                              "shard_blocked_ell_fill_ratios",
                              H5T_NATIVE_FLOAT,
                              (hsize_t) execution->shard_count,
                              execution->shard_blocked_ell_fill_ratios)) goto done;
        if (!write_dataset_1d(root,
                              "shard_execution_bytes",
                              H5T_NATIVE_UINT64,
                              (hsize_t) execution->shard_count,
                              execution->shard_execution_bytes)) goto done;
        if (!write_dataset_1d(root,
                              "shard_bucketed_blocked_ell_bytes",
                              H5T_NATIVE_UINT64,
                              (hsize_t) execution->shard_count,
                              execution->shard_bucketed_blocked_ell_bytes)) goto done;
        if (execution->shard_sliced_ell_slice_counts != 0
            && !write_dataset_1d(root,
                                 "shard_sliced_ell_slice_counts",
                                 H5T_NATIVE_UINT32,
                                 (hsize_t) execution->shard_count,
                                 execution->shard_sliced_ell_slice_counts)) goto done;
        if (execution->shard_sliced_ell_slice_rows != 0
            && !write_dataset_1d(root,
                                 "shard_sliced_ell_slice_rows",
                                 H5T_NATIVE_UINT32,
                                 (hsize_t) execution->shard_count,
                                 execution->shard_sliced_ell_slice_rows)) goto done;
        if (execution->shard_bucketed_sliced_ell_bytes != 0
            && !write_dataset_1d(root,
                                 "shard_bucketed_sliced_ell_bytes",
                                 H5T_NATIVE_UINT64,
                                 (hsize_t) execution->shard_count,
                                 execution->shard_bucketed_sliced_ell_bytes)) goto done;
        if (!write_dataset_1d(root,
                              "shard_preferred_pair_ids",
                              H5T_NATIVE_UINT32,
                              (hsize_t) execution->shard_count,
                              execution->shard_preferred_pair_ids)) goto done;
        if (execution->shard_owner_node_ids != 0
            && !write_dataset_1d(root,
                                 "shard_owner_node_ids",
                                 H5T_NATIVE_UINT32,
                                 (hsize_t) execution->shard_count,
                                 execution->shard_owner_node_ids)) goto done;
        if (execution->shard_owner_rank_ids != 0
            && !write_dataset_1d(root,
                                 "shard_owner_rank_ids",
                                 H5T_NATIVE_UINT32,
                                 (hsize_t) execution->shard_count,
                                 execution->shard_owner_rank_ids)) goto done;
    }

    ok = 1;

done:
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_dataset_runtime_service_h5(const char *filename,
                                      const dataset_runtime_service_view *runtime_service) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    dataset_runtime_service_view defaults;
    const dataset_runtime_service_view *view = runtime_service;
    int ok = 0;

    if (filename == 0) return 0;
    init(&defaults);
    if (view == 0) {
        defaults.service_mode = dataset_runtime_service_mode_local_cache;
        defaults.live_write_mode = dataset_live_write_mode_read_only;
        defaults.prefer_pack_delivery = 1u;
        defaults.maintenance_lock_blocks_overwrite = 1u;
        defaults.canonical_generation = 1u;
        defaults.execution_plan_generation = 1u;
        defaults.pack_generation = 1u;
        defaults.service_epoch = 1u;
        defaults.active_read_generation = 1u;
        defaults.staged_write_generation = 1u;
        view = &defaults;
    }

    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    root = create_group(file, runtime_service_group);
    if (root < 0) goto done;

    if (!write_attr_u32(root, "service_mode", view->service_mode)
        || !write_attr_u32(root, "live_write_mode", view->live_write_mode)
        || !write_attr_u32(root, "prefer_pack_delivery", view->prefer_pack_delivery)
        || !write_attr_u32(root, "remote_pack_delivery", view->remote_pack_delivery)
        || !write_attr_u32(root, "single_reader_coordinator", view->single_reader_coordinator)
        || !write_attr_u32(root, "maintenance_lock_blocks_overwrite", view->maintenance_lock_blocks_overwrite)
        || !write_attr_u64(root, "canonical_generation", view->canonical_generation)
        || !write_attr_u64(root, "execution_plan_generation", view->execution_plan_generation)
        || !write_attr_u64(root, "pack_generation", view->pack_generation)
        || !write_attr_u64(root, "service_epoch", view->service_epoch)
        || !write_attr_u64(root, "active_read_generation", view->active_read_generation)
        || !write_attr_u64(root, "staged_write_generation", view->staged_write_generation)) {
        goto done;
    }

    ok = 1;

done:
    if (root >= 0) H5Gclose(root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

} // namespace cellshard
