#include "execution_internal.hh"

int finalize_preprocessed_blocked_ell_dataset_h5_to_output(const char *source_filename,
                                                           const char *output_filename,
                                                           const std::uint8_t *cell_keep,
                                                           const std::uint8_t *gene_keep,
                                                           const dataset_embedded_metadata_view *embedded_metadata,
                                                           const dataset_annotation_view *observation_metadata,
                                                           const dataset_feature_metadata_view *feature_metadata,
                                                           const dataset_user_attribute_view *attributes,
                                                           const dataset_preprocess_view *preprocess,
                                                           const char *working_root,
                                                           std::uint64_t *rows_out,
                                                           std::uint64_t *cols_out,
                                                           std::uint64_t *nnz_out) {
    namespace fs = std::filesystem;
    sharded<sparse::blocked_ell> matrix;
    shard_storage storage;
    hid_t file = (hid_t) -1, datasets = (hid_t) -1, provenance = (hid_t) -1, matrix_group_h5 = (hid_t) -1;
    owned_text_column dataset_ids, matrix_paths, feature_paths, barcode_paths, metadata_paths;
    owned_text_column global_barcodes, filtered_global_barcodes;
    owned_text_column feature_ids, feature_names, feature_types;
    owned_text_column filtered_feature_ids, filtered_feature_names, filtered_feature_types;
    std::vector<std::uint32_t> dataset_formats;
    std::vector<std::uint64_t> dataset_rows, filtered_dataset_row_begin, filtered_dataset_row_end,
        filtered_dataset_rows, filtered_dataset_cols, filtered_dataset_nnz;
    std::vector<std::uint32_t> cell_dataset_ids;
    std::vector<std::uint64_t> cell_local_indices;
    std::vector<std::uint32_t> filtered_cell_dataset_ids;
    std::vector<std::uint64_t> filtered_cell_local_indices;
    std::vector<std::uint32_t> feature_dataset_ids;
    std::vector<std::uint64_t> feature_local_indices;
    std::vector<std::uint32_t> filtered_feature_dataset_ids;
    std::vector<std::uint64_t> filtered_feature_local_indices, dataset_feature_offsets;
    std::vector<std::uint32_t> dataset_feature_to_global;
    std::vector<std::uint64_t> filtered_dataset_feature_offsets;
    std::vector<std::uint32_t> filtered_dataset_feature_to_global;
    std::vector<std::uint32_t> partition_dataset_ids;
    std::vector<std::uint32_t> col_remap;
    std::vector<sparse::blocked_ell> filtered_parts;
    std::vector<sparse::blocked_ell *> filtered_part_ptrs, compact_part_ptrs;
    std::vector<std::uint64_t> part_rows, part_nnz, part_aux, part_row_offsets;
    std::vector<std::uint32_t> part_dataset_ids, part_codec_ids;
    std::vector<std::uint64_t> compact_part_rows, compact_part_nnz, compact_part_aux,
        compact_part_row_offsets, compact_shard_offsets;
    std::vector<std::uint32_t> compact_part_dataset_ids, compact_part_codec_ids;
    std::vector<std::uint32_t> compact_shard_part_begin, compact_shard_part_end;
    std::vector<std::uint64_t> shard_offsets;
    std::vector<std::uint32_t> part_formats, part_block_sizes, part_bucket_counts;
    std::vector<float> part_fill_ratios;
    std::vector<std::uint64_t> part_execution_bytes, part_blocked_ell_bytes, part_bucketed_blocked_ell_bytes;
    std::vector<std::uint32_t> zero_part_sliced_u32;
    std::vector<std::uint64_t> zero_part_sliced_u64;
    std::vector<std::uint32_t> shard_formats, shard_block_sizes, shard_bucketed_partition_counts,
        shard_bucketed_segment_counts;
    std::vector<float> shard_fill_ratios;
    std::vector<std::uint64_t> shard_execution_bytes, shard_bucketed_blocked_ell_bytes;
    std::vector<std::uint32_t> zero_shard_sliced_u32;
    std::vector<std::uint64_t> zero_shard_sliced_u64;
    std::vector<std::uint32_t> shard_pair_ids, shard_owner_node_ids, shard_owner_rank_ids;
    std::vector<float> filtered_pre_cell_total_counts, filtered_pre_cell_mito_counts, filtered_pre_cell_max_counts;
    std::vector<std::uint32_t> filtered_pre_cell_detected_genes;
    std::vector<std::uint8_t> filtered_pre_cell_keep;
    std::vector<float> filtered_pre_gene_sum, filtered_pre_gene_sq_sum, filtered_pre_gene_detected_cells;
    std::vector<std::uint8_t> filtered_pre_gene_keep, filtered_pre_gene_flags;
    dataset_layout_view layout{};
    dataset_dataset_table_view dataset_view{};
    dataset_provenance_view provenance_view{};
    dataset_execution_view execution_view{};
    dataset_runtime_service_view runtime_service{};
    dataset_preprocess_view finalized_preprocess{};
    dataset_codec_descriptor codec{};
    std::string temp_path;
    std::uint64_t dataset_count = 0u, final_rows = 0u, final_cols = 0u, final_nnz = 0u;
    int ok = 0;

    (void) working_root;
    if (rows_out != 0) *rows_out = 0u;
    if (cols_out != 0) *cols_out = 0u;
    if (nnz_out != 0) *nnz_out = 0u;
    if (source_filename == 0 || output_filename == 0 || cell_keep == 0 || gene_keep == 0) return 0;

    init(&runtime_service);
    init(&matrix);
    init(&storage);
    if (!load_header(source_filename, &matrix, &storage)) goto done;
    if (!get_dataset_h5_runtime_service(&storage, &runtime_service)) {
        init(&runtime_service);
        runtime_service.service_mode = dataset_runtime_service_mode_owner_hosted;
        runtime_service.live_write_mode = dataset_live_write_mode_append_only;
        runtime_service.prefer_pack_delivery = 1u;
        runtime_service.remote_pack_delivery = 0u;
        runtime_service.single_reader_coordinator = 1u;
        runtime_service.maintenance_lock_blocks_overwrite = 1u;
        runtime_service.canonical_generation = 1u;
        runtime_service.execution_plan_generation = 1u;
        runtime_service.pack_generation = 1u;
        runtime_service.service_epoch = 1u;
        runtime_service.active_read_generation = 1u;
        runtime_service.staged_write_generation = 1u;
    }

    file = H5Fopen(source_filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) goto done;
    datasets = H5Gopen2(file, datasets_group, H5P_DEFAULT);
    provenance = H5Gopen2(file, provenance_group, H5P_DEFAULT);
    matrix_group_h5 = H5Gopen2(file, matrix_group, H5P_DEFAULT);
    if (datasets < 0 || provenance < 0 || matrix_group_h5 < 0) goto done;
    if (!read_attr_u64(file, "num_datasets", &dataset_count)) goto done;
    dataset_formats.assign((std::size_t) dataset_count, 0u);
    dataset_rows.assign((std::size_t) dataset_count, 0u);
    cell_dataset_ids.assign((std::size_t) matrix.rows, 0u);
    cell_local_indices.assign((std::size_t) matrix.rows, 0u);
    feature_dataset_ids.assign((std::size_t) matrix.cols, 0u);
    feature_local_indices.assign((std::size_t) matrix.cols, 0u);
    dataset_feature_offsets.assign((std::size_t) dataset_count + 1u, 0u);
    partition_dataset_ids.assign((std::size_t) matrix.num_partitions, 0u);
    if (!read_text_column(datasets, "dataset_ids", &dataset_ids)
        || !read_text_column(datasets, "matrix_paths", &matrix_paths)
        || !read_text_column(datasets, "feature_paths", &feature_paths)
        || !read_text_column(datasets, "barcode_paths", &barcode_paths)
        || !read_text_column(datasets, "metadata_paths", &metadata_paths)
        || !read_dataset_1d(datasets, "formats", H5T_NATIVE_UINT32, dataset_count, dataset_formats.data())
        || !read_dataset_1d(datasets, "rows", H5T_NATIVE_UINT64, dataset_count, dataset_rows.data())
        || !read_text_column(provenance, "global_barcodes", &global_barcodes)
        || !read_dataset_1d(provenance, "cell_dataset_ids", H5T_NATIVE_UINT32, matrix.rows, cell_dataset_ids.data())
        || !read_dataset_1d(provenance, "cell_local_indices", H5T_NATIVE_UINT64, matrix.rows, cell_local_indices.data())
        || !read_text_column(provenance, "feature_ids", &feature_ids)
        || !read_text_column(provenance, "feature_names", &feature_names)
        || !read_text_column(provenance, "feature_types", &feature_types)
        || !read_dataset_1d(provenance, "feature_dataset_ids", H5T_NATIVE_UINT32, matrix.cols, feature_dataset_ids.data())
        || !read_dataset_1d(provenance, "feature_local_indices", H5T_NATIVE_UINT64, matrix.cols, feature_local_indices.data())
        || !read_dataset_1d(provenance, "dataset_feature_offsets", H5T_NATIVE_UINT64, dataset_count + 1u, dataset_feature_offsets.data())) {
        goto done;
    }
    dataset_feature_to_global.assign((std::size_t) (dataset_feature_offsets.empty() ? 0u : dataset_feature_offsets.back()), 0u);
    if ((!dataset_feature_to_global.empty()
         && !read_dataset_1d(provenance,
                             "dataset_feature_to_global",
                             H5T_NATIVE_UINT32,
                             dataset_feature_offsets.back(),
                             dataset_feature_to_global.data()))
        || !read_dataset_1d(matrix_group_h5,
                            "partition_dataset_ids",
                            H5T_NATIVE_UINT32,
                            matrix.num_partitions,
                            partition_dataset_ids.data())) {
        goto done;
    }
    H5Gclose(matrix_group_h5);
    matrix_group_h5 = (hid_t) -1;
    H5Gclose(provenance);
    provenance = (hid_t) -1;
    H5Gclose(datasets);
    datasets = (hid_t) -1;
    H5Fclose(file);
    file = (hid_t) -1;

    filtered_dataset_rows.assign((std::size_t) dataset_count, 0u);
    filtered_dataset_cols.assign((std::size_t) dataset_count, 0u);
    filtered_dataset_nnz.assign((std::size_t) dataset_count, 0u);
    filtered_dataset_row_begin.assign((std::size_t) dataset_count, 0u);
    filtered_dataset_row_end.assign((std::size_t) dataset_count, 0u);
    col_remap.assign((std::size_t) matrix.cols, std::numeric_limits<std::uint32_t>::max());

    for (std::uint64_t row = 0u; row < matrix.rows; ++row) {
        if (cell_keep[row] == 0u) continue;
        const std::uint32_t dataset_id = row < cell_dataset_ids.size() ? cell_dataset_ids[(std::size_t) row] : 0u;
        append_text_value(&filtered_global_barcodes, text_column_value(global_barcodes, (std::uint32_t) row));
        filtered_cell_dataset_ids.push_back(dataset_id);
        filtered_cell_local_indices.push_back(row < cell_local_indices.size() ? cell_local_indices[(std::size_t) row] : 0u);
        if (dataset_id < filtered_dataset_rows.size()) ++filtered_dataset_rows[(std::size_t) dataset_id];
        ++final_rows;
    }
    {
        std::uint64_t row_cursor = 0u;
        for (std::size_t dataset_idx = 0; dataset_idx < (std::size_t) dataset_count; ++dataset_idx) {
            filtered_dataset_row_begin[dataset_idx] = row_cursor;
            row_cursor += filtered_dataset_rows[dataset_idx];
            filtered_dataset_row_end[dataset_idx] = row_cursor;
        }
    }

    for (std::uint32_t gene = 0u; gene < matrix.cols; ++gene) {
        if (gene_keep[gene] == 0u) continue;
        col_remap[(std::size_t) gene] = (std::uint32_t) final_cols;
        append_text_value(&filtered_feature_ids, text_column_value(feature_ids, gene));
        append_text_value(&filtered_feature_names, text_column_value(feature_names, gene));
        append_text_value(&filtered_feature_types, text_column_value(feature_types, gene));
        filtered_feature_dataset_ids.push_back(gene < feature_dataset_ids.size() ? feature_dataset_ids[(std::size_t) gene] : 0u);
        filtered_feature_local_indices.push_back(gene < feature_local_indices.size() ? feature_local_indices[(std::size_t) gene] : 0u);
        ++final_cols;
    }

    filtered_dataset_feature_offsets.assign(1u, 0u);
    for (std::size_t dataset_idx = 0; dataset_idx < (std::size_t) dataset_count; ++dataset_idx) {
        const std::uint64_t begin = dataset_idx < dataset_feature_offsets.size() ? dataset_feature_offsets[dataset_idx] : 0u;
        const std::uint64_t end = dataset_idx + 1u < dataset_feature_offsets.size() ? dataset_feature_offsets[dataset_idx + 1u] : begin;
        for (std::uint64_t idx = begin; idx < end && idx < dataset_feature_to_global.size(); ++idx) {
            const std::uint32_t global_col = dataset_feature_to_global[(std::size_t) idx];
            if (global_col >= col_remap.size()) continue;
            if (col_remap[(std::size_t) global_col] == std::numeric_limits<std::uint32_t>::max()) continue;
            filtered_dataset_feature_to_global.push_back(col_remap[(std::size_t) global_col]);
        }
        filtered_dataset_cols[dataset_idx] =
            (std::uint64_t) filtered_dataset_feature_to_global.size() - filtered_dataset_feature_offsets.back();
        filtered_dataset_feature_offsets.push_back((std::uint64_t) filtered_dataset_feature_to_global.size());
    }

    filtered_parts.resize((std::size_t) matrix.num_partitions);
    filtered_part_ptrs.resize((std::size_t) matrix.num_partitions, 0);
    part_rows.assign((std::size_t) matrix.num_partitions, 0u);
    part_nnz.assign((std::size_t) matrix.num_partitions, 0u);
    part_aux.assign((std::size_t) matrix.num_partitions, 0u);
    part_row_offsets.assign((std::size_t) matrix.num_partitions + 1u, 0u);
    part_dataset_ids.assign((std::size_t) matrix.num_partitions, 0u);
    part_codec_ids.assign((std::size_t) matrix.num_partitions, 0u);
    part_formats.assign((std::size_t) matrix.num_partitions, dataset_execution_format_bucketed_blocked_ell);
    part_block_sizes.assign((std::size_t) matrix.num_partitions, 0u);
    part_bucket_counts.assign((std::size_t) matrix.num_partitions, 1u);
    part_fill_ratios.assign((std::size_t) matrix.num_partitions, 0.0f);
    part_execution_bytes.assign((std::size_t) matrix.num_partitions, 0u);
    part_blocked_ell_bytes.assign((std::size_t) matrix.num_partitions, 0u);
    part_bucketed_blocked_ell_bytes.assign((std::size_t) matrix.num_partitions, 0u);

    for (unsigned long part_id = 0u; part_id < matrix.num_partitions; ++part_id) {
        std::uint32_t live_rows = 0u;
        std::uint32_t live_nnz = 0u;
        if (!fetch_partition(&matrix, &storage, part_id)) goto done;
        if (!build_filtered_blocked_ell_part_from_blocked(matrix.parts[part_id],
                                                          cell_keep,
                                                          matrix.partition_offsets[part_id],
                                                          col_remap.data(),
                                                          (std::uint32_t) final_cols,
                                                          &filtered_parts[(std::size_t) part_id],
                                                          &live_rows,
                                                          &live_nnz)) {
            goto done;
        }
        filtered_part_ptrs[(std::size_t) part_id] = &filtered_parts[(std::size_t) part_id];
        part_rows[(std::size_t) part_id] = live_rows;
        part_nnz[(std::size_t) part_id] = live_nnz;
        part_aux[(std::size_t) part_id] = partition_aux(&filtered_parts[(std::size_t) part_id]);
        part_row_offsets[(std::size_t) part_id + 1u] = part_row_offsets[(std::size_t) part_id] + live_rows;
        part_dataset_ids[(std::size_t) part_id] =
            part_id < partition_dataset_ids.size() ? partition_dataset_ids[(std::size_t) part_id] : 0u;
        if (part_dataset_ids[(std::size_t) part_id] < filtered_dataset_nnz.size()) {
            filtered_dataset_nnz[(std::size_t) part_dataset_ids[(std::size_t) part_id]] += live_nnz;
        }
        final_nnz += live_nnz;
    }

    compact_part_row_offsets.assign(1u, 0u);
    for (unsigned long part_id = 0u; part_id < matrix.num_partitions; ++part_id) {
        if (part_rows[(std::size_t) part_id] == 0u) continue;
        compact_part_ptrs.push_back(filtered_part_ptrs[(std::size_t) part_id]);
        compact_part_rows.push_back(part_rows[(std::size_t) part_id]);
        compact_part_nnz.push_back(part_nnz[(std::size_t) part_id]);
        compact_part_aux.push_back(part_aux[(std::size_t) part_id]);
        compact_part_dataset_ids.push_back(part_dataset_ids[(std::size_t) part_id]);
        compact_part_codec_ids.push_back(part_codec_ids[(std::size_t) part_id]);
        compact_part_row_offsets.push_back(compact_part_row_offsets.back() + part_rows[(std::size_t) part_id]);
    }

    compact_shard_offsets.assign(1u, 0u);
    {
        std::uint32_t compact_part_cursor = 0u;
        for (unsigned long shard_id = 0u; shard_id < matrix.num_shards; ++shard_id) {
            const unsigned long part_begin = matrix.shard_parts != 0 ? matrix.shard_parts[shard_id] : shard_id;
            const unsigned long part_end = matrix.shard_parts != 0
                ? matrix.shard_parts[shard_id + 1u]
                : std::min<unsigned long>(matrix.num_partitions, shard_id + 1u);
            std::uint32_t kept_in_shard = 0u;
            std::uint64_t shard_rows = 0u;
            for (unsigned long part_id = part_begin; part_id < part_end; ++part_id) {
                if (part_rows[(std::size_t) part_id] == 0u) continue;
                ++kept_in_shard;
                shard_rows += part_rows[(std::size_t) part_id];
            }
            if (kept_in_shard == 0u) continue;
            compact_shard_part_begin.push_back(compact_part_cursor);
            compact_part_cursor += kept_in_shard;
            compact_shard_part_end.push_back(compact_part_cursor);
            compact_shard_offsets.push_back(compact_shard_offsets.back() + shard_rows);
        }
    }

    dataset_view.count = (std::uint32_t) dataset_count;
    dataset_view.dataset_ids = dataset_ids.view();
    dataset_view.matrix_paths = matrix_paths.view();
    dataset_view.feature_paths = feature_paths.view();
    dataset_view.barcode_paths = barcode_paths.view();
    dataset_view.metadata_paths = metadata_paths.view();
    dataset_view.formats = dataset_formats.empty() ? nullptr : dataset_formats.data();
    dataset_view.row_begin = filtered_dataset_row_begin.empty() ? nullptr : filtered_dataset_row_begin.data();
    dataset_view.row_end = filtered_dataset_row_end.empty() ? nullptr : filtered_dataset_row_end.data();
    dataset_view.rows = filtered_dataset_rows.empty() ? nullptr : filtered_dataset_rows.data();
    dataset_view.cols = filtered_dataset_cols.empty() ? nullptr : filtered_dataset_cols.data();
    dataset_view.nnz = filtered_dataset_nnz.empty() ? nullptr : filtered_dataset_nnz.data();

    provenance_view.global_barcodes = filtered_global_barcodes.view();
    provenance_view.cell_dataset_ids = filtered_cell_dataset_ids.empty() ? nullptr : filtered_cell_dataset_ids.data();
    provenance_view.cell_local_indices = filtered_cell_local_indices.empty() ? nullptr : filtered_cell_local_indices.data();
    provenance_view.feature_ids = filtered_feature_ids.view();
    provenance_view.feature_names = filtered_feature_names.view();
    provenance_view.feature_types = filtered_feature_types.view();
    provenance_view.feature_dataset_ids = filtered_feature_dataset_ids.empty() ? nullptr : filtered_feature_dataset_ids.data();
    provenance_view.feature_local_indices = filtered_feature_local_indices.empty() ? nullptr : filtered_feature_local_indices.data();
    provenance_view.dataset_feature_offsets = filtered_dataset_feature_offsets.empty() ? nullptr : filtered_dataset_feature_offsets.data();
    provenance_view.dataset_feature_to_global = filtered_dataset_feature_to_global.empty() ? nullptr : filtered_dataset_feature_to_global.data();

    codec.codec_id = 0u;
    codec.family = dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = final_rows;
    layout.cols = final_cols;
    layout.nnz = final_nnz;
    layout.num_partitions = (std::uint64_t) compact_part_ptrs.size();
    layout.num_shards = compact_shard_offsets.empty() ? 0u : (std::uint64_t) compact_shard_offsets.size() - 1u;
    layout.partition_rows = compact_part_rows.empty() ? nullptr : compact_part_rows.data();
    layout.partition_nnz = compact_part_nnz.empty() ? nullptr : compact_part_nnz.data();
    layout.partition_axes = nullptr;
    layout.partition_aux = compact_part_aux.empty() ? nullptr : compact_part_aux.data();
    layout.partition_row_offsets = compact_part_row_offsets.empty() ? nullptr : compact_part_row_offsets.data();
    layout.partition_dataset_ids = compact_part_dataset_ids.empty() ? nullptr : compact_part_dataset_ids.data();
    layout.partition_codec_ids = compact_part_codec_ids.empty() ? nullptr : compact_part_codec_ids.data();
    layout.shard_offsets = compact_shard_offsets.empty() ? nullptr : compact_shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    temp_path = (fs::path(output_filename).parent_path() / (fs::path(output_filename).filename().string()
        + ".preprocess_finalize." + std::to_string((unsigned long long) ::getpid()) + ".tmp")).string();
    {
        std::error_code ec;
        fs::remove(temp_path, ec);
    }
    if (!create_dataset_blocked_ell_h5(temp_path.c_str(), &layout, &dataset_view, &provenance_view)) goto done;

    part_formats.assign(compact_part_ptrs.size(), dataset_execution_format_bucketed_blocked_ell);
    part_block_sizes.assign(compact_part_ptrs.size(), 0u);
    part_bucket_counts.assign(compact_part_ptrs.size(), 1u);
    part_fill_ratios.assign(compact_part_ptrs.size(), 0.0f);
    part_execution_bytes.assign(compact_part_ptrs.size(), 0u);
    part_blocked_ell_bytes.assign(compact_part_ptrs.size(), 0u);
    part_bucketed_blocked_ell_bytes.assign(compact_part_ptrs.size(), 0u);
    for (std::size_t part_id = 0u; part_id < compact_part_ptrs.size(); ++part_id) {
        part_block_sizes[part_id] = compact_part_ptrs[part_id] != nullptr ? compact_part_ptrs[part_id]->block_size : 0u;
        part_blocked_ell_bytes[part_id] = compact_part_ptrs[part_id] != nullptr
            ? (std::uint64_t) packed_bytes((const sparse::blocked_ell *) 0,
                                           compact_part_ptrs[part_id]->rows,
                                           compact_part_ptrs[part_id]->cols,
                                           compact_part_ptrs[part_id]->nnz,
                                           partition_aux(compact_part_ptrs[part_id]),
                                           sizeof(real::storage_t))
            : 0u;
    }

    shard_formats.assign(compact_shard_part_begin.size(), dataset_execution_format_bucketed_blocked_ell);
    shard_block_sizes.assign(compact_shard_part_begin.size(), 0u);
    shard_bucketed_partition_counts.assign(compact_shard_part_begin.size(), 0u);
    shard_bucketed_segment_counts.assign(compact_shard_part_begin.size(), 0u);
    shard_fill_ratios.assign(compact_shard_part_begin.size(), 0.0f);
    shard_execution_bytes.assign(compact_shard_part_begin.size(), 0u);
    shard_bucketed_blocked_ell_bytes.assign(compact_shard_part_begin.size(), 0u);
    zero_part_sliced_u32.assign(compact_part_ptrs.size(), 0u);
    zero_part_sliced_u64.assign(compact_part_ptrs.size(), 0u);
    zero_shard_sliced_u32.assign(compact_shard_part_begin.size(), 0u);
    zero_shard_sliced_u64.assign(compact_shard_part_begin.size(), 0u);
    shard_pair_ids.assign(compact_shard_part_begin.size(), 0u);
    shard_owner_node_ids.assign(compact_shard_part_begin.size(), 0u);
    shard_owner_rank_ids.assign(compact_shard_part_begin.size(), 0u);
    for (std::size_t shard_id = 0u; shard_id < compact_shard_part_begin.size(); ++shard_id) {
        const std::uint32_t part_begin = compact_shard_part_begin[shard_id];
        const std::uint32_t part_end = compact_shard_part_end[shard_id];
        bucketed_blocked_ell_shard optimized_shard;
        std::uint32_t shard_segments = 0u;
        init(&optimized_shard);
        if (!build_bucketed_optimized_shard_from_parts(compact_part_ptrs.data() + part_begin,
                                                       part_end - part_begin,
                                                       (std::uint32_t) final_cols,
                                                       &optimized_shard,
                                                       part_block_sizes.data() + part_begin,
                                                       part_bucket_counts.data() + part_begin,
                                                       part_fill_ratios.data() + part_begin,
                                                       part_execution_bytes.data() + part_begin,
                                                       part_blocked_ell_bytes.data() + part_begin,
                                                       part_bucketed_blocked_ell_bytes.data() + part_begin,
                                                       shard_block_sizes.data() + shard_id,
                                                       &shard_segments,
                                                       shard_fill_ratios.data() + shard_id,
                                                       shard_execution_bytes.data() + shard_id,
                                                       shard_bucketed_blocked_ell_bytes.data() + shard_id)) {
            clear(&optimized_shard);
            goto done;
        }
        shard_bucketed_partition_counts[shard_id] = part_end - part_begin;
        shard_bucketed_segment_counts[shard_id] = shard_segments;
        if (!append_bucketed_blocked_ell_shard_h5(temp_path.c_str(), (unsigned long) shard_id, &optimized_shard)) {
            clear(&optimized_shard);
            goto done;
        }
        clear(&optimized_shard);
    }

    execution_view.partition_count = (std::uint32_t) compact_part_ptrs.size();
    execution_view.partition_execution_formats = part_formats.empty() ? nullptr : part_formats.data();
    execution_view.partition_blocked_ell_block_sizes = part_block_sizes.empty() ? nullptr : part_block_sizes.data();
    execution_view.partition_blocked_ell_bucket_counts = part_bucket_counts.empty() ? nullptr : part_bucket_counts.data();
    execution_view.partition_blocked_ell_fill_ratios = part_fill_ratios.empty() ? nullptr : part_fill_ratios.data();
    execution_view.partition_execution_bytes = part_execution_bytes.empty() ? nullptr : part_execution_bytes.data();
    execution_view.partition_blocked_ell_bytes = part_blocked_ell_bytes.empty() ? nullptr : part_blocked_ell_bytes.data();
    execution_view.partition_bucketed_blocked_ell_bytes = part_bucketed_blocked_ell_bytes.empty() ? nullptr : part_bucketed_blocked_ell_bytes.data();
    execution_view.partition_sliced_ell_slice_counts = zero_part_sliced_u32.empty() ? nullptr : zero_part_sliced_u32.data();
    execution_view.partition_sliced_ell_slice_rows = zero_part_sliced_u32.empty() ? nullptr : zero_part_sliced_u32.data();
    execution_view.partition_sliced_ell_bytes = zero_part_sliced_u64.empty() ? nullptr : zero_part_sliced_u64.data();
    execution_view.partition_bucketed_sliced_ell_bytes = zero_part_sliced_u64.empty() ? nullptr : zero_part_sliced_u64.data();
    execution_view.shard_count = (std::uint32_t) compact_shard_part_begin.size();
    execution_view.shard_execution_formats = shard_formats.empty() ? nullptr : shard_formats.data();
    execution_view.shard_blocked_ell_block_sizes = shard_block_sizes.empty() ? nullptr : shard_block_sizes.data();
    execution_view.shard_bucketed_partition_counts = shard_bucketed_partition_counts.empty() ? nullptr : shard_bucketed_partition_counts.data();
    execution_view.shard_bucketed_segment_counts = shard_bucketed_segment_counts.empty() ? nullptr : shard_bucketed_segment_counts.data();
    execution_view.shard_blocked_ell_fill_ratios = shard_fill_ratios.empty() ? nullptr : shard_fill_ratios.data();
    execution_view.shard_execution_bytes = shard_execution_bytes.empty() ? nullptr : shard_execution_bytes.data();
    execution_view.shard_bucketed_blocked_ell_bytes = shard_bucketed_blocked_ell_bytes.empty() ? nullptr : shard_bucketed_blocked_ell_bytes.data();
    execution_view.shard_sliced_ell_slice_counts = zero_shard_sliced_u32.empty() ? nullptr : zero_shard_sliced_u32.data();
    execution_view.shard_sliced_ell_slice_rows = zero_shard_sliced_u32.empty() ? nullptr : zero_shard_sliced_u32.data();
    execution_view.shard_bucketed_sliced_ell_bytes = zero_shard_sliced_u64.empty() ? nullptr : zero_shard_sliced_u64.data();
    execution_view.shard_preferred_pair_ids = shard_pair_ids.empty() ? nullptr : shard_pair_ids.data();
    execution_view.shard_owner_node_ids = shard_owner_node_ids.empty() ? nullptr : shard_owner_node_ids.data();
    execution_view.shard_owner_rank_ids = shard_owner_rank_ids.empty() ? nullptr : shard_owner_rank_ids.data();
    execution_view.preferred_base_format = dataset_execution_format_bucketed_blocked_ell;
    if (!append_dataset_execution_h5(temp_path.c_str(), &execution_view)) goto done;
    if (!append_dataset_runtime_service_h5(temp_path.c_str(), &runtime_service)) goto done;
    if (embedded_metadata != nullptr && !append_dataset_embedded_metadata_h5(temp_path.c_str(), embedded_metadata)) goto done;
    if (observation_metadata != nullptr && !append_dataset_observation_annotations_h5(temp_path.c_str(), observation_metadata)) goto done;
    if (feature_metadata != nullptr && !append_dataset_feature_metadata_h5(temp_path.c_str(), feature_metadata)) goto done;
    if (attributes != nullptr && !append_dataset_user_attributes_h5(temp_path.c_str(), attributes)) goto done;
    if (preprocess != nullptr) {
        finalized_preprocess = *preprocess;
        finalized_preprocess.processed_matrix_available = 1u;
        finalized_preprocess.rows = final_rows;
        finalized_preprocess.cols = (std::uint32_t) final_cols;
        finalized_preprocess.nnz = final_nnz;
        filtered_pre_cell_total_counts.reserve((std::size_t) final_rows);
        filtered_pre_cell_mito_counts.reserve((std::size_t) final_rows);
        filtered_pre_cell_max_counts.reserve((std::size_t) final_rows);
        filtered_pre_cell_detected_genes.reserve((std::size_t) final_rows);
        filtered_pre_cell_keep.reserve((std::size_t) final_rows);
        for (std::uint64_t row = 0u; row < matrix.rows; ++row) {
            if (cell_keep[row] == 0u) continue;
            if (preprocess->cell_total_counts != nullptr) filtered_pre_cell_total_counts.push_back(preprocess->cell_total_counts[row]);
            if (preprocess->cell_mito_counts != nullptr) filtered_pre_cell_mito_counts.push_back(preprocess->cell_mito_counts[row]);
            if (preprocess->cell_max_counts != nullptr) filtered_pre_cell_max_counts.push_back(preprocess->cell_max_counts[row]);
            if (preprocess->cell_detected_genes != nullptr) filtered_pre_cell_detected_genes.push_back(preprocess->cell_detected_genes[row]);
            if (preprocess->cell_keep != nullptr) filtered_pre_cell_keep.push_back(1u);
        }
        filtered_pre_gene_sum.reserve((std::size_t) final_cols);
        filtered_pre_gene_sq_sum.reserve((std::size_t) final_cols);
        filtered_pre_gene_detected_cells.reserve((std::size_t) final_cols);
        filtered_pre_gene_keep.reserve((std::size_t) final_cols);
        filtered_pre_gene_flags.reserve((std::size_t) final_cols);
        for (std::uint32_t gene = 0u; gene < matrix.cols; ++gene) {
            if (gene_keep[gene] == 0u) continue;
            if (preprocess->gene_sum != nullptr) filtered_pre_gene_sum.push_back(preprocess->gene_sum[gene]);
            if (preprocess->gene_sq_sum != nullptr) filtered_pre_gene_sq_sum.push_back(preprocess->gene_sq_sum[gene]);
            if (preprocess->gene_detected_cells != nullptr) filtered_pre_gene_detected_cells.push_back(preprocess->gene_detected_cells[gene]);
            if (preprocess->gene_keep != nullptr) filtered_pre_gene_keep.push_back(1u);
            if (preprocess->gene_flags != nullptr) filtered_pre_gene_flags.push_back(preprocess->gene_flags[gene]);
        }
        finalized_preprocess.cell_total_counts = filtered_pre_cell_total_counts.empty() ? nullptr : filtered_pre_cell_total_counts.data();
        finalized_preprocess.cell_mito_counts = filtered_pre_cell_mito_counts.empty() ? nullptr : filtered_pre_cell_mito_counts.data();
        finalized_preprocess.cell_max_counts = filtered_pre_cell_max_counts.empty() ? nullptr : filtered_pre_cell_max_counts.data();
        finalized_preprocess.cell_detected_genes = filtered_pre_cell_detected_genes.empty() ? nullptr : filtered_pre_cell_detected_genes.data();
        finalized_preprocess.cell_keep = filtered_pre_cell_keep.empty() ? nullptr : filtered_pre_cell_keep.data();
        finalized_preprocess.gene_sum = filtered_pre_gene_sum.empty() ? nullptr : filtered_pre_gene_sum.data();
        finalized_preprocess.gene_sq_sum = filtered_pre_gene_sq_sum.empty() ? nullptr : filtered_pre_gene_sq_sum.data();
        finalized_preprocess.gene_detected_cells = filtered_pre_gene_detected_cells.empty() ? nullptr : filtered_pre_gene_detected_cells.data();
        finalized_preprocess.gene_keep = filtered_pre_gene_keep.empty() ? nullptr : filtered_pre_gene_keep.data();
        finalized_preprocess.gene_flags = filtered_pre_gene_flags.empty() ? nullptr : filtered_pre_gene_flags.data();
        if (!append_dataset_preprocess_h5(temp_path.c_str(), &finalized_preprocess)) goto done;
    }

    {
        std::error_code ec;
        fs::rename(temp_path, output_filename, ec);
        if (ec) goto done;
    }
    if (rows_out != nullptr) *rows_out = final_rows;
    if (cols_out != nullptr) *cols_out = final_cols;
    if (nnz_out != nullptr) *nnz_out = final_nnz;
    ok = 1;

done:
    if (!ok && !temp_path.empty()) {
        std::error_code ec;
        fs::remove(temp_path, ec);
    }
    for (std::size_t i = 0; i < filtered_parts.size(); ++i) sparse::clear(&filtered_parts[i]);
    clear(&storage);
    clear(&matrix);
    if (matrix_group_h5 >= 0) H5Gclose(matrix_group_h5);
    if (provenance >= 0) H5Gclose(provenance);
    if (datasets >= 0) H5Gclose(datasets);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int finalize_preprocessed_sliced_ell_dataset_h5_to_output(const char *source_filename,
                                                          const char *output_filename,
                                                          const std::uint8_t *cell_keep,
                                                          const std::uint8_t *gene_keep,
                                                          const dataset_embedded_metadata_view *embedded_metadata,
                                                          const dataset_annotation_view *observation_metadata,
                                                          const dataset_feature_metadata_view *feature_metadata,
                                                          const dataset_user_attribute_view *attributes,
                                                          const dataset_preprocess_view *preprocess,
                                                          const char *working_root,
                                                          std::uint64_t *rows_out,
                                                          std::uint64_t *cols_out,
                                                          std::uint64_t *nnz_out) {
    namespace fs = std::filesystem;
    sharded<sparse::sliced_ell> matrix;
    shard_storage storage;
    hid_t file = (hid_t) -1, datasets = (hid_t) -1, provenance = (hid_t) -1, matrix_group_h5 = (hid_t) -1;
    owned_text_column dataset_ids, matrix_paths, feature_paths, barcode_paths, metadata_paths;
    owned_text_column global_barcodes, filtered_global_barcodes;
    owned_text_column feature_ids, feature_names, feature_types;
    owned_text_column filtered_feature_ids, filtered_feature_names, filtered_feature_types;
    std::vector<std::uint32_t> dataset_formats;
    std::vector<std::uint64_t> dataset_rows, filtered_dataset_row_begin, filtered_dataset_row_end,
        filtered_dataset_rows, filtered_dataset_cols, filtered_dataset_nnz;
    std::vector<std::uint32_t> cell_dataset_ids;
    std::vector<std::uint64_t> cell_local_indices;
    std::vector<std::uint32_t> filtered_cell_dataset_ids;
    std::vector<std::uint64_t> filtered_cell_local_indices;
    std::vector<std::uint32_t> feature_dataset_ids;
    std::vector<std::uint64_t> feature_local_indices;
    std::vector<std::uint32_t> filtered_feature_dataset_ids;
    std::vector<std::uint64_t> filtered_feature_local_indices, dataset_feature_offsets;
    std::vector<std::uint32_t> dataset_feature_to_global;
    std::vector<std::uint64_t> filtered_dataset_feature_offsets;
    std::vector<std::uint32_t> filtered_dataset_feature_to_global;
    std::vector<std::uint32_t> partition_dataset_ids;
    std::vector<std::uint32_t> col_remap;
    std::vector<sparse::sliced_ell> filtered_parts;
    std::vector<bucketed_sliced_ell_partition> filtered_bucketed_parts;
    std::vector<bucketed_sliced_ell_partition *> filtered_part_ptrs, compact_part_ptrs;
    std::vector<std::uint64_t> part_rows, part_nnz, part_aux, part_row_offsets;
    std::vector<std::uint32_t> part_dataset_ids, part_codec_ids;
    std::vector<std::uint64_t> compact_part_rows, compact_part_nnz, compact_part_aux,
        compact_part_row_offsets, compact_shard_offsets;
    std::vector<std::uint32_t> compact_part_dataset_ids, compact_part_codec_ids;
    std::vector<std::uint32_t> compact_shard_part_begin, compact_shard_part_end;
    std::vector<std::uint32_t> part_formats, part_block_sizes, part_bucket_counts;
    std::vector<float> part_fill_ratios;
    std::vector<std::uint64_t> part_execution_bytes, part_blocked_ell_bytes, part_bucketed_blocked_ell_bytes;
    std::vector<std::uint32_t> part_slice_counts, part_slice_rows;
    std::vector<std::uint64_t> part_sliced_bytes, part_bucketed_sliced_bytes;
    std::vector<std::uint32_t> shard_formats, shard_block_sizes, shard_bucketed_partition_counts,
        shard_bucketed_segment_counts;
    std::vector<float> shard_fill_ratios;
    std::vector<std::uint64_t> shard_execution_bytes, shard_bucketed_blocked_ell_bytes;
    std::vector<std::uint32_t> shard_sliced_ell_slice_counts, shard_sliced_ell_slice_rows;
    std::vector<std::uint64_t> shard_bucketed_sliced_ell_bytes;
    std::vector<std::uint32_t> shard_pair_ids, shard_owner_node_ids, shard_owner_rank_ids;
    std::vector<float> filtered_pre_cell_total_counts, filtered_pre_cell_mito_counts, filtered_pre_cell_max_counts;
    std::vector<std::uint32_t> filtered_pre_cell_detected_genes;
    std::vector<std::uint8_t> filtered_pre_cell_keep;
    std::vector<float> filtered_pre_gene_sum, filtered_pre_gene_sq_sum, filtered_pre_gene_detected_cells;
    std::vector<std::uint8_t> filtered_pre_gene_keep, filtered_pre_gene_flags;
    dataset_layout_view layout{};
    dataset_dataset_table_view dataset_view{};
    dataset_provenance_view provenance_view{};
    dataset_execution_view execution_view{};
    dataset_runtime_service_view runtime_service{};
    dataset_preprocess_view finalized_preprocess{};
    dataset_codec_descriptor codec{};
    std::string temp_path;
    std::uint64_t dataset_count = 0u, final_rows = 0u, final_cols = 0u, final_nnz = 0u;
    int ok = 0;

    (void) working_root;
    if (rows_out != 0) *rows_out = 0u;
    if (cols_out != 0) *cols_out = 0u;
    if (nnz_out != 0) *nnz_out = 0u;
    if (source_filename == 0 || output_filename == 0 || cell_keep == 0 || gene_keep == 0) return 0;

    init(&runtime_service);
    init(&matrix);
    init(&storage);
    if (!load_header(source_filename, &matrix, &storage)) goto done;
    if (!get_dataset_h5_runtime_service(&storage, &runtime_service)) {
        init(&runtime_service);
        runtime_service.service_mode = dataset_runtime_service_mode_owner_hosted;
        runtime_service.live_write_mode = dataset_live_write_mode_append_only;
        runtime_service.prefer_pack_delivery = 1u;
        runtime_service.remote_pack_delivery = 0u;
        runtime_service.single_reader_coordinator = 1u;
        runtime_service.maintenance_lock_blocks_overwrite = 1u;
        runtime_service.canonical_generation = 1u;
        runtime_service.execution_plan_generation = 1u;
        runtime_service.pack_generation = 1u;
        runtime_service.service_epoch = 1u;
        runtime_service.active_read_generation = 1u;
        runtime_service.staged_write_generation = 1u;
    }

    file = H5Fopen(source_filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) goto done;
    datasets = H5Gopen2(file, datasets_group, H5P_DEFAULT);
    provenance = H5Gopen2(file, provenance_group, H5P_DEFAULT);
    matrix_group_h5 = H5Gopen2(file, matrix_group, H5P_DEFAULT);
    if (datasets < 0 || provenance < 0 || matrix_group_h5 < 0) goto done;
    if (!read_attr_u64(file, "num_datasets", &dataset_count)) goto done;
    dataset_formats.assign((std::size_t) dataset_count, 0u);
    dataset_rows.assign((std::size_t) dataset_count, 0u);
    cell_dataset_ids.assign((std::size_t) matrix.rows, 0u);
    cell_local_indices.assign((std::size_t) matrix.rows, 0u);
    feature_dataset_ids.assign((std::size_t) matrix.cols, 0u);
    feature_local_indices.assign((std::size_t) matrix.cols, 0u);
    dataset_feature_offsets.assign((std::size_t) dataset_count + 1u, 0u);
    partition_dataset_ids.assign((std::size_t) matrix.num_partitions, 0u);
    if (!read_text_column(datasets, "dataset_ids", &dataset_ids)
        || !read_text_column(datasets, "matrix_paths", &matrix_paths)
        || !read_text_column(datasets, "feature_paths", &feature_paths)
        || !read_text_column(datasets, "barcode_paths", &barcode_paths)
        || !read_text_column(datasets, "metadata_paths", &metadata_paths)
        || !read_dataset_1d(datasets, "formats", H5T_NATIVE_UINT32, dataset_count, dataset_formats.data())
        || !read_dataset_1d(datasets, "rows", H5T_NATIVE_UINT64, dataset_count, dataset_rows.data())
        || !read_text_column(provenance, "global_barcodes", &global_barcodes)
        || !read_dataset_1d(provenance, "cell_dataset_ids", H5T_NATIVE_UINT32, matrix.rows, cell_dataset_ids.data())
        || !read_dataset_1d(provenance, "cell_local_indices", H5T_NATIVE_UINT64, matrix.rows, cell_local_indices.data())
        || !read_text_column(provenance, "feature_ids", &feature_ids)
        || !read_text_column(provenance, "feature_names", &feature_names)
        || !read_text_column(provenance, "feature_types", &feature_types)
        || !read_dataset_1d(provenance, "feature_dataset_ids", H5T_NATIVE_UINT32, matrix.cols, feature_dataset_ids.data())
        || !read_dataset_1d(provenance, "feature_local_indices", H5T_NATIVE_UINT64, matrix.cols, feature_local_indices.data())
        || !read_dataset_1d(provenance, "dataset_feature_offsets", H5T_NATIVE_UINT64, dataset_count + 1u, dataset_feature_offsets.data())) {
        goto done;
    }
    dataset_feature_to_global.assign((std::size_t) (dataset_feature_offsets.empty() ? 0u : dataset_feature_offsets.back()), 0u);
    if ((!dataset_feature_to_global.empty()
         && !read_dataset_1d(provenance,
                             "dataset_feature_to_global",
                             H5T_NATIVE_UINT32,
                             dataset_feature_offsets.back(),
                             dataset_feature_to_global.data()))
        || !read_dataset_1d(matrix_group_h5,
                            "partition_dataset_ids",
                            H5T_NATIVE_UINT32,
                            matrix.num_partitions,
                            partition_dataset_ids.data())) {
        goto done;
    }
    H5Gclose(matrix_group_h5);
    matrix_group_h5 = (hid_t) -1;
    H5Gclose(provenance);
    provenance = (hid_t) -1;
    H5Gclose(datasets);
    datasets = (hid_t) -1;
    H5Fclose(file);
    file = (hid_t) -1;

    filtered_dataset_rows.assign((std::size_t) dataset_count, 0u);
    filtered_dataset_cols.assign((std::size_t) dataset_count, 0u);
    filtered_dataset_nnz.assign((std::size_t) dataset_count, 0u);
    filtered_dataset_row_begin.assign((std::size_t) dataset_count, 0u);
    filtered_dataset_row_end.assign((std::size_t) dataset_count, 0u);
    col_remap.assign((std::size_t) matrix.cols, std::numeric_limits<std::uint32_t>::max());

    for (std::uint64_t row = 0u; row < matrix.rows; ++row) {
        if (cell_keep[row] == 0u) continue;
        const std::uint32_t dataset_id = row < cell_dataset_ids.size() ? cell_dataset_ids[(std::size_t) row] : 0u;
        append_text_value(&filtered_global_barcodes, text_column_value(global_barcodes, (std::uint32_t) row));
        filtered_cell_dataset_ids.push_back(dataset_id);
        filtered_cell_local_indices.push_back(row < cell_local_indices.size() ? cell_local_indices[(std::size_t) row] : 0u);
        if (dataset_id < filtered_dataset_rows.size()) ++filtered_dataset_rows[(std::size_t) dataset_id];
        ++final_rows;
    }
    {
        std::uint64_t row_cursor = 0u;
        for (std::size_t dataset_idx = 0; dataset_idx < (std::size_t) dataset_count; ++dataset_idx) {
            filtered_dataset_row_begin[dataset_idx] = row_cursor;
            row_cursor += filtered_dataset_rows[dataset_idx];
            filtered_dataset_row_end[dataset_idx] = row_cursor;
        }
    }

    for (std::uint32_t gene = 0u; gene < matrix.cols; ++gene) {
        if (gene_keep[gene] == 0u) continue;
        col_remap[(std::size_t) gene] = (std::uint32_t) final_cols;
        append_text_value(&filtered_feature_ids, text_column_value(feature_ids, gene));
        append_text_value(&filtered_feature_names, text_column_value(feature_names, gene));
        append_text_value(&filtered_feature_types, text_column_value(feature_types, gene));
        filtered_feature_dataset_ids.push_back(gene < feature_dataset_ids.size() ? feature_dataset_ids[(std::size_t) gene] : 0u);
        filtered_feature_local_indices.push_back(gene < feature_local_indices.size() ? feature_local_indices[(std::size_t) gene] : 0u);
        ++final_cols;
    }

    filtered_dataset_feature_offsets.assign(1u, 0u);
    for (std::size_t dataset_idx = 0; dataset_idx < (std::size_t) dataset_count; ++dataset_idx) {
        const std::uint64_t begin = dataset_idx < dataset_feature_offsets.size() ? dataset_feature_offsets[dataset_idx] : 0u;
        const std::uint64_t end = dataset_idx + 1u < dataset_feature_offsets.size() ? dataset_feature_offsets[dataset_idx + 1u] : begin;
        for (std::uint64_t idx = begin; idx < end && idx < dataset_feature_to_global.size(); ++idx) {
            const std::uint32_t global_col = dataset_feature_to_global[(std::size_t) idx];
            if (global_col >= col_remap.size()) continue;
            if (col_remap[(std::size_t) global_col] == std::numeric_limits<std::uint32_t>::max()) continue;
            filtered_dataset_feature_to_global.push_back(col_remap[(std::size_t) global_col]);
        }
        filtered_dataset_cols[dataset_idx] =
            (std::uint64_t) filtered_dataset_feature_to_global.size() - filtered_dataset_feature_offsets.back();
        filtered_dataset_feature_offsets.push_back((std::uint64_t) filtered_dataset_feature_to_global.size());
    }

    filtered_parts.resize((std::size_t) matrix.num_partitions);
    filtered_bucketed_parts.resize((std::size_t) matrix.num_partitions);
    filtered_part_ptrs.resize((std::size_t) matrix.num_partitions, 0);
    part_rows.assign((std::size_t) matrix.num_partitions, 0u);
    part_nnz.assign((std::size_t) matrix.num_partitions, 0u);
    part_aux.assign((std::size_t) matrix.num_partitions, 0u);
    part_row_offsets.assign((std::size_t) matrix.num_partitions + 1u, 0u);
    part_dataset_ids.assign((std::size_t) matrix.num_partitions, 0u);
    part_codec_ids.assign((std::size_t) matrix.num_partitions, 0u);
    part_formats.assign((std::size_t) matrix.num_partitions, dataset_execution_format_bucketed_sliced_ell);
    part_block_sizes.assign((std::size_t) matrix.num_partitions, 0u);
    part_bucket_counts.assign((std::size_t) matrix.num_partitions, 0u);
    part_fill_ratios.assign((std::size_t) matrix.num_partitions, 0.0f);
    part_execution_bytes.assign((std::size_t) matrix.num_partitions, 0u);
    part_blocked_ell_bytes.assign((std::size_t) matrix.num_partitions, 0u);
    part_bucketed_blocked_ell_bytes.assign((std::size_t) matrix.num_partitions, 0u);
    part_slice_counts.assign((std::size_t) matrix.num_partitions, 0u);
    part_slice_rows.assign((std::size_t) matrix.num_partitions, 0u);
    part_sliced_bytes.assign((std::size_t) matrix.num_partitions, 0u);
    part_bucketed_sliced_bytes.assign((std::size_t) matrix.num_partitions, 0u);

    for (unsigned long part_id = 0u; part_id < matrix.num_partitions; ++part_id) {
        std::uint32_t live_rows = 0u;
        std::uint32_t live_nnz = 0u;
        std::uint32_t bucket_count = 1u;
        std::uint64_t bucketed_bytes = 0u;
        if (!fetch_partition(&matrix, &storage, part_id)) goto done;
        init(&filtered_bucketed_parts[(std::size_t) part_id]);
        if (!build_filtered_sliced_ell_part_from_sliced(matrix.parts[part_id],
                                                        cell_keep,
                                                        matrix.partition_offsets[part_id],
                                                        col_remap.data(),
                                                        (std::uint32_t) final_cols,
                                                        &filtered_parts[(std::size_t) part_id],
                                                        &live_rows,
                                                        &live_nnz)) {
            goto done;
        }
        part_rows[(std::size_t) part_id] = live_rows;
        part_nnz[(std::size_t) part_id] = live_nnz;
        if (live_rows != 0u) {
            if (!choose_bucket_count_for_sliced_part(&filtered_parts[(std::size_t) part_id], &bucket_count, &bucketed_bytes)
                || !build_bucketed_sliced_ell_partition(&filtered_bucketed_parts[(std::size_t) part_id],
                                                        &filtered_parts[(std::size_t) part_id],
                                                        bucket_count,
                                                        &bucketed_bytes)) {
                goto done;
            }
            filtered_part_ptrs[(std::size_t) part_id] = &filtered_bucketed_parts[(std::size_t) part_id];
            part_aux[(std::size_t) part_id] = partition_aux(&filtered_parts[(std::size_t) part_id]);
            part_slice_counts[(std::size_t) part_id] = filtered_parts[(std::size_t) part_id].slice_count;
            part_slice_rows[(std::size_t) part_id] = sparse::uniform_slice_rows(&filtered_parts[(std::size_t) part_id]);
            part_sliced_bytes[(std::size_t) part_id] = (std::uint64_t) sparse::bytes(&filtered_parts[(std::size_t) part_id]);
            part_bucketed_sliced_bytes[(std::size_t) part_id] = bucketed_bytes;
            part_execution_bytes[(std::size_t) part_id] = bucketed_bytes;
            part_bucket_counts[(std::size_t) part_id] = filtered_bucketed_parts[(std::size_t) part_id].segment_count;
        }
        part_row_offsets[(std::size_t) part_id + 1u] = part_row_offsets[(std::size_t) part_id] + live_rows;
        part_dataset_ids[(std::size_t) part_id] =
            part_id < partition_dataset_ids.size() ? partition_dataset_ids[(std::size_t) part_id] : 0u;
        if (part_dataset_ids[(std::size_t) part_id] < filtered_dataset_nnz.size()) {
            filtered_dataset_nnz[(std::size_t) part_dataset_ids[(std::size_t) part_id]] += live_nnz;
        }
        final_nnz += live_nnz;
    }

    compact_part_row_offsets.assign(1u, 0u);
    for (unsigned long part_id = 0u; part_id < matrix.num_partitions; ++part_id) {
        if (part_rows[(std::size_t) part_id] == 0u) continue;
        compact_part_ptrs.push_back(filtered_part_ptrs[(std::size_t) part_id]);
        compact_part_rows.push_back(part_rows[(std::size_t) part_id]);
        compact_part_nnz.push_back(part_nnz[(std::size_t) part_id]);
        compact_part_aux.push_back(part_aux[(std::size_t) part_id]);
        compact_part_dataset_ids.push_back(part_dataset_ids[(std::size_t) part_id]);
        compact_part_codec_ids.push_back(part_codec_ids[(std::size_t) part_id]);
        compact_part_row_offsets.push_back(compact_part_row_offsets.back() + part_rows[(std::size_t) part_id]);
    }

    compact_shard_offsets.assign(1u, 0u);
    {
        std::uint32_t compact_part_cursor = 0u;
        for (unsigned long shard_id = 0u; shard_id < matrix.num_shards; ++shard_id) {
            const unsigned long part_begin = matrix.shard_parts != 0 ? matrix.shard_parts[shard_id] : shard_id;
            const unsigned long part_end = matrix.shard_parts != 0
                ? matrix.shard_parts[shard_id + 1u]
                : std::min<unsigned long>(matrix.num_partitions, shard_id + 1u);
            std::uint32_t kept_in_shard = 0u;
            std::uint64_t shard_rows = 0u;
            for (unsigned long part_id = part_begin; part_id < part_end; ++part_id) {
                if (part_rows[(std::size_t) part_id] == 0u) continue;
                ++kept_in_shard;
                shard_rows += part_rows[(std::size_t) part_id];
            }
            if (kept_in_shard == 0u) continue;
            compact_shard_part_begin.push_back(compact_part_cursor);
            compact_part_cursor += kept_in_shard;
            compact_shard_part_end.push_back(compact_part_cursor);
            compact_shard_offsets.push_back(compact_shard_offsets.back() + shard_rows);
        }
    }

    dataset_view.count = (std::uint32_t) dataset_count;
    dataset_view.dataset_ids = dataset_ids.view();
    dataset_view.matrix_paths = matrix_paths.view();
    dataset_view.feature_paths = feature_paths.view();
    dataset_view.barcode_paths = barcode_paths.view();
    dataset_view.metadata_paths = metadata_paths.view();
    dataset_view.formats = dataset_formats.empty() ? nullptr : dataset_formats.data();
    dataset_view.row_begin = filtered_dataset_row_begin.empty() ? nullptr : filtered_dataset_row_begin.data();
    dataset_view.row_end = filtered_dataset_row_end.empty() ? nullptr : filtered_dataset_row_end.data();
    dataset_view.rows = filtered_dataset_rows.empty() ? nullptr : filtered_dataset_rows.data();
    dataset_view.cols = filtered_dataset_cols.empty() ? nullptr : filtered_dataset_cols.data();
    dataset_view.nnz = filtered_dataset_nnz.empty() ? nullptr : filtered_dataset_nnz.data();

    provenance_view.global_barcodes = filtered_global_barcodes.view();
    provenance_view.cell_dataset_ids = filtered_cell_dataset_ids.empty() ? nullptr : filtered_cell_dataset_ids.data();
    provenance_view.cell_local_indices = filtered_cell_local_indices.empty() ? nullptr : filtered_cell_local_indices.data();
    provenance_view.feature_ids = filtered_feature_ids.view();
    provenance_view.feature_names = filtered_feature_names.view();
    provenance_view.feature_types = filtered_feature_types.view();
    provenance_view.feature_dataset_ids = filtered_feature_dataset_ids.empty() ? nullptr : filtered_feature_dataset_ids.data();
    provenance_view.feature_local_indices = filtered_feature_local_indices.empty() ? nullptr : filtered_feature_local_indices.data();
    provenance_view.dataset_feature_offsets = filtered_dataset_feature_offsets.empty() ? nullptr : filtered_dataset_feature_offsets.data();
    provenance_view.dataset_feature_to_global = filtered_dataset_feature_to_global.empty() ? nullptr : filtered_dataset_feature_to_global.data();

    codec.codec_id = 0u;
    codec.family = dataset_codec_family_sliced_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = final_rows;
    layout.cols = final_cols;
    layout.nnz = final_nnz;
    layout.num_partitions = (std::uint64_t) compact_part_ptrs.size();
    layout.num_shards = compact_shard_offsets.empty() ? 0u : (std::uint64_t) compact_shard_offsets.size() - 1u;
    layout.partition_rows = compact_part_rows.empty() ? nullptr : compact_part_rows.data();
    layout.partition_nnz = compact_part_nnz.empty() ? nullptr : compact_part_nnz.data();
    layout.partition_axes = nullptr;
    layout.partition_aux = compact_part_aux.empty() ? nullptr : compact_part_aux.data();
    layout.partition_row_offsets = compact_part_row_offsets.empty() ? nullptr : compact_part_row_offsets.data();
    layout.partition_dataset_ids = compact_part_dataset_ids.empty() ? nullptr : compact_part_dataset_ids.data();
    layout.partition_codec_ids = compact_part_codec_ids.empty() ? nullptr : compact_part_codec_ids.data();
    layout.shard_offsets = compact_shard_offsets.empty() ? nullptr : compact_shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    temp_path = (fs::path(output_filename).parent_path() / (fs::path(output_filename).filename().string()
        + ".preprocess_finalize." + std::to_string((unsigned long long) ::getpid()) + ".tmp")).string();
    {
        std::error_code ec;
        fs::remove(temp_path, ec);
    }
    if (!create_dataset_sliced_ell_h5(temp_path.c_str(), &layout, &dataset_view, &provenance_view)) goto done;

    {
    std::vector<std::uint32_t> compact_part_formats(compact_part_ptrs.size(), dataset_execution_format_bucketed_sliced_ell);
    std::vector<std::uint32_t> compact_part_block_sizes(compact_part_ptrs.size(), 0u);
    std::vector<std::uint32_t> compact_part_bucket_counts(compact_part_ptrs.size(), 0u);
    std::vector<float> compact_part_fill_ratios(compact_part_ptrs.size(), 0.0f);
    std::vector<std::uint64_t> compact_part_execution_bytes(compact_part_ptrs.size(), 0u);
    std::vector<std::uint64_t> compact_part_blocked_ell_bytes(compact_part_ptrs.size(), 0u);
    std::vector<std::uint64_t> compact_part_bucketed_blocked_ell_bytes(compact_part_ptrs.size(), 0u);
    std::vector<std::uint32_t> compact_part_slice_counts(compact_part_ptrs.size(), 0u);
    std::vector<std::uint32_t> compact_part_slice_rows(compact_part_ptrs.size(), 0u);
    std::vector<std::uint64_t> compact_part_sliced_bytes(compact_part_ptrs.size(), 0u);
    std::vector<std::uint64_t> compact_part_bucketed_sliced_bytes(compact_part_ptrs.size(), 0u);
    for (std::size_t compact_part_id = 0u, source_part_id = 0u; source_part_id < (std::size_t) matrix.num_partitions; ++source_part_id) {
        if (part_rows[source_part_id] == 0u) continue;
        if (!append_sliced_ell_partition_h5(temp_path.c_str(), (unsigned long) compact_part_id, compact_part_ptrs[compact_part_id])) goto done;
        compact_part_bucket_counts[compact_part_id] = filtered_bucketed_parts[source_part_id].segment_count;
        compact_part_execution_bytes[compact_part_id] = part_execution_bytes[source_part_id];
        compact_part_slice_counts[compact_part_id] = part_slice_counts[source_part_id];
        compact_part_slice_rows[compact_part_id] = part_slice_rows[source_part_id];
        compact_part_sliced_bytes[compact_part_id] = part_sliced_bytes[source_part_id];
        compact_part_bucketed_sliced_bytes[compact_part_id] = part_bucketed_sliced_bytes[source_part_id];
        ++compact_part_id;
    }

    std::vector<std::uint32_t> compact_shard_formats(compact_shard_part_begin.size(), dataset_execution_format_bucketed_sliced_ell);
    std::vector<std::uint32_t> compact_shard_block_sizes(compact_shard_part_begin.size(), 0u);
    std::vector<std::uint32_t> compact_shard_bucketed_partition_counts(compact_shard_part_begin.size(), 0u);
    std::vector<std::uint32_t> compact_shard_bucketed_segment_counts(compact_shard_part_begin.size(), 0u);
    std::vector<float> compact_shard_fill_ratios(compact_shard_part_begin.size(), 0.0f);
    std::vector<std::uint64_t> compact_shard_execution_bytes(compact_shard_part_begin.size(), 0u);
    std::vector<std::uint64_t> compact_shard_bucketed_blocked_ell_bytes(compact_shard_part_begin.size(), 0u);
    std::vector<std::uint32_t> compact_shard_slice_counts(compact_shard_part_begin.size(), 0u);
    std::vector<std::uint32_t> compact_shard_slice_rows(compact_shard_part_begin.size(), 0u);
    std::vector<std::uint64_t> compact_shard_bucketed_sliced_bytes(compact_shard_part_begin.size(), 0u);
    std::vector<std::uint32_t> compact_shard_pair_ids(compact_shard_part_begin.size(), 0u);
    std::vector<std::uint32_t> compact_shard_owner_node_ids(compact_shard_part_begin.size(), 0u);
    std::vector<std::uint32_t> compact_shard_owner_rank_ids(compact_shard_part_begin.size(), 0u);
    for (std::size_t shard_id = 0u; shard_id < compact_shard_part_begin.size(); ++shard_id) {
        const std::uint32_t part_begin = compact_shard_part_begin[shard_id];
        const std::uint32_t part_end = compact_shard_part_end[shard_id];
        std::uint32_t uniform_rows = 0u;
        int same_uniform_rows = 1;
        for (std::uint32_t part_id = part_begin; part_id < part_end; ++part_id) {
            compact_shard_bucketed_partition_counts[shard_id] += 1u;
            compact_shard_bucketed_segment_counts[shard_id] += compact_part_bucket_counts[part_id];
            compact_shard_slice_counts[shard_id] += compact_part_slice_counts[part_id];
            compact_shard_execution_bytes[shard_id] += compact_part_execution_bytes[part_id];
            compact_shard_bucketed_sliced_bytes[shard_id] += compact_part_bucketed_sliced_bytes[part_id];
            if (part_id == part_begin) {
                uniform_rows = compact_part_slice_rows[part_id];
            } else if (uniform_rows != compact_part_slice_rows[part_id]) {
                same_uniform_rows = 0;
            }
        }
        compact_shard_slice_rows[shard_id] = same_uniform_rows ? uniform_rows : 0u;
    }

    execution_view.partition_count = (std::uint32_t) compact_part_ptrs.size();
    execution_view.partition_execution_formats = compact_part_formats.empty() ? nullptr : compact_part_formats.data();
    execution_view.partition_blocked_ell_block_sizes = compact_part_block_sizes.empty() ? nullptr : compact_part_block_sizes.data();
    execution_view.partition_blocked_ell_bucket_counts = compact_part_bucket_counts.empty() ? nullptr : compact_part_bucket_counts.data();
    execution_view.partition_blocked_ell_fill_ratios = compact_part_fill_ratios.empty() ? nullptr : compact_part_fill_ratios.data();
    execution_view.partition_execution_bytes = compact_part_execution_bytes.empty() ? nullptr : compact_part_execution_bytes.data();
    execution_view.partition_blocked_ell_bytes = compact_part_blocked_ell_bytes.empty() ? nullptr : compact_part_blocked_ell_bytes.data();
    execution_view.partition_bucketed_blocked_ell_bytes =
        compact_part_bucketed_blocked_ell_bytes.empty() ? nullptr : compact_part_bucketed_blocked_ell_bytes.data();
    execution_view.partition_sliced_ell_slice_counts = compact_part_slice_counts.empty() ? nullptr : compact_part_slice_counts.data();
    execution_view.partition_sliced_ell_slice_rows = compact_part_slice_rows.empty() ? nullptr : compact_part_slice_rows.data();
    execution_view.partition_sliced_ell_bytes = compact_part_sliced_bytes.empty() ? nullptr : compact_part_sliced_bytes.data();
    execution_view.partition_bucketed_sliced_ell_bytes =
        compact_part_bucketed_sliced_bytes.empty() ? nullptr : compact_part_bucketed_sliced_bytes.data();
    execution_view.shard_count = (std::uint32_t) compact_shard_part_begin.size();
    execution_view.shard_execution_formats = compact_shard_formats.empty() ? nullptr : compact_shard_formats.data();
    execution_view.shard_blocked_ell_block_sizes = compact_shard_block_sizes.empty() ? nullptr : compact_shard_block_sizes.data();
    execution_view.shard_bucketed_partition_counts =
        compact_shard_bucketed_partition_counts.empty() ? nullptr : compact_shard_bucketed_partition_counts.data();
    execution_view.shard_bucketed_segment_counts =
        compact_shard_bucketed_segment_counts.empty() ? nullptr : compact_shard_bucketed_segment_counts.data();
    execution_view.shard_blocked_ell_fill_ratios = compact_shard_fill_ratios.empty() ? nullptr : compact_shard_fill_ratios.data();
    execution_view.shard_execution_bytes = compact_shard_execution_bytes.empty() ? nullptr : compact_shard_execution_bytes.data();
    execution_view.shard_bucketed_blocked_ell_bytes =
        compact_shard_bucketed_blocked_ell_bytes.empty() ? nullptr : compact_shard_bucketed_blocked_ell_bytes.data();
    execution_view.shard_sliced_ell_slice_counts =
        compact_shard_slice_counts.empty() ? nullptr : compact_shard_slice_counts.data();
    execution_view.shard_sliced_ell_slice_rows =
        compact_shard_slice_rows.empty() ? nullptr : compact_shard_slice_rows.data();
    execution_view.shard_bucketed_sliced_ell_bytes =
        compact_shard_bucketed_sliced_bytes.empty() ? nullptr : compact_shard_bucketed_sliced_bytes.data();
    execution_view.shard_preferred_pair_ids = compact_shard_pair_ids.empty() ? nullptr : compact_shard_pair_ids.data();
    execution_view.shard_owner_node_ids = compact_shard_owner_node_ids.empty() ? nullptr : compact_shard_owner_node_ids.data();
    execution_view.shard_owner_rank_ids = compact_shard_owner_rank_ids.empty() ? nullptr : compact_shard_owner_rank_ids.data();
    execution_view.preferred_base_format = dataset_execution_format_bucketed_sliced_ell;
    if (!append_dataset_execution_h5(temp_path.c_str(), &execution_view)) goto done;
    if (!append_dataset_runtime_service_h5(temp_path.c_str(), &runtime_service)) goto done;
    if (embedded_metadata != nullptr && !append_dataset_embedded_metadata_h5(temp_path.c_str(), embedded_metadata)) goto done;
    if (observation_metadata != nullptr && !append_dataset_observation_annotations_h5(temp_path.c_str(), observation_metadata)) goto done;
    if (feature_metadata != nullptr && !append_dataset_feature_metadata_h5(temp_path.c_str(), feature_metadata)) goto done;
    if (attributes != nullptr && !append_dataset_user_attributes_h5(temp_path.c_str(), attributes)) goto done;
    if (preprocess != nullptr) {
        finalized_preprocess = *preprocess;
        finalized_preprocess.processed_matrix_available = 1u;
        finalized_preprocess.rows = final_rows;
        finalized_preprocess.cols = (std::uint32_t) final_cols;
        finalized_preprocess.nnz = final_nnz;
        filtered_pre_cell_total_counts.reserve((std::size_t) final_rows);
        filtered_pre_cell_mito_counts.reserve((std::size_t) final_rows);
        filtered_pre_cell_max_counts.reserve((std::size_t) final_rows);
        filtered_pre_cell_detected_genes.reserve((std::size_t) final_rows);
        filtered_pre_cell_keep.reserve((std::size_t) final_rows);
        for (std::uint64_t row = 0u; row < matrix.rows; ++row) {
            if (cell_keep[row] == 0u) continue;
            if (preprocess->cell_total_counts != nullptr) filtered_pre_cell_total_counts.push_back(preprocess->cell_total_counts[row]);
            if (preprocess->cell_mito_counts != nullptr) filtered_pre_cell_mito_counts.push_back(preprocess->cell_mito_counts[row]);
            if (preprocess->cell_max_counts != nullptr) filtered_pre_cell_max_counts.push_back(preprocess->cell_max_counts[row]);
            if (preprocess->cell_detected_genes != nullptr) filtered_pre_cell_detected_genes.push_back(preprocess->cell_detected_genes[row]);
            if (preprocess->cell_keep != nullptr) filtered_pre_cell_keep.push_back(1u);
        }
        filtered_pre_gene_sum.reserve((std::size_t) final_cols);
        filtered_pre_gene_sq_sum.reserve((std::size_t) final_cols);
        filtered_pre_gene_detected_cells.reserve((std::size_t) final_cols);
        filtered_pre_gene_keep.reserve((std::size_t) final_cols);
        filtered_pre_gene_flags.reserve((std::size_t) final_cols);
        for (std::uint32_t gene = 0u; gene < matrix.cols; ++gene) {
            if (gene_keep[gene] == 0u) continue;
            if (preprocess->gene_sum != nullptr) filtered_pre_gene_sum.push_back(preprocess->gene_sum[gene]);
            if (preprocess->gene_sq_sum != nullptr) filtered_pre_gene_sq_sum.push_back(preprocess->gene_sq_sum[gene]);
            if (preprocess->gene_detected_cells != nullptr) filtered_pre_gene_detected_cells.push_back(preprocess->gene_detected_cells[gene]);
            if (preprocess->gene_keep != nullptr) filtered_pre_gene_keep.push_back(1u);
            if (preprocess->gene_flags != nullptr) filtered_pre_gene_flags.push_back(preprocess->gene_flags[gene]);
        }
        finalized_preprocess.cell_total_counts = filtered_pre_cell_total_counts.empty() ? nullptr : filtered_pre_cell_total_counts.data();
        finalized_preprocess.cell_mito_counts = filtered_pre_cell_mito_counts.empty() ? nullptr : filtered_pre_cell_mito_counts.data();
        finalized_preprocess.cell_max_counts = filtered_pre_cell_max_counts.empty() ? nullptr : filtered_pre_cell_max_counts.data();
        finalized_preprocess.cell_detected_genes = filtered_pre_cell_detected_genes.empty() ? nullptr : filtered_pre_cell_detected_genes.data();
        finalized_preprocess.cell_keep = filtered_pre_cell_keep.empty() ? nullptr : filtered_pre_cell_keep.data();
        finalized_preprocess.gene_sum = filtered_pre_gene_sum.empty() ? nullptr : filtered_pre_gene_sum.data();
        finalized_preprocess.gene_sq_sum = filtered_pre_gene_sq_sum.empty() ? nullptr : filtered_pre_gene_sq_sum.data();
        finalized_preprocess.gene_detected_cells = filtered_pre_gene_detected_cells.empty() ? nullptr : filtered_pre_gene_detected_cells.data();
        finalized_preprocess.gene_keep = filtered_pre_gene_keep.empty() ? nullptr : filtered_pre_gene_keep.data();
        finalized_preprocess.gene_flags = filtered_pre_gene_flags.empty() ? nullptr : filtered_pre_gene_flags.data();
        if (!append_dataset_preprocess_h5(temp_path.c_str(), &finalized_preprocess)) goto done;
    }

    {
        std::error_code ec;
        fs::rename(temp_path, output_filename, ec);
        if (ec) goto done;
    }
    if (rows_out != nullptr) *rows_out = final_rows;
    if (cols_out != nullptr) *cols_out = final_cols;
    if (nnz_out != nullptr) *nnz_out = final_nnz;
    ok = 1;
    }

done:
    if (!ok && !temp_path.empty()) {
        std::error_code ec;
        fs::remove(temp_path, ec);
    }
    for (std::size_t i = 0; i < filtered_bucketed_parts.size(); ++i) clear(&filtered_bucketed_parts[i]);
    for (std::size_t i = 0; i < filtered_parts.size(); ++i) sparse::clear(&filtered_parts[i]);
    clear(&storage);
    clear(&matrix);
    if (matrix_group_h5 >= 0) H5Gclose(matrix_group_h5);
    if (provenance >= 0) H5Gclose(provenance);
    if (datasets >= 0) H5Gclose(datasets);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int finalize_preprocessed_dataset_h5_to_output(const char *source_filename,
                                               const char *output_filename,
                                               const std::uint8_t *cell_keep,
                                               const std::uint8_t *gene_keep,
                                               const dataset_embedded_metadata_view *embedded_metadata,
                                               const dataset_annotation_view *observation_metadata,
                                               const dataset_feature_metadata_view *feature_metadata,
                                               const dataset_user_attribute_view *attributes,
                                               const dataset_preprocess_view *preprocess,
                                               const char *working_root,
                                               std::uint64_t *rows_out,
                                               std::uint64_t *cols_out,
                                               std::uint64_t *nnz_out) {
    hid_t file = (hid_t) -1;
    char matrix_format[64];
    int ok = 0;

    if (source_filename == 0 || output_filename == 0) return 0;
    matrix_format[0] = '\0';
    file = H5Fopen(source_filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) goto done;
    if (!read_attr_string(file, "matrix_format", matrix_format, sizeof(matrix_format))) goto done;
    H5Fclose(file);
    file = (hid_t) -1;

    if (std::strcmp(matrix_format, "sliced_ell") == 0) {
        ok = finalize_preprocessed_sliced_ell_dataset_h5_to_output(source_filename,
                                                                   output_filename,
                                                                   cell_keep,
                                                                   gene_keep,
                                                                   embedded_metadata,
                                                                   observation_metadata,
                                                                   feature_metadata,
                                                                   attributes,
                                                                   preprocess,
                                                                   working_root,
                                                                   rows_out,
                                                                   cols_out,
                                                                   nnz_out);
    } else if (std::strcmp(matrix_format, "blocked_ell") == 0) {
        ok = finalize_preprocessed_blocked_ell_dataset_h5_to_output(source_filename,
                                                                    output_filename,
                                                                    cell_keep,
                                                                    gene_keep,
                                                                    embedded_metadata,
                                                                    observation_metadata,
                                                                    feature_metadata,
                                                                    attributes,
                                                                    preprocess,
                                                                    working_root,
                                                                    rows_out,
                                                                    cols_out,
                                                                    nnz_out);
    }

done:
    if (file >= 0) H5Fclose(file);
    return ok;
}

int finalize_preprocessed_blocked_ell_dataset_h5(const char *filename,
                                                 const std::uint8_t *cell_keep,
                                                 const std::uint8_t *gene_keep,
                                                 const dataset_embedded_metadata_view *embedded_metadata,
                                                 const dataset_annotation_view *observation_metadata,
                                                 const dataset_feature_metadata_view *feature_metadata,
                                                 const dataset_user_attribute_view *attributes,
                                                 const dataset_preprocess_view *preprocess,
                                                 const char *working_root,
                                                 std::uint64_t *rows_out,
                                                 std::uint64_t *cols_out,
                                                 std::uint64_t *nnz_out) {
    return finalize_preprocessed_blocked_ell_dataset_h5_to_output(filename,
                                                                  filename,
                                                                  cell_keep,
                                                                  gene_keep,
                                                                  embedded_metadata,
                                                                  observation_metadata,
                                                                  feature_metadata,
                                                                  attributes,
                                                                  preprocess,
                                                                  working_root,
                                                                  rows_out,
                                                                  cols_out,
                                                                  nnz_out);
}

} // namespace cellshard
