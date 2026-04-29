#pragma once

static inline int convert_manifest_dataset_to_hdf5(const manifest *m,
                                                   const char *out_path,
                                                   const dataset_h5_convert_options *opts) {
    namespace fs = std::filesystem;
    std::vector<dataset_dataset_plan> plans;
    common::text_column dataset_ids;
    common::text_column matrix_paths;
    common::text_column feature_paths;
    common::text_column barcode_paths;
    common::text_column metadata_paths;
    common::text_column global_barcodes;
    common::text_column global_feature_ids;
    common::text_column global_feature_names;
    common::text_column global_feature_types;
    owned_buffer<std::uint32_t> dataset_formats;
    owned_buffer<std::uint64_t> dataset_row_begin;
    owned_buffer<std::uint64_t> dataset_row_end;
    owned_buffer<std::uint64_t> dataset_rows;
    owned_buffer<std::uint64_t> dataset_cols;
    owned_buffer<std::uint64_t> dataset_nnz;
    owned_buffer<std::uint32_t> cell_dataset_ids;
    owned_buffer<std::uint64_t> cell_local_indices;
    owned_buffer<std::uint32_t> feature_dataset_ids;
    owned_buffer<std::uint64_t> feature_local_indices;
    owned_buffer<std::uint64_t> dataset_feature_offsets;
    owned_buffer<std::uint32_t> dataset_feature_to_global;
    owned_buffer<std::uint64_t> part_rows;
    owned_buffer<std::uint64_t> part_nnz;
    owned_buffer<std::uint32_t> part_axes;
    owned_buffer<std::uint64_t> part_aux;
    owned_buffer<std::uint64_t> part_row_offsets;
    owned_buffer<std::uint32_t> part_dataset_ids;
    owned_buffer<std::uint32_t> part_codec_ids;
    owned_buffer<std::uint64_t> part_bytes;
    owned_buffer<std::uint64_t> shard_offsets;
    std::unordered_map<std::string, std::uint32_t> feature_map;
    cellshard::dataset_codec_descriptor codec;
    cellshard::dataset_layout_view layout;
    cellshard::dataset_dataset_table_view dataset_view;
    cellshard::dataset_provenance_view provenance_view;
    unsigned int manifest_i = 0;
    unsigned int dataset_idx = 0;
    unsigned long global_rows = 0;
    unsigned long global_parts = 0;
    std::string spool_root;
    int spool_ready = 0;
    partition shard_plan;
    int ok = 0;

    if (m == 0 || out_path == 0 || opts == 0) return 0;

    common::init(&dataset_ids);
    common::init(&matrix_paths);
    common::init(&feature_paths);
    common::init(&barcode_paths);
    common::init(&metadata_paths);
    common::init(&global_barcodes);
    common::init(&global_feature_ids);
    common::init(&global_feature_names);
    common::init(&global_feature_types);
    init(&shard_plan);
    plans.reserve(m->count);
    part_row_offsets.push_back(0ull);
    spool_root = build_ingest_spool_root(out_path, opts->working_root, opts->cache_root);
    if (!prepare_ingest_spool_root(spool_root)) goto done;
    spool_ready = 1;

    for (manifest_i = 0; manifest_i < m->count; ++manifest_i) {
        common::barcode_table barcodes;
        common::feature_table features;
        mtx::header header;
        unsigned long *row_nnz = 0;
        unsigned long *row_offsets = 0;
        unsigned long *part_nnz_raw = 0;
        unsigned long num_parts = 0;
        dataset_dataset_plan plan;
        unsigned long local_part = 0;
        unsigned int feature_i = 0;

        if (format_at(m, manifest_i) != source_mtx
            && format_at(m, manifest_i) != source_tenx_mtx
            && format_at(m, manifest_i) != source_h5ad) continue;

        common::init(&barcodes);
        common::init(&features);
        mtx::init(&header);
        if (!scan_source_row_nnz(m, manifest_i, &header, &row_nnz, opts->reader_bytes)) {
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }
        if (!mtx::plan_row_partitions_by_nnz(row_nnz, header.rows, opts->max_part_nnz, &row_offsets, &num_parts)) {
            std::free(row_nnz);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }
        if (!mtx::build_part_nnz_from_row_nnz(row_nnz, row_offsets, num_parts, &part_nnz_raw)) {
            std::free(row_nnz);
            std::free(row_offsets);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }

        if (!load_source_barcodes(m, manifest_i, &barcodes)) {
            std::free(row_nnz);
            std::free(row_offsets);
            std::free(part_nnz_raw);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }
        if (!load_source_features(m, manifest_i, &features)) {
            std::free(row_nnz);
            std::free(row_offsets);
            std::free(part_nnz_raw);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }
        if (common::count(&barcodes) != header.rows || common::count(&features) != header.cols) {
            std::free(row_nnz);
            std::free(row_offsets);
            std::free(part_nnz_raw);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }

        plan.manifest_idx = manifest_i;
        plan.dataset_idx = dataset_idx;
        plan.header = header;
        plan.global_row_begin = global_rows;
        plan.global_part_begin = global_parts;
        plan.row_offsets.assign_copy(row_offsets, (std::size_t) num_parts + 1u);
        plan.part_nnz.assign_copy(part_nnz_raw, (std::size_t) num_parts);
        plan.part_rows.resize((std::size_t) num_parts);
        plan.part_bytes.resize((std::size_t) num_parts);
        plan.part_aux.resize((std::size_t) num_parts);
        plan.spool_paths.resize((std::size_t) num_parts);
        plan.feature_to_global.resize((std::size_t) header.cols);

        for (local_part = 0; local_part < num_parts; ++local_part) {
            const unsigned long rows = row_offsets[local_part + 1ul] - row_offsets[local_part];
            plan.part_rows[local_part] = rows;
            plan.part_bytes[local_part] = (unsigned long) standard_csr_bytes(rows, part_nnz_raw[local_part]);
            part_rows.push_back((std::uint64_t) rows);
            part_nnz.push_back((std::uint64_t) part_nnz_raw[local_part]);
            part_axes.push_back(0u);
            part_aux.push_back(0ull);
            part_dataset_ids.push_back((std::uint32_t) dataset_idx);
            part_codec_ids.push_back(0u);
            part_bytes.push_back((std::uint64_t) plan.part_bytes[local_part]);
            ++global_parts;
        }

        for (local_part = 0; local_part < num_parts; ++local_part) {
            global_rows += plan.part_rows[local_part];
            part_row_offsets.push_back((std::uint64_t) global_rows);
        }

        if (!common::append(&dataset_ids, dataset_id_at(m, manifest_i), std::strlen(dataset_id_at(m, manifest_i)))) goto done;
        if (!common::append(&matrix_paths, matrix_path_at(m, manifest_i), std::strlen(matrix_path_at(m, manifest_i)))) goto done;
        if (!common::append(&feature_paths,
                            feature_path_at(m, manifest_i) != 0 ? feature_path_at(m, manifest_i) : "",
                            std::strlen(feature_path_at(m, manifest_i) != 0 ? feature_path_at(m, manifest_i) : ""))) goto done;
        if (!common::append(&barcode_paths,
                            barcode_path_at(m, manifest_i) != 0 ? barcode_path_at(m, manifest_i) : "",
                            std::strlen(barcode_path_at(m, manifest_i) != 0 ? barcode_path_at(m, manifest_i) : ""))) goto done;
        if (!common::append(&metadata_paths,
                            metadata_path_at(m, manifest_i) != 0 ? metadata_path_at(m, manifest_i) : "",
                            std::strlen(metadata_path_at(m, manifest_i) != 0 ? metadata_path_at(m, manifest_i) : ""))) goto done;
        dataset_formats.push_back(format_at(m, manifest_i));
        dataset_row_begin.push_back((std::uint64_t) plan.global_row_begin);
        dataset_row_end.push_back((std::uint64_t) (plan.global_row_begin + header.rows));
        dataset_rows.push_back((std::uint64_t) header.rows);
        dataset_cols.push_back((std::uint64_t) header.cols);
        dataset_nnz.push_back((std::uint64_t) header.nnz_file);

        for (feature_i = 0; feature_i < common::count(&features); ++feature_i) {
            const char *feature_id = common::id(&features, feature_i);
            const char *feature_name = common::name(&features, feature_i);
            const char *feature_type = common::type(&features, feature_i);
            std::string key = std::string(feature_id != 0 ? feature_id : "")
                + "\t" + std::string(feature_name != 0 ? feature_name : "")
                + "\t" + std::string(feature_type != 0 ? feature_type : "");
            std::unordered_map<std::string, std::uint32_t>::const_iterator hit = feature_map.find(key);
            std::uint32_t global_feature = 0u;

            if (hit == feature_map.end()) {
                global_feature = (std::uint32_t) feature_dataset_ids.size();
                feature_map.insert(std::make_pair(key, global_feature));
                if (!common::append(&global_feature_ids, feature_id != 0 ? feature_id : "", std::strlen(feature_id != 0 ? feature_id : ""))) goto done;
                if (!common::append(&global_feature_names, feature_name != 0 ? feature_name : "", std::strlen(feature_name != 0 ? feature_name : ""))) goto done;
                if (!common::append(&global_feature_types, feature_type != 0 ? feature_type : "", std::strlen(feature_type != 0 ? feature_type : ""))) goto done;
                feature_dataset_ids.push_back((std::uint32_t) dataset_idx);
                feature_local_indices.push_back((std::uint64_t) feature_i);
            } else {
                global_feature = hit->second;
            }
            plan.feature_to_global[feature_i] = global_feature;
            dataset_feature_to_global.push_back(global_feature);
        }
        dataset_feature_offsets.push_back((std::uint64_t) dataset_feature_to_global.size() - (std::uint64_t) header.cols);

        for (feature_i = 0; feature_i < common::count(&barcodes); ++feature_i) {
            const char *barcode = common::get(&barcodes, feature_i);
            if (!common::append(&global_barcodes, barcode != 0 ? barcode : "", std::strlen(barcode != 0 ? barcode : ""))) goto done;
            cell_dataset_ids.push_back((std::uint32_t) dataset_idx);
            cell_local_indices.push_back((std::uint64_t) feature_i);
        }

        plans.push_back(plan);
        ++dataset_idx;

        std::free(row_nnz);
        std::free(row_offsets);
        std::free(part_nnz_raw);
        common::clear(&barcodes);
        common::clear(&features);
    }

    if (plans.empty()) goto done;
    dataset_feature_offsets.push_back((std::uint64_t) dataset_feature_to_global.size());
    for (manifest_i = 0; manifest_i < plans.size(); ++manifest_i) {
        partition windows;
        sharded<sparse::compressed> window_compressed;
        sharded<sparse::coo> window_view;
        sparse::sliced_ell sliced_part;
        unsigned long window_i = 0;

        init(&windows);
        init(&window_compressed);
        init(&window_view);
        sparse::init(&sliced_part);
        if (!build_by_bytes(&windows,
                            plans[manifest_i].part_rows.data(),
                            plans[manifest_i].part_bytes.data(),
                            (unsigned long) plans[manifest_i].part_rows.size(),
                            opts->convert_window_bytes)) {
            clear(&windows);
            goto done;
        }

        for (window_i = 0; window_i < windows.count; ++window_i) {
            unsigned long local_part = 0;
            const int have_compressed =
                load_source_part_window_compressed(m,
                                                  plans[manifest_i].manifest_idx,
                                                  &plans[manifest_i].header,
                                                  plans[manifest_i].row_offsets.data(),
                                                  plans[manifest_i].part_nnz.data(),
                                                  (unsigned long) plans[manifest_i].part_rows.size(),
                                                  windows.ranges[window_i].part_begin,
                                                  windows.ranges[window_i].part_end,
                                                  &window_compressed,
                                                  opts->reader_bytes);
            if (!have_compressed
                && !load_source_part_window_coo(m,
                                               plans[manifest_i].manifest_idx,
                                               &plans[manifest_i].header,
                                               plans[manifest_i].row_offsets.data(),
                                               plans[manifest_i].part_nnz.data(),
                                               (unsigned long) plans[manifest_i].part_rows.size(),
                                               windows.ranges[window_i].part_begin,
                                               windows.ranges[window_i].part_end,
                                               &window_view,
                                               opts->reader_bytes)) {
                clear(&windows);
                clear(&window_compressed);
                clear(&window_view);
                sparse::clear(&sliced_part);
                goto done;
            }

            for (local_part = 0;
                 local_part < (have_compressed ? window_compressed.num_partitions : window_view.num_partitions);
                 ++local_part) {
                const unsigned long global_part_id = plans[manifest_i].global_part_begin + windows.ranges[window_i].part_begin + local_part;
                cellshard::convert::sliced_ell_tune_result tune = {};
                const int converted = have_compressed
                    ? convert_compressed_part_to_sliced_ell_auto(window_compressed.parts[local_part],
                                                                 opts->device,
                                                                 &sliced_part,
                                                                 &tune)
                    : convert_coo_part_to_sliced_ell_auto(window_view.parts[local_part],
                                                          &sliced_part,
                                                          &tune);
                if (!converted) {
                    clear(&windows);
                    clear(&window_compressed);
                    clear(&window_view);
                    sparse::clear(&sliced_part);
                    goto done;
                }
                plans[manifest_i].part_aux[(std::size_t) (windows.ranges[window_i].part_begin + local_part)] =
                    cellshard::sparse::pack_sliced_ell_aux(sliced_part.slice_count, cellshard::sparse::total_slots(&sliced_part));
                plans[manifest_i].part_nnz[(std::size_t) (windows.ranges[window_i].part_begin + local_part)] =
                    (unsigned long) sliced_part.nnz;
                plans[manifest_i].part_bytes[(std::size_t) (windows.ranges[window_i].part_begin + local_part)] =
                    (unsigned long) sparse::bytes(&sliced_part);
                plans[manifest_i].spool_paths[(std::size_t) (windows.ranges[window_i].part_begin + local_part)] =
                    build_ingest_spool_part_path(spool_root, global_part_id);
                part_nnz[(std::size_t) global_part_id] = (std::uint64_t) sliced_part.nnz;
                part_aux[(std::size_t) global_part_id] = plans[manifest_i].part_aux[(std::size_t) (windows.ranges[window_i].part_begin + local_part)];
                part_bytes[(std::size_t) global_part_id] = (std::uint64_t) plans[manifest_i].part_bytes[(std::size_t) (windows.ranges[window_i].part_begin + local_part)];
                if (plans[manifest_i].spool_paths[(std::size_t) (windows.ranges[window_i].part_begin + local_part)].empty()
                    || !cellshard::store(plans[manifest_i].spool_paths[(std::size_t) (windows.ranges[window_i].part_begin + local_part)].c_str(), &sliced_part)) {
                    clear(&windows);
                    clear(&window_compressed);
                    clear(&window_view);
                    sparse::clear(&sliced_part);
                    goto done;
                }
                sparse::clear(&sliced_part);
                sparse::init(&sliced_part);
            }
            clear(&window_compressed);
            init(&window_compressed);
            clear(&window_view);
            init(&window_view);
        }

        clear(&windows);
        clear(&window_compressed);
        sparse::clear(&sliced_part);
    }
    if (!build_sliced_ell_shards(&shard_plan,
                                 (const unsigned long *) part_rows.data(),
                                 (const unsigned long *) part_nnz.data(),
                                 (const unsigned long *) part_aux.data(),
                                 (const unsigned long *) part_bytes.data(),
                                 (unsigned long) part_rows.size(),
                                 opts->target_shard_bytes)) goto done;
    shard_offsets.resize((std::size_t) shard_plan.count + 1u, 0ull);
    for (manifest_i = 0; manifest_i < shard_plan.count; ++manifest_i) {
        shard_offsets[manifest_i] = (std::uint64_t) shard_plan.ranges[manifest_i].row_begin;
    }
    shard_offsets[shard_plan.count] = (std::uint64_t) global_rows;

    codec.codec_id = 0u;
    codec.family = cellshard::dataset_codec_family_sliced_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = (std::uint64_t) global_rows;
    layout.cols = (std::uint64_t) feature_dataset_ids.size();
    layout.nnz = 0u;
    for (manifest_i = 0; manifest_i < part_nnz.size(); ++manifest_i) layout.nnz += part_nnz[manifest_i];
    layout.num_partitions = (std::uint64_t) part_rows.size();
    layout.num_shards = (std::uint64_t) shard_plan.count;
    layout.partition_rows = part_rows.data();
    layout.partition_nnz = part_nnz.data();
    layout.partition_axes = part_axes.data();
    layout.partition_aux = part_aux.data();
    layout.partition_row_offsets = part_row_offsets.data();
    layout.partition_dataset_ids = part_dataset_ids.data();
    layout.partition_codec_ids = part_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    dataset_view.count = dataset_idx;
    dataset_view.dataset_ids = as_text_view(&dataset_ids);
    dataset_view.matrix_paths = as_text_view(&matrix_paths);
    dataset_view.feature_paths = as_text_view(&feature_paths);
    dataset_view.barcode_paths = as_text_view(&barcode_paths);
    dataset_view.metadata_paths = as_text_view(&metadata_paths);
    dataset_view.formats = dataset_formats.data();
    dataset_view.row_begin = dataset_row_begin.data();
    dataset_view.row_end = dataset_row_end.data();
    dataset_view.rows = dataset_rows.data();
    dataset_view.cols = dataset_cols.data();
    dataset_view.nnz = dataset_nnz.data();

    provenance_view.global_barcodes = as_text_view(&global_barcodes);
    provenance_view.cell_dataset_ids = cell_dataset_ids.data();
    provenance_view.cell_local_indices = cell_local_indices.data();
    provenance_view.feature_ids = as_text_view(&global_feature_ids);
    provenance_view.feature_names = as_text_view(&global_feature_names);
    provenance_view.feature_types = as_text_view(&global_feature_types);
    provenance_view.feature_dataset_ids = feature_dataset_ids.data();
    provenance_view.feature_local_indices = feature_local_indices.data();
    provenance_view.dataset_feature_offsets = dataset_feature_offsets.data();
    provenance_view.dataset_feature_to_global = dataset_feature_to_global.data();

    if (!cellshard::create_dataset_sliced_ell_h5(out_path, &layout, &dataset_view, &provenance_view)) goto done;

    for (unsigned long shard_id = 0; shard_id < shard_plan.count; ++shard_id) {
        const shard_range &range = shard_plan.ranges[shard_id];
        std::vector<sparse::sliced_ell> shard_parts;
        shard_parts.resize((std::size_t) (range.part_end - range.part_begin));
        for (sparse::sliced_ell &part : shard_parts) sparse::init(&part);
        for (unsigned long global_part_id = range.part_begin; global_part_id < range.part_end; ++global_part_id) {
            const std::string *spool_path = find_spool_path_for_part(plans, global_part_id);
            const std::size_t local_part = (std::size_t) (global_part_id - range.part_begin);
            if (spool_path == nullptr
                || spool_path->empty()
                || !cellshard::load(spool_path->c_str(), &shard_parts[local_part])) {
                for (sparse::sliced_ell &part : shard_parts) sparse::clear(&part);
                goto done;
            }
            {
                std::uint32_t bucket_count = 1u;
                std::uint64_t execution_bytes = 0u;
                cellshard::bucketed_sliced_ell_partition persisted_part;
                cellshard::init(&persisted_part);
                if (!choose_bucket_count_for_sliced_part_exact(&shard_parts[local_part], &bucket_count, &execution_bytes)
                    || !cellshard::build_bucketed_sliced_ell_partition(&persisted_part,
                                                                       &shard_parts[local_part],
                                                                       bucket_count,
                                                                       &execution_bytes)
                    || !cellshard::append_sliced_ell_partition_h5(out_path, global_part_id, &persisted_part)) {
                    cellshard::clear(&persisted_part);
                    for (sparse::sliced_ell &part : shard_parts) sparse::clear(&part);
                    goto done;
                }
                cellshard::clear(&persisted_part);
            }
        }
        for (sparse::sliced_ell &part : shard_parts) sparse::clear(&part);
    }

    if (!opts->cache_root.empty()
        && !cellshard::warm_dataset_sliced_ell_h5_cache(out_path, opts->cache_root.c_str())) {
        goto done;
    }

    ok = 1;

done:
    if (ok && spool_ready) {
        std::error_code ec;
        fs::remove_all(spool_root, ec);
    }
    clear(&shard_plan);
    common::clear(&dataset_ids);
    common::clear(&matrix_paths);
    common::clear(&feature_paths);
    common::clear(&barcode_paths);
    common::clear(&metadata_paths);
    common::clear(&global_barcodes);
    common::clear(&global_feature_ids);
    common::clear(&global_feature_names);
    common::clear(&global_feature_types);
    return ok;
}
