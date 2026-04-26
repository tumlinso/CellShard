#include "shared.hh"

template<typename MatrixT, typename ConfigureStateFn>
inline int load_dataset_h5_header_common(const char *filename,
                                         sharded<MatrixT> *m,
                                         shard_storage *s,
                                         const char *expected_matrix_format,
                                         const char *expected_payload_layout,
                                         ConfigureStateFn configure_state) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint64_t num_partitions = 0;
    std::uint64_t num_shards = 0;
    std::uint64_t num_codecs = 0;
    unsigned long rows_ul = 0ul;
    unsigned long cols_ul = 0ul;
    unsigned long nnz_ul = 0ul;
    int ok = 0;
    char matrix_format[64];
    char payload_layout[64];
    dataset_header_layout_buffers buffers;

    matrix_format[0] = '\0';
    payload_layout[0] = '\0';
    if (filename == 0 || m == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    if (!read_attr_u64(file, "rows", &rows)) goto done;
    if (!read_attr_u64(file, "cols", &cols)) goto done;
    if (!read_attr_u64(file, "nnz", &nnz)) goto done;
    if (!read_attr_u64(file, "num_partitions", &num_partitions)) goto done;
    if (!read_attr_u64(file, "num_shards", &num_shards)) goto done;
    if (!read_attr_u64(file, "num_codecs", &num_codecs)) goto done;
    if (!validate_dataset_header_scalars(filename, rows, cols, nnz, num_codecs)) goto done;
    if (expected_matrix_format != 0) {
        if (!read_attr_string(file, "matrix_format", matrix_format, sizeof(matrix_format))) goto done;
        if (std::strcmp(matrix_format, expected_matrix_format) != 0) goto done;
    }
    if (expected_payload_layout != 0) {
        if (!read_attr_string(file, "payload_layout", payload_layout, sizeof(payload_layout))) goto done;
        if (std::strcmp(payload_layout, expected_payload_layout) != 0) goto done;
    } else if (!read_attr_string(file, "payload_layout", payload_layout, sizeof(payload_layout))) {
        payload_layout[0] = '\0';
    }
    if (!sharded_from_u64(rows, &rows_ul, "rows", filename)) goto done;
    if (!sharded_from_u64(cols, &cols_ul, "cols", filename)) goto done;
    if (!sharded_from_u64(nnz, &nnz_ul, "nnz", filename)) goto done;

    matrix = H5Gopen2(file, matrix_group, H5P_DEFAULT);
    codecs = H5Gopen2(file, codecs_group, H5P_DEFAULT);
    if (matrix < 0 || codecs < 0) goto done;
    if (!read_header_layout_tables(matrix, filename, rows, nnz, num_partitions, num_shards, &buffers)) goto done;
    if (!initialize_sharded_header_view(filename, m, rows_ul, cols_ul, nnz_ul, num_partitions, num_shards, buffers)) goto done;

    if (s != 0) {
        dataset_h5_state *state = 0;
        if (!bind_dataset_h5_owner_runtime(s, filename)) goto done;
        state = dataset_h5_state_from_storage(s);
        if (state == 0) goto done;
        if (!populate_common_dataset_h5_state(filename,
                                              matrix,
                                              codecs,
                                              state,
                                              rows,
                                              cols,
                                              nnz,
                                              num_partitions,
                                              num_shards,
                                              num_codecs,
                                              buffers)) {
            goto done;
        }
        if (!configure_state(file, state, payload_layout)) goto done;
        if (!build_shard_partition_spans(state)) goto done;
        if (!load_dataset_execution_metadata(file, state)) goto done;
        if (!load_dataset_runtime_service_metadata(file, state)) goto done;
    }

    ok = 1;

done:
    if (!ok && s != 0) clear(s);
    buffers.clear();
    if (codecs >= 0) H5Gclose(codecs);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int load_dataset_blocked_ell_h5_header(const char *filename,
                                       sharded<sparse::blocked_ell> *m,
                                       shard_storage *s) {
    return load_dataset_h5_header_common(
        filename,
        m,
        s,
        matrix_traits<sparse::blocked_ell>::matrix_format_name(),
        0,
        [](hid_t file, dataset_h5_state *state, const char *payload_layout) -> int {
            const int optimized_codec = std::strcmp(payload_layout, payload_layout_optimized_blocked_ell) == 0;
            state->matrix_family = matrix_traits<sparse::blocked_ell>::matrix_family;
            state->blocked_ell_optimized_payload = optimized_codec;
            if (optimized_codec) return 1;

            state->partition_block_idx_offsets =
                (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
            state->partition_value_offsets =
                (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
            state->shard_block_idx_offsets =
                (std::uint64_t *) std::calloc((std::size_t) state->num_shards + 1u, sizeof(std::uint64_t));
            state->shard_value_offsets =
                (std::uint64_t *) std::calloc((std::size_t) state->num_shards + 1u, sizeof(std::uint64_t));
            if ((state->num_partitions != 0u
                 && (state->partition_block_idx_offsets == 0 || state->partition_value_offsets == 0))
                || (state->num_shards != 0u && (state->shard_block_idx_offsets == 0 || state->shard_value_offsets == 0))) {
                return 0;
            }
            {
                hid_t payload = H5Gopen2(file, payload_blocked_ell_group, H5P_DEFAULT);
                if (payload < 0) return 0;
                if (!read_dataset_1d(payload,
                                     "partition_block_idx_offsets",
                                     H5T_NATIVE_UINT64,
                                     state->num_partitions,
                                     state->partition_block_idx_offsets)
                    || !read_dataset_1d(payload,
                                        "partition_value_offsets",
                                        H5T_NATIVE_UINT64,
                                        state->num_partitions,
                                        state->partition_value_offsets)
                    || !read_dataset_1d(payload,
                                        "shard_block_idx_offsets",
                                        H5T_NATIVE_UINT64,
                                        state->num_shards + 1u,
                                        state->shard_block_idx_offsets)
                    || !read_dataset_1d(payload,
                                        "shard_value_offsets",
                                        H5T_NATIVE_UINT64,
                                        state->num_shards + 1u,
                                        state->shard_value_offsets)) {
                    H5Gclose(payload);
                    return 0;
                }
                H5Gclose(payload);
            }
            return 1;
        });
}

int load_dataset_quantized_blocked_ell_h5_header(const char *filename,
                                                 sharded<sparse::quantized_blocked_ell> *m,
                                                 shard_storage *s) {
    const int ok = load_dataset_h5_header_common(
        filename,
        m,
        s,
        matrix_traits<sparse::quantized_blocked_ell>::matrix_format_name(),
        payload_layout_shard_packed,
        [](hid_t, dataset_h5_state *state, const char *) -> int {
            state->matrix_family = matrix_traits<sparse::quantized_blocked_ell>::matrix_family;
            return 1;
        });
    if (!ok) clear(m);
    return ok;
}

int load_dataset_sliced_ell_h5_header(const char *filename,
                                      sharded<sparse::sliced_ell> *m,
                                      shard_storage *s) {
    return load_dataset_h5_header_common(
        filename,
        m,
        s,
        matrix_traits<sparse::sliced_ell>::matrix_format_name(),
        payload_layout_optimized_sliced_ell,
        [](hid_t, dataset_h5_state *state, const char *) -> int {
            state->matrix_family = matrix_traits<sparse::sliced_ell>::matrix_family;
            return 1;
        });
}

} // namespace cellshard
