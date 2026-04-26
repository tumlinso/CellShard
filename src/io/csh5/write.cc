#include "execution_internal.hh"

int append_blocked_ell_partition_h5(const char *filename,
                                    unsigned long partition_id,
                                    const sparse::blocked_ell *part) {
    hid_t file = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    hid_t d_block_idx = (hid_t) -1;
    hid_t d_values = (hid_t) -1;
    std::uint64_t *partition_block_idx_offsets = 0;
    std::uint64_t *partition_value_offsets = 0;
    std::uint64_t num_partitions = 0;
    char payload_layout[64];
    const std::size_t row_blocks = sparse::row_block_count(part);
    const std::size_t ell_width = sparse::ell_width_blocks(part);
    int ok = 0;

    if (filename == 0 || part == 0) return 0;

    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    payload_layout[0] = '\0';
    if (!read_attr_string(file, "payload_layout", payload_layout, sizeof(payload_layout))) goto done;
    if (std::strcmp(payload_layout, payload_layout_shard_packed) != 0) {
        std::fprintf(stderr,
                     "cellshard: append_blocked_ell_partition_h5 only supports legacy shard_packed blocked files; blocked now defaults to optimized execution payload\n");
        goto done;
    }
    if (!read_attr_u64(file, "num_partitions", &num_partitions)) goto done;
    if (partition_id >= num_partitions) goto done;

    partition_block_idx_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    partition_value_offsets = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    if ((num_partitions != 0) && (partition_block_idx_offsets == 0 || partition_value_offsets == 0)) goto done;

    payload = H5Gopen2(file, payload_blocked_ell_group, H5P_DEFAULT);
    if (payload < 0) goto done;
    if (!read_dataset_1d(payload, "partition_block_idx_offsets", H5T_NATIVE_UINT64, num_partitions, partition_block_idx_offsets)) goto done;
    if (!read_dataset_1d(payload, "partition_value_offsets", H5T_NATIVE_UINT64, num_partitions, partition_value_offsets)) goto done;
    d_block_idx = H5Dopen2(payload, "block_col_idx", H5P_DEFAULT);
    d_values = H5Dopen2(payload, "values", H5P_DEFAULT);
    if (d_block_idx < 0 || d_values < 0) goto done;

    {
        hsize_t off[1];
        hsize_t dims[1];
        hid_t filespace = (hid_t) -1;
        hid_t memspace = (hid_t) -1;

        off[0] = (hsize_t) partition_block_idx_offsets[partition_id];
        dims[0] = (hsize_t) (row_blocks * ell_width);
        filespace = H5Dget_space(d_block_idx);
        if (filespace < 0) goto done;
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) {
            H5Sclose(filespace);
            goto done;
        }
        memspace = H5Screate_simple(1, dims, 0);
        if (memspace < 0) {
            H5Sclose(filespace);
            goto done;
        }
        if (H5Dwrite(d_block_idx, H5T_NATIVE_UINT32, memspace, filespace, H5P_DEFAULT, part->blockColIdx) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            goto done;
        }
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    {
        hsize_t off[1];
        hsize_t dims[1];
        hid_t filespace = (hid_t) -1;
        hid_t memspace = (hid_t) -1;

        off[0] = (hsize_t) partition_value_offsets[partition_id];
        dims[0] = (hsize_t) ((std::size_t) part->rows * (std::size_t) part->ell_cols);
        filespace = H5Dget_space(d_values);
        if (filespace < 0) goto done;
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) {
            H5Sclose(filespace);
            goto done;
        }
        memspace = H5Screate_simple(1, dims, 0);
        if (memspace < 0) {
            H5Sclose(filespace);
            goto done;
        }
        if (H5Dwrite(d_values, H5T_NATIVE_UINT16, memspace, filespace, H5P_DEFAULT, part->val) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            goto done;
        }
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    ok = 1;

done:
    std::free(partition_block_idx_offsets);
    std::free(partition_value_offsets);
    if (d_values >= 0) H5Dclose(d_values);
    if (d_block_idx >= 0) H5Dclose(d_block_idx);
    if (payload >= 0) H5Gclose(payload);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_quantized_blocked_ell_partition_h5(const char *filename,
                                              unsigned long partition_id,
                                              const sparse::quantized_blocked_ell *part) {
    hid_t file = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    char *buffer = 0;
    unsigned char *blob = 0;
    std::size_t blob_bytes = 0u;
    char dataset_name[64];
    std::FILE *fp = 0;
    int ok = 0;

    if (filename == 0 || part == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    payload = H5Gopen2(file, payload_quantized_blocked_ell_group, H5P_DEFAULT);
    if (payload < 0) goto done;
    if (!build_partition_blob_dataset_name(partition_id, dataset_name, sizeof(dataset_name))) goto done;
    fp = open_memstream(&buffer, &blob_bytes);
    if (fp == 0) goto done;
    if (!::cellshard::store(fp, part) || std::fclose(fp) != 0) {
        fp = 0;
        goto done;
    }
    fp = 0;
    blob = (unsigned char *) buffer;
    if (!write_blob_dataset(payload, dataset_name, blob, blob_bytes)) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    std::free(buffer);
    if (payload >= 0) H5Gclose(payload);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_sliced_ell_partition_h5(const char *filename,
                                   unsigned long partition_id,
                                   const bucketed_sliced_ell_partition *part) {
    hid_t file = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    char *buffer = 0;
    unsigned char *blob = 0;
    std::size_t blob_bytes = 0u;
    char dataset_name[64];
    std::FILE *fp = 0;
    int ok = 0;

    if (filename == 0 || part == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    payload = H5Gopen2(file, payload_sliced_ell_group, H5P_DEFAULT);
    if (payload < 0) goto done;
    if (!build_partition_blob_dataset_name(partition_id, dataset_name, sizeof(dataset_name))) goto done;
    fp = open_memstream(&buffer, &blob_bytes);
    if (fp == 0) goto done;
    if (!write_sliced_execution_partition_blob(fp, part) || std::fclose(fp) != 0) {
        fp = 0;
        goto done;
    }
    fp = 0;
    blob = (unsigned char *) buffer;
    if (!write_blob_dataset(payload, dataset_name, blob, blob_bytes)) goto done;
    if (H5Aexists(file, "payload_layout") > 0 && H5Adelete(file, "payload_layout") < 0) goto done;
    if (!write_attr_string(file, "payload_layout", payload_layout_optimized_sliced_ell)) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    std::free(buffer);
    if (payload >= 0) H5Gclose(payload);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_bucketed_blocked_ell_shard_h5(const char *filename,
                                         unsigned long shard_id,
                                         const bucketed_blocked_ell_shard *shard) {
    hid_t file = (hid_t) -1;
    hid_t payload_root = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    unsigned char *blob = 0;
    std::size_t blob_bytes = 0u;
    char dataset_name[64];
    int ok = 0;

    if (filename == 0 || shard == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_dataset_identity(file)) goto done;
    payload_root = H5Gopen2(file, payload_group, H5P_DEFAULT);
    if (payload_root < 0) payload_root = create_group(file, payload_group);
    if (payload_root < 0) goto done;
    if (H5Lexists(payload_root, matrix_traits<sparse::blocked_ell>::matrix_format_name(), H5P_DEFAULT) > 0) {
        payload = H5Gopen2(payload_root, matrix_traits<sparse::blocked_ell>::matrix_format_name(), H5P_DEFAULT);
    } else {
        payload = create_group(payload_root, matrix_traits<sparse::blocked_ell>::matrix_format_name());
    }
    if (payload < 0) goto done;
    if (!build_optimized_shard_dataset_name(shard_id, dataset_name, sizeof(dataset_name))) goto done;
    if (!serialize_optimized_shard(shard, &blob, &blob_bytes)) goto done;
    if (!write_blob_dataset(payload, dataset_name, blob, blob_bytes)) goto done;
    if (H5Aexists(file, "payload_layout") > 0 && H5Adelete(file, "payload_layout") < 0) goto done;
    if (!write_attr_string(file, "payload_layout", payload_layout_optimized_blocked_ell)) goto done;
    ok = 1;

done:
    std::free(blob);
    if (payload >= 0) H5Gclose(payload);
    if (payload_root >= 0) H5Gclose(payload_root);
    if (file >= 0) H5Fclose(file);
    return ok;
}

} // namespace cellshard
