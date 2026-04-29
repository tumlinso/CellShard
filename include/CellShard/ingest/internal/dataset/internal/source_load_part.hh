#pragma once

static inline int scan_source_row_nnz(const manifest *m,
                                      unsigned int idx,
                                      mtx::header *header,
                                      unsigned long **row_nnz_out,
                                      std::size_t reader_bytes) {
    std::string error;
    const unsigned int format = format_at(m, idx);
    if (format == source_mtx || format == source_tenx_mtx) {
        return mtx::scan_row_nnz(matrix_path_at(m, idx), header, row_nnz_out, reader_bytes);
    }
    if (format == source_h5ad) {
        h5ad::selected_matrix_info info;
        if (!h5ad::probe_selected_matrix(matrix_path_at(m, idx), matrix_source_at(m, idx), &info, &error)) return 0;
        if (info.processed_like && !allow_processed_at(m, idx)) return 0;
        return h5ad::scan_row_nnz(matrix_path_at(m, idx), matrix_source_at(m, idx), header, row_nnz_out, &error);
    }
    return 0;
}

static inline int load_source_barcodes(const manifest *m,
                                       unsigned int idx,
                                       common::barcode_table *barcodes) {
    std::string error;
    const unsigned int format = format_at(m, idx);
    if (format == source_mtx || format == source_tenx_mtx) {
        return barcode_path_at(m, idx) != 0 && *barcode_path_at(m, idx) != 0
            && common::load_lines(barcode_path_at(m, idx), barcodes);
    }
    if (format == source_h5ad) {
        return h5ad::load_barcodes(matrix_path_at(m, idx), barcodes, &error);
    }
    return 0;
}

static inline int load_source_features(const manifest *m,
                                       unsigned int idx,
                                       common::feature_table *features) {
    std::string error;
    const unsigned int format = format_at(m, idx);
    if (format == source_mtx || format == source_tenx_mtx) {
        return feature_path_at(m, idx) != 0 && *feature_path_at(m, idx) != 0
            && common::load_tsv(feature_path_at(m, idx), features, 0);
    }
    if (format == source_h5ad) {
        return h5ad::load_feature_table(matrix_path_at(m, idx), matrix_source_at(m, idx), features, &error);
    }
    return 0;
}

static inline int load_source_part_window_coo(const manifest *m,
                                              unsigned int idx,
                                              const mtx::header *header,
                                              const unsigned long *row_offsets,
                                              const unsigned long *part_nnz,
                                              unsigned long num_parts,
                                              unsigned long part_begin,
                                              unsigned long part_end,
                                              sharded<sparse::coo> *out,
                                              std::size_t reader_bytes) {
    std::string error;
    const unsigned int format = format_at(m, idx);
    if (format == source_mtx || format == source_tenx_mtx) {
        return mtx::load_part_window_coo(matrix_path_at(m, idx),
                                         header,
                                         row_offsets,
                                         part_nnz,
                                         num_parts,
                                         part_begin,
                                         part_end,
                                         out,
                                         reader_bytes);
    }
    if (format == source_h5ad) {
        return h5ad::load_part_window_coo(matrix_path_at(m, idx),
                                          matrix_source_at(m, idx),
                                          header,
                                          row_offsets,
                                          part_nnz,
                                          num_parts,
                                          part_begin,
                                          part_end,
                                          out,
                                          &error);
    }
    return 0;
}

static inline int load_source_part_window_compressed(const manifest *m,
                                                     unsigned int idx,
                                                     const mtx::header *header,
                                                     const unsigned long *row_offsets,
                                                     const unsigned long *part_nnz,
                                                     unsigned long num_parts,
                                                     unsigned long part_begin,
                                                     unsigned long part_end,
                                                     sharded<sparse::compressed> *out,
                                                     std::size_t reader_bytes) {
    std::string error;
    const unsigned int format = format_at(m, idx);
    (void) reader_bytes;
    if (format == source_h5ad) {
        return h5ad::load_part_window_compressed(matrix_path_at(m, idx),
                                                 matrix_source_at(m, idx),
                                                 header,
                                                 row_offsets,
                                                 part_nnz,
                                                 num_parts,
                                                 part_begin,
                                                 part_end,
                                                 out,
                                                 &error);
    }
    return 0;
}
