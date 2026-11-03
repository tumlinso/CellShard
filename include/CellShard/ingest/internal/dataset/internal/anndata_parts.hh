#pragma once

// This boundary materializes only caller-selected CellShard partitions. The
// source owner performs native module I/O; these adapters make the FP16/u32
// payload conversion explicit without exposing modules to CUDA compilation.
static inline anndata_io::matrix_selection select_anndata_matrix(const char* source) {
    const std::string s = source ? source : "x";
    if (s.empty() || s == "x") return {};
    if (s == "raw_x") return {anndata_io::matrix_slot::raw, {}};
    if (s.starts_with("layer:") && s.size() > 6) return {anndata_io::matrix_slot::layer, s.substr(6)};
    throw std::invalid_argument("AnnData matrix_source must be x, raw_x, or layer:<name>");
}

static inline int scan_anndata_rows(const manifest* m, unsigned int idx,
                                    mtx::header* h, unsigned long** row_nnz,
                                    std::size_t reader_bytes) {
    *row_nnz = nullptr;
    try {
        anndata_io::source source(matrix_path_at(m, idx), select_anndata_matrix(matrix_source_at(m, idx)), reader_bytes);
        const auto info = source.info();
        if (info.rows > std::numeric_limits<unsigned long>::max() || info.cols > std::numeric_limits<unsigned long>::max()) return 0;
        std::vector<std::uint64_t> counts(info.rows);
        const auto scan = source.row_nnz(counts);
        if ((!scan.count_like && !allow_processed_at(m, idx)) || scan.nnz > std::numeric_limits<unsigned long>::max()) return 0;
        auto* out = static_cast<unsigned long*>(std::calloc(info.rows ? info.rows : 1, sizeof(unsigned long)));
        if (!out) return 0;
        std::copy(counts.begin(), counts.end(), out);
        mtx::init(h);
        h->rows = info.rows; h->cols = info.cols;
        h->nnz_file = h->nnz_loaded = scan.nnz; h->row_sorted = info.csr ? 1 : 0;
        *row_nnz = out;
        return 1;
    } catch (const std::exception&) { return 0; }
}

static inline int load_anndata_barcodes(const manifest* m, unsigned int idx, common::barcode_table* out) {
    common::clear(out); common::init(out);
    try {
        anndata_io::source source(matrix_path_at(m, idx), select_anndata_matrix(matrix_source_at(m, idx)));
        const auto info = source.info();
        for (std::size_t r = 0; r < info.rows; ++r) {
            const auto s = source.observation_name(r);
            if (!common::append(out, s.data(), s.size())) throw std::bad_alloc();
        }
        return 1;
    } catch (const std::exception&) { common::clear(out); return 0; }
}

static inline int load_anndata_features(const manifest* m, unsigned int idx, common::feature_table* out) {
    common::clear(out); common::init(out);
    try {
        anndata_io::source source(matrix_path_at(m, idx), select_anndata_matrix(matrix_source_at(m, idx)));
        const auto columns = source.feature_columns();
        const auto info = source.info();
        const bool names = std::find(columns.begin(), columns.end(), "feature_name") != columns.end();
        const bool types = std::find(columns.begin(), columns.end(), "feature_type") != columns.end();
        for (std::size_t c = 0; c < info.cols; ++c) {
            const auto id = source.feature_name(c);
            const auto name = names ? source.feature_text("feature_name", c).value_or("") : id;
            const auto type = types ? source.feature_text("feature_type", c).value_or("") : std::string();
            if (!common::append(out, id.data(), id.size(), name.data(), name.size(), type.data(), type.size())) throw std::bad_alloc();
        }
        return 1;
    } catch (const std::exception&) { common::clear(out); return 0; }
}

template<class F> static inline void each_anndata_entry(const anndata_io::sparse_chunk& b, F&& f) {
    const auto major = b.csr ? b.rows : b.cols;
    for (std::size_t j = 0; j < major; ++j)
        for (auto k = b.offsets[j]; k < b.offsets[j + 1]; ++k) {
            const auto r = b.row_begin + (b.csr ? j : b.indices[k]);
            const auto c = b.col_begin + (b.csr ? b.indices[k] : j);
            f(r, c, b.values[k]);
        }
}

static inline bool anndata_partition_window(const anndata_io::matrix_info& info, const mtx::header* h,
                                            const unsigned long* offsets, const unsigned long* nnz,
                                            unsigned long count, unsigned long first, unsigned long last) {
    if (!h || !offsets || !nnz || first >= last || last > count || h->rows != info.rows || h->cols != info.cols
        || info.cols > std::numeric_limits<::cellshard::types::idx_t>::max()) return false;
    for (auto p = first; p < last; ++p)
        if (offsets[p] > offsets[p + 1] || offsets[p + 1] > info.rows
            || offsets[p + 1] - offsets[p] > std::numeric_limits<::cellshard::types::dim_t>::max()
            || nnz[p] > std::numeric_limits<::cellshard::types::ptr_t>::max()) return false;
    return true;
}

static inline int load_anndata_coo(const manifest* m, unsigned int idx, const mtx::header* h,
                                   const unsigned long* offsets, const unsigned long* nnz,
                                   unsigned long count, unsigned long first, unsigned long last,
                                   sharded<sparse::coo>* out, std::size_t bytes) {
    clear(out); init(out);
    try {
        anndata_io::source source(matrix_path_at(m, idx), select_anndata_matrix(matrix_source_at(m, idx)), bytes);
        if (!anndata_partition_window(source.info(), h, offsets, nnz, count, first, last)) return 0;
        if (!mtx::allocate_part_window_coo(h, offsets, nnz, count, first, last, out)) return 0;
        struct output {
            sharded<sparse::coo>* parts;
            const unsigned long* offsets;
            unsigned long count, first;
            std::vector<unsigned long> cursors;
        } data{out, offsets, count, first, std::vector<unsigned long>(last - first)};
        auto copy = [](const anndata_io::sparse_chunk& b, void* context) {
            auto& d = *static_cast<output*>(context);
            each_anndata_entry(b, [&](auto r, auto c, double x) {
                const auto p = find_offset_span(r, d.offsets, d.count), local = p - d.first;
                auto* part = d.parts->parts[local];
                const auto k = d.cursors[local]++;
                if (k >= d.parts->partition_nnz[local]) throw std::runtime_error("AnnData partition NNZ changed");
                part->rowIdx[k] = ::cellshard::types::idx_t(r - d.offsets[p]);
                part->colIdx[k] = ::cellshard::types::idx_t(c);
                part->val[k] = __float2half(float(x));
            });
        };
        source.visit_rows(offsets[first], offsets[last], copy, &data);
        for (auto p = first; p < last; ++p)
            if (data.cursors[p - first] != nnz[p]) throw std::runtime_error("AnnData partition NNZ differs from plan");
        return 1;
    } catch (const std::exception&) { clear(out); return 0; }
}

static inline int load_anndata_compressed(const manifest* m, unsigned int idx, const mtx::header* h,
                                          const unsigned long* offsets, const unsigned long* nnz,
                                          unsigned long count, unsigned long first, unsigned long last,
                                          sharded<sparse::compressed>* out, std::size_t bytes) {
    clear(out); init(out);
    try {
        anndata_io::source source(matrix_path_at(m, idx), select_anndata_matrix(matrix_source_at(m, idx)), bytes);
        const auto info = source.info();
        if (!anndata_partition_window(info, h, offsets, nnz, count, first, last)) return 0;
        if (!reserve_partitions(out, last - first)) return 0;
        out->num_partitions = last - first; out->cols = info.cols;
        for (auto p = first; p < last; ++p) {
            const auto local = p - first, rows = offsets[p + 1] - offsets[p];
            auto* part = new sparse::compressed;
            sparse::init(part, rows, info.cols, nnz[p], info.csr ? sparse::compressed_by_row : sparse::compressed_by_col);
            out->parts[local] = part;
            if (!sparse::allocate(part)) throw std::bad_alloc();
            out->partition_rows[local] = rows; out->partition_nnz[local] = nnz[p]; out->partition_aux[local] = 0;
            std::fill(part->majorPtr, part->majorPtr + (info.csr ? rows : info.cols) + 1, 0);
        }
        rebuild_partition_offsets(out);
        if (!set_shards_to_partitions(out)) throw std::bad_alloc();
        struct output {
            sharded<sparse::compressed>* parts;
            const unsigned long* offsets;
            unsigned long count, first;
            bool csr, fill;
        } data{out, offsets, count, first, info.csr, false};
        auto copy = [](const anndata_io::sparse_chunk& b, void* context) {
            auto& d = *static_cast<output*>(context);
            each_anndata_entry(b, [&](auto r, auto c, double x) {
                const auto p = find_offset_span(r, d.offsets, d.count), local = p - d.first;
                auto* part = d.parts->parts[local];
                const auto major = d.csr ? r - d.offsets[p] : c;
                if (!d.fill) { ++part->majorPtr[major + 1]; return; }
                const auto k = part->majorPtr[major]++;
                if (k >= d.parts->partition_nnz[local]) throw std::runtime_error("AnnData partition NNZ changed");
                part->minorIdx[k] = ::cellshard::types::idx_t(d.csr ? c : r - d.offsets[p]);
                part->val[k] = __float2half(float(x));
            });
        };
        source.visit_rows(offsets[first], offsets[last], copy, &data);
        for (auto p = first; p < last; ++p) {
            auto* part = out->parts[p - first];
            const auto major = info.csr ? offsets[p + 1] - offsets[p] : info.cols;
            for (std::size_t j = 0; j < major; ++j) part->majorPtr[j + 1] += part->majorPtr[j];
            if (part->majorPtr[major] != nnz[p]) throw std::runtime_error("AnnData partition NNZ differs from plan");
        }
        data.fill = true;
        source.visit_rows(offsets[first], offsets[last], copy, &data);
        for (auto p = first; p < last; ++p) {
            auto* part = out->parts[p - first];
            const auto major = info.csr ? offsets[p + 1] - offsets[p] : info.cols;
            for (auto j = major; j > 0; --j) part->majorPtr[j] = part->majorPtr[j - 1];
            part->majorPtr[0] = 0;
        }
        return 1;
    } catch (const std::exception&) { clear(out); return 0; }
}
