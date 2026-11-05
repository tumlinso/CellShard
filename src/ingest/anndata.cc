#include <CellShard/ingest/anndata.hh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

import anndatapp.h5ad;

namespace cellshard::ingest::anndata_io {
namespace {
using ::anndatapp::DType;

const char* dtype_name(DType t) {
    switch (t) {
    case DType::boolean: return "bool";
    case DType::int8: return "int8";
    case DType::uint8: return "uint8";
    case DType::int16: return "int16";
    case DType::uint16: return "uint16";
    case DType::int32: return "int32";
    case DType::uint32: return "uint32";
    case DType::int64: return "int64";
    case DType::uint64: return "uint64";
    case DType::float32: return "float32";
    case DType::float64: return "float64";
    }
    throw std::invalid_argument("unknown AnnData dtype");
}

template<class T> T scalar(const std::byte* p) {
    T x; std::memcpy(&x, p, sizeof(T)); return x;
}

template<class T, class U> void convert(const std::byte* p, std::span<U> out) {
    for (std::size_t i = 0; i < out.size(); ++i) {
        const T x = scalar<T>(p + i*sizeof(T));
        if constexpr (std::is_same_v<U, std::uint64_t> && std::is_signed_v<T>)
            if (x < 0) throw std::runtime_error("negative AnnData sparse index or offset");
        out[i] = U(x);
    }
}

template<class U> void convert(DType t, const std::byte* p, std::span<U> out) {
    switch (t) {
    case DType::boolean: return convert<std::uint8_t>(p, out);
    case DType::int8: return convert<std::int8_t>(p, out);
    case DType::uint8: return convert<std::uint8_t>(p, out);
    case DType::int16: return convert<std::int16_t>(p, out);
    case DType::uint16: return convert<std::uint16_t>(p, out);
    case DType::int32: return convert<std::int32_t>(p, out);
    case DType::uint32: return convert<std::uint32_t>(p, out);
    case DType::int64: return convert<std::int64_t>(p, out);
    case DType::uint64: return convert<std::uint64_t>(p, out);
    case DType::float32:
        if constexpr (std::is_same_v<U, double>) return convert<float>(p, out);
        break;
    case DType::float64:
        if constexpr (std::is_same_v<U, double>) return convert<double>(p, out);
        break;
    }
    throw std::runtime_error("AnnData sparse index/offset must be an integer");
}

std::optional<std::string> text(const ::anndatapp::ColumnView& c, std::size_t row) {
    if (!c.valid(row)) return std::nullopt;
    if (c.kind() == ::anndatapp::ColumnKind::string) return std::string(c.string(row));
    if (c.kind() == ::anndatapp::ColumnKind::categorical) {
        const auto k = c.code(row);
        return k < 0 ? std::nullopt : text(c.categories(), std::size_t(k));
    }
    const auto a = c.values();
    const auto b = a.bytes();
    const auto* p = b.data() + std::ptrdiff_t(row)*b.stride_bytes(0);
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    switch (a.dtype()) {
    case DType::boolean: out << (scalar<std::uint8_t>(p) ? "true" : "false"); break;
    case DType::int8: out << int(scalar<std::int8_t>(p)); break;
    case DType::uint8: out << unsigned(scalar<std::uint8_t>(p)); break;
    case DType::int16: out << scalar<std::int16_t>(p); break;
    case DType::uint16: out << scalar<std::uint16_t>(p); break;
    case DType::int32: out << scalar<std::int32_t>(p); break;
    case DType::uint32: out << scalar<std::uint32_t>(p); break;
    case DType::int64: out << scalar<std::int64_t>(p); break;
    case DType::uint64: out << scalar<std::uint64_t>(p); break;
    case DType::float32: out << scalar<float>(p); break;
    case DType::float64: out << scalar<double>(p); break;
    }
    return out.str();
}

std::optional<std::string> table_text(const ::anndatapp::BackedTable& t,
                                    std::string_view name, std::size_t row) {
    if (row >= t.size()) throw std::out_of_range("AnnData metadata row out of bounds");
    const auto c = t.column(name).select(::anndatapp::range{std::ptrdiff_t(row), std::ptrdiff_t(row + 1)}).read();
    return text(c.view(), 0);
}
}

struct source::impl {
    ::anndatapp::BackedAnnData owner;
    ::anndatapp::BackedArray array;
    ::anndatapp::BackedTable features;
    std::size_t budget;
    bool csr;

    impl(std::string_view path, const matrix_selection& selection, std::size_t bytes)
        : owner(::anndatapp::open_h5ad(path, {.scratch_bytes = bytes})), budget(bytes) {
        if (bytes < 128) throw std::invalid_argument("AnnData working_bytes must be at least 128");
        switch (selection.slot) {
        case matrix_slot::x: array = owner.x(); features = owner.var(); break;
        case matrix_slot::layer:
            if (selection.layer.empty()) throw std::invalid_argument("AnnData layer selection needs a name");
            array = owner.layer(selection.layer); features = owner.var(); break;
        case matrix_slot::raw: array = owner.raw_x(); features = owner.raw_var(); break;
        }
        if (array.rank() != 2 || array.layout() == ::anndatapp::Layout::dense)
            throw std::invalid_argument("CellShard AnnData ingest requires a CSR or CSC matrix");
        if (array.extent(0) > std::size_t(std::numeric_limits<std::ptrdiff_t>::max())
            || array.extent(1) > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
            throw std::overflow_error("AnnData dimensions exceed selection index range");
        csr = array.layout() == ::anndatapp::Layout::csr;
    }

    read_stats visit(std::size_t first, std::size_t last, chunk_visitor visitor, void* context) const {
        const auto rows = array.extent(0), cols = array.extent(1);
        if (first > last || last > rows) throw std::out_of_range("AnnData row window out of bounds");
        read_stats stats;
        if (first == last) return stats;
        const std::size_t major_first = csr ? first : 0, major_last = csr ? last : cols;
        for (std::size_t begin = major_first; begin < major_last;) {
            std::size_t end = std::min(major_last, begin + std::min<std::size_t>(4096, major_last - begin));
            for (;;) {
                const auto r0 = csr ? begin : first, r1 = csr ? end : last;
                const auto c0 = csr ? 0 : begin, c1 = csr ? cols : end;
                const auto block = array.select({::anndatapp::range{std::ptrdiff_t(r0), std::ptrdiff_t(r1)},
                                                  ::anndatapp::range{std::ptrdiff_t(c0), std::ptrdiff_t(c1)}});
                const auto prepared = block.prepare_read();
                const auto req = prepared.requirements();
                if (!req.nnz) { begin = end; break; }
                const auto major = end - begin;
                const auto vw = ::anndatapp::dtype_size(req.dtype), iw = ::anndatapp::dtype_size(req.index_dtype),
                           ow = ::anndatapp::dtype_size(req.offset_dtype);
                const auto per_value = vw + iw + sizeof(double) + sizeof(std::uint64_t);
                const auto per_offset = ow + sizeof(std::uint64_t);
                const bool fits = major < budget/per_offset
                    && req.nnz <= (budget - (major + 1)*per_offset)/per_value;
                if (!fits) {
                    if (end - begin == 1)
                        throw std::length_error("AnnData sparse major segment exceeds working_bytes; increase the explicit budget");
                    end = begin + (end - begin)/2;
                    continue;
                }
                std::vector<std::byte> v(req.nnz*vw), i(req.nnz*iw), o((major + 1)*ow);
                std::vector<double> values(req.nnz);
                std::vector<std::uint64_t> indices(req.nnz), offsets(major + 1);
                prepared.read_into(::anndatapp::SparseBuffer{v.data(), i.data(), o.data(),
                    req.dtype, req.index_dtype, req.offset_dtype, req.nnz, req.nnz, major + 1});
                convert(req.dtype, v.data(), std::span(values));
                convert(req.index_dtype, i.data(), std::span(indices));
                convert(req.offset_dtype, o.data(), std::span(offsets));
                for (double x : values)
                    if (!std::isfinite(x) || x < -1e-6 || std::fabs(x - std::round(x)) > 1e-4) stats.count_like = false;
                if (req.nnz > std::numeric_limits<std::uint64_t>::max() - stats.nnz)
                    throw std::overflow_error("AnnData NNZ count overflow");
                stats.nnz += req.nnz; ++stats.batches;
                stats.peak_buffer_bytes = std::max(stats.peak_buffer_bytes,
                    req.nnz*per_value + (major + 1)*per_offset);
                if (visitor) visitor({r0, c0, r1 - r0, c1 - c0, csr, values, indices, offsets}, context);
                begin = end;
                break;
            }
        }
        return stats;
    }
};

source::source(std::string_view path, matrix_selection matrix, std::size_t bytes)
    : p_(std::make_unique<impl>(path, matrix, bytes)) {}
source::~source() = default;
source::source(source&&) noexcept = default;
source& source::operator=(source&&) noexcept = default;
matrix_info source::info() const {
    return {p_->array.extent(0), p_->array.extent(1), p_->csr, dtype_name(p_->array.dtype())};
}
read_stats source::row_nnz(std::span<std::uint64_t> counts) const {
    if (counts.size() != p_->array.extent(0)) throw std::invalid_argument("AnnData row-count output has wrong size");
    std::fill(counts.begin(), counts.end(), 0);
    auto count = [](const sparse_chunk& b, void* context) {
        auto out = *static_cast<std::span<std::uint64_t>*>(context);
        if (b.csr) {
            for (std::size_t r = 0; r < b.rows; ++r) out[b.row_begin + r] += b.offsets[r + 1] - b.offsets[r];
        } else for (auto r : b.indices) ++out[b.row_begin + r];
    };
    return p_->visit(0, counts.size(), count, &counts);
}
read_stats source::visit_rows(std::size_t first, std::size_t last, chunk_visitor visitor, void* context) const {
    if (!visitor) throw std::invalid_argument("AnnData visitor must be non-null");
    return p_->visit(first, last, visitor, context);
}
std::string source::observation_name(std::size_t row) const { return p_->owner.obs().name(row); }
std::string source::feature_name(std::size_t col) const { return p_->features.name(col); }
std::vector<std::string> source::observation_columns() const { return p_->owner.obs().column_names(); }
std::vector<std::string> source::feature_columns() const { return p_->features.column_names(); }
std::optional<std::string> source::observation_text(std::string_view name, std::size_t row) const {
    return table_text(p_->owner.obs(), name, row);
}
std::optional<std::string> source::feature_text(std::string_view name, std::size_t col) const {
    return table_text(p_->features, name, col);
}

} // namespace cellshard::ingest::anndata_io
