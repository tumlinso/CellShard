#pragma once

#include "../core/types.cuh"
#include "../io/common/generation.hh"
#include "../io/common/layout.hh"
#include "../io/common/partition.hh"
#include "../io/common/raw_format.hh"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace cellshard {
namespace access {

enum adapter_capability : std::uint64_t {
    capability_none = 0ull,
    capability_contiguous_payload = 1ull << 0u,
    capability_cell_span_copy = 1ull << 1u,
    capability_indexed_cell_copy = 1ull << 2u,
    capability_feature_span_copy = 1ull << 3u,
    capability_pinned_host_staging = 1ull << 4u,
    capability_async_staging = 1ull << 5u,
    capability_pack_read = 1ull << 6u,
    capability_pack_write = 1ull << 7u,
    capability_archive_to_pack = 1ull << 8u,
    capability_debug_at = 1ull << 63u
};

enum selection_kind : std::uint32_t {
    selection_none = 0u,
    selection_spans = 1u,
    selection_indices = 2u
};

struct cell_span {
    std::uint64_t begin;
    std::uint64_t end;

    std::uint64_t size() const {
        return end > begin ? end - begin : 0u;
    }
};

using feature_span = cell_span;

struct cell_selection_view {
    const cell_span *spans;
    std::uint32_t span_count;
    const std::uint64_t *indices;
    std::uint64_t index_count;

    selection_kind kind() const {
        return index_count != 0u ? selection_indices : (span_count != 0u ? selection_spans : selection_none);
    }
};

using feature_selection_view = cell_selection_view;

struct byte_span {
    const void *data;
    std::size_t bytes;
    std::uint32_t value_code;
};

struct mutable_byte_span {
    void *data;
    std::size_t bytes;
    std::uint32_t value_code;
};

struct fallback_payload_view {
    byte_span header;
    byte_span major_ptr;
    byte_span minor_idx;
    byte_span values;
    byte_span auxiliary;
};

struct archive_descriptor {
    std::uint64_t cell_count;
    std::uint64_t feature_count;
    std::uint64_t nnz;
    std::uint32_t assay_id;
    std::uint32_t value_code;
    disk_format archive_format;
    std::uint32_t execution_format;
    std::uint64_t capabilities;
};

struct pack_descriptor {
    std::uint64_t cell_count;
    std::uint64_t feature_count;
    std::uint64_t nnz;
    std::uint32_t value_code;
    std::uint32_t execution_format;
    std::uint64_t payload_bytes;
    std::uint64_t capabilities;
};

struct copy_preflight {
    std::uint64_t cell_count;
    std::uint64_t feature_count;
    std::uint64_t nnz;
    std::size_t output_bytes;
    std::size_t scratch_bytes;
};

struct cell_span_copy_request {
    cell_selection_view cells;
    feature_selection_view features;
    mutable_byte_span output;
    mutable_byte_span scratch;
};

struct pack_build_request {
    dataset_generation_ref generation;
    dataset_partition_ref partition;
    cell_selection_view cells;
    feature_selection_view features;
    mutable_byte_span output;
    mutable_byte_span scratch;
    std::uint32_t requested_execution_format;
    std::uint32_t flags;
    void *stream;
};

struct pack_build_result {
    std::uint64_t cell_count;
    std::uint64_t feature_count;
    std::uint64_t nnz;
    std::size_t payload_bytes;
    std::uint32_t execution_format;
    std::uint32_t status;
};

struct passthrough_pack_policy {
    enum : std::uint32_t {
        allow_layout_change = 0u,
        prefer_pinned_host = 1u
    };
};

template<class Binding>
struct adapter_view {
    using binding_type = Binding;

    Binding *binding;
    const void *archive_context;
    void *pack_context;
    const void *metadata_context;
};

template<class Binding>
inline adapter_view<Binding> make_adapter_view(
    Binding &binding,
    const void *archive_context = nullptr,
    void *pack_context = nullptr,
    const void *metadata_context = nullptr) {
    return adapter_view<Binding>{&binding, archive_context, pack_context, metadata_context};
}

template<class Binding>
struct archive_adapter;

template<class Binding>
struct pack_adapter;

template<class ArchiveBinding, class PackBinding, class Policy = passthrough_pack_policy>
struct archive_to_pack;

template<class MatrixT>
struct payload_traits {
    __host__ __device__ __forceinline__ static std::uint64_t rows(const MatrixT *matrix) {
        return matrix != nullptr ? matrix->rows : 0u;
    }

    __host__ __device__ __forceinline__ static std::uint64_t cols(const MatrixT *matrix) {
        return matrix != nullptr ? matrix->cols : 0u;
    }

    __host__ __device__ __forceinline__ static std::uint64_t nnz(const MatrixT *matrix) {
        return matrix != nullptr ? matrix->nnz : 0u;
    }

    __host__ __device__ __forceinline__ static std::uint64_t aux(const MatrixT *) {
        return 0u;
    }

    __host__ __device__ __forceinline__ static std::size_t host_bytes(
        const MatrixT *matrix,
        std::uint64_t,
        std::uint64_t,
        std::uint64_t,
        std::uint64_t) {
        return matrix != nullptr ? bytes(matrix) : 0u;
    }

    __host__ __device__ __forceinline__ static const types::storage_value_t *debug_at(
        const MatrixT *matrix,
        std::uint64_t row,
        types::idx_t col) {
        (void) matrix;
        (void) row;
        (void) col;
        return nullptr;
    }
};

namespace detail {

template<class...>
using void_t = void;

template<class Binding, class = void>
struct has_archive_describe : std::false_type {};

template<class Binding>
struct has_archive_describe<Binding, void_t<decltype(archive_adapter<Binding>::describe(std::declval<const adapter_view<Binding> &>()))>> : std::true_type {};

template<class Binding, class = void>
struct has_archive_contiguous_payload : std::false_type {};

template<class Binding>
struct has_archive_contiguous_payload<Binding, void_t<decltype(archive_adapter<Binding>::contiguous_payload(std::declval<const adapter_view<Binding> &>()))>> : std::true_type {};

template<class Binding, class = void>
struct has_archive_preflight_cell_spans : std::false_type {};

template<class Binding>
struct has_archive_preflight_cell_spans<Binding, void_t<decltype(archive_adapter<Binding>::preflight_cell_spans(
    std::declval<const adapter_view<Binding> &>(),
    std::declval<const cell_selection_view &>(),
    std::declval<const feature_selection_view &>()))>> : std::true_type {};

template<class Binding, class = void>
struct has_archive_copy_cell_spans : std::false_type {};

template<class Binding>
struct has_archive_copy_cell_spans<Binding, void_t<decltype(archive_adapter<Binding>::copy_cell_spans(
    std::declval<const adapter_view<Binding> &>(),
    std::declval<const cell_span_copy_request &>()))>> : std::true_type {};

template<class Binding, class = void>
struct has_pack_describe : std::false_type {};

template<class Binding>
struct has_pack_describe<Binding, void_t<decltype(pack_adapter<Binding>::describe(std::declval<const adapter_view<Binding> &>()))>> : std::true_type {};

template<class ArchiveBinding, class PackBinding, class Policy, class = void>
struct has_archive_to_pack_build : std::false_type {};

template<class ArchiveBinding, class PackBinding, class Policy>
struct has_archive_to_pack_build<ArchiveBinding, PackBinding, Policy, void_t<decltype(archive_to_pack<ArchiveBinding, PackBinding, Policy>::build(
    std::declval<const adapter_view<ArchiveBinding> &>(),
    std::declval<const adapter_view<PackBinding> &>(),
    std::declval<const pack_build_request &>()))>> : std::true_type {};

} // namespace detail

template<class Binding>
struct is_archive_adapter : detail::has_archive_describe<Binding> {};

template<class Binding>
struct is_pack_adapter : detail::has_pack_describe<Binding> {};

template<class ArchiveBinding, class PackBinding, class Policy = passthrough_pack_policy>
struct is_archive_to_pack_adapter : detail::has_archive_to_pack_build<ArchiveBinding, PackBinding, Policy> {};

template<class Binding>
inline archive_descriptor describe_archive(const adapter_view<Binding> &view) {
    static_assert(is_archive_adapter<Binding>::value,
                  "cellshard::access::archive_adapter<Binding> must provide describe(const adapter_view<Binding>&)");
    return archive_adapter<Binding>::describe(view);
}

template<class Binding>
inline pack_descriptor describe_pack(const adapter_view<Binding> &view) {
    static_assert(is_pack_adapter<Binding>::value,
                  "cellshard::access::pack_adapter<Binding> must provide describe(const adapter_view<Binding>&)");
    return pack_adapter<Binding>::describe(view);
}

template<class Binding>
inline fallback_payload_view contiguous_payload(const adapter_view<Binding> &view) {
    static_assert(detail::has_archive_contiguous_payload<Binding>::value,
                  "cellshard::access::archive_adapter<Binding> must provide contiguous_payload(const adapter_view<Binding>&)");
    return archive_adapter<Binding>::contiguous_payload(view);
}

template<class Binding>
inline copy_preflight preflight_cell_spans(
    const adapter_view<Binding> &view,
    const cell_selection_view &cells,
    const feature_selection_view &features) {
    static_assert(detail::has_archive_preflight_cell_spans<Binding>::value,
                  "cellshard::access::archive_adapter<Binding> must provide preflight_cell_spans(view, cells, features)");
    return archive_adapter<Binding>::preflight_cell_spans(view, cells, features);
}

template<class Binding>
inline pack_build_result copy_cell_spans(
    const adapter_view<Binding> &view,
    const cell_span_copy_request &request) {
    static_assert(detail::has_archive_copy_cell_spans<Binding>::value,
                  "cellshard::access::archive_adapter<Binding> must provide copy_cell_spans(view, request)");
    return archive_adapter<Binding>::copy_cell_spans(view, request);
}

template<class ArchiveBinding, class PackBinding, class Policy = passthrough_pack_policy>
inline pack_build_result build_pack(
    const adapter_view<ArchiveBinding> &archive,
    const adapter_view<PackBinding> &pack,
    const pack_build_request &request) {
    static_assert(is_archive_to_pack_adapter<ArchiveBinding, PackBinding, Policy>::value,
                  "cellshard::access::archive_to_pack<ArchiveBinding, PackBinding, Policy> must provide build(archive, pack, request)");
    return archive_to_pack<ArchiveBinding, PackBinding, Policy>::build(archive, pack, request);
}

inline cell_selection_view make_cell_spans(const cell_span *spans, std::uint32_t count) {
    return cell_selection_view{spans, count, nullptr, 0u};
}

inline cell_selection_view make_cell_indices(const std::uint64_t *indices, std::uint64_t count) {
    return cell_selection_view{nullptr, 0u, indices, count};
}

inline std::uint64_t selected_count(const cell_selection_view &selection, std::uint64_t fallback_all_count) {
    if (selection.index_count != 0u) return selection.index_count;
    if (selection.span_count == 0u) return fallback_all_count;
    std::uint64_t out = 0u;
    for (std::uint32_t i = 0u; i < selection.span_count; ++i) out += selection.spans[i].size();
    return out;
}

inline bool has_capability(std::uint64_t capabilities, adapter_capability capability) {
    return (capabilities & static_cast<std::uint64_t>(capability)) != 0u;
}

} // namespace access
} // namespace cellshard
