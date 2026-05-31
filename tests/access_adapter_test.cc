#include <CellShard/access.hh>

#include <cassert>
#include <cstdint>
#include <cstring>

namespace {

struct fake_external_archive {
    std::uint64_t cells;
    std::uint64_t features;
    std::uint64_t nnz;
};

struct fake_archive_binding {
    const fake_external_archive *archive;
};

struct fake_pack_binding {
    std::uint32_t selected_format;
    std::uint64_t build_calls;
};

struct incomplete_binding {};

} // namespace

namespace cellshard {
namespace access {

template<>
struct archive_adapter<fake_archive_binding> {
    static archive_descriptor describe(const adapter_view<fake_archive_binding> &view) {
        return archive_descriptor{
            view.binding->archive->cells,
            view.binding->archive->features,
            view.binding->archive->nnz,
            42u,
            types::value_f32,
            disk_format_none,
            dataset_execution_format_unknown,
            capability_archive_to_pack
        };
    }
};

template<>
struct pack_adapter<fake_pack_binding> {
    static pack_descriptor describe(const adapter_view<fake_pack_binding> &view) {
        return pack_descriptor{0u, 0u, 0u, types::value_f32, view.binding->selected_format, 0u, capability_pack_write};
    }
};

template<>
struct archive_to_pack<fake_archive_binding, fake_pack_binding, passthrough_pack_policy> {
    static pack_build_result build(
        const adapter_view<fake_archive_binding> &archive,
        const adapter_view<fake_pack_binding> &pack,
        const pack_build_request &request) {
        ++pack.binding->build_calls;
        pack.binding->selected_format = request.requested_execution_format;
        const archive_descriptor desc = describe_archive(archive);
        return pack_build_result{
            desc.cell_count,
            desc.feature_count,
            desc.nnz,
            0u,
            request.requested_execution_format,
            1u
        };
    }
};

} // namespace access
} // namespace cellshard

int main() {
    using namespace cellshard;
    using namespace cellshard::access;

    static_assert(is_archive_adapter<fake_archive_binding>::value, "fake archive binding should satisfy archive contract");
    static_assert(is_pack_adapter<fake_pack_binding>::value, "fake pack binding should satisfy pack contract");
    static_assert(is_archive_to_pack_adapter<fake_archive_binding, fake_pack_binding>::value,
                  "fake archive-to-pack override should be visible");
    static_assert(!is_archive_adapter<incomplete_binding>::value, "missing describe must fail the archive contract");

    fake_external_archive fake_archive{11u, 7u, 19u};
    fake_archive_binding fake_archive_binding_value{&fake_archive};
    fake_pack_binding fake_pack_binding_value{dataset_execution_format_unknown, 0u};
    auto fake_archive_view = make_adapter_view(fake_archive_binding_value);
    auto fake_pack_view = make_adapter_view(fake_pack_binding_value);
    const pack_build_result fake_result = build_pack(
        fake_archive_view,
        fake_pack_view,
        pack_build_request{
            dataset_generation_ref{1u, 2u, 3u, 4u},
            dataset_partition_ref{5u, 6u},
            cell_selection_view{},
            feature_selection_view{},
            mutable_byte_span{},
            mutable_byte_span{},
            dataset_execution_format_dense,
            0u,
            nullptr
        });
    assert(fake_result.status == 1u);
    assert(fake_result.cell_count == 11u);
    assert(fake_pack_binding_value.build_calls == 1u);
    assert(fake_pack_binding_value.selected_format == dataset_execution_format_dense);

    types::storage_value_t dense_values[6] = {
        types::storage_value_t(1), types::storage_value_t(2), types::storage_value_t(3),
        types::storage_value_t(4), types::storage_value_t(5), types::storage_value_t(6)
    };
    dense dense_matrix{};
    attach(&dense_matrix, 2u, 3u, dense_values, dense_row_major);
    dense_fallback_binding dense_binding{&dense_matrix, nullptr, 9u};
    auto dense_view = make_adapter_view(dense_binding);
    const archive_descriptor dense_desc = describe_archive(dense_view);
    assert(dense_desc.cell_count == 2u);
    assert(dense_desc.feature_count == 3u);
    assert(dense_desc.nnz == 6u);
    assert(dense_desc.archive_format == disk_format_dense);
    assert(has_capability(dense_desc.capabilities, capability_cell_span_copy));

    const cell_span dense_cell_spans[] = {cell_span{1u, 2u}};
    const feature_span dense_feature_spans[] = {feature_span{1u, 3u}};
    types::storage_value_t dense_out[2] = {};
    const copy_preflight dense_need = preflight_cell_spans(
        dense_view,
        make_cell_spans(dense_cell_spans, 1u),
        make_cell_spans(dense_feature_spans, 1u));
    assert(dense_need.cell_count == 1u);
    assert(dense_need.feature_count == 2u);
    assert(dense_need.output_bytes == sizeof(dense_out));

    const pack_build_result dense_copy = build_pack(
        dense_view,
        dense_view,
        pack_build_request{
            dataset_generation_ref{},
            dataset_partition_ref{},
            make_cell_spans(dense_cell_spans, 1u),
            make_cell_spans(dense_feature_spans, 1u),
            mutable_byte_span{dense_out, sizeof(dense_out), types::value_code<types::storage_value_t>::code},
            mutable_byte_span{},
            dataset_execution_format_dense,
            0u,
            nullptr
        });
    assert(dense_copy.status == 1u);
    assert(dense_out[0] == types::storage_value_t(5));
    assert(dense_out[1] == types::storage_value_t(6));

    types::ptr_t ptrs[] = {0u, 1u, 1u, 2u};
    types::idx_t idx[] = {2u, 0u};
    types::storage_value_t vals[] = {types::storage_value_t(7), types::storage_value_t(8)};
    sparse::compressed compressed_matrix{};
    sparse::init(&compressed_matrix, 3u, 4u, 2u, sparse::compressed_by_row);
    compressed_matrix.majorPtr = ptrs;
    compressed_matrix.minorIdx = idx;
    compressed_matrix.val = vals;
    compressed_fallback_binding compressed_binding{&compressed_matrix, nullptr, 10u};
    auto compressed_view = make_adapter_view(compressed_binding);
    const archive_descriptor compressed_desc = describe_archive(compressed_view);
    assert(compressed_desc.cell_count == 3u);
    assert(compressed_desc.feature_count == 4u);
    assert(compressed_desc.nnz == 2u);
    assert(compressed_desc.archive_format == disk_format_compressed);

    const fallback_payload_view compressed_payload = contiguous_payload(compressed_view);
    assert(compressed_payload.major_ptr.bytes == sizeof(ptrs));
    assert(compressed_payload.minor_idx.bytes == sizeof(idx));
    assert(compressed_payload.values.bytes == sizeof(vals));

    const pack_descriptor compressed_pack = describe_pack(compressed_view);
    assert(compressed_pack.execution_format == dataset_execution_format_compressed);
    assert(compressed_pack.nnz == 2u);

    return 0;
}
