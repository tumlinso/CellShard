#include <CellShard/compiler/atom/relation_edge_spine_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using namespace cellshard::compiler::atom;

void write_compact(std::byte *destination, std::uint64_t index,
                   compact_edge_index_width_v1 width,
                   std::uint64_t value) {
    const auto bytes = static_cast<std::uint8_t>(width);
    for (std::uint8_t byte = 0; byte < bytes; ++byte) {
        destination[index * bytes + byte] = static_cast<std::byte>(
            (value >> (byte * 8)) & UINT64_C(0xff));
    }
}

void test_stable_global_spine() {
    const std::uint64_t edges[]{19, UINT64_C(1) << 40, (UINT64_C(1) << 63) + 7};
    const relation_edge_spine_view_v1 spine{edges, 3, {81, 901}, 17};
    const auto result = validate_relation_edge_spine_v1(spine);
    assert(result.valid());
    assert(result.index == 3);
}

void test_compact_local_orders() {
    const std::uint64_t edges[]{11, 22, 33, 44};
    const relation_edge_spine_view_v1 spine{edges, 4, {81, 901}, 17};
    const std::array<std::byte, 4> forward{
        std::byte{0}, std::byte{1}, std::byte{2}, std::byte{3}};
    const std::array<std::byte, 4> transpose{
        std::byte{3}, std::byte{1}, std::byte{0}, std::byte{2}};
    std::array<std::uint8_t, 4> marks{};
    relation_edge_local_map_view_v1 mapping{
        forward.data(), forward.size(), forward.size(),
        compact_edge_index_width_v1::u8,
        {}, {81, 901}, 17};
    assert(validate_relation_edge_local_map_v1(
               spine, mapping, marks.data(), marks.size())
               .valid());
    mapping.canonical_ordinals = transpose.data();
    assert(validate_relation_edge_local_map_v1(
               spine, mapping, marks.data(), marks.size())
               .valid());
}

void test_deterministic_rejections() {
    std::uint64_t edges[]{11, 22, 33};
    relation_edge_spine_view_v1 spine{edges, 3, {81, 901}, 17};
    auto malformed = spine;
    malformed.relation_identity = {};
    assert(validate_relation_edge_spine_v1(malformed).code
           == relation_edge_spine_validation_code_v1::
                  invalid_relation_identity);
    malformed = spine;
    malformed.structure_epoch = 0;
    assert(validate_relation_edge_spine_v1(malformed).code
           == relation_edge_spine_validation_code_v1::
                  missing_structure_epoch);
    malformed = spine;
    malformed.edge_count = 0;
    assert(validate_relation_edge_spine_v1(malformed).code
           == relation_edge_spine_validation_code_v1::empty_spine);
    malformed = spine;
    malformed.global_edge_ids = nullptr;
    assert(validate_relation_edge_spine_v1(malformed).code
           == relation_edge_spine_validation_code_v1::
                  missing_global_edge_ids);
    edges[1] = 0;
    assert(validate_relation_edge_spine_v1(spine).code
           == relation_edge_spine_validation_code_v1::
                  zero_global_edge_identity);
    edges[1] = 11;
    assert(validate_relation_edge_spine_v1(spine).code
           == relation_edge_spine_validation_code_v1::
                  unordered_or_duplicate_global_edge);
    edges[1] = 22;

    std::array<std::byte, 3> indices{
        std::byte{0}, std::byte{1}, std::byte{2}};
    std::array<std::uint8_t, 3> marks{};
    relation_edge_local_map_view_v1 mapping{
        indices.data(), indices.size(), indices.size(),
        compact_edge_index_width_v1::u8,
        {}, {81, 901}, 17};
    auto result = validate_relation_edge_local_map_v1(
        spine, mapping, nullptr, marks.size());
    assert(result.code
           == relation_edge_spine_validation_code_v1::missing_marks);
    result = validate_relation_edge_local_map_v1(
        spine, mapping, marks.data(), marks.size() - 1);
    assert(result.code
           == relation_edge_spine_validation_code_v1::insufficient_marks);
    mapping.canonical_ordinal_bytes = 2;
    result = validate_relation_edge_local_map_v1(
        spine, mapping, marks.data(), marks.size());
    assert(result.code
           == relation_edge_spine_validation_code_v1::invalid_index_bytes);
    mapping.canonical_ordinal_bytes = indices.size();
    mapping.local_edge_count = std::numeric_limits<std::uint64_t>::max();
    mapping.index_width = compact_edge_index_width_v1::u16;
    result = validate_relation_edge_local_map_v1(
        spine, mapping, marks.data(), marks.size());
    assert(result.code
           == relation_edge_spine_validation_code_v1::index_bytes_overflow);
    mapping.local_edge_count = indices.size();
    mapping.index_width = compact_edge_index_width_v1::u8;
    indices[2] = std::byte{3};
    result = validate_relation_edge_local_map_v1(
        spine, mapping, marks.data(), marks.size());
    assert(result.code
           == relation_edge_spine_validation_code_v1::local_index_out_of_range);
    indices[2] = std::byte{1};
    result = validate_relation_edge_local_map_v1(
        spine, mapping, marks.data(), marks.size());
    assert(result.code
           == relation_edge_spine_validation_code_v1::duplicate_local_index);
}

void test_randomized_compact_widths_and_orders() {
    for (std::uint64_t count : {UINT64_C(5), UINT64_C(255), UINT64_C(257)}) {
        const auto width = count <= 256
            ? compact_edge_index_width_v1::u8
            : compact_edge_index_width_v1::u16;
        const auto width_bytes = static_cast<std::uint8_t>(width);
        std::vector<std::uint64_t> edges(count);
        std::vector<std::byte> indices(count * width_bytes);
        std::vector<std::uint8_t> marks(count);
        for (std::uint64_t index = 0; index < count; ++index) {
            edges[index] = 100 + index * 3;
            write_compact(indices.data(), index, width, count - index - 1);
        }
        const relation_edge_spine_view_v1 spine{
            edges.data(), count, {81, 901}, 17};
        const relation_edge_local_map_view_v1 mapping{
            indices.data(), count, indices.size(), width, {}, {81, 901}, 17};
        assert(validate_relation_edge_local_map_v1(
                   spine, mapping, marks.data(), marks.size())
                   .valid());
    }

    const std::uint64_t edge = 9;
    std::byte index{};
    std::uint8_t mark{};
    const relation_edge_spine_view_v1 oversized{
        &edge, 257, {81, 901}, 17};
    const relation_edge_local_map_view_v1 too_narrow{
        &index, 1, 1, compact_edge_index_width_v1::u8, {}, {81, 901}, 17};
    // The pointer is intentionally not traversed: width capacity fails first
    // after the separately valid spine precondition is supplied below.
    std::vector<std::uint64_t> edges(257);
    for (std::size_t i = 0; i < edges.size(); ++i) edges[i] = i + 1;
    const relation_edge_spine_view_v1 valid_oversized{
        edges.data(), edges.size(), {81, 901}, 17};
    assert(validate_relation_edge_local_map_v1(
               valid_oversized, too_narrow, &mark, 1)
               .code == relation_edge_spine_validation_code_v1::width_too_small);
    (void)oversized;
}

} // namespace

int main() {
    test_stable_global_spine();
    test_compact_local_orders();
    test_deterministic_rejections();
    test_randomized_compact_widths_and_orders();
    return 0;
}
