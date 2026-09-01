#include <CellShard/compiler/certification/partial_result_compatibility_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace cellshard::compiler;

namespace {

alignas(16) float values[4]{};
alignas(8) std::byte coverage_record[248]{};
const std::array<std::byte, 4> identity_map{
    std::byte{0}, std::byte{1}, std::byte{2}, std::byte{3}};
const atom::atom_dependency_generation_v1 dependency{{7, 1}, {8, 1}, 11};

atom::atom_partial_result_plane_v1 make_partial(std::uint64_t owner) {
    atom::atom_partial_result_plane_v1 partial{};
    auto &layout = partial.partial_layout;
    layout.values = values;
    layout.value_bytes = sizeof(values);
    layout.element_count = 4;
    layout.element_stride_bytes = sizeof(float);
    layout.element_bytes = sizeof(float);
    layout.value_alignment = 16;
    layout.plane_identity = {1, owner};
    layout.structure_plane_identity = {1, 12};
    layout.subject_space_identity = {1, 13};
    layout.persistent_order_identity = {1, 14};
    layout.numeric = {{2, 1}, {2, 1}, {2, 2}};
    layout.structure_epoch = 7;
    layout.value_generation = 9;
    layout.canonical_element_count = 4;
    layout.local_to_canonical = {
        identity_map.data(), 4, 4, atom::compact_edge_index_width_v1::u8, {}};
    partial.exact_contribution_coverage = {
        coverage_record,
        {3, owner},
        4,
        1,
        248,
        atom::atom_certified_exact_coverage_role_v1
            | atom::atom_partial_contribution_owner_role_v1,
        atom::atom_logical_coverage_kind_v1::relation_edge_ids,
        0};
    partial.contribution_owner_identity = {4, owner};
    partial.reconstruction_algebra_identity = {4, 32};
    partial.numerical_policy_identity = {4, 33};
    partial.dependencies = &dependency;
    partial.dependency_count = 1;
    partial.status = atom::atom_partial_result_status_v1::ready_to_merge;
    return partial;
}

} // namespace

int main() {
    atom::atom_partial_result_plane_v1 partials[]{make_partial(1),
                                                   make_partial(2)};
    std::array<std::uint8_t, 4> marks{};
    assert(certification::validate_partial_result_algebra_compatibility_v1(
               partials, 2, 0, marks.data(), marks.size())
               .compatible());

    partials[1].reconstruction_algebra_identity = {4, 34};
    assert(certification::validate_partial_result_algebra_compatibility_v1(
               partials, 2, 0, marks.data(), marks.size())
               .code
           == certification::partial_result_compatibility_code_v1::
               reconstruction_algebra_mismatch);

    partials[1] = make_partial(2);
    partials[1].partial_layout.numeric.accumulation_type = {2, 3};
    assert(certification::validate_partial_result_algebra_compatibility_v1(
               partials, 2, 0, marks.data(), marks.size())
               .code
           == certification::partial_result_compatibility_code_v1::
               numeric_mismatch);
}
