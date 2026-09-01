#include <CellShard/compiler/certification/atom_certification_v1.hh>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace {

using namespace cellshard::compiler;

certification::atom_certification_request_v1 make_valid_request(
    atom::common_atom_view_v1 *atom_view,
    std::uint64_t *workspace) {
    certification::atom_certification_request_v1 request{};
    request.proposed_atoms = atom_view;
    request.workspace = workspace;
    request.proposed_atom_count = 1;
    request.workspace_bytes = sizeof(*workspace);
    request.request_identity = {1, 1};
    request.proposal_provider_identity = {2, 1};
    request.certification_authority_identity = {3, 1};
    request.canonical_source_identity = {4, 1};
    request.canonical_source_generation = 1;
    request.maximum_local_index_width =
        certification::certification_local_index_width_v1::u32;
    return request;
}

} // namespace

int main() {
    static_assert(std::is_same<
                      decltype(certification::atom_certification_request_v1::
                                   proposed_atom_count),
                      std::uint64_t>::value);
    static_assert(std::is_same<
                      decltype(certification::atom_certification_result_v1::
                                   certified_entity_count),
                      std::uint64_t>::value);
    static_assert(std::is_same<
                      decltype(certification::atom_certification_result_v1::
                                   certified_relation_edge_count),
                      std::uint64_t>::value);

    atom::common_atom_view_v1 atom_view{};
    std::uint64_t workspace = 0;
    auto request = make_valid_request(&atom_view, &workspace);
    assert(certification::validate_atom_certification_request_v1(request)
               .valid());

    request.certification_authority_identity =
        request.proposal_provider_identity;
    assert(certification::validate_atom_certification_request_v1(request).code
           == certification::atom_certification_request_validation_code_v1::
               provider_self_certification);

    request = make_valid_request(&atom_view, &workspace);
    request.proposed_atom_count = UINT64_C(1) << 32;
    assert(certification::validate_atom_certification_request_v1(request)
               .valid());

    request.maximum_local_index_width =
        static_cast<certification::certification_local_index_width_v1>(3);
    assert(certification::validate_atom_certification_request_v1(request).code
           == certification::atom_certification_request_validation_code_v1::
               invalid_local_index_width);

    request = make_valid_request(&atom_view, &workspace);
    request.workspace = nullptr;
    assert(certification::validate_atom_certification_request_v1(request).code
           == certification::atom_certification_request_validation_code_v1::
               missing_workspace);

    request = make_valid_request(&atom_view, &workspace);
    request.reserved[2] = 1;
    assert(certification::validate_atom_certification_request_v1(request).code
           == certification::atom_certification_request_validation_code_v1::
               nonzero_reserved);

    certification::atom_certification_result_v1 result{};
    assert(!result.certified());
    result.outcome = certification::atom_certification_outcome_v1::certified;
    assert(result.certified());
}
