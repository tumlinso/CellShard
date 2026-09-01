#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

using namespace cellshard::compiler::atom;
using namespace cellshard::compiler::partial;

alignas(16) std::array<float, 4> payload{};
alignas(8) std::array<std::byte, 64> coverage{};

partial_atom_view_v1 valid_partial() {
    partial_atom_view_v1 partial{};
    partial.payload = payload.data();
    partial.payload_bytes = sizeof(payload);
    partial.payload_alignment = alignof(decltype(payload));
    partial.header.partial_identity = {1, 1};
    partial.header.source_atom_semantic_identity = {1, 2};
    partial.header.partial_kind_identity = {1, 3};
    partial.header.payload_schema_identity = {1, 4};
    partial.header.contribution_coverage_identity = {1, 5};
    partial.header.dependency_closure_identity = {1, 6};
    partial.header.reconstruction_algebra_identity = {1, 7};
    partial.header.numerical_policy_identity = {1, 8};
    partial.header.complete_cost_evidence_identity = {1, 9};
    partial.header.structure_generation = 2;
    partial.header.value_generation = 3;
    partial.header.state_generation = 4;
    partial.header.materialization_generation = 5;
    partial.header.cost_model_generation = 6;
    partial.result.partial_layout.values = payload.data();
    partial.result.partial_layout.value_bytes = sizeof(payload);
    partial.result.partial_layout.value_alignment = alignof(decltype(payload));
    partial.result.exact_contribution_coverage.cellerator_coverage =
        coverage.data();
    partial.result.exact_contribution_coverage.coverage_identity = {1, 5};
    partial.result.reconstruction_algebra_identity = {1, 7};
    partial.result.numerical_policy_identity = {1, 8};
    partial.result.status = atom_partial_result_status_v1::ready_to_merge;
    return partial;
}

void test_layout_and_valid_envelope() {
    static_assert(std::is_standard_layout<partial_atom_header_v1>::value);
    static_assert(std::is_trivially_copyable<partial_atom_header_v1>::value);
    static_assert(std::is_standard_layout<partial_atom_view_v1>::value);
    static_assert(std::is_trivially_copyable<partial_atom_view_v1>::value);
    assert(validate_partial_atom_envelope_v1(valid_partial()).valid());
}

void test_version_and_identity_rejections() {
    auto partial = valid_partial();
    partial.header.schema_version = 2;
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::unsupported_schema);
    partial = valid_partial();
    partial.header.record_bytes = 0;
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::invalid_record_bytes);
    partial = valid_partial();
    partial.header.dependency_closure_identity = {};
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::
                  invalid_dependency_closure_identity);
    partial = valid_partial();
    partial.header.complete_cost_evidence_identity = {};
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::
                  invalid_complete_cost_evidence);
}

void test_generation_and_scientific_rejections() {
    auto partial = valid_partial();
    partial.header.value_generation = 0;
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::
                  missing_value_generation);
    partial = valid_partial();
    partial.header.persistence_class =
        static_cast<partial_persistence_class_v1>(2);
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::
                  scientifically_unknown_partial);
    partial = valid_partial();
    partial.result.status = atom_partial_result_status_v1::accumulating;
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::incomplete_partial);
}

void test_payload_and_contract_rejections() {
    auto partial = valid_partial();
    partial.payload = nullptr;
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::missing_payload);
    partial = valid_partial();
    partial.payload_alignment = 3;
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::
                  invalid_payload_alignment);
    partial = valid_partial();
    partial.result.partial_layout.value_bytes -= sizeof(float);
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::
                  payload_binding_mismatch);
    partial = valid_partial();
    partial.result.exact_contribution_coverage.coverage_identity = {2, 5};
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::
                  coverage_binding_mismatch);
    partial = valid_partial();
    partial.result.reconstruction_algebra_identity = {2, 7};
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::
                  algebra_binding_mismatch);
    partial = valid_partial();
    partial.result.numerical_policy_identity = {2, 8};
    assert(validate_partial_atom_envelope_v1(partial).code
           == partial_atom_envelope_validation_code_v1::
                  numerical_policy_binding_mismatch);
}

} // namespace

int main() {
    test_layout_and_valid_envelope();
    test_version_and_identity_rejections();
    test_generation_and_scientific_rejections();
    test_payload_and_contract_rejections();
    return 0;
}
