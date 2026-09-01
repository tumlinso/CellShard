#include <CellShard/compiler/discovery/multimodal/exact_certification_v1.hh>

#include <cassert>

namespace multimodal = cellshard::compiler::discovery::multimodal;

int main() {
    const multimodal::modality_identity_binding_v1 bindings[] = {
        {10, 30, 31, 40, 41, 0, 1,
         multimodal::modality_kind_v1::transcriptome, 0},
        {11, 30, 31, 60, 61, 0, 1,
         multimodal::modality_kind_v1::protein, 0},
    };
    const multimodal::multimodal_identity_spine_view_v1 spine{
        bindings, 2, 0, 1, 2, 30, 31, 4};
    const multimodal::multimodal_payload_descriptor_v1 payloads[] = {
        {10, 100, 1, 0, 64, 8,
         multimodal::multimodal_payload_kind_v1::sparse_counts, 8},
        {11, 101, 1, 64, 32, 4,
         multimodal::multimodal_payload_kind_v1::continuous_values, 8},
    };
    const multimodal::multi_payload_atom_v1 atom{
        200, 201, 1, 4, 0, 8, 0, 2, 0};
    multimodal::payload_exact_check_v1 checks[] = {
        {100, 8, 0, 300, 300}, {101, 4, 0, 301, 301},
    };
    multimodal::multimodal_exact_certificate_v1 certificate{};
    auto result = multimodal::certify_multimodal_atom_v1(
        spine, atom, payloads, 2, 96, checks, 2, 500, &certificate);
    assert(result.certified());
    assert(certificate.certified == 1);
    assert(certificate.checked_element_count == 12);
    checks[1].observed_checksum = 302;
    result = multimodal::certify_multimodal_atom_v1(
        spine, atom, payloads, 2, 96, checks, 2, 500, &certificate);
    assert(result.code
           == multimodal::multimodal_certification_code_v1::oracle_mismatch);
    assert(certificate.certified == 0);
    return 0;
}
