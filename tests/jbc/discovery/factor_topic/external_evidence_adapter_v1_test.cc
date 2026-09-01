#include <CellShard/compiler/discovery/factor_topic/external_evidence_adapter_v1.hh>

#include <cassert>

namespace factor_topic = cellshard::compiler::discovery::factor_topic;
namespace evidence = cellshard::compiler::evidence;

int main() {
    factor_topic::external_factor_topic_evidence_v1 source{};
    source.evidence_identity = {11, 12};
    source.subject_atom_identity = {21, 22};
    source.source_identity = {31, 32};
    source.observation_generation = 7;
    source.observation_count = 128;
    source.kind = factor_topic::external_factor_topic_kind_v1::topic;

    evidence::atom_evidence_record_v1 adapted{};
    auto result = factor_topic::adapt_external_factor_topic_evidence_v1(
        source, &adapted);
    assert(result.adapted());
    assert(evidence::validate_atom_evidence_record_v1(adapted).valid());
    assert(adapted.evidence_identity == source.evidence_identity);
    assert(adapted.subject_atom_identity == source.subject_atom_identity);
    assert(adapted.source_identity == source.source_identity);
    assert(adapted.kind == evidence::evidence_kind::factor_membership);
    assert(adapted.disposition == evidence::evidence_disposition_v1::proposal_only);
    assert(!factor_topic::authorizes_execution(source));

    source.observation_generation = 0;
    result = factor_topic::adapt_external_factor_topic_evidence_v1(source, &adapted);
    assert(result.code
           == factor_topic::external_factor_topic_adapter_code_v1::
               missing_observation_generation);
    assert(!evidence::validate_atom_evidence_record_v1(adapted).valid());

    source.observation_generation = 7;
    source.kind = static_cast<factor_topic::external_factor_topic_kind_v1>(99);
    result = factor_topic::adapt_external_factor_topic_evidence_v1(source, &adapted);
    assert(result.code
           == factor_topic::external_factor_topic_adapter_code_v1::invalid_kind);

    result = factor_topic::adapt_external_factor_topic_evidence_v1(source, nullptr);
    assert(result.code
           == factor_topic::external_factor_topic_adapter_code_v1::null_destination);
    return 0;
}
