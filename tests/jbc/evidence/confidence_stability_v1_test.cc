#include <CellShard/compiler/evidence/confidence_stability_v1.hh>

#include <cassert>
#include <cstdint>
#include <initializer_list>

namespace evidence = cellshard::compiler::evidence;

int main() {
    evidence::confidence_stability_v1 record{};
    record.evidence_identity = {1, 2};
    record.confidence_numerator = 9;
    record.confidence_denominator = 10;
    record.stable_resamples = 8;
    record.total_resamples = 10;
    record.supporting_strata = 3;
    record.observed_strata = 4;
    record.assessment = evidence::evidence_assessment_v1::candidate_supported;
    assert(evidence::validate_confidence_stability_v1(record).valid());
    assert(!evidence::authorizes_execution(record));

    for (const auto assessment : {
             evidence::evidence_assessment_v1::weak_evidence,
             evidence::evidence_assessment_v1::unstable_evidence,
             evidence::evidence_assessment_v1::null_result,
             evidence::evidence_assessment_v1::no_promotion}) {
        record.assessment = assessment;
        assert(evidence::validate_confidence_stability_v1(record).valid());
        assert(!evidence::authorizes_execution(record));
    }

    auto malformed = record;
    malformed.stable_resamples = malformed.total_resamples + 1;
    assert(evidence::validate_confidence_stability_v1(malformed).code
           == evidence::confidence_stability_validation_code_v1::
               invalid_stability);
    malformed = record;
    malformed.confidence_denominator = 0;
    assert(evidence::validate_confidence_stability_v1(malformed).code
           == evidence::confidence_stability_validation_code_v1::
               invalid_confidence);
    malformed = record;
    malformed.assessment = static_cast<evidence::evidence_assessment_v1>(
        UINT32_MAX);
    assert(evidence::validate_confidence_stability_v1(malformed).code
           == evidence::confidence_stability_validation_code_v1::
               invalid_assessment);
}
