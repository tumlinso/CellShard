#include <CellShard/compiler/evidence/algorithm_provenance_v1.hh>

#include <cassert>
#include <cstddef>

namespace evidence = cellshard::compiler::evidence;

namespace {

cellshard::content_digest digest(std::byte value) {
    cellshard::content_digest result{};
    result.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    result.used_bytes = 8;
    result.bytes[0] = value;
    return result;
}

evidence::algorithm_provenance_v1 valid_record() {
    evidence::algorithm_provenance_v1 record{};
    record.provenance_identity = {1, 2};
    record.algorithm_identity = {1, 3};
    record.execution_environment_identity = {1, 4};
    record.implementation_digest = digest(std::byte{1});
    record.parameter_digest = digest(std::byte{2});
    record.algorithm_revision = 5;
    return record;
}

} // namespace

int main() {
    auto record = valid_record();
    assert(evidence::validate_algorithm_provenance_v1(record).valid());

    record.algorithm_revision = 0;
    assert(evidence::validate_algorithm_provenance_v1(record).code
           == evidence::algorithm_provenance_validation_code_v1::
               missing_algorithm_revision);

    record = valid_record();
    record.parameter_digest.bytes[31] = std::byte{1};
    assert(evidence::validate_algorithm_provenance_v1(record).code
           == evidence::algorithm_provenance_validation_code_v1::
               invalid_parameter_digest);

    record = valid_record();
    record.implementation_digest = {};
    assert(evidence::validate_algorithm_provenance_v1(record).code
           == evidence::algorithm_provenance_validation_code_v1::
               missing_implementation_digest);

    record = valid_record();
    record.execution_environment_identity = {};
    assert(evidence::validate_algorithm_provenance_v1(record).code
           == evidence::algorithm_provenance_validation_code_v1::
               invalid_environment_identity);
}
