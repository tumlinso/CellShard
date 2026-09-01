#include <CellShard/compiler/evidence/biological_stratum_v1.hh>

#include <cassert>

namespace evidence = cellshard::compiler::evidence;

int main() {
    evidence::biological_stratum_ref_v1 record{};
    record.stratum_identity = {1, 1};
    record.domain_identity = {1, 2};
    record.order_identity = {1, 3};
    record.selection_identity = {1, 4};
    record.domain_generation = 5;
    record.selection_generation = 6;
    record.selected_element_count = 7;
    assert(evidence::validate_biological_stratum_v1(record).valid());

    auto malformed = record;
    malformed.order_identity = {};
    assert(evidence::validate_biological_stratum_v1(malformed).code
           == evidence::biological_stratum_validation_code_v1::
               invalid_order_identity);
    malformed = record;
    malformed.selection_generation = 0;
    assert(evidence::validate_biological_stratum_v1(malformed).code
           == evidence::biological_stratum_validation_code_v1::
               missing_selection_generation);
    malformed = record;
    malformed.selected_element_count = 0;
    assert(evidence::validate_biological_stratum_v1(malformed).code
           == evidence::biological_stratum_validation_code_v1::empty_selection);
}
