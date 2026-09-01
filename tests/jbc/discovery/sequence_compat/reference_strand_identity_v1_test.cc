#include <CellShard/compiler/discovery/sequence_compat/reference_strand_identity_v1.hh>

#include <cassert>
#include <initializer_list>

namespace sequence = cellshard::compiler::discovery::sequence_compat;

int main() {
    sequence::reference_strand_identity_v1 identity{};
    identity.assembly_identity = {1u, 1u};
    identity.sequence_identity = {1u, 2u};

    for (const auto strand : {sequence::strand_identity_v1::forward,
             sequence::strand_identity_v1::reverse,
             sequence::strand_identity_v1::both,
             sequence::strand_identity_v1::unknown}) {
        identity.strand = strand;
        assert(sequence::validate_reference_strand_identity_v1(identity)
                   .valid());
    }

    auto malformed = identity;
    malformed.sequence_identity = malformed.assembly_identity;
    assert(sequence::validate_reference_strand_identity_v1(malformed).code
        == sequence::reference_strand_identity_validation_code_v1::
            collapsed_assembly_sequence_identity);

    malformed = identity;
    malformed.strand = static_cast<sequence::strand_identity_v1>(0u);
    assert(sequence::validate_reference_strand_identity_v1(malformed).code
        == sequence::reference_strand_identity_validation_code_v1::
            invalid_strand);

    malformed = identity;
    malformed.assembly_identity = {};
    assert(sequence::validate_reference_strand_identity_v1(malformed).code
        == sequence::reference_strand_identity_validation_code_v1::
            invalid_assembly_identity);

    malformed = identity;
    malformed.reserved[3] = 1u;
    assert(sequence::validate_reference_strand_identity_v1(malformed).code
        == sequence::reference_strand_identity_validation_code_v1::
            nonzero_reserved);
    return 0;
}
