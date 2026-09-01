#include <CellShard/compiler/discovery/sequence_compat/mock_sequence_provider_v1.hh>

#include <cassert>
#include <initializer_list>

namespace sequence = cellshard::compiler::discovery::sequence_compat;

int main() {
    sequence::mock_baseplane_shaped_provider_v1 provider;
    for (const auto strand : {sequence::strand_identity_v1::forward,
             sequence::strand_identity_v1::reverse,
             sequence::strand_identity_v1::both,
             sequence::strand_identity_v1::unknown}) {
        provider.build(strand);
        assert(sequence::validate_mock_baseplane_shaped_provider_v1(provider)
                   .valid());
        assert(provider.reference().strand == strand);
        assert(provider.coordinate().owned_count == 10u);
        assert(provider.owned_halos().interval_count == 3u);
        assert(provider.hierarchy().interval_count == 3u);
        assert(provider.bridge().production_count == 1u);
    }
    return 0;
}
