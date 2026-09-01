#include <CellShard/compiler/discovery/multimodal/shared_destination_bundle_v1.hh>

#include <cassert>

namespace multimodal = cellshard::compiler::discovery::multimodal;

int main() {
    const multimodal::cross_modal_relation_atom_v1 atoms[] = {
        {1, 10, 20, 1, 101, 3, 301, 1, 1,
         multimodal::cross_modal_relation_kind_v1::regulatory, 1},
        {2, 11, 21, 2, 201, 3, 301, 1, 1,
         multimodal::cross_modal_relation_kind_v1::regulatory, 1},
        {3, 12, 22, 1, 102, 3, 302, 1, 1,
         multimodal::cross_modal_relation_kind_v1::regulatory, 1},
    };
    multimodal::shared_destination_bundle_v1 bundles[2]{};
    std::uint64_t members[3]{};
    auto result = multimodal::propose_shared_destination_bundles_v1(
        atoms, 3, 2, 100, bundles, 2, members, 3, 16);
    assert(result.proposed());
    assert(result.bundle_count == 1 && result.member_count == 2);
    assert(bundles[0].destination_entity_identity == 301);
    assert(bundles[0].source_modality_count == 2);
    assert(members[0] == 1 && members[1] == 2);
    return 0;
}
