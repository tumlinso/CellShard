#include <CellShard/compiler/discovery/multimodal/persistent_order_proposal_v1.hh>

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
    const multimodal::cross_modal_order_key_v1 keys[] = {
        {10, 2, 3}, {11, 9, 4}, {10, 1, 3}, {10, 0, 5},
    };
    multimodal::persistent_order_entry_v1 entries[4]{};
    multimodal::persistent_order_proposal_v1 proposals[2]{};
    auto result = multimodal::propose_persistent_orders_v1(
        spine, keys, 4, 100, 90, entries, 4, proposals, 2);
    assert(result.proposed());
    assert(entries[0].canonical_entity_id == 0);
    assert(entries[1].canonical_entity_id == 1);
    assert(entries[2].canonical_entity_id == 2);
    assert(entries[0].execution_ordinal == 0);
    assert(proposals[0].entry_count == 3);
    assert(proposals[1].entry_offset == 3);
    return 0;
}
