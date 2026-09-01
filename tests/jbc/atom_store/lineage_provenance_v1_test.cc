#include <CellShard/artifact/atom_store/lineage_provenance_v1.hh>
#include <cassert>
using namespace cellshard::artifact::atom_store;
int main() {
    composition_lineage_record_v1 l{{1,2},{3,4},0,2,1,lineage_operation_v1::compose,0};
    assert(valid_composition_lineage_record_v1(l)); l.parent_count=0; assert(!valid_composition_lineage_record_v1(l));
    provenance_record_v1 p{}; p.subject={1,2}; p.source_content.bytes[0]=std::byte{1};
    p.evidence_content.bytes[0]=std::byte{2}; p.provider_identity=3; p.evidence_generation=4; p.source_epoch=5;
    assert(valid_provenance_record_v1(p)); p.provider_identity=0; assert(!valid_provenance_record_v1(p));
}
