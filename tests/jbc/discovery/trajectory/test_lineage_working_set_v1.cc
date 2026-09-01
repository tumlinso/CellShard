#include <CellShard/compiler/discovery/trajectory/lineage_working_set_v1.hh>
#include <array>
#include <cassert>
namespace tr=cellshard::compiler::discovery::trajectory;
int main(){
 tr::lineage_working_set_observation_v1 in[]={{{1,1},64,3,7},{{1,1},96,5,7},{{1,2},32,2,7}};
 cellshard::compiler::evidence::evidence_identity_v1 ids[]={{2,1},{2,2}};
 std::array<tr::lineage_working_set_evidence_v1,2> out{};
 auto r=tr::build_lineage_working_set_evidence_v1(in,3,ids,2,out.data(),2);
 assert(r.built()&&r.evidence_count==2&&out[0].maximum_resident_bytes==96&&out[0].total_access_count==8&&out[0].observation_count==2);
 assert(!tr::authorizes_execution(out[0]));
}
