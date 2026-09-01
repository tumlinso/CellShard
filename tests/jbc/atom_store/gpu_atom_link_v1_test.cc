#include <CellShard/artifact/atom_store/gpu_atom_link_v1.hh>
#include <array>
#include <cassert>
using namespace cellshard::artifact::atom_store;
int main(){std::array<semantic_identity_v1,3>d{{{1,1},{1,3},{2,1}}};std::array<semantic_identity_v1,4>q{{{1,3},{1,2},{2,1},{9,9}}};std::array<std::uint64_t,4>out{};assert(link_atoms_cpu_reference_v1(d.data(),d.size(),q.data(),q.size(),out.data())==atom_link_status_v1::success);assert(out[0]==1&&out[1]==atom_link_missing_v1&&out[2]==2&&out[3]==atom_link_missing_v1);std::swap(d[0],d[1]);assert(link_atoms_cpu_reference_v1(d.data(),d.size(),q.data(),q.size(),out.data())==atom_link_status_v1::unsorted_dictionary);}
