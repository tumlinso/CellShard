#include <CellShard/artifact/atom_store/consolidation_v1.hh>
#include <array>
#include <cassert>
using namespace cellshard::artifact::atom_store;
int main(){std::array<consolidation_source_v1,3>s{};for(auto&i:s){i.content.bytes[0]=std::byte{1};i.bytes=10;i.alignment=8;i.live=1;}s[0].source_offset=10;s[1].live=0;s[2].source_offset=30;std::array<consolidation_copy_v1,2>c{};std::size_t n=0;std::uint64_t bytes=0;assert(plan_consolidation_v1(s.data(),s.size(),c.data(),c.size(),&n,&bytes)==consolidation_status_v1::success);assert(n==2&&c[0].target_offset==0&&c[1].target_offset==16&&bytes==26);assert(plan_consolidation_v1(s.data(),s.size(),c.data(),1,&n,&bytes)==consolidation_status_v1::insufficient_output);}
