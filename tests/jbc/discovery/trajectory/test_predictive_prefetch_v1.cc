#include <CellShard/compiler/discovery/trajectory/predictive_prefetch_v1.hh>
#include <cassert>
#include <cstdint>
namespace tr=cellshard::compiler::discovery::trajectory;
int main(){
 tr::predictive_prefetch_policy_v1 p{10,9,10,3,4};
 for(std::uint64_t base=100;base<500;++base){
  tr::predictive_prefetch_measurement_v1 x{{1,1},{2,1},{2,2},3,10,9,100,80,base,10,20,30,5};
  const auto r=tr::evaluate_predictive_prefetch_v1(x,p);
  assert(r.evaluated()&&r.complete_prefetch_ns==65&&r.promote==(65<base)&&!tr::authorizes_execution(r));
 }
}
