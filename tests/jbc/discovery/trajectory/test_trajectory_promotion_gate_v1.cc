#include <CellShard/compiler/discovery/trajectory/trajectory_promotion_gate_v1.hh>
#include <array>
#include <cassert>
#include <cstdint>
namespace tr=cellshard::compiler::discovery::trajectory;
int main(){
 tr::predictive_prefetch_measurement_v1 observed{{1,1},{2,1},{2,2},3,20,19,1000,900,200,10,20,30,5};
 tr::predictive_prefetch_policy_v1 pp{20,9,10,4,5};tr::trajectory_promotion_policy_v1 gp{31,9,10};
 std::array<std::uint64_t,63>nulls{};std::uint64_t state=17;
 for(auto&x:nulls){state=state*6364136223846793005ULL+1;x=100+(state%101);}
 std::uint64_t reference=0;for(auto x:nulls)if(65<x)++reference;
 tr::trajectory_null_controls_v1 controls{nulls.data(),nulls.size(),nulls.size(),{3,1},{1,1},3};
 const auto r=tr::evaluate_trajectory_promotion_v1(observed,pp,controls,gp);
 assert(r.evaluated()&&r.null_trials_slower==reference&&r.promote==(reference*10>=nulls.size()*9)&&!tr::authorizes_execution(r));
 controls.observation_generation=4;assert(tr::evaluate_trajectory_promotion_v1(observed,pp,controls,gp).code==tr::trajectory_promotion_code_v1::invalid_generation);
}
