#include <CellShard/compiler/discovery/trajectory/trajectory_promotion_gate_v1.hh>
#include <array>
#include <iostream>
namespace tr=cellshard::compiler::discovery::trajectory;
int main(){
 tr::predictive_prefetch_measurement_v1 observed{{1,1},{2,1},{2,2},3,32,31,4096,3900,240,10,20,30,5};
 std::array<std::uint64_t,127>nulls{};std::uint64_t state=23;for(auto&x:nulls){state=state*6364136223846793005ULL+1442695040888963407ULL;x=100+(state%201);}
 const auto result=tr::evaluate_trajectory_promotion_v1(observed,{32,19,20,9,10},{nulls.data(),nulls.size(),nulls.size(),{3,1},{1,1},3},{127,19,20});
 if(!result.evaluated())return 2;
 std::cout<<"{\"promote\":"<<(result.promote?"true":"false")<<",\"observed_complete_ns\":"<<result.observed_complete_cost_ns<<",\"null_trials_slower\":"<<result.null_trials_slower<<",\"null_trial_count\":"<<result.null_trial_count<<"}\n";
 return 0;
}
