#include <compiler/grammar/induced/repeated_candidate_v1.hh>
#include <array>
#include <cassert>
namespace ig=cellshard::compiler::grammar::induced;
int main(){ig::execution_observation_v1 x[4]{};for(std::uint64_t i=0;i<4;++i)x[i]={i+1,{1,i%2+1},{2,i%2+1},{3,1},{3,2},4,i+1,i*10,i*10+10,64,1};ig::induced_grammar_observation_view_v1 v{x,4,4,{4,1},{4,2},7};ig::induced_identity_v1 ids[]={{5,1}};std::array<ig::repeated_production_candidate_v1,1>out{};auto r=ig::mine_repeated_candidates_v1(v,2,4,ids,1,out.data(),1);assert(r.mined()&&r.candidate_count==1&&out[0].first_begin==0&&out[0].second_begin==2&&out[0].symbol_count==2&&!ig::authorizes_execution(out[0]));}
