#include <compiler/grammar/induced/canonical_production_v1.hh>
#include <array>
#include <cassert>
namespace ig=cellshard::compiler::grammar::induced;
int main(){ig::execution_observation_v1 x[4]{};for(std::uint64_t i=0;i<4;++i)x[i]={i+1,{1,i%2+1},{2,i%2+1},{3,1},{3,2},4,i+1,i*10,i*10+10,64,1};ig::induced_grammar_observation_view_v1 v{x,4,4,{4,1},{4,2},7};ig::repeated_production_candidate_v1 c{0,2,2,{5,1},{4,2},7};std::array<ig::canonical_symbol_v1,2>out{};auto r=ig::encode_canonical_production_v1(v,c,{6,1},out.data(),2);const ig::induced_identity_v1 expected{1,2};assert(r.encoded()&&r.production.encoding_version==1&&out[1].operation_identity==expected&&!ig::authorizes_execution(r.production));}
