#include <CellShard/compiler/schedule/replay_mode.hh>
#include <cassert>
using namespace cellshard::compiler::schedule;
int main(){replay_context c{true,true,true,true,true,true};assert(select_replay_mode(c)==replay_mode::exact);c.all_bindings_available=false;assert(select_replay_mode(c)==replay_mode::relink);c.target_matches=false;assert(select_replay_mode(c)==replay_mode::retarget);c.provider_sources_match=false;assert(select_replay_mode(c)==replay_mode::recompile);c.provider_sources_match=true;c.logical_graph_matches=false;assert(select_replay_mode(c)==replay_mode::recompile);}
