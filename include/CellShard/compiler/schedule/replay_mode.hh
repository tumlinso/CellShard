#pragma once
#include <CellShard/compiler/schedule/portable_artifact.hh>
namespace cellshard::compiler::schedule {
enum class replay_mode : std::uint32_t { exact=1, relink=2, retarget=3, recompile=4 };
struct replay_context {
    bool logical_graph_matches=false; bool target_matches=false;
    bool provider_sources_match=false; bool all_bindings_available=false;
    bool compatible_providers_available=false; bool compatible_target_available=false;
};
[[nodiscard]] constexpr replay_mode select_replay_mode(const replay_context&c)noexcept{
    if(c.logical_graph_matches&&c.target_matches&&c.provider_sources_match&&c.all_bindings_available)return replay_mode::exact;
    if(c.logical_graph_matches&&c.target_matches&&c.provider_sources_match&&c.compatible_providers_available)return replay_mode::relink;
    if(c.logical_graph_matches&&c.provider_sources_match&&c.compatible_providers_available&&c.compatible_target_available)return replay_mode::retarget;
    return replay_mode::recompile;
}
}
