#pragma once
#include <CellShard/compiler/graph/atom_dependency.hh>
#include <cstdint>
#include <type_traits>
namespace cellshard::compiler::graph {
struct graph_family_id_tag;
using graph_family_id=strong_id<graph_family_id_tag>;
struct graph_family_descriptor {
    graph_family_id id{}; content_digest logical_graph_digest{};
    std::uint64_t provider_count=0; std::uint64_t node_class_count=0;
    std::uint64_t structure_epoch=0;
};
struct workload_distribution {
    graph_family_id family{}; std::uint64_t sample_count=0;
    std::uint64_t min_nodes=0; std::uint64_t median_nodes=0; std::uint64_t max_nodes=0;
    std::uint64_t min_edges=0; std::uint64_t median_edges=0; std::uint64_t max_edges=0;
    std::uint64_t median_atom_reuse_q16=0;
};
[[nodiscard]] constexpr bool valid_graph_family_descriptor(const graph_family_descriptor&d)noexcept{return d.id.valid()&&valid_content_digest(d.logical_graph_digest)&&d.logical_graph_digest.algorithm!=digest_algorithm::none&&d.provider_count!=0&&d.node_class_count!=0&&d.structure_epoch!=0;}
[[nodiscard]] constexpr bool valid_workload_distribution(const workload_distribution&w)noexcept{return w.family.valid()&&w.sample_count!=0&&w.min_nodes!=0&&w.min_nodes<=w.median_nodes&&w.median_nodes<=w.max_nodes&&w.min_edges<=w.median_edges&&w.median_edges<=w.max_edges;}
static_assert(std::is_trivially_copyable<graph_family_descriptor>::value);
static_assert(std::is_trivially_copyable<workload_distribution>::value);
}
