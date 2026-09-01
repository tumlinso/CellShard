#pragma once
#include <CellShard/compiler/graph/graph_family.hh>
#include <cstdint>
#include <type_traits>
namespace cellshard::compiler::graph {
enum class graph_rewrite_kind : std::uint32_t { fuse=1, eliminate=2, reorder=3, substitute_projection=4, materialize=5 };
enum graph_rewrite_proof : std::uint64_t { proof_exact_values=UINT64_C(1)<<0,proof_domains=UINT64_C(1)<<1,proof_orders=UINT64_C(1)<<2,proof_effects=UINT64_C(1)<<3,proof_dependencies=UINT64_C(1)<<4 };
struct graph_rewrite_descriptor {
    graph_rewrite_kind kind{}; std::uint32_t reserved=0;
    content_digest before_digest{}; content_digest after_digest{};
    std::uint64_t source_node_offset=0; std::uint64_t source_node_count=0;
    std::uint64_t result_node_offset=0; std::uint64_t result_node_count=0;
    std::uint64_t proof_flags=0; std::uint64_t cost_evidence_identity=0;
};
[[nodiscard]] constexpr bool valid_graph_rewrite_descriptor(const graph_rewrite_descriptor&r)noexcept{
    constexpr auto required=proof_exact_values|proof_domains|proof_orders|proof_effects|proof_dependencies;
    const bool kind=r.kind==graph_rewrite_kind::fuse||r.kind==graph_rewrite_kind::eliminate||r.kind==graph_rewrite_kind::reorder||r.kind==graph_rewrite_kind::substitute_projection||r.kind==graph_rewrite_kind::materialize;
    return kind&&valid_content_digest(r.before_digest)&&valid_content_digest(r.after_digest)&&r.before_digest.algorithm!=digest_algorithm::none&&r.after_digest.algorithm!=digest_algorithm::none&&r.before_digest!=r.after_digest&&r.source_node_count!=0&&r.result_node_count!=0&&(r.proof_flags&required)==required&&(r.proof_flags&~required)==0&&r.cost_evidence_identity!=0;
}
static_assert(std::is_trivially_copyable<graph_rewrite_descriptor>::value);
}
