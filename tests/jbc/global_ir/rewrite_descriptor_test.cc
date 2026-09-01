#include <CellShard/compiler/graph/rewrite_descriptor.hh>
#include <cassert>
using namespace cellshard::compiler::graph;
cellshard::content_digest digest(std::byte b){cellshard::content_digest d{};d.algorithm=cellshard::digest_algorithm::legacy_fnv1a64;d.used_bytes=8;d.bytes[0]=b;return d;}
int main(){graph_rewrite_descriptor r{};r.kind=graph_rewrite_kind::fuse;r.before_digest=digest(std::byte{1});r.after_digest=digest(std::byte{2});r.source_node_count=2;r.result_node_count=1;r.proof_flags=proof_exact_values|proof_domains|proof_orders|proof_effects|proof_dependencies;r.cost_evidence_identity=3;assert(valid_graph_rewrite_descriptor(r));r.proof_flags&=~proof_orders;assert(!valid_graph_rewrite_descriptor(r));}
