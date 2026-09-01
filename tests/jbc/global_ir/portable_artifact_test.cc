#include <CellShard/compiler/schedule/portable_artifact.hh>
#include <cassert>
using namespace cellshard::compiler::schedule;
int main(){portable_schedule_header h{};h.id=portable_schedule_id{1};h.realization=cellshard::compiler::graph::physical_realization_id{2};h.graph_digest.algorithm=cellshard::digest_algorithm::legacy_fnv1a64;h.graph_digest.used_bytes=8;h.graph_digest.bytes[0]=std::byte{1};h.command_count=1;h.binding_count=2;h.transient_bytes=64;assert(valid_portable_schedule_header(h));portable_schedule_command c{};c.node=cellshard::compiler::graph::operation_node_id{1};c.kind=portable_command_kind::launch;c.binding_count=2;c.transient_bytes=64;assert(valid_portable_schedule_command(c,0,2,64));c.transient_bytes=65;assert(!valid_portable_schedule_command(c,0,2,64));}
