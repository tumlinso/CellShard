#include <CellShard/compiler/graph/global_ir_serialization.hh>
#include <cstring>
namespace cellshard::compiler::graph {
namespace {std::uint64_t mix(std::uint64_t h,std::uint64_t v){for(unsigned i=0;i<8;++i){h^=(v>>(i*8))&0xffu;h*=UINT64_C(1099511628211);}return h;}std::uint64_t hash_bytes(const std::byte*p,std::size_t n){std::uint64_t h=UINT64_C(1469598103934665603);for(std::size_t i=0;i<n;++i){h^=static_cast<std::uint64_t>(p[i]);h*=UINT64_C(1099511628211);}return h;}}
std::size_t global_ir_serialized_bytes(std::size_t n,std::size_t p,std::size_t e)noexcept{std::size_t total=sizeof(global_ir_header);if(n>SIZE_MAX/sizeof(operation_node_descriptor)||p>SIZE_MAX/sizeof(typed_port_descriptor)||e>SIZE_MAX/sizeof(atom_dependency_edge))return 0;for(auto bytes:{n*sizeof(operation_node_descriptor),p*sizeof(typed_port_descriptor),e*sizeof(atom_dependency_edge)}){if(total>SIZE_MAX-bytes)return 0;total+=bytes;}return total;}
global_ir_serialize_status serialize_global_ir(graph_family_id family,const operation_node_descriptor*nodes,std::size_t n,const typed_port_descriptor*ports,std::size_t p,const atom_dependency_edge*edges,std::size_t e,std::byte*out,std::size_t cap)noexcept{
    if(!family.valid()||nodes==nullptr||ports==nullptr||edges==nullptr||n==0||p==0||e==0||out==nullptr)return global_ir_serialize_status::invalid_input;
    const auto total=global_ir_serialized_bytes(n,p,e);if(total==0)return global_ir_serialize_status::overflow;if(cap<total)return global_ir_serialize_status::insufficient_output;
    for(std::size_t i=0;i<n;++i)if(!valid_operation_node_descriptor(nodes[i]))return global_ir_serialize_status::invalid_input;
    for(std::size_t i=0;i<p;++i)if(!valid_typed_port_descriptor(ports[i]))return global_ir_serialize_status::invalid_input;
    for(std::size_t i=0;i<e;++i)if(!valid_atom_dependency_edge(edges[i]))return global_ir_serialize_status::invalid_input;
    global_ir_header h{};h.family=family;h.node_count=n;h.port_count=p;h.edge_count=e;h.node_offset=sizeof(h);h.port_offset=h.node_offset+n*sizeof(*nodes);h.edge_offset=h.port_offset+p*sizeof(*ports);h.total_bytes=total;std::memset(out,0,total);std::memcpy(out,&h,sizeof(h));std::memcpy(out+h.node_offset,nodes,n*sizeof(*nodes));std::memcpy(out+h.port_offset,ports,p*sizeof(*ports));std::memcpy(out+h.edge_offset,edges,e*sizeof(*edges));const auto digest=hash_bytes(out,total);h.content.algorithm=digest_algorithm::legacy_fnv1a64;h.content.used_bytes=8;for(unsigned i=0;i<8;++i)h.content.bytes[i]=static_cast<std::byte>((digest>>(i*8))&0xffu);std::memcpy(out,&h,sizeof(h));return global_ir_serialize_status::success;
}
profiler_event_identity emit_profiler_identity(graph_family_id family,schedule::portable_schedule_id schedule,operation_node_id node,std::uint64_t ordinal)noexcept{if(!family.valid()||!schedule.valid()||!node.valid())return{};auto low=mix(UINT64_C(1469598103934665603),schedule.value());low=mix(low,node.value());low=mix(low,ordinal);return{family.value(),low};}
}
