#pragma once
#include <CellShard/compiler/grammar/explicit_grammar_builder_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar {
struct derivation_edge_v1{std::uint64_t prerequisite_symbol_index=0,derived_symbol_index=0;};
struct derivation_dag_v1{const derivation_edge_v1*edges=nullptr;const std::uint64_t*topological_symbol_indices=nullptr;std::uint64_t symbol_count=0,edge_count=0;grammar_identity_v1 grammar_identity{};std::uint64_t grammar_generation=0;};
struct derivation_dag_buffers_v1{derivation_edge_v1*edges=nullptr;std::uint64_t edge_capacity=0;std::uint64_t*topological_indices=nullptr,*indegrees=nullptr;std::uint64_t symbol_capacity=0;};
enum class derivation_dag_code_v1:std::uint32_t{built,invalid_grammar,edge_overflow,missing_output,insufficient_output,cycle};
struct derivation_dag_result_v1{derivation_dag_code_v1 code=derivation_dag_code_v1::built;derivation_dag_v1 dag{};std::uint64_t required_edges=0;[[nodiscard]]constexpr bool built()const noexcept{return code==derivation_dag_code_v1::built;}};
[[nodiscard]]inline std::uint64_t symbol_index_v1(typed_symbol_table_v1 t,grammar_identity_v1 id)noexcept{const auto*p=find_symbol_v1(t,id);return p==nullptr?t.symbol_count:static_cast<std::uint64_t>(p-t.symbols);}
[[nodiscard]]inline derivation_dag_result_v1 build_derivation_dag_v1(explicit_grammar_v1 g,derivation_dag_buffers_v1 out)noexcept{
 if(!valid(g.grammar_identity)||g.grammar_generation==0||!validate_typed_symbol_table_v1(g.symbols).valid()||!validate_explicit_production_registry_v1(g.productions,g.symbols).valid())return{derivation_dag_code_v1::invalid_grammar};
 std::uint64_t required=0;for(std::uint64_t i=0;i<g.productions.production_count;++i)for(std::uint64_t j=0;j<g.productions.productions[i].rhs_count;++j){const auto*x=find_symbol_v1(g.symbols,g.productions.productions[i].rhs_symbols[j]);if(x->symbol_kind==grammar_symbol_kind_v1::nonterminal){if(required==UINT64_MAX)return{derivation_dag_code_v1::edge_overflow};++required;}}
 if(out.edges==nullptr||out.topological_indices==nullptr||out.indegrees==nullptr)return{derivation_dag_code_v1::missing_output,{},required};
 if(out.edge_capacity<required||out.symbol_capacity<g.symbols.symbol_count)return{derivation_dag_code_v1::insufficient_output,{},required};
 for(std::uint64_t i=0;i<g.symbols.symbol_count;++i)out.indegrees[i]=0;
 std::uint64_t cursor=0;for(std::uint64_t i=0;i<g.productions.production_count;++i){const auto&p=g.productions.productions[i];const auto lhs=symbol_index_v1(g.symbols,p.lhs_symbol);for(std::uint64_t j=0;j<p.rhs_count;++j){const auto rhs=symbol_index_v1(g.symbols,p.rhs_symbols[j]);if(g.symbols.symbols[rhs].symbol_kind==grammar_symbol_kind_v1::nonterminal){out.edges[cursor++]={rhs,lhs};++out.indegrees[lhs];}}}
 std::uint64_t emitted=0;while(emitted<g.symbols.symbol_count){std::uint64_t node=g.symbols.symbol_count;for(std::uint64_t i=0;i<g.symbols.symbol_count;++i)if(out.indegrees[i]==0){node=i;break;}if(node==g.symbols.symbol_count)return{derivation_dag_code_v1::cycle,{},required};out.indegrees[node]=UINT64_MAX;out.topological_indices[emitted++]=node;for(std::uint64_t i=0;i<cursor;++i)if(out.edges[i].prerequisite_symbol_index==node)--out.indegrees[out.edges[i].derived_symbol_index];}
 return{derivation_dag_code_v1::built,{out.edges,out.topological_indices,g.symbols.symbol_count,cursor,g.grammar_identity,g.grammar_generation},required};
}
[[nodiscard]]constexpr bool authorizes_execution(derivation_dag_v1)noexcept{return false;}
}
