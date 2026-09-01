#pragma once
#include <CellShard/compiler/grammar/derivation_dag_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar {
struct simplified_production_view_v1{const explicit_production_v1*productions=nullptr;std::uint64_t production_count=0;grammar_identity_v1 source_grammar_identity{},root_symbol_identity{};std::uint64_t source_grammar_generation=0;};
struct grammar_simplification_buffers_v1{explicit_production_v1*productions=nullptr;std::uint64_t production_capacity=0;grammar_identity_v1*rhs_symbols=nullptr;std::uint64_t rhs_capacity=0;std::uint8_t*live_symbols=nullptr;std::uint64_t live_capacity=0;};
enum class grammar_simplification_code_v1:std::uint32_t{simplified,invalid_grammar,invalid_root,missing_output,insufficient_output,rhs_overflow};
struct grammar_simplification_result_v1{grammar_simplification_code_v1 code=grammar_simplification_code_v1::simplified;simplified_production_view_v1 view{};std::uint64_t required_productions=0,required_rhs=0;[[nodiscard]]constexpr bool simplified()const noexcept{return code==grammar_simplification_code_v1::simplified;}};
[[nodiscard]]constexpr bool same_production_body_v1(const explicit_production_v1&a,const explicit_production_v1&b)noexcept{if(!(a.lhs_symbol==b.lhs_symbol)||a.algebra!=b.algebra||a.rhs_count!=b.rhs_count)return false;for(std::uint64_t i=0;i<a.rhs_count;++i)if(!(a.rhs_symbols[i]==b.rhs_symbols[i]))return false;return true;}
[[nodiscard]]inline grammar_simplification_result_v1 simplify_explicit_grammar_v1(explicit_grammar_v1 g,grammar_identity_v1 root,grammar_simplification_buffers_v1 out)noexcept{
 if(!valid(g.grammar_identity)||!validate_typed_symbol_table_v1(g.symbols).valid()||!validate_explicit_production_registry_v1(g.productions,g.symbols).valid())return{grammar_simplification_code_v1::invalid_grammar};
 const auto root_index=symbol_index_v1(g.symbols,root);
 if(root_index==g.symbols.symbol_count)return{grammar_simplification_code_v1::invalid_root};
 if(out.live_symbols==nullptr||out.productions==nullptr||out.rhs_symbols==nullptr)return{grammar_simplification_code_v1::missing_output};
 if(out.live_capacity<g.symbols.symbol_count)return{grammar_simplification_code_v1::insufficient_output};
 for(std::uint64_t i=0;i<g.symbols.symbol_count;++i)out.live_symbols[i]=0;
 out.live_symbols[root_index]=1;
 bool changed=true;while(changed){changed=false;for(std::uint64_t i=0;i<g.productions.production_count;++i){const auto&p=g.productions.productions[i];if(!out.live_symbols[symbol_index_v1(g.symbols,p.lhs_symbol)])continue;for(std::uint64_t j=0;j<p.rhs_count;++j){const auto index=symbol_index_v1(g.symbols,p.rhs_symbols[j]);if(!out.live_symbols[index]){out.live_symbols[index]=1;changed=true;}}}}
 std::uint64_t required_p=0,required_rhs=0;for(std::uint64_t i=0;i<g.productions.production_count;++i){const auto&p=g.productions.productions[i];if(!out.live_symbols[symbol_index_v1(g.symbols,p.lhs_symbol)])continue;bool duplicate=false;for(std::uint64_t j=0;j<i;++j)if(out.live_symbols[symbol_index_v1(g.symbols,g.productions.productions[j].lhs_symbol)]&&same_production_body_v1(g.productions.productions[j],p)){duplicate=true;break;}if(duplicate)continue;if(required_rhs>UINT64_MAX-p.rhs_count)return{grammar_simplification_code_v1::rhs_overflow};required_rhs+=p.rhs_count;++required_p;}
 if(out.production_capacity<required_p||out.rhs_capacity<required_rhs)return{grammar_simplification_code_v1::insufficient_output,{},required_p,required_rhs};
 std::uint64_t pc=0,rc=0;
 for(std::uint64_t i=0;i<g.productions.production_count;++i){const auto&p=g.productions.productions[i];if(!out.live_symbols[symbol_index_v1(g.symbols,p.lhs_symbol)])continue;bool duplicate=false;for(std::uint64_t j=0;j<i;++j)if(out.live_symbols[symbol_index_v1(g.symbols,g.productions.productions[j].lhs_symbol)]&&same_production_body_v1(g.productions.productions[j],p)){duplicate=true;break;}if(duplicate)continue;out.productions[pc]=p;out.productions[pc].rhs_symbols=out.rhs_symbols+rc;for(std::uint64_t j=0;j<p.rhs_count;++j)out.rhs_symbols[rc++]=p.rhs_symbols[j];++pc;}
 return{grammar_simplification_code_v1::simplified,{out.productions,pc,g.grammar_identity,root,g.grammar_generation},required_p,required_rhs};
}
[[nodiscard]]constexpr bool authorizes_execution(simplified_production_view_v1)noexcept{return false;}
}
