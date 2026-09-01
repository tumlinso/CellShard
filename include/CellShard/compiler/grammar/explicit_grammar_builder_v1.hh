#pragma once
#include <CellShard/compiler/grammar/explicit_production_registry_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar {
struct explicit_grammar_v1{typed_symbol_table_v1 symbols{};explicit_production_registry_v1 productions{};grammar_identity_v1 grammar_identity{};std::uint64_t grammar_generation=0;};
struct explicit_grammar_buffers_v1{typed_grammar_symbol_v1*symbols=nullptr;std::uint64_t symbol_capacity=0;explicit_production_v1*productions=nullptr;std::uint64_t production_capacity=0;grammar_identity_v1*rhs_symbols=nullptr;std::uint64_t rhs_capacity=0;};
enum class explicit_grammar_build_code_v1:std::uint32_t{built,invalid_identity,missing_generation,invalid_symbols,invalid_productions,rhs_count_overflow,missing_output,insufficient_output};
struct explicit_grammar_build_result_v1{explicit_grammar_build_code_v1 code=explicit_grammar_build_code_v1::built;explicit_grammar_v1 grammar{};std::uint64_t required_rhs=0;[[nodiscard]]constexpr bool built()const noexcept{return code==explicit_grammar_build_code_v1::built;}};
[[nodiscard]]inline explicit_grammar_build_result_v1 build_explicit_grammar_v1(grammar_identity_v1 grammar_identity,std::uint64_t grammar_generation,typed_symbol_table_v1 symbols,explicit_production_registry_v1 productions,explicit_grammar_buffers_v1 out)noexcept{
 if(!valid(grammar_identity))return{explicit_grammar_build_code_v1::invalid_identity};
 if(grammar_generation==0)return{explicit_grammar_build_code_v1::missing_generation};
 if(!validate_typed_symbol_table_v1(symbols).valid())return{explicit_grammar_build_code_v1::invalid_symbols};
 if(!validate_explicit_production_registry_v1(productions,symbols).valid())return{explicit_grammar_build_code_v1::invalid_productions};
 std::uint64_t required_rhs=0;for(std::uint64_t i=0;i<productions.production_count;++i){if(required_rhs>UINT64_MAX-productions.productions[i].rhs_count)return{explicit_grammar_build_code_v1::rhs_count_overflow};required_rhs+=productions.productions[i].rhs_count;}
 if(out.symbols==nullptr||out.productions==nullptr||out.rhs_symbols==nullptr)return{explicit_grammar_build_code_v1::missing_output,{},required_rhs};
 if(out.symbol_capacity<symbols.symbol_count||out.production_capacity<productions.production_count||out.rhs_capacity<required_rhs)return{explicit_grammar_build_code_v1::insufficient_output,{},required_rhs};
 for(std::uint64_t i=0;i<symbols.symbol_count;++i)out.symbols[i]=symbols.symbols[i];
 std::uint64_t cursor=0;
 for(std::uint64_t i=0;i<productions.production_count;++i){out.productions[i]=productions.productions[i];out.productions[i].rhs_symbols=out.rhs_symbols+cursor;for(std::uint64_t j=0;j<productions.productions[i].rhs_count;++j)out.rhs_symbols[cursor++]=productions.productions[i].rhs_symbols[j];}
 typed_symbol_table_v1 symbol_view{out.symbols,symbols.symbol_count,out.symbol_capacity,symbols.table_identity,symbols.table_generation};explicit_production_registry_v1 production_view{out.productions,productions.production_count,out.production_capacity,productions.registry_identity,productions.symbol_table_identity,productions.registry_generation,productions.symbol_table_generation};return{explicit_grammar_build_code_v1::built,{symbol_view,production_view,grammar_identity,grammar_generation},required_rhs};
}
[[nodiscard]]constexpr bool authorizes_execution(explicit_grammar_v1)noexcept{return false;}
}
