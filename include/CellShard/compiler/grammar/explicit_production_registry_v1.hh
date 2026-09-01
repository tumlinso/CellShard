#pragma once
#include <CellShard/compiler/grammar/typed_symbol_table_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar {
enum class production_algebra_v1:std::uint32_t{ordered_composition=1,branch_alternative=2,exact_repetition=3};
struct explicit_production_v1{grammar_identity_v1 identity{},lhs_symbol{};const grammar_identity_v1*rhs_symbols=nullptr;std::uint64_t rhs_count=0,maximum_rhs_count=0,production_generation=0;production_algebra_v1 algebra=production_algebra_v1::ordered_composition;};
struct explicit_production_registry_v1{const explicit_production_v1*productions=nullptr;std::uint64_t production_count=0,production_capacity=0;grammar_identity_v1 registry_identity{},symbol_table_identity{};std::uint64_t registry_generation=0,symbol_table_generation=0;};
enum class production_registry_code_v1:std::uint32_t{valid,invalid_registry_identity,invalid_symbol_table,stale_symbol_table,empty_registry,missing_productions,capacity_overflow,invalid_production_identity,unordered_or_duplicate_production,invalid_lhs,terminal_lhs,empty_rhs,missing_rhs,rhs_capacity_overflow,invalid_rhs,invalid_algebra,missing_generation};
struct production_registry_validation_v1{production_registry_code_v1 code=production_registry_code_v1::valid;std::uint64_t production_index=0,rhs_index=0;[[nodiscard]]constexpr bool valid()const noexcept{return code==production_registry_code_v1::valid;}};
[[nodiscard]]constexpr production_registry_validation_v1 validate_explicit_production_registry_v1(explicit_production_registry_v1 registry,typed_symbol_table_v1 symbols)noexcept{
 if(!valid(registry.registry_identity))return{production_registry_code_v1::invalid_registry_identity};
 if(!validate_typed_symbol_table_v1(symbols).valid())return{production_registry_code_v1::invalid_symbol_table};
 if(!(registry.symbol_table_identity==symbols.table_identity)||registry.symbol_table_generation!=symbols.table_generation)return{production_registry_code_v1::stale_symbol_table};
 if(registry.production_count==0)return{production_registry_code_v1::empty_registry};
 if(registry.productions==nullptr)return{production_registry_code_v1::missing_productions};
 if(registry.production_count>registry.production_capacity)return{production_registry_code_v1::capacity_overflow};
 if(registry.registry_generation==0)return{production_registry_code_v1::missing_generation};
 for(std::uint64_t i=0;i<registry.production_count;++i){const auto&p=registry.productions[i];
  if(!valid(p.identity))return{production_registry_code_v1::invalid_production_identity,i};
  if(i&&!less(registry.productions[i-1].identity,p.identity))return{production_registry_code_v1::unordered_or_duplicate_production,i};
  const auto*lhs=find_symbol_v1(symbols,p.lhs_symbol);
  if(lhs==nullptr)return{production_registry_code_v1::invalid_lhs,i};
  if(lhs->symbol_kind!=grammar_symbol_kind_v1::nonterminal)return{production_registry_code_v1::terminal_lhs,i};
  if(p.rhs_count==0)return{production_registry_code_v1::empty_rhs,i};
  if(p.rhs_symbols==nullptr)return{production_registry_code_v1::missing_rhs,i};
  if(p.rhs_count>p.maximum_rhs_count)return{production_registry_code_v1::rhs_capacity_overflow,i};
  const auto algebra=static_cast<std::uint32_t>(p.algebra);
  if(algebra<1||algebra>3)return{production_registry_code_v1::invalid_algebra,i};
  if(p.production_generation==0)return{production_registry_code_v1::missing_generation,i};
  for(std::uint64_t j=0;j<p.rhs_count;++j)if(find_symbol_v1(symbols,p.rhs_symbols[j])==nullptr)return{production_registry_code_v1::invalid_rhs,i,j};
 }
 return{production_registry_code_v1::valid,registry.production_count};
}
[[nodiscard]]constexpr bool authorizes_execution(explicit_production_registry_v1)noexcept{return false;}
}
