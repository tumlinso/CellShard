#pragma once
#include <CellShard/compiler/grammar/derivation_validation_receipt_v1.hh>
#include <CellShard/compiler/grammar/exact_coverage_equation_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar {
struct flat_basis_contract_v1{const grammar_identity_v1*terminal_symbols=nullptr;std::uint64_t terminal_count=0,terminal_capacity=0;grammar_identity_v1 basis_identity{},domain_identity{},order_identity{};std::uint64_t structure_generation=0,value_generation=0;};
struct complete_strategy_cost_v1{std::uint64_t preparation_ns=0,execution_ns=0,canonicalization_ns=0,synchronization_ns=0;};
enum class grammar_strategy_v1:std::uint32_t{flat_basis=1,explicit_grammar=2};
enum class flat_basis_selection_code_v1:std::uint32_t{selected,invalid_basis,invalid_cost,cost_overflow};
struct flat_basis_selection_result_v1{flat_basis_selection_code_v1 code=flat_basis_selection_code_v1::selected;grammar_strategy_v1 strategy=grammar_strategy_v1::flat_basis;std::uint64_t selected_complete_ns=0,flat_complete_ns=0,grammar_complete_ns=0;[[nodiscard]]constexpr bool selected()const noexcept{return code==flat_basis_selection_code_v1::selected;}};
[[nodiscard]]constexpr bool validate_flat_basis_v1(flat_basis_contract_v1 b,typed_symbol_table_v1 symbols)noexcept{if(b.terminal_count==0||b.terminal_symbols==nullptr||b.terminal_count>b.terminal_capacity||!valid(b.basis_identity)||!valid(b.domain_identity)||!valid(b.order_identity)||b.structure_generation==0)return false;for(std::uint64_t i=0;i<b.terminal_count;++i){const auto*x=find_symbol_v1(symbols,b.terminal_symbols[i]);if(x==nullptr||x->symbol_kind!=grammar_symbol_kind_v1::terminal_atom||!(x->domain_identity==b.domain_identity)||!(x->order_identity==b.order_identity)||x->structure_generation!=b.structure_generation)return false;if(i&&!less(b.terminal_symbols[i-1],b.terminal_symbols[i]))return false;}return true;}
[[nodiscard]]constexpr bool sum_complete_cost_v1(complete_strategy_cost_v1 x,std::uint64_t&total)noexcept{total=x.preparation_ns;if(total>UINT64_MAX-x.execution_ns)return false;total+=x.execution_ns;if(total>UINT64_MAX-x.canonicalization_ns)return false;total+=x.canonicalization_ns;if(total>UINT64_MAX-x.synchronization_ns)return false;total+=x.synchronization_ns;return total!=0;}
[[nodiscard]]constexpr flat_basis_selection_result_v1 select_with_flat_basis_fallback_v1(flat_basis_contract_v1 basis,typed_symbol_table_v1 symbols,coverage_equation_result_v1 coverage,derivation_validation_result_v1 derivation,complete_strategy_cost_v1 flat_cost,complete_strategy_cost_v1 grammar_cost)noexcept{
 if(!validate_flat_basis_v1(basis,symbols))return{flat_basis_selection_code_v1::invalid_basis};
 std::uint64_t flat=0,grammar=0;
 if(!sum_complete_cost_v1(flat_cost,flat)||!sum_complete_cost_v1(grammar_cost,grammar))return{flat_basis_selection_code_v1::cost_overflow};
 const bool grammar_correct=coverage.exact()&&derivation.validated();
 const auto strategy=grammar_correct&&grammar<flat?grammar_strategy_v1::explicit_grammar:grammar_strategy_v1::flat_basis;
 return{flat_basis_selection_code_v1::selected,strategy,strategy==grammar_strategy_v1::explicit_grammar?grammar:flat,flat,grammar};
}
[[nodiscard]]constexpr bool authorizes_execution(flat_basis_selection_result_v1 r)noexcept{return r.selected();}
}
