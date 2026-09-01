#pragma once
#include <CellShard/compiler/grammar/derivation_dag_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar {
struct derivation_validation_receipt_v1{grammar_identity_v1 receipt_identity{},grammar_identity{},validator_identity{};std::uint64_t grammar_generation=0,validator_generation=0,symbol_count=0,production_count=0,edge_count=0;};
enum class derivation_validation_code_v1:std::uint32_t{validated,invalid_identity,invalid_generation,grammar_mismatch,invalid_dag,missing_scratch,insufficient_scratch,invalid_topological_index,duplicate_topological_index,invalid_edge,order_violation};
struct derivation_validation_result_v1{derivation_validation_code_v1 code=derivation_validation_code_v1::validated;derivation_validation_receipt_v1 receipt{};std::uint64_t index=0;[[nodiscard]]constexpr bool validated()const noexcept{return code==derivation_validation_code_v1::validated;}};
[[nodiscard]]inline derivation_validation_result_v1 validate_derivation_independently_v1(explicit_grammar_v1 grammar,derivation_dag_v1 dag,grammar_identity_v1 receipt_identity,grammar_identity_v1 validator_identity,std::uint64_t validator_generation,std::uint64_t*positions,std::uint64_t position_capacity)noexcept{
 if(!valid(receipt_identity)||!valid(validator_identity))return{derivation_validation_code_v1::invalid_identity};
 if(validator_generation==0)return{derivation_validation_code_v1::invalid_generation};
 if(!(dag.grammar_identity==grammar.grammar_identity)||dag.grammar_generation!=grammar.grammar_generation)return{derivation_validation_code_v1::grammar_mismatch};
 if(dag.symbol_count!=grammar.symbols.symbol_count||dag.topological_symbol_indices==nullptr||(dag.edge_count&&dag.edges==nullptr))return{derivation_validation_code_v1::invalid_dag};
 if(positions==nullptr)return{derivation_validation_code_v1::missing_scratch};
 if(position_capacity<dag.symbol_count)return{derivation_validation_code_v1::insufficient_scratch};
 for(std::uint64_t i=0;i<dag.symbol_count;++i)positions[i]=UINT64_MAX;
 for(std::uint64_t i=0;i<dag.symbol_count;++i){const auto node=dag.topological_symbol_indices[i];if(node>=dag.symbol_count)return{derivation_validation_code_v1::invalid_topological_index,{},i};if(positions[node]!=UINT64_MAX)return{derivation_validation_code_v1::duplicate_topological_index,{},i};positions[node]=i;}
 for(std::uint64_t i=0;i<dag.edge_count;++i){const auto&e=dag.edges[i];if(e.prerequisite_symbol_index>=dag.symbol_count||e.derived_symbol_index>=dag.symbol_count)return{derivation_validation_code_v1::invalid_edge,{},i};if(positions[e.prerequisite_symbol_index]>=positions[e.derived_symbol_index])return{derivation_validation_code_v1::order_violation,{},i};}
 return{derivation_validation_code_v1::validated,{receipt_identity,grammar.grammar_identity,validator_identity,grammar.grammar_generation,validator_generation,grammar.symbols.symbol_count,grammar.productions.production_count,dag.edge_count},dag.edge_count};
}
[[nodiscard]]constexpr bool authorizes_execution(derivation_validation_receipt_v1)noexcept{return false;}
}
