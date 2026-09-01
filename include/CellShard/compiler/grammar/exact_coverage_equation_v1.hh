#pragma once
#include <CellShard/compiler/grammar/typed_symbol_table_v1.hh>
#include <cstdint>
#include <limits>

namespace cellshard::compiler::grammar {
struct coverage_term_v1{grammar_identity_v1 contribution_identity{},symbol_identity{};std::int64_t numerator=0;std::uint64_t denominator=0,structure_generation=0,value_generation=0;};
struct exact_coverage_equation_v1{const coverage_term_v1*terms=nullptr;std::uint64_t term_count=0,term_capacity=0;std::int64_t target_numerator=0;std::uint64_t denominator=0;grammar_identity_v1 equation_identity{},domain_identity{},order_identity{};std::uint64_t equation_generation=0;};
enum class coverage_equation_code_v1:std::uint32_t{exact,not_exact,invalid_identity,missing_generation,empty_terms,missing_terms,capacity_overflow,invalid_denominator,invalid_term,unordered_or_duplicate_contribution,arithmetic_overflow};
struct coverage_equation_result_v1{coverage_equation_code_v1 code=coverage_equation_code_v1::exact;std::int64_t evaluated_numerator=0;std::uint64_t index=0;[[nodiscard]]constexpr bool exact()const noexcept{return code==coverage_equation_code_v1::exact;}};
[[nodiscard]]constexpr coverage_equation_result_v1 evaluate_exact_coverage_equation_v1(exact_coverage_equation_v1 e)noexcept{
 if(!valid(e.equation_identity)||!valid(e.domain_identity)||!valid(e.order_identity))return{coverage_equation_code_v1::invalid_identity};
 if(e.equation_generation==0)return{coverage_equation_code_v1::missing_generation};
 if(e.term_count==0)return{coverage_equation_code_v1::empty_terms};
 if(e.terms==nullptr)return{coverage_equation_code_v1::missing_terms};
 if(e.term_count>e.term_capacity)return{coverage_equation_code_v1::capacity_overflow};
 if(e.denominator==0)return{coverage_equation_code_v1::invalid_denominator};
 std::int64_t sum=0;for(std::uint64_t i=0;i<e.term_count;++i){const auto&t=e.terms[i];if(!valid(t.contribution_identity)||!valid(t.symbol_identity)||t.numerator==0||t.denominator!=e.denominator||t.structure_generation==0)return{coverage_equation_code_v1::invalid_term,sum,i};if(i&&!less(e.terms[i-1].contribution_identity,t.contribution_identity))return{coverage_equation_code_v1::unordered_or_duplicate_contribution,sum,i};if((t.numerator>0&&sum>std::numeric_limits<std::int64_t>::max()-t.numerator)||(t.numerator<0&&sum<std::numeric_limits<std::int64_t>::min()-t.numerator))return{coverage_equation_code_v1::arithmetic_overflow,sum,i};sum+=t.numerator;}
 return{sum==e.target_numerator?coverage_equation_code_v1::exact:coverage_equation_code_v1::not_exact,sum,e.term_count};
}
[[nodiscard]]constexpr bool authorizes_execution(coverage_equation_result_v1 r)noexcept{return r.exact();}
}
