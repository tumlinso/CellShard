#pragma once
#include <CellShard/compiler/grammar/explicit_production_registry_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar {
enum class production_parameter_kind_v1:std::uint32_t{unsigned_integer=1,signed_integer=2,rational=3,identity=4,enumeration=5};
struct production_parameter_spec_v1{grammar_identity_v1 parameter_identity{};production_parameter_kind_v1 kind=production_parameter_kind_v1::unsigned_integer;std::int64_t minimum_numerator=0,maximum_numerator=0;std::uint64_t denominator=1;bool required=true;std::uint8_t reserved[7]{};};
struct production_parameter_schema_v1{grammar_identity_v1 schema_identity{},production_identity{};const production_parameter_spec_v1*parameters=nullptr;std::uint64_t parameter_count=0,parameter_capacity=0,schema_generation=0,production_generation=0;};
enum class parameter_schema_code_v1:std::uint32_t{valid,invalid_identity,missing_generation,empty_parameters,missing_parameters,capacity_overflow,invalid_parameter_identity,unordered_or_duplicate_parameter,invalid_kind,invalid_range,nonzero_reserved,unknown_production,stale_production};
struct parameter_schema_validation_v1{parameter_schema_code_v1 code=parameter_schema_code_v1::valid;std::uint64_t index=0;[[nodiscard]]constexpr bool valid()const noexcept{return code==parameter_schema_code_v1::valid;}};
[[nodiscard]]constexpr const explicit_production_v1*find_production_v1(explicit_production_registry_v1 r,grammar_identity_v1 id)noexcept{std::uint64_t first=0,last=r.production_count;while(first<last){const auto middle=first+(last-first)/2;if(less(r.productions[middle].identity,id))first=middle+1;else last=middle;}return first<r.production_count&&r.productions[first].identity==id?&r.productions[first]:nullptr;}
[[nodiscard]]constexpr parameter_schema_validation_v1 validate_production_parameter_schema_v1(production_parameter_schema_v1 s,explicit_production_registry_v1 registry)noexcept{
 if(!valid(s.schema_identity)||!valid(s.production_identity))return{parameter_schema_code_v1::invalid_identity};
 if(s.schema_generation==0||s.production_generation==0)return{parameter_schema_code_v1::missing_generation};
 const auto*p=find_production_v1(registry,s.production_identity);
 if(p==nullptr)return{parameter_schema_code_v1::unknown_production};
 if(p->production_generation!=s.production_generation)return{parameter_schema_code_v1::stale_production};
 if(s.parameter_count==0)return{parameter_schema_code_v1::empty_parameters};
 if(s.parameters==nullptr)return{parameter_schema_code_v1::missing_parameters};
 if(s.parameter_count>s.parameter_capacity)return{parameter_schema_code_v1::capacity_overflow};
 for(std::uint64_t i=0;i<s.parameter_count;++i){const auto&x=s.parameters[i];if(!valid(x.parameter_identity))return{parameter_schema_code_v1::invalid_parameter_identity,i};if(i&&!less(s.parameters[i-1].parameter_identity,x.parameter_identity))return{parameter_schema_code_v1::unordered_or_duplicate_parameter,i};const auto kind=static_cast<std::uint32_t>(x.kind);if(kind<1||kind>5)return{parameter_schema_code_v1::invalid_kind,i};if(x.denominator==0||x.minimum_numerator>x.maximum_numerator)return{parameter_schema_code_v1::invalid_range,i};for(auto byte:x.reserved)if(byte!=0)return{parameter_schema_code_v1::nonzero_reserved,i};}
 return{parameter_schema_code_v1::valid,s.parameter_count};
}
[[nodiscard]]constexpr bool authorizes_execution(production_parameter_schema_v1)noexcept{return false;}
}
