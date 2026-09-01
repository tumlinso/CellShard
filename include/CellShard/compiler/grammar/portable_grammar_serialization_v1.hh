#pragma once
#include <CellShard/compiler/grammar/explicit_grammar_builder_v1.hh>
#include <cstdint>

namespace cellshard::compiler::grammar {
inline constexpr std::uint64_t portable_grammar_header_bytes_v1=112,symbol_record_bytes_v1=104,production_record_bytes_v1=72,identity_record_bytes_v1=16;
enum class grammar_serialization_code_v1:std::uint32_t{written,invalid_grammar,size_overflow,missing_output,insufficient_output};
struct grammar_serialization_result_v1{grammar_serialization_code_v1 code=grammar_serialization_code_v1::written;std::uint64_t bytes=0;[[nodiscard]]constexpr bool written()const noexcept{return code==grammar_serialization_code_v1::written;}};
inline void put_u32_le_v1(std::uint8_t*&p,std::uint32_t x)noexcept{for(unsigned i=0;i<4;++i)*p++=static_cast<std::uint8_t>(x>>(i*8));}
inline void put_u64_le_v1(std::uint8_t*&p,std::uint64_t x)noexcept{for(unsigned i=0;i<8;++i)*p++=static_cast<std::uint8_t>(x>>(i*8));}
inline void put_identity_le_v1(std::uint8_t*&p,grammar_identity_v1 x)noexcept{put_u64_le_v1(p,x.producer_namespace);put_u64_le_v1(p,x.local_identity);}
[[nodiscard]]inline grammar_serialization_result_v1 serialize_explicit_grammar_v1(explicit_grammar_v1 g,std::uint8_t*out,std::uint64_t capacity)noexcept{
 if(!valid(g.grammar_identity)||g.grammar_generation==0||!validate_typed_symbol_table_v1(g.symbols).valid()||!validate_explicit_production_registry_v1(g.productions,g.symbols).valid())return{grammar_serialization_code_v1::invalid_grammar};
 std::uint64_t rhs=0;for(std::uint64_t i=0;i<g.productions.production_count;++i){if(rhs>UINT64_MAX-g.productions.productions[i].rhs_count)return{grammar_serialization_code_v1::size_overflow};rhs+=g.productions.productions[i].rhs_count;}
 if(g.symbols.symbol_count>(UINT64_MAX-portable_grammar_header_bytes_v1)/symbol_record_bytes_v1)return{grammar_serialization_code_v1::size_overflow};
 auto bytes=portable_grammar_header_bytes_v1+g.symbols.symbol_count*symbol_record_bytes_v1;
 if(g.productions.production_count>(UINT64_MAX-bytes)/production_record_bytes_v1)return{grammar_serialization_code_v1::size_overflow};
 bytes+=g.productions.production_count*production_record_bytes_v1;
 if(rhs>(UINT64_MAX-bytes)/identity_record_bytes_v1)return{grammar_serialization_code_v1::size_overflow};
 bytes+=rhs*identity_record_bytes_v1;
 if(out==nullptr)return{grammar_serialization_code_v1::missing_output,bytes};
 if(capacity<bytes)return{grammar_serialization_code_v1::insufficient_output,bytes};
 auto*p=out;put_u32_le_v1(p,0x3147424aU);put_u32_le_v1(p,1);put_u64_le_v1(p,bytes);put_identity_le_v1(p,g.grammar_identity);put_u64_le_v1(p,g.grammar_generation);put_identity_le_v1(p,g.symbols.table_identity);put_u64_le_v1(p,g.symbols.table_generation);put_identity_le_v1(p,g.productions.registry_identity);put_u64_le_v1(p,g.productions.registry_generation);put_u64_le_v1(p,g.symbols.symbol_count);put_u64_le_v1(p,g.productions.production_count);put_u64_le_v1(p,rhs);
 for(std::uint64_t i=0;i<g.symbols.symbol_count;++i){const auto&x=g.symbols.symbols[i];put_identity_le_v1(p,x.identity);put_identity_le_v1(p,x.domain_identity);put_identity_le_v1(p,x.order_identity);put_identity_le_v1(p,x.relation_identity);put_identity_le_v1(p,x.scalar_encoding_identity);put_u64_le_v1(p,x.structure_generation);put_u64_le_v1(p,x.value_generation);put_u32_le_v1(p,static_cast<std::uint32_t>(x.symbol_kind));put_u32_le_v1(p,static_cast<std::uint32_t>(x.value_kind));}
 std::uint64_t begin=0;for(std::uint64_t i=0;i<g.productions.production_count;++i){const auto&x=g.productions.productions[i];put_identity_le_v1(p,x.identity);put_identity_le_v1(p,x.lhs_symbol);put_u64_le_v1(p,begin);put_u64_le_v1(p,x.rhs_count);put_u64_le_v1(p,x.maximum_rhs_count);put_u64_le_v1(p,x.production_generation);put_u32_le_v1(p,static_cast<std::uint32_t>(x.algebra));put_u32_le_v1(p,0);begin+=x.rhs_count;}
 for(std::uint64_t i=0;i<g.productions.production_count;++i)for(std::uint64_t j=0;j<g.productions.productions[i].rhs_count;++j)put_identity_le_v1(p,g.productions.productions[i].rhs_symbols[j]);
 return{grammar_serialization_code_v1::written,bytes};
}
}
