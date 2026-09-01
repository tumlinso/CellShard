#include <CellShard/compiler/grammar/typed_symbol_table_v1.hh>

#include <array>
#include <cassert>

namespace grammar = cellshard::compiler::grammar;

int main() {
    std::array<grammar::typed_grammar_symbol_v1, 2> symbols{{
        {{1, 1}, {2, 1}, {2, 2}, {2, 3}, {}, 3, 0,
         grammar::grammar_symbol_kind_v1::terminal_atom,
         grammar::grammar_value_kind_v1::immutable_structure},
        {{1, 2}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, 3, 5,
         grammar::grammar_symbol_kind_v1::nonterminal,
         grammar::grammar_value_kind_v1::partial_result}}};
    grammar::typed_symbol_table_v1 table{
        symbols.data(), symbols.size(), symbols.size(), {3, 1}, 7};
    assert(grammar::validate_typed_symbol_table_v1(table).valid());
    assert(grammar::find_symbol_v1(table, {1, 2}) == &symbols[1]);
    assert(grammar::find_symbol_v1(table, {1, 9}) == nullptr);
    assert(!grammar::authorizes_execution(table));

    symbols[1].identity = symbols[0].identity;
    assert(grammar::validate_typed_symbol_table_v1(table).code
           == grammar::typed_symbol_table_code_v1::unordered_or_duplicate_symbol);
}
