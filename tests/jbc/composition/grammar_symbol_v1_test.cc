#include <CellShard/compiler/composition/grammar_symbol_v1.hh>

#include <array>
#include <cassert>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<composition::atom_port_signature_v1, 2> inputs{{
        {1, composition::atom_port_direction_v1::input,
         composition::atom_port_kind_v1::immutable_structure, 0,
         cellshard::domain_id{10}, cellshard::order_id{11},
         cellshard::structure_id{12}, {}},
        {3, composition::atom_port_direction_v1::input_output,
         composition::atom_port_kind_v1::mutable_value, 0,
         cellshard::domain_id{10}, cellshard::order_id{11},
         cellshard::structure_id{12}, cellshard::scalar_encoding_id{13}}}};
    const std::array<composition::atom_port_signature_v1, 1> outputs{{
        {2, composition::atom_port_direction_v1::output,
         composition::atom_port_kind_v1::partial_result, 0,
         cellshard::domain_id{20}, cellshard::order_id{21},
         cellshard::structure_id{22}, cellshard::scalar_encoding_id{23}}}};
    const composition::atom_interface_signature_v1 interface{
        composition::atom_interface_id{30}, inputs.data(), outputs.data(),
        inputs.size(), outputs.size()};
    assert(composition::validate_atom_interface_signature_v1(interface).valid());

    const composition::typed_grammar_symbol_v1 symbol{
        composition::grammar_symbol_id{40},
        composition::grammar_symbol_kind_v1::terminal_atom, {},
        interface.identity, composition::composition_lineage_id{41}};
    assert(composition::validate_typed_grammar_symbol_v1(symbol).valid());

    auto unordered = inputs;
    unordered[1].port_identity = 1;
    auto malformed = interface;
    malformed.inputs = unordered.data();
    assert(composition::validate_atom_interface_signature_v1(malformed).code
           == composition::grammar_signature_code_v1::
                  unordered_port_identity);

    auto conflicting_output = outputs;
    conflicting_output[0].port_identity = 3;
    malformed = interface;
    malformed.outputs = conflicting_output.data();
    assert(composition::validate_atom_interface_signature_v1(malformed).code
           == composition::grammar_signature_code_v1::duplicate_port_identity);

    auto hidden_encoding = inputs;
    hidden_encoding[0].encoding = cellshard::scalar_encoding_id{99};
    malformed = interface;
    malformed.inputs = hidden_encoding.data();
    assert(composition::validate_atom_interface_signature_v1(malformed).code
           == composition::grammar_signature_code_v1::unexpected_encoding);
}
