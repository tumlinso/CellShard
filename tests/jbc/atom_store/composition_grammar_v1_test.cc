#include <CellShard/artifact/atom_store/composition_grammar_v1.hh>
#include <cassert>
namespace atom_store = cellshard::artifact::atom_store;
int main() {
    const std::byte bytes[] = {std::byte{5}};
    const atom_store::composition_record_v1 composition{{1, 1}, {2, 2}, 0, 2, 1};
    atom_store::grammar_record_v1 grammar{
        3, 4, 0, 2, {1, 1}, atom_store::sha256_digest_v1(bytes, 1), 1, 1};
    assert(atom_store::valid_composition_record_v1(composition));
    assert(atom_store::valid_grammar_record_v1(grammar));
    grammar.certified = 0;
    assert(!atom_store::valid_grammar_record_v1(grammar));
    return 0;
}
