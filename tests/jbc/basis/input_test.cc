#include "CellShard/compiler/basis/input.hpp"

#include <cassert>

int main() {
    using namespace cellshard::compiler::basis;
    const atom_input atoms[] = {{UINT64_C(0x100000001), 4, 5, 6, 7},
                                {UINT64_C(0x200000002), 8, 9, 10, 11}};
    const local_index required[] = {0, 1};
    const workload_family_input families[] = {
        {UINT64_C(0xf00000001), 3, 0, 2}};
    basis_input_view input{families, 1, atoms, 2, required, 2};
    assert(validate_input(input) == input_error::none);
    const local_index bad_required[] = {0, 2};
    input.required_atoms = bad_required;
    assert(validate_input(input) == input_error::invalid_atom_reference);
    assert(valid_weight({3, 7}));
    assert(!valid_weight({1, 0}));
    assert(compare_weight({UINT64_MAX, UINT64_MAX}, {1, 2}) > 0);
    assert(compare_weight({2, 4}, {1, 2}) == 0);
    return 0;
}
