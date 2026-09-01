#include "CellShard/compiler/composition/superatom/candidate.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::composition::superatom;
    const global_id atoms[] = {UINT64_C(0x100000001), UINT64_C(0x200000002)};
    const candidate candidates[] = {{{UINT64_C(0x300000003), 4, 5, 6}, 0, 2}};
    assert(validate_candidates({candidates, 1, atoms, 2}) == candidate_error::none);
    const global_id reversed[] = {9, 8};
    assert(validate_candidates({candidates, 1, reversed, 2}) == candidate_error::invalid_atoms);
    return 0;
}
