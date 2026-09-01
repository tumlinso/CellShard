#include "CellShard/compiler/basis/multi_basis.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    const family_basis_offer offers[] = {{0, UINT64_C(0x200000001), 4},
                                         {0, UINT64_C(0x100000001), 4},
                                         {1, UINT64_C(0x300000001), 7}};
    global_id assignment[2]{}; std::uint64_t utility[2]{};
    assert(assign_family_bases(offers, 3, assignment, utility, 2));
    assert(assignment[0] == UINT64_C(0x100000001) && utility[1] == 7);
    return 0;
}
