#include "CellShard/compiler/basis/exact_oracle.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    const atom_coverage atoms[] = {{10, 0, 1}, {20, 1, 1}};
    const local_index refs[] = {0, 1}; const std::uint64_t freq[] = {5, 7};
    const std::uint64_t costs[] = {2, 10}; bool scratch[2]{};
    const auto result = exact_coverage_oracle({atoms, 2, refs, 2, freq, 2}, costs, scratch);
    assert(result.status == oracle_status::success && result.mask == 1 && result.utility == 3);
    coverage_view too_large{}; too_large.atom_count = exact_oracle_max_atoms + 1;
    assert(exact_coverage_oracle(too_large, costs, scratch).status == oracle_status::instance_too_large);
    return 0;
}
