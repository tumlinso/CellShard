#include "atom_store_faults.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    assert(complete_fault_matrix());
    assert(atom_store_fault_matrix[6].expected == recovery_outcome::retain_pinned_atom);
    return 0;
}
