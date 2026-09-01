#include "CellShard/compiler/basis/manifest.hpp"
#include <cassert>
int main() {
    using namespace cellshard::compiler::basis;
    const global_id atoms[] = {10, 20};
    basis_manifest manifest{}; manifest.basis_id = 4; manifest.structure_epoch = 5;
    manifest.workload_epoch = 6; manifest.solver_id = 7; manifest.atom_count = 2;
    assert(validate_manifest(manifest, atoms, 2) == manifest_validity::valid);
    assert(freshness(manifest, 5, 7) == manifest_freshness::stale_workload);
    assert(freshness(manifest, 8, 6) == manifest_freshness::stale_structure);
    return 0;
}
