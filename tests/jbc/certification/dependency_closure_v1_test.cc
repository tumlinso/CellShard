#include <CellShard/compiler/certification/dependency_closure_v1.hh>

#include <cassert>

using namespace cellshard::compiler;

int main() {
    atom::atom_dependency_requirement_v1 dependencies[]{
        {{1, 1},
         {2, 1},
         5,
         5,
         atom::atom_dependency_generation_kind_v1::structure,
         atom::atom_dependency_effect_v1::correctness},
        {{1, 2},
         {2, 1},
         7,
         8,
         atom::atom_dependency_generation_kind_v1::value,
         atom::atom_dependency_effect_v1::preference}};
    atom::common_atom_view_v1 common{};
    common.dependencies = {dependencies, 2, {3, 1}, 1};
    certification::authoritative_generation_v1 authority[]{
        {{1, 1}, {2, 1}, 5, atom::atom_dependency_generation_kind_v1::structure, 0},
        {{1, 2}, {2, 1}, 8, atom::atom_dependency_generation_kind_v1::value, 0}};
    assert(certification::validate_generation_dependency_closure_v1(
               &common, 1, authority, 2)
               .valid());

    authority[1].generation = 9;
    assert(certification::validate_generation_dependency_closure_v1(
               &common, 1, authority, 2)
               .code
           == certification::dependency_closure_validation_code_v1::
               observed_generation_stale);

    authority[1].generation = 8;
    authority[1].dependency_identity = {1, 3};
    assert(certification::validate_generation_dependency_closure_v1(
               &common, 1, authority, 2)
               .code
           == certification::dependency_closure_validation_code_v1::
               dependency_missing_from_authority);
}
