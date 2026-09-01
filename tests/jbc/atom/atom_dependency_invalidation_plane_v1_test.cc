#include <CellShard/compiler/atom/dependency_invalidation_plane_v1.hh>

#include <array>
#include <cassert>

namespace {

using namespace cellshard::compiler::atom;

atom_dependency_requirement_v1 make_dependency(
    std::uint64_t identity,
    atom_dependency_generation_kind_v1 kind,
    atom_dependency_effect_v1 effect) {
    return {{1, identity}, {2, 1}, 7, 7, kind, effect};
}

void test_all_generation_classes_and_freshness() {
    std::array<atom_dependency_requirement_v1, 6> dependencies{
        make_dependency(1, atom_dependency_generation_kind_v1::structure,
                        atom_dependency_effect_v1::correctness),
        make_dependency(2, atom_dependency_generation_kind_v1::value,
                        atom_dependency_effect_v1::correctness),
        make_dependency(3, atom_dependency_generation_kind_v1::state,
                        atom_dependency_effect_v1::correctness),
        make_dependency(4, atom_dependency_generation_kind_v1::graph,
                        atom_dependency_effect_v1::preference),
        make_dependency(5, atom_dependency_generation_kind_v1::topology,
                        atom_dependency_effect_v1::performance),
        make_dependency(6, atom_dependency_generation_kind_v1::build,
                        atom_dependency_effect_v1::performance)};
    atom_dependency_invalidation_plane_v1 plane{
        dependencies.data(), dependencies.size(), {3, 1}, 9};
    auto result = validate_atom_dependency_invalidation_plane_v1(plane);
    assert(result.valid());
    assert(result.preference_fresh());
    assert(result.performance_fresh());

    dependencies[3].observed_generation = 8;
    dependencies[4].observed_generation = 6;
    result = validate_atom_dependency_invalidation_plane_v1(plane);
    assert(result.valid());
    assert(!result.preference_fresh());
    assert(!result.performance_fresh());
    assert(result.stale_preference_count == 1);
    assert(result.stale_performance_count == 1);
}

void test_correctness_staleness_invalidates() {
    auto dependency = make_dependency(
        1, atom_dependency_generation_kind_v1::structure,
        atom_dependency_effect_v1::correctness);
    dependency.observed_generation = 8;
    const atom_dependency_invalidation_plane_v1 plane{
        &dependency, 1, {3, 1}, 9};
    const auto result = validate_atom_dependency_invalidation_plane_v1(plane);
    assert(result.code
           == atom_dependency_invalidation_code_v1::
                  stale_correctness_dependency);
    assert(!result.valid());
}

void test_deterministic_rejections() {
    std::array<atom_dependency_requirement_v1, 2> dependencies{
        make_dependency(1, atom_dependency_generation_kind_v1::structure,
                        atom_dependency_effect_v1::correctness),
        make_dependency(2, atom_dependency_generation_kind_v1::value,
                        atom_dependency_effect_v1::preference)};
    atom_dependency_invalidation_plane_v1 plane{
        dependencies.data(), dependencies.size(), {3, 1}, 9};

    dependencies[0].generation_namespace = {};
    assert(validate_atom_dependency_invalidation_plane_v1(plane).code
           == atom_dependency_invalidation_code_v1::
                  invalid_generation_namespace);

    dependencies[0] = make_dependency(
        1, atom_dependency_generation_kind_v1::structure,
        atom_dependency_effect_v1::correctness);
    dependencies[1].dependency_identity = dependencies[0].dependency_identity;
    dependencies[1].generation_kind = dependencies[0].generation_kind;
    assert(validate_atom_dependency_invalidation_plane_v1(plane).code
           == atom_dependency_invalidation_code_v1::
                  unordered_or_duplicate_dependency);

    dependencies[1] = make_dependency(
        2, atom_dependency_generation_kind_v1::value,
        atom_dependency_effect_v1::preference);
    dependencies[1].required_generation = 0;
    assert(validate_atom_dependency_invalidation_plane_v1(plane).code
           == atom_dependency_invalidation_code_v1::
                  missing_required_generation);

    dependencies[1] = make_dependency(
        2, atom_dependency_generation_kind_v1::value,
        atom_dependency_effect_v1::preference);
    plane.validation_generation = 0;
    assert(validate_atom_dependency_invalidation_plane_v1(plane).code
           == atom_dependency_invalidation_code_v1::
                  missing_validation_generation);
}

} // namespace

int main() {
    test_all_generation_classes_and_freshness();
    test_correctness_staleness_invalidates();
    test_deterministic_rejections();
    return 0;
}
