#include <CellShard/compiler/partial/dependency_freshness_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace {

using namespace cellshard::compiler::atom;
using namespace cellshard::compiler::partial;

partial_atom_view_v1 partial() {
    partial_atom_view_v1 result{};
    result.header.partial_identity = {1, 50};
    result.header.dependency_closure_identity = {1, 60};
    return result;
}

std::array<partial_dependency_requirement_v1, 3> dependencies() {
    return {{{{2, 1}, {3, 1}, 10,
              atom_dependency_generation_kind_v1::structure,
              partial_dependency_role_v1::direct},
             {{2, 2}, {3, 1}, 20,
              atom_dependency_generation_kind_v1::value,
              partial_dependency_role_v1::direct},
             {{2, 3}, {3, 2}, 30,
              atom_dependency_generation_kind_v1::state,
              partial_dependency_role_v1::transitive}}};
}

partial_dependency_closure_view_v1 closure(
    const std::array<partial_dependency_requirement_v1, 3> &items) {
    return {items.data(), items.size(), {1, 60}, {1, 61}, 1, 0};
}

std::array<partial_dependency_observation_v1, 3> observations() {
    return {{{{2, 1}, {3, 1}, 10,
              atom_dependency_generation_kind_v1::structure, 0},
             {{2, 2}, {3, 1}, 20,
              atom_dependency_generation_kind_v1::value, 0},
             {{2, 3}, {3, 2}, 30,
              atom_dependency_generation_kind_v1::state, 0}}};
}

void test_exact_current_closure() {
    const auto deps = dependencies();
    const auto seen = observations();
    const auto result = evaluate_partial_freshness_v1(
        partial(), closure(deps), {seen.data(), seen.size()});
    assert(result.reusable());
    assert(result.reason
           == partial_freshness_reason_v1::all_generations_match);
}

void test_invalid_closure_rejections() {
    auto deps = dependencies();
    auto view = closure(deps);
    view.exact_certification_identity = {};
    assert(validate_partial_dependency_closure_v1(
               view, partial().header.partial_identity)
               .code == partial_dependency_validation_code_v1::
                            invalid_exact_certification);
    deps = dependencies();
    deps[1].dependency_identity = deps[0].dependency_identity;
    deps[1].generation_kind = deps[0].generation_kind;
    assert(validate_partial_dependency_closure_v1(
               closure(deps), partial().header.partial_identity)
               .code == partial_dependency_validation_code_v1::
                            unordered_or_duplicate_dependency);
    deps = dependencies();
    deps[1].dependency_identity = partial().header.partial_identity;
    assert(validate_partial_dependency_closure_v1(
               closure(deps), partial().header.partial_identity)
               .code == partial_dependency_validation_code_v1::
                            self_dependency);
}

void test_freshness_fails_closed() {
    const auto deps = dependencies();
    auto seen = observations();
    seen[1].current_generation = 21;
    auto result = evaluate_partial_freshness_v1(
        partial(), closure(deps), {seen.data(), seen.size()});
    assert(result.freshness == partial_freshness_v1::stale);
    assert(!result.reusable());

    seen = observations();
    result = evaluate_partial_freshness_v1(
        partial(), closure(deps), {seen.data(), seen.size() - 1});
    assert(result.freshness == partial_freshness_v1::unproven);
    assert(!result.reusable());

    auto wrong_partial = partial();
    wrong_partial.header.dependency_closure_identity = {9, 9};
    result = evaluate_partial_freshness_v1(
        wrong_partial, closure(deps), {seen.data(), seen.size()});
    assert(result.freshness == partial_freshness_v1::invalid);
    assert(!result.reusable());
}

void test_randomized_differential_generations() {
    auto deps = dependencies();
    auto seen = observations();
    std::uint64_t state = 0x4d595df4d0f33173ULL;
    for (std::uint32_t trial = 0; trial < 4096; ++trial) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const std::size_t changed = static_cast<std::size_t>(state % seen.size());
        seen = observations();
        const bool equal = (state & 7U) == 0;
        if (!equal) {
            seen[changed].current_generation += 1 + ((state >> 8) & 15U);
        }
        const auto result = evaluate_partial_freshness_v1(
            partial(), closure(deps), {seen.data(), seen.size()});
        assert(result.reusable() == equal);
        assert(result.freshness
               == (equal ? partial_freshness_v1::current
                         : partial_freshness_v1::stale));
    }
}

} // namespace

int main() {
    test_exact_current_closure();
    test_invalid_closure_rejections();
    test_freshness_fails_closed();
    test_randomized_differential_generations();
    return 0;
}
