#include <CellShard/compiler/atom/atomicity_v1.hh>
#include <CellShard/compiler/atom/level_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

using cellshard::compiler::atom::atom_atomicity_bit_v1;
using cellshard::compiler::atom::atom_atomicity_capability_v1;
using cellshard::compiler::atom::atom_atomicity_known_mask_v1;
using cellshard::compiler::atom::atom_atomicity_name_v1;
using cellshard::compiler::atom::atom_atomicity_set_v1;
using cellshard::compiler::atom::atom_atomicity_validation_code_v1;
using cellshard::compiler::atom::atom_has_atomicity_v1;
using cellshard::compiler::atom::atom_level_v1;
using cellshard::compiler::atom::valid_atom_atomicity_capability_v1;
using cellshard::compiler::atom::validate_atom_atomicity_v1;

constexpr std::array<atom_atomicity_capability_v1, 11> all_capabilities{{
    atom_atomicity_capability_v1::semantic,
    atom_atomicity_capability_v1::ownership,
    atom_atomicity_capability_v1::materialization,
    atom_atomicity_capability_v1::transfer,
    atom_atomicity_capability_v1::cache_reuse,
    atom_atomicity_capability_v1::order,
    atom_atomicity_capability_v1::algebraic,
    atom_atomicity_capability_v1::executable,
    atom_atomicity_capability_v1::grammar,
    atom_atomicity_capability_v1::cellerator_local,
    atom_atomicity_capability_v1::cellshard_global,
}};

constexpr auto local_executable =
    atom_atomicity_capability_v1::executable
    | atom_atomicity_capability_v1::cellerator_local;
constexpr auto global_transfer =
    atom_atomicity_capability_v1::transfer
    | atom_atomicity_capability_v1::cellshard_global;
constexpr auto cross_boundary =
    atom_atomicity_capability_v1::cellerator_local
    | atom_atomicity_capability_v1::cellshard_global;

static_assert(cellshard::compiler::atom::atom_atomicity_contract_version_v1 == 1);
static_assert(std::is_trivially_copyable<atom_atomicity_set_v1>::value);
static_assert(validate_atom_atomicity_v1(local_executable).valid());
static_assert(validate_atom_atomicity_v1(global_transfer).valid());
static_assert(validate_atom_atomicity_v1(cross_boundary).valid());
static_assert(atom_has_atomicity_v1(
    local_executable, atom_atomicity_capability_v1::executable));
static_assert(!atom_has_atomicity_v1(
    local_executable, atom_atomicity_capability_v1::ownership));

void test_capabilities_are_independent_bits() {
    std::uint64_t observed = 0;
    for (std::size_t index = 0; index < all_capabilities.size(); ++index) {
        const auto capability = all_capabilities[index];
        const auto bit = atom_atomicity_bit_v1(capability);
        assert(valid_atom_atomicity_capability_v1(capability));
        assert(bit == (UINT64_C(1) << index));
        assert((observed & bit) == 0);
        assert(std::strcmp(atom_atomicity_name_v1(capability), "invalid") != 0);
        observed |= bit;
    }
    assert(observed == atom_atomicity_known_mask_v1);
    assert(!valid_atom_atomicity_capability_v1(
        static_cast<atom_atomicity_capability_v1>(0)));
    assert(!valid_atom_atomicity_capability_v1(
        static_cast<atom_atomicity_capability_v1>(UINT64_C(3))));
    assert(!valid_atom_atomicity_capability_v1(
        static_cast<atom_atomicity_capability_v1>(UINT64_C(1) << 63)));
}

void test_all_known_combinations() {
    for (std::uint64_t bits = 1; bits <= atom_atomicity_known_mask_v1; ++bits) {
        const atom_atomicity_set_v1 set{bits};
        const auto result = validate_atom_atomicity_v1(set);
        assert(result.valid());
        assert(result.unknown_capabilities == 0);
        for (const auto capability : all_capabilities) {
            const auto expected = (bits & atom_atomicity_bit_v1(capability)) != 0;
            assert(atom_has_atomicity_v1(set, capability) == expected);
        }
    }
}

void test_deterministic_rejections() {
    auto result = validate_atom_atomicity_v1({0});
    assert(result.code == atom_atomicity_validation_code_v1::empty);
    assert(result.unknown_capabilities == 0);

    constexpr std::uint64_t unknown_a = UINT64_C(1) << 40;
    constexpr std::uint64_t unknown_b = UINT64_C(1) << 63;
    result = validate_atom_atomicity_v1(
        {atom_atomicity_known_mask_v1 | unknown_a | unknown_b});
    assert(result.code
           == atom_atomicity_validation_code_v1::unknown_capability);
    assert(result.unknown_capabilities == (unknown_a | unknown_b));
}

void test_level_and_atomicity_are_orthogonal() {
    struct atom_interpretation_fixture {
        atom_level_v1 level;
        atom_atomicity_set_v1 atomicity;
    };

    const atom_interpretation_fixture shared{
        atom_level_v1::materialized,
        atom_atomicity_capability_v1::materialization
            | atom_atomicity_capability_v1::cache_reuse};
    const atom_interpretation_fixture owned{
        atom_level_v1::materialized,
        atom_atomicity_capability_v1::materialization
            | atom_atomicity_capability_v1::ownership};

    assert(shared.level == owned.level);
    assert(atom_has_atomicity_v1(
        shared.atomicity, atom_atomicity_capability_v1::cache_reuse));
    assert(!atom_has_atomicity_v1(
        shared.atomicity, atom_atomicity_capability_v1::ownership));
    assert(atom_has_atomicity_v1(
        owned.atomicity, atom_atomicity_capability_v1::ownership));
    assert(!atom_has_atomicity_v1(
        owned.atomicity, atom_atomicity_capability_v1::cache_reuse));
}

} // namespace

int main() {
    test_capabilities_are_independent_bits();
    test_all_known_combinations();
    test_deterministic_rejections();
    test_level_and_atomicity_are_orthogonal();
    return 0;
}
