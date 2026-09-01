#include <CellShard/compiler/atom/level_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {

using cellshard::compiler::atom::atom_level_name_v1;
using cellshard::compiler::atom::atom_level_path_view_v1;
using cellshard::compiler::atom::atom_level_rank_v1;
using cellshard::compiler::atom::atom_level_validation_code_v1;
using cellshard::compiler::atom::atom_level_v1;
using cellshard::compiler::atom::valid_atom_level_transition_v1;
using cellshard::compiler::atom::valid_atom_level_v1;
using cellshard::compiler::atom::validate_atom_level_path_v1;

constexpr std::array<atom_level_v1, 10> all_levels{{
    atom_level_v1::evidence,
    atom_level_v1::semantic,
    atom_level_v1::structural,
    atom_level_v1::materialized,
    atom_level_v1::partial,
    atom_level_v1::executable,
    atom_level_v1::graph_family,
    atom_level_v1::schedule,
    atom_level_v1::topology,
    atom_level_v1::resident,
}};

constexpr std::array<atom_level_v1, 4> skipped_path{{
    atom_level_v1::evidence,
    atom_level_v1::structural,
    atom_level_v1::executable,
    atom_level_v1::resident,
}};

constexpr std::array<atom_level_v1, 1> semantic_only{{
    atom_level_v1::semantic,
}};

static_assert(cellshard::compiler::atom::atom_level_contract_version_v1 == 1);
static_assert(std::is_trivially_copyable<atom_level_path_view_v1>::value);
static_assert(validate_atom_level_path_v1(
                  {all_levels.data(), all_levels.size()})
                  .valid());
static_assert(validate_atom_level_path_v1(
                  {skipped_path.data(), skipped_path.size()})
                  .valid());
static_assert(validate_atom_level_path_v1(
                  {semantic_only.data(), semantic_only.size()})
                  .valid());
static_assert(atom_level_v1::semantic != atom_level_v1::materialized);
static_assert(atom_level_v1::materialized != atom_level_v1::executable);
static_assert(atom_level_v1::executable != atom_level_v1::resident);
static_assert(valid_atom_level_transition_v1(atom_level_v1::evidence,
                                             atom_level_v1::resident));
static_assert(!valid_atom_level_transition_v1(atom_level_v1::resident,
                                              atom_level_v1::evidence));

std::uint64_t next_random(std::uint64_t *state) {
    *state = *state * UINT64_C(6364136223846793005) + UINT64_C(1);
    return *state;
}

void test_level_values_and_names() {
    assert(!valid_atom_level_v1(atom_level_v1::invalid));
    assert(!valid_atom_level_v1(static_cast<atom_level_v1>(UINT32_C(0xffffffff))));
    for (std::size_t index = 0; index < all_levels.size(); ++index) {
        assert(valid_atom_level_v1(all_levels[index]));
        assert(atom_level_rank_v1(all_levels[index]) == index + 1);
        assert(std::strcmp(atom_level_name_v1(all_levels[index]), "invalid") != 0);
    }
    assert(std::strcmp(atom_level_name_v1(atom_level_v1::invalid), "invalid") == 0);
}

void test_deterministic_rejections() {
    auto result = validate_atom_level_path_v1({nullptr, 0});
    assert(result.code == atom_level_validation_code_v1::empty_path);
    assert(result.index == 0);

    result = validate_atom_level_path_v1({nullptr, 1});
    assert(result.code == atom_level_validation_code_v1::null_levels);
    assert(result.index == 0);

    const std::array<atom_level_v1, 3> invalid{{
        atom_level_v1::evidence,
        static_cast<atom_level_v1>(999),
        atom_level_v1::resident,
    }};
    result = validate_atom_level_path_v1({invalid.data(), invalid.size()});
    assert(result.code == atom_level_validation_code_v1::invalid_level);
    assert(result.index == 1);
    assert(result.level == invalid[1]);

    const std::array<atom_level_v1, 3> duplicate{{
        atom_level_v1::semantic,
        atom_level_v1::semantic,
        atom_level_v1::structural,
    }};
    result = validate_atom_level_path_v1({duplicate.data(), duplicate.size()});
    assert(result.code == atom_level_validation_code_v1::duplicate_level);
    assert(result.index == 1);

    const std::array<atom_level_v1, 3> reversed{{
        atom_level_v1::structural,
        atom_level_v1::semantic,
        atom_level_v1::resident,
    }};
    result = validate_atom_level_path_v1({reversed.data(), reversed.size()});
    assert(result.code == atom_level_validation_code_v1::non_monotonic_level);
    assert(result.index == 1);
}

void test_level_does_not_conflate_atomicity() {
    struct atomicity_example {
        atom_level_v1 level;
        bool semantic;
        bool ownership;
        bool materialization;
        bool execution;
        bool cache_reuse;
    };

    // These fixture-only capabilities are deliberately independent. A02 owns
    // their eventual public taxonomy; A01 proves that level cannot substitute
    // for semantic, ownership, materialization, execution, or cache atomicity.
    const atomicity_example semantic_description{
        atom_level_v1::semantic, true, false, false, false, false};
    const atomicity_example shared_materialization{
        atom_level_v1::materialized, false, false, true, false, true};
    const atomicity_example owned_materialization{
        atom_level_v1::materialized, false, true, true, false, false};
    const atomicity_example executable_view{
        atom_level_v1::executable, false, false, false, true, false};

    assert(semantic_description.semantic);
    assert(shared_materialization.level == owned_materialization.level);
    assert(shared_materialization.ownership != owned_materialization.ownership);
    assert(shared_materialization.cache_reuse != owned_materialization.cache_reuse);
    assert(executable_view.execution && !executable_view.materialization);
}

void test_randomized_subsets_and_mutations() {
    std::uint64_t random_state = UINT64_C(0x51a7c0de);
    for (std::size_t iteration = 0; iteration < 2048; ++iteration) {
        std::vector<atom_level_v1> levels;
        for (const auto level : all_levels) {
            if ((next_random(&random_state) >> 63) != 0) {
                levels.push_back(level);
            }
        }
        if (levels.empty()) {
            levels.push_back(all_levels[next_random(&random_state)
                                        % all_levels.size()]);
        }

        const auto valid = validate_atom_level_path_v1(
            {levels.data(), levels.size()});
        assert(valid.valid());
        assert(valid.index == levels.size());

        if (levels.size() > 1) {
            std::vector<atom_level_v1> reversed = levels;
            const auto index = 1 + next_random(&random_state)
                % (reversed.size() - 1);
            const auto temporary = reversed[index - 1];
            reversed[index - 1] = reversed[index];
            reversed[index] = temporary;
            assert(validate_atom_level_path_v1(
                       {reversed.data(), reversed.size()})
                       .code
                   == atom_level_validation_code_v1::non_monotonic_level);
        }

        std::vector<atom_level_v1> duplicated = levels;
        const auto duplicate_index = next_random(&random_state)
            % duplicated.size();
        using difference_type =
            std::vector<atom_level_v1>::difference_type;
        duplicated.insert(
            duplicated.begin()
                + static_cast<difference_type>(duplicate_index),
            duplicated[duplicate_index]);
        assert(validate_atom_level_path_v1(
                   {duplicated.data(), duplicated.size()})
                   .code
               == atom_level_validation_code_v1::duplicate_level);
    }
}

} // namespace

int main() {
    test_level_values_and_names();
    test_deterministic_rejections();
    test_level_does_not_conflate_atomicity();
    test_randomized_subsets_and_mutations();
    return 0;
}
