#include <CellShard/compiler/atom/species_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using namespace cellshard::compiler::atom;

std::array<atom_species_descriptor_v1, core_atom_species_count_v1>
make_core_registry() {
    std::array<atom_species_descriptor_v1, core_atom_species_count_v1> core{};
    for (std::size_t index = 0; index < core.size(); ++index) {
        core[index] = core_atom_species_descriptor_v1(
            static_cast<core_atom_species_v1>(index + 1));
    }
    return core;
}

void test_core_registry_and_names() {
    const auto core = make_core_registry();
    const atom_species_registry_view_v1 registry{core.data(), core.size()};
    const auto result = validate_atom_species_registry_v1(registry);
    assert(result.valid());
    assert(result.index == core.size());

    for (std::size_t index = 0; index < core.size(); ++index) {
        const auto species = static_cast<core_atom_species_v1>(index + 1);
        assert(valid_core_atom_species_v1(species));
        assert(core[index].id == core_atom_species_id_v1(species));
        assert(core[index].stable_name_size != 0);
        assert(std::strcmp(core[index].stable_name, "invalid") != 0);
        assert(find_atom_species_v1(registry, core[index].id) == &core[index]);
    }
    assert(!valid_core_atom_species_v1(static_cast<core_atom_species_v1>(0)));
    assert(!valid_core_atom_species_v1(static_cast<core_atom_species_v1>(23)));
    assert(find_atom_species_v1(registry, {2, 1}) == nullptr);
}

void test_provider_extensions() {
    const auto core = make_core_registry();
    std::vector<atom_species_descriptor_v1> descriptors(core.begin(), core.end());
    descriptors.push_back({{2, 1}, "provider_alpha", 14});
    descriptors.push_back({{2, 9}, "provider_beta", 13});
    descriptors.push_back({{77, 3}, "provider_gamma", 14});

    const atom_species_registry_view_v1 registry{
        descriptors.data(), descriptors.size()};
    assert(validate_atom_species_registry_v1(registry).valid());
    const auto *found = find_atom_species_v1(registry, {2, 9});
    assert(found != nullptr);
    assert(std::strcmp(found->stable_name, "provider_beta") == 0);
}

void test_deterministic_rejections() {
    auto result = validate_atom_species_registry_v1({nullptr, 0});
    assert(result.code == atom_species_validation_code_v1::empty_registry);
    result = validate_atom_species_registry_v1({nullptr, core_atom_species_count_v1});
    assert(result.code == atom_species_validation_code_v1::null_species);

    auto core = make_core_registry();
    result = validate_atom_species_registry_v1({core.data(), core.size() - 1});
    assert(result.code == atom_species_validation_code_v1::missing_core_species);

    auto malformed = core;
    malformed[5].id = {0, 6};
    result = validate_atom_species_registry_v1(
        {malformed.data(), malformed.size()});
    assert(result.code == atom_species_validation_code_v1::invalid_id);
    assert(result.index == 5);

    malformed = core;
    malformed[5].id = malformed[4].id;
    result = validate_atom_species_registry_v1(
        {malformed.data(), malformed.size()});
    assert(result.code
           == atom_species_validation_code_v1::unsorted_or_duplicate_id);
    assert(result.index == 5);

    malformed = core;
    malformed[5].stable_name = nullptr;
    result = validate_atom_species_registry_v1(
        {malformed.data(), malformed.size()});
    assert(result.code == atom_species_validation_code_v1::invalid_name);
    assert(result.index == 5);

    malformed = core;
    malformed[5].stable_name = "wrong";
    malformed[5].stable_name_size = 5;
    result = validate_atom_species_registry_v1(
        {malformed.data(), malformed.size()});
    assert(result.code == atom_species_validation_code_v1::core_name_mismatch);
    assert(result.index == 5);

    malformed = core;
    malformed[5].stable_name = "mo\0tif";
    malformed[5].stable_name_size = 6;
    result = validate_atom_species_registry_v1(
        {malformed.data(), malformed.size()});
    assert(result.code == atom_species_validation_code_v1::invalid_name);
    assert(result.index == 5);
}

std::uint64_t next_random(std::uint64_t *state) {
    *state = *state * UINT64_C(2862933555777941757) + UINT64_C(3037000493);
    return *state;
}

void test_randomized_sorted_extensions() {
    const auto core = make_core_registry();
    std::uint64_t state = UINT64_C(0xa03c0de);
    for (std::size_t iteration = 0; iteration < 2048; ++iteration) {
        std::vector<atom_species_descriptor_v1> descriptors(
            core.begin(), core.end());
        const auto provider_count = 1 + next_random(&state) % 4;
        for (std::uint64_t provider = 2;
             provider < 2 + provider_count; ++provider) {
            const auto species_count = 1 + next_random(&state) % 8;
            std::uint64_t local_id = 0;
            for (std::uint64_t index = 0; index < species_count; ++index) {
                local_id += 1 + next_random(&state) % 5;
                descriptors.push_back(
                    {{provider, local_id}, "provider_extension", 18});
            }
        }
        const atom_species_registry_view_v1 registry{
            descriptors.data(), descriptors.size()};
        assert(validate_atom_species_registry_v1(registry).valid());
        const auto probe = descriptors[core.size()
            + next_random(&state) % (descriptors.size() - core.size())];
        assert(find_atom_species_v1(registry, probe.id) != nullptr);

        auto duplicated = descriptors;
        duplicated.push_back(duplicated.back());
        assert(validate_atom_species_registry_v1(
                   {duplicated.data(), duplicated.size()})
                   .code
               == atom_species_validation_code_v1::unsorted_or_duplicate_id);
    }
}

} // namespace

int main() {
    test_core_registry_and_names();
    test_provider_extensions();
    test_deterministic_rejections();
    test_randomized_sorted_extensions();
    return 0;
}
