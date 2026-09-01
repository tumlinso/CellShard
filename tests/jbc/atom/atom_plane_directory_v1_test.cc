#include <CellShard/compiler/atom/plane_directory_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using namespace cellshard::compiler::atom;

alignas(16) std::byte descriptor_payload[64]{};

atom_plane_descriptor_v1 primary(std::uint64_t kind,
                                 std::uint64_t identity) {
    return {{1, kind}, {2, identity}, {3, kind}, {}, descriptor_payload,
            sizeof(descriptor_payload), 16,
            atom_plane_representation_role_v1::primary};
}

atom_plane_descriptor_v1 mirror(std::uint64_t kind, std::uint64_t identity,
                                std::uint64_t primary_identity) {
    return {{1, kind}, {2, identity}, {3, kind}, {2, primary_identity},
            descriptor_payload, sizeof(descriptor_payload),
            16,
            atom_plane_representation_role_v1::alternate_physical_mirror};
}

void test_primary_and_explicit_mirrors() {
    std::array<atom_plane_descriptor_v1, 5> planes{
        primary(1, 10), mirror(1, 11, 10), mirror(1, 12, 10),
        primary(2, 20), mirror(2, 21, 20)};
    const auto result = validate_atom_plane_directory_v1(
        {planes.data(), planes.size()});
    assert(result.valid());
    assert(result.index == planes.size());
}

void test_deterministic_rejections() {
    assert(validate_atom_plane_directory_v1({nullptr, 0}).code
           == atom_plane_directory_validation_code_v1::empty_directory);
    assert(validate_atom_plane_directory_v1({nullptr, 1}).code
           == atom_plane_directory_validation_code_v1::missing_planes);

    auto base = primary(1, 10);
    auto expect = [&](atom_plane_descriptor_v1 plane,
                      atom_plane_directory_validation_code_v1 code) {
        const auto result = validate_atom_plane_directory_v1({&plane, 1});
        assert(result.code == code);
    };
    auto malformed = base;
    malformed.plane_kind = {};
    expect(malformed,
           atom_plane_directory_validation_code_v1::invalid_plane_kind);
    malformed = base;
    malformed.plane_identity = {};
    expect(malformed,
           atom_plane_directory_validation_code_v1::invalid_plane_identity);
    malformed = base;
    malformed.descriptor_schema = {};
    expect(malformed,
           atom_plane_directory_validation_code_v1::invalid_descriptor_schema);
    malformed = base;
    malformed.descriptor = nullptr;
    expect(malformed,
           atom_plane_directory_validation_code_v1::missing_descriptor);
    malformed = base;
    malformed.descriptor_bytes = 0;
    expect(malformed, atom_plane_directory_validation_code_v1::empty_descriptor);
    malformed = base;
    malformed.descriptor_alignment = 3;
    expect(malformed, atom_plane_directory_validation_code_v1::
                          invalid_descriptor_alignment);
    malformed = base;
    malformed.descriptor = descriptor_payload + 1;
    expect(malformed,
           atom_plane_directory_validation_code_v1::misaligned_descriptor);
    malformed = base;
    malformed.primary_plane_identity = {2, 10};
    expect(malformed, atom_plane_directory_validation_code_v1::
                          unexpected_primary_reference);

    std::array<atom_plane_descriptor_v1, 2> duplicate{base, base};
    assert(validate_atom_plane_directory_v1(
               {duplicate.data(), duplicate.size()})
               .code == atom_plane_directory_validation_code_v1::
                            unordered_or_duplicate_plane);

    std::array<atom_plane_descriptor_v1, 2> two_primary{
        primary(1, 10), primary(1, 11)};
    assert(validate_atom_plane_directory_v1(
               {two_primary.data(), two_primary.size()})
               .code == atom_plane_directory_validation_code_v1::
                            duplicate_primary_plane);

    auto only_mirror = mirror(1, 11, 10);
    expect(only_mirror,
           atom_plane_directory_validation_code_v1::missing_primary_plane);

    std::array<atom_plane_descriptor_v1, 2> bad_mirror{
        primary(1, 10), mirror(1, 11, 99)};
    assert(validate_atom_plane_directory_v1(
               {bad_mirror.data(), bad_mirror.size()})
               .code == atom_plane_directory_validation_code_v1::
                            mirror_primary_mismatch);
    bad_mirror[1].primary_plane_identity = {};
    assert(validate_atom_plane_directory_v1(
               {bad_mirror.data(), bad_mirror.size()})
               .code == atom_plane_directory_validation_code_v1::
                            missing_primary_reference);
    bad_mirror[1].primary_plane_identity = bad_mirror[1].plane_identity;
    assert(validate_atom_plane_directory_v1(
               {bad_mirror.data(), bad_mirror.size()})
               .code == atom_plane_directory_validation_code_v1::
                            mirror_self_reference);
}

void test_randomized_sorted_directories() {
    std::uint64_t state = UINT64_C(0xa09c0de);
    for (std::size_t iteration = 0; iteration < 4096; ++iteration) {
        std::vector<atom_plane_descriptor_v1> planes;
        const std::uint64_t kind_count = 1 + state % 8;
        for (std::uint64_t kind = 1; kind <= kind_count; ++kind) {
            state = state * UINT64_C(6364136223846793005) + 1;
            const std::uint64_t mirror_count = state % 5;
            const std::uint64_t primary_identity = kind * 100;
            planes.push_back(primary(kind, primary_identity));
            for (std::uint64_t index = 1; index <= mirror_count; ++index) {
                planes.push_back(mirror(
                    kind, primary_identity + index, primary_identity));
            }
        }
        assert(validate_atom_plane_directory_v1(
                   {planes.data(), planes.size()})
                   .valid());
        state = state * UINT64_C(6364136223846793005) + 1;
    }
}

} // namespace

int main() {
    test_primary_and_explicit_mirrors();
    test_deterministic_rejections();
    test_randomized_sorted_directories();
    return 0;
}
