#include <CellShard/compiler/atom/persistent_identity_v1.hh>

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

using namespace cellshard::compiler::atom;

void test_exact_adaptation() {
    atom_persistent_identity_record_v1 destination{};
    const auto result = adapt_cellerator_persistent_identity_v1(
        1, 24, UINT64_C(0x1020304050607080),
        UINT64_C(0x8877665544332211), &destination);
    assert(result.valid());
    assert(validate_atom_persistent_identity_record_v1(destination).valid());
    assert(destination.identity.producer_namespace
           == UINT64_C(0x1020304050607080));
    assert(destination.identity.local_identity
           == UINT64_C(0x8877665544332211));

    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    assert(adapt_cellerator_persistent_identity_v1(
               1, 24, maximum, maximum, &destination)
               .valid());
    assert((destination.identity
            == atom_persistent_identity_v1{maximum, maximum}));
}

void test_namespace_is_not_folded_into_local_identity() {
    const atom_persistent_identity_v1 first{7, 41};
    const atom_persistent_identity_v1 other_namespace{8, 41};
    const atom_persistent_identity_v1 other_local{7, 42};
    assert(first != other_namespace);
    assert(first != other_local);
    assert(atom_persistent_identity_less_v1(first, other_namespace));
    assert(atom_persistent_identity_less_v1(first, other_local));
}

void test_deterministic_rejections() {
    atom_persistent_identity_record_v1 destination{};
    auto result = adapt_cellerator_persistent_identity_v1(
        2, 24, 7, 41, &destination);
    assert(result.code
           == atom_persistent_identity_validation_code_v1::unsupported_schema);
    assert(destination.identity == atom_persistent_identity_v1{});

    result = adapt_cellerator_persistent_identity_v1(
        1, 16, 7, 41, &destination);
    assert(result.code
           == atom_persistent_identity_validation_code_v1::
                  invalid_record_bytes);
    assert(destination.identity == atom_persistent_identity_v1{});

    result = adapt_cellerator_persistent_identity_v1(
        1, 24, 0, 41, &destination);
    assert(result.code
           == atom_persistent_identity_validation_code_v1::
                  missing_producer_namespace);

    result = adapt_cellerator_persistent_identity_v1(
        1, 24, 7, 0, &destination);
    assert(result.code
           == atom_persistent_identity_validation_code_v1::
                  missing_local_identity);

    result = adapt_cellerator_persistent_identity_v1(
        1, 24, 7, 41, nullptr);
    assert(result.code
           == atom_persistent_identity_validation_code_v1::null_destination);

    destination = {};
    destination.schema_version = 2;
    assert(validate_atom_persistent_identity_record_v1(destination).code
           == atom_persistent_identity_validation_code_v1::unsupported_schema);
    destination = {};
    destination.record_bytes = 16;
    assert(validate_atom_persistent_identity_record_v1(destination).code
           == atom_persistent_identity_validation_code_v1::
                  invalid_record_bytes);
}

std::uint64_t next_random(std::uint64_t *state) {
    *state = *state * UINT64_C(2862933555777941757) + UINT64_C(3037000493);
    return *state;
}

void test_randomized_bit_preservation() {
    std::uint64_t state = UINT64_C(0xa04c0de);
    for (std::size_t iteration = 0; iteration < 10000; ++iteration) {
        auto producer_namespace = next_random(&state);
        auto local_identity = next_random(&state);
        producer_namespace |= 1;
        local_identity |= 1;
        atom_persistent_identity_record_v1 destination{};
        assert(adapt_cellerator_persistent_identity_v1(
                   1, 24, producer_namespace, local_identity, &destination)
                   .valid());
        assert(destination.identity.producer_namespace == producer_namespace);
        assert(destination.identity.local_identity == local_identity);
    }
}

} // namespace

int main() {
    static_assert(std::is_standard_layout<atom_persistent_identity_v1>::value,
                  "identity must be standard layout");
    static_assert(
        std::is_trivially_copyable<atom_persistent_identity_record_v1>::value,
        "record must be trivially copyable");
    test_exact_adaptation();
    test_namespace_is_not_folded_into_local_identity();
    test_deterministic_rejections();
    test_randomized_bit_preservation();
    return 0;
}
