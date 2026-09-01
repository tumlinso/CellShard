#include <CellShard/compiler/atom/identity_classes_v1.hh>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

using namespace cellshard;
using namespace cellshard::compiler::atom;

content_digest digest_with(std::uint64_t value) {
    content_digest digest{};
    digest.algorithm = digest_algorithm::legacy_fnv1a64;
    digest.used_bytes = sizeof(value);
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        digest.bytes[index] = static_cast<std::byte>(
            (value >> (index * 8)) & UINT64_C(0xff));
    }
    return digest;
}

atom_identity_binding_v1 valid_binding() {
    return {
        {{11, 101}},
        {digest_with(UINT64_C(0x0123456789abcdef))},
        {{12, 202}},
        {{13, 303}},
        {404, 505},
    };
}

void test_all_classes_are_independent() {
    const auto base = valid_binding();
    assert(validate_atom_identity_binding_v1(base).valid());

    auto changed = base;
    changed.semantic_family.persistent.local_identity += 1;
    assert(changed.content == base.content);
    assert(changed.materialization == base.materialization);
    assert(changed.replica == base.replica);
    assert(changed.resident == base.resident);

    changed = base;
    changed.content.digest = digest_with(UINT64_C(0xfedcba9876543210));
    assert(changed.semantic_family == base.semantic_family);
    assert(changed.materialization == base.materialization);
    assert(changed.replica == base.replica);
    assert(changed.resident == base.resident);

    changed = base;
    changed.materialization.persistent.local_identity += 1;
    assert(changed.semantic_family == base.semantic_family);
    assert(changed.content == base.content);
    assert(changed.replica == base.replica);
    assert(changed.resident == base.resident);

    changed = base;
    changed.replica.persistent.local_identity += 1;
    assert(changed.semantic_family == base.semantic_family);
    assert(changed.content == base.content);
    assert(changed.materialization == base.materialization);
    assert(changed.resident == base.resident);

    changed = base;
    changed.resident.local_identity += 1;
    assert(changed.semantic_family == base.semantic_family);
    assert(changed.content == base.content);
    assert(changed.materialization == base.materialization);
    assert(changed.replica == base.replica);
}

void test_resident_identity_is_session_scoped() {
    const atom_resident_id_v1 first{7, 19};
    const atom_resident_id_v1 same_local_other_session{8, 19};
    assert(!(first == same_local_other_session));
    assert((first == atom_resident_id_v1{7, 19}));
}

void test_deterministic_rejections() {
    auto binding = valid_binding();
    binding.semantic_family = {};
    auto result = validate_atom_identity_binding_v1(binding);
    assert(result.code
           == atom_identity_validation_code_v1::invalid_semantic_family);
    assert(result.field == atom_identity_field_v1::semantic_family);

    binding = valid_binding();
    binding.content = {};
    result = validate_atom_identity_binding_v1(binding);
    assert(result.code
           == atom_identity_validation_code_v1::missing_content_digest);

    binding = valid_binding();
    binding.content.digest.used_bytes = 9;
    result = validate_atom_identity_binding_v1(binding);
    assert(result.code
           == atom_identity_validation_code_v1::invalid_content_digest);

    binding = valid_binding();
    binding.materialization = {};
    result = validate_atom_identity_binding_v1(binding);
    assert(result.code
           == atom_identity_validation_code_v1::invalid_materialization);

    binding = valid_binding();
    binding.replica = {};
    result = validate_atom_identity_binding_v1(binding);
    assert(result.code == atom_identity_validation_code_v1::invalid_replica);

    binding = valid_binding();
    binding.resident.session_identity = 0;
    result = validate_atom_identity_binding_v1(binding);
    assert(result.code
           == atom_identity_validation_code_v1::missing_resident_session);

    binding = valid_binding();
    binding.resident.local_identity = 0;
    result = validate_atom_identity_binding_v1(binding);
    assert(result.code
           == atom_identity_validation_code_v1::
                  missing_resident_local_identity);
}

std::uint64_t next_random(std::uint64_t *state) {
    *state = *state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return *state;
}

void test_randomized_valid_bindings() {
    std::uint64_t state = UINT64_C(0xa05c0de);
    for (std::size_t iteration = 0; iteration < 10000; ++iteration) {
        auto binding = valid_binding();
        binding.semantic_family.persistent = {
            next_random(&state) | 1, next_random(&state) | 1};
        binding.content.digest = digest_with(next_random(&state));
        binding.materialization.persistent = {
            next_random(&state) | 1, next_random(&state) | 1};
        binding.replica.persistent = {
            next_random(&state) | 1, next_random(&state) | 1};
        binding.resident = {next_random(&state) | 1,
                            next_random(&state) | 1};
        assert(validate_atom_identity_binding_v1(binding).valid());
    }
}

} // namespace

int main() {
    test_all_classes_are_independent();
    test_resident_identity_is_session_scoped();
    test_deterministic_rejections();
    test_randomized_valid_bindings();
    return 0;
}
