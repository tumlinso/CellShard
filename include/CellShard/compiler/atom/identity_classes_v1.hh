#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>
#include <CellShard/identity/digest.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_identity_classes_contract_version_v1 = 1;

// What the atom means. This identity is stable across materializations,
// encodings, replicas, and runtime sessions.
struct atom_semantic_family_id_v1 {
    atom_persistent_identity_v1 persistent{};
};

// Which exact bytes are present. A digest is evidence about payload bytes, not
// semantic equivalence and not the identity of a physical view.
struct atom_content_id_v1 {
    content_digest digest{};
};

// Which persistent physical view was produced from a semantic family. This is
// intentionally independent of byte content so a replacement/correction is
// not silently treated as the same bytes.
struct atom_materialization_id_v1 {
    atom_persistent_identity_v1 persistent{};
};

// Which persistent encoding/location copy of a materialization is addressed.
// Operational location details remain outside this portable identity.
struct atom_replica_id_v1 {
    atom_persistent_identity_v1 persistent{};
};

// Runtime-only identity. It is valid solely within one nonzero execution
// session and must never be persisted or used as a semantic/content key.
struct atom_resident_id_v1 {
    std::uint64_t session_identity = 0;
    std::uint64_t local_identity = 0;
};

// A binding makes all five identities visible at a boundary without deriving
// one class from another. Changing one field never implicitly changes another.
struct atom_identity_binding_v1 {
    atom_semantic_family_id_v1 semantic_family{};
    atom_content_id_v1 content{};
    atom_materialization_id_v1 materialization{};
    atom_replica_id_v1 replica{};
    atom_resident_id_v1 resident{};
};

enum class atom_identity_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_semantic_family,
    invalid_content_digest,
    missing_content_digest,
    invalid_materialization,
    invalid_replica,
    missing_resident_session,
    missing_resident_local_identity,
};

enum class atom_identity_field_v1 : std::uint32_t {
    none = 0,
    semantic_family,
    content,
    materialization,
    replica,
    resident,
};

struct atom_identity_validation_v1 {
    atom_identity_validation_code_v1 code =
        atom_identity_validation_code_v1::valid;
    atom_identity_field_v1 field = atom_identity_field_v1::none;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_identity_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<atom_semantic_family_id_v1>::value);
static_assert(std::is_trivially_copyable<atom_semantic_family_id_v1>::value);
static_assert(std::is_standard_layout<atom_content_id_v1>::value);
static_assert(std::is_trivially_copyable<atom_content_id_v1>::value);
static_assert(std::is_standard_layout<atom_materialization_id_v1>::value);
static_assert(std::is_trivially_copyable<atom_materialization_id_v1>::value);
static_assert(std::is_standard_layout<atom_replica_id_v1>::value);
static_assert(std::is_trivially_copyable<atom_replica_id_v1>::value);
static_assert(std::is_standard_layout<atom_resident_id_v1>::value);
static_assert(std::is_trivially_copyable<atom_resident_id_v1>::value);
static_assert(std::is_standard_layout<atom_identity_binding_v1>::value);
static_assert(std::is_trivially_copyable<atom_identity_binding_v1>::value);
static_assert(!std::is_convertible<atom_semantic_family_id_v1,
                                   atom_materialization_id_v1>::value,
              "semantic and materialization identities must not mix");
static_assert(!std::is_convertible<atom_materialization_id_v1,
                                   atom_replica_id_v1>::value,
              "materialization and replica identities must not mix");
static_assert(!std::is_convertible<atom_replica_id_v1,
                                   atom_resident_id_v1>::value,
              "persistent and resident identities must not mix");

[[nodiscard]] constexpr bool operator==(
    atom_semantic_family_id_v1 lhs,
    atom_semantic_family_id_v1 rhs) noexcept {
    return lhs.persistent == rhs.persistent;
}

[[nodiscard]] constexpr bool operator==(
    atom_content_id_v1 lhs, atom_content_id_v1 rhs) noexcept {
    return lhs.digest == rhs.digest;
}

[[nodiscard]] constexpr bool operator==(
    atom_materialization_id_v1 lhs,
    atom_materialization_id_v1 rhs) noexcept {
    return lhs.persistent == rhs.persistent;
}

[[nodiscard]] constexpr bool operator==(
    atom_replica_id_v1 lhs, atom_replica_id_v1 rhs) noexcept {
    return lhs.persistent == rhs.persistent;
}

[[nodiscard]] constexpr bool operator==(
    atom_resident_id_v1 lhs, atom_resident_id_v1 rhs) noexcept {
    return lhs.session_identity == rhs.session_identity
        && lhs.local_identity == rhs.local_identity;
}

[[nodiscard]] constexpr atom_identity_validation_v1
validate_atom_identity_binding_v1(
    const atom_identity_binding_v1 &binding) noexcept {
    if (!validate_atom_persistent_identity_v1(
             binding.semantic_family.persistent)
             .valid()) {
        return {atom_identity_validation_code_v1::invalid_semantic_family,
                atom_identity_field_v1::semantic_family};
    }
    if (!valid_content_digest(binding.content.digest)) {
        return {atom_identity_validation_code_v1::invalid_content_digest,
                atom_identity_field_v1::content};
    }
    if (binding.content.digest.algorithm == digest_algorithm::none) {
        return {atom_identity_validation_code_v1::missing_content_digest,
                atom_identity_field_v1::content};
    }
    if (!validate_atom_persistent_identity_v1(
             binding.materialization.persistent)
             .valid()) {
        return {atom_identity_validation_code_v1::invalid_materialization,
                atom_identity_field_v1::materialization};
    }
    if (!validate_atom_persistent_identity_v1(binding.replica.persistent)
             .valid()) {
        return {atom_identity_validation_code_v1::invalid_replica,
                atom_identity_field_v1::replica};
    }
    if (binding.resident.session_identity == 0) {
        return {atom_identity_validation_code_v1::missing_resident_session,
                atom_identity_field_v1::resident};
    }
    if (binding.resident.local_identity == 0) {
        return {
            atom_identity_validation_code_v1::missing_resident_local_identity,
            atom_identity_field_v1::resident};
    }
    return {};
}

} // namespace cellshard::compiler::atom
