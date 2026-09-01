#pragma once

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_persistent_identity_schema_version_v1 = 1;

// This is the CellShard-side representation of Cellerator's namespace-
// qualified persistent identity. The producer namespace and local identity
// remain separate exact values. Neither field is a content digest, pointer,
// device ordinal, registry slot, nor legacy process-local counter.
struct atom_persistent_identity_v1 {
    std::uint64_t producer_namespace = 0;
    std::uint64_t local_identity = 0;
};

// Persistence is field-oriented. Callers encode these fields in the declared
// artifact byte order rather than copying native struct bytes.
struct atom_persistent_identity_record_v1 {
    std::uint32_t schema_version =
        atom_persistent_identity_schema_version_v1;
    std::uint32_t record_bytes = sizeof(atom_persistent_identity_record_v1);
    atom_persistent_identity_v1 identity{};
};

enum class atom_persistent_identity_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    missing_producer_namespace,
    missing_local_identity,
    null_destination,
};

struct atom_persistent_identity_validation_v1 {
    atom_persistent_identity_validation_code_v1 code =
        atom_persistent_identity_validation_code_v1::valid;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_persistent_identity_validation_code_v1::valid;
    }
};

static_assert(sizeof(atom_persistent_identity_v1) == 16,
              "persistent identity must remain two explicit u64 fields");
static_assert(sizeof(atom_persistent_identity_record_v1) == 24,
              "persistent identity record layout is part of atom ABI v1");
static_assert(std::is_standard_layout<atom_persistent_identity_v1>::value,
              "persistent identity must remain standard layout");
static_assert(std::is_trivially_copyable<atom_persistent_identity_v1>::value,
              "persistent identity must remain trivially copyable");
static_assert(
    std::is_standard_layout<atom_persistent_identity_record_v1>::value,
    "persistent identity record must remain standard layout");
static_assert(
    std::is_trivially_copyable<atom_persistent_identity_record_v1>::value,
    "persistent identity record must remain trivially copyable");

[[nodiscard]] constexpr bool operator==(
    atom_persistent_identity_v1 lhs,
    atom_persistent_identity_v1 rhs) noexcept {
    return lhs.producer_namespace == rhs.producer_namespace
        && lhs.local_identity == rhs.local_identity;
}

[[nodiscard]] constexpr bool operator!=(
    atom_persistent_identity_v1 lhs,
    atom_persistent_identity_v1 rhs) noexcept {
    return !(lhs == rhs);
}

[[nodiscard]] constexpr bool atom_persistent_identity_less_v1(
    atom_persistent_identity_v1 lhs,
    atom_persistent_identity_v1 rhs) noexcept {
    return lhs.producer_namespace < rhs.producer_namespace
        || (lhs.producer_namespace == rhs.producer_namespace
            && lhs.local_identity < rhs.local_identity);
}

[[nodiscard]] constexpr atom_persistent_identity_validation_v1
validate_atom_persistent_identity_v1(
    atom_persistent_identity_v1 identity) noexcept {
    if (identity.producer_namespace == 0) {
        return {atom_persistent_identity_validation_code_v1::
                    missing_producer_namespace};
    }
    if (identity.local_identity == 0) {
        return {atom_persistent_identity_validation_code_v1::
                    missing_local_identity};
    }
    return {};
}

[[nodiscard]] constexpr atom_persistent_identity_validation_v1
validate_atom_persistent_identity_record_v1(
    atom_persistent_identity_record_v1 record) noexcept {
    if (record.schema_version
        != atom_persistent_identity_schema_version_v1) {
        return {atom_persistent_identity_validation_code_v1::
                    unsupported_schema};
    }
    if (record.record_bytes != sizeof(atom_persistent_identity_record_v1)) {
        return {atom_persistent_identity_validation_code_v1::
                    invalid_record_bytes};
    }
    return validate_atom_persistent_identity_v1(record.identity);
}

// Cellerator's v1 bridge has the same explicit schema/version/identity fields.
// Taking scalar fields keeps CellShard independent of Cellerator headers while
// still making every source bit visible. This cold O(1) adapter allocates no
// storage and deliberately exposes no pointer- or hash-based identity input.
[[nodiscard]] constexpr atom_persistent_identity_validation_v1
adapt_cellerator_persistent_identity_v1(
    std::uint32_t source_schema_version,
    std::uint32_t source_record_bytes,
    std::uint64_t producer_namespace,
    std::uint64_t local_identity,
    atom_persistent_identity_record_v1 *destination) noexcept {
    if (destination == nullptr) {
        return {atom_persistent_identity_validation_code_v1::null_destination};
    }
    *destination = {};
    if (source_schema_version
        != atom_persistent_identity_schema_version_v1) {
        return {atom_persistent_identity_validation_code_v1::
                    unsupported_schema};
    }
    if (source_record_bytes != sizeof(atom_persistent_identity_record_v1)) {
        return {atom_persistent_identity_validation_code_v1::
                    invalid_record_bytes};
    }
    destination->identity = {producer_namespace, local_identity};
    return validate_atom_persistent_identity_v1(destination->identity);
}

} // namespace cellshard::compiler::atom
