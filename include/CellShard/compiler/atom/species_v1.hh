#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_species_contract_version_v1 = 1;
inline constexpr std::uint64_t core_atom_species_provider_v1 = 1;

// This enumeration names only the stable core provider's species. The public
// species ID below remains provider-qualified so extensions never require
// editing this list or a central registry source file.
enum class core_atom_species_v1 : std::uint64_t {
    identity_spine = 1,
    support_signature,
    co_support_source,
    destination_convergence,
    divergence,
    motif,
    program,
    state_neighborhood,
    trajectory_prefix,
    trajectory_branch,
    trajectory_delta,
    multimodal,
    sequence,
    halo,
    stable_structure,
    stable_value,
    operation_polymorphic,
    segment,
    transform,
    partial,
    executable,
    superatom,
};

inline constexpr std::size_t core_atom_species_count_v1 = 22;

struct atom_species_id_v1 {
    std::uint64_t provider_namespace = 0;
    std::uint64_t local_id = 0;
};

struct atom_species_descriptor_v1 {
    atom_species_id_v1 id{};
    const char *stable_name = nullptr;
    std::size_t stable_name_size = 0;
};

struct atom_species_registry_view_v1 {
    const atom_species_descriptor_v1 *species = nullptr;
    std::size_t species_count = 0;
};

enum class atom_species_validation_code_v1 : std::uint32_t {
    valid = 0,
    empty_registry,
    null_species,
    missing_core_species,
    invalid_id,
    unsorted_or_duplicate_id,
    invalid_name,
    core_name_mismatch,
};

struct atom_species_validation_v1 {
    atom_species_validation_code_v1 code =
        atom_species_validation_code_v1::valid;
    std::size_t index = 0;
    atom_species_id_v1 id{};

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_species_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<atom_species_id_v1>::value,
              "species IDs must remain standard-layout values");
static_assert(std::is_trivially_copyable<atom_species_id_v1>::value,
              "species IDs must remain trivially copyable");
static_assert(std::is_standard_layout<atom_species_descriptor_v1>::value,
              "species descriptors must remain standard-layout values");
static_assert(std::is_trivially_copyable<atom_species_descriptor_v1>::value,
              "species descriptors must remain trivially copyable");
static_assert(std::is_standard_layout<atom_species_registry_view_v1>::value,
              "species registries must remain standard-layout values");
static_assert(std::is_trivially_copyable<atom_species_registry_view_v1>::value,
              "species registries must remain trivially copyable");
static_assert(offsetof(atom_species_registry_view_v1, species) == 0,
              "species registries must remain pointer-first");

[[nodiscard]] constexpr bool operator==(
    atom_species_id_v1 lhs, atom_species_id_v1 rhs) noexcept {
    return lhs.provider_namespace == rhs.provider_namespace
        && lhs.local_id == rhs.local_id;
}

[[nodiscard]] constexpr bool operator!=(
    atom_species_id_v1 lhs, atom_species_id_v1 rhs) noexcept {
    return !(lhs == rhs);
}

[[nodiscard]] constexpr bool atom_species_id_less_v1(
    atom_species_id_v1 lhs, atom_species_id_v1 rhs) noexcept {
    return lhs.provider_namespace < rhs.provider_namespace
        || (lhs.provider_namespace == rhs.provider_namespace
            && lhs.local_id < rhs.local_id);
}

[[nodiscard]] constexpr bool valid_atom_species_id_v1(
    atom_species_id_v1 id) noexcept {
    return id.provider_namespace != 0 && id.local_id != 0;
}

[[nodiscard]] constexpr bool valid_core_atom_species_v1(
    core_atom_species_v1 species) noexcept {
    const auto value = static_cast<std::uint64_t>(species);
    return value >= 1 && value <= core_atom_species_count_v1;
}

[[nodiscard]] constexpr atom_species_id_v1 core_atom_species_id_v1(
    core_atom_species_v1 species) noexcept {
    return valid_core_atom_species_v1(species)
        ? atom_species_id_v1{core_atom_species_provider_v1,
                             static_cast<std::uint64_t>(species)}
        : atom_species_id_v1{};
}

[[nodiscard]] constexpr const char *core_atom_species_name_v1(
    core_atom_species_v1 species) noexcept {
    switch (species) {
    case core_atom_species_v1::identity_spine:
        return "identity_spine";
    case core_atom_species_v1::support_signature:
        return "support_signature";
    case core_atom_species_v1::co_support_source:
        return "co_support_source";
    case core_atom_species_v1::destination_convergence:
        return "destination_convergence";
    case core_atom_species_v1::divergence:
        return "divergence";
    case core_atom_species_v1::motif:
        return "motif";
    case core_atom_species_v1::program:
        return "program";
    case core_atom_species_v1::state_neighborhood:
        return "state_neighborhood";
    case core_atom_species_v1::trajectory_prefix:
        return "trajectory_prefix";
    case core_atom_species_v1::trajectory_branch:
        return "trajectory_branch";
    case core_atom_species_v1::trajectory_delta:
        return "trajectory_delta";
    case core_atom_species_v1::multimodal:
        return "multimodal";
    case core_atom_species_v1::sequence:
        return "sequence";
    case core_atom_species_v1::halo:
        return "halo";
    case core_atom_species_v1::stable_structure:
        return "stable_structure";
    case core_atom_species_v1::stable_value:
        return "stable_value";
    case core_atom_species_v1::operation_polymorphic:
        return "operation_polymorphic";
    case core_atom_species_v1::segment:
        return "segment";
    case core_atom_species_v1::transform:
        return "transform";
    case core_atom_species_v1::partial:
        return "partial";
    case core_atom_species_v1::executable:
        return "executable";
    case core_atom_species_v1::superatom:
        return "superatom";
    }
    return "invalid";
}

[[nodiscard]] constexpr std::size_t atom_species_name_size_v1(
    const char *name) noexcept {
    if (name == nullptr) {
        return 0;
    }
    std::size_t size = 0;
    while (name[size] != '\0') {
        ++size;
    }
    return size;
}

[[nodiscard]] constexpr atom_species_descriptor_v1 core_atom_species_descriptor_v1(
    core_atom_species_v1 species) noexcept {
    const auto *name = core_atom_species_name_v1(species);
    return {core_atom_species_id_v1(species), name,
            atom_species_name_size_v1(name)};
}

namespace detail {

[[nodiscard]] constexpr bool valid_atom_species_name_v1(
    const char *name, std::size_t size) noexcept {
    if (name == nullptr || size == 0) {
        return false;
    }
    for (std::size_t index = 0; index < size; ++index) {
        if (name[index] == '\0') {
            return false;
        }
    }
    return true;
}

[[nodiscard]] constexpr bool atom_species_name_equal_v1(
    const char *lhs, std::size_t lhs_size,
    const char *rhs, std::size_t rhs_size) noexcept {
    if (lhs == nullptr || rhs == nullptr || lhs_size != rhs_size) {
        return false;
    }
    for (std::size_t index = 0; index < lhs_size; ++index) {
        if (lhs[index] != rhs[index]) {
            return false;
        }
    }
    return true;
}

} // namespace detail

// Registries are sorted by provider namespace and local ID. The stable core
// occupies the first 22 entries; source-linked providers append sorted
// extensions with provider namespaces greater than the core namespace.
// Validation is O(species_count + total_name_bytes) time and O(1) storage.
[[nodiscard]] constexpr atom_species_validation_v1 validate_atom_species_registry_v1(
    atom_species_registry_view_v1 registry) noexcept {
    if (registry.species_count == 0) {
        return {atom_species_validation_code_v1::empty_registry, 0, {}};
    }
    if (registry.species == nullptr) {
        return {atom_species_validation_code_v1::null_species, 0, {}};
    }
    if (registry.species_count < core_atom_species_count_v1) {
        return {atom_species_validation_code_v1::missing_core_species,
                registry.species_count, {}};
    }

    atom_species_id_v1 previous{};
    for (std::size_t index = 0; index < registry.species_count; ++index) {
        const auto &descriptor = registry.species[index];
        if (!valid_atom_species_id_v1(descriptor.id)
            || descriptor.id.provider_namespace < core_atom_species_provider_v1) {
            return {atom_species_validation_code_v1::invalid_id, index,
                    descriptor.id};
        }
        if (index != 0 && !atom_species_id_less_v1(previous, descriptor.id)) {
            return {atom_species_validation_code_v1::unsorted_or_duplicate_id,
                    index, descriptor.id};
        }
        if (!detail::valid_atom_species_name_v1(
                descriptor.stable_name, descriptor.stable_name_size)) {
            return {atom_species_validation_code_v1::invalid_name, index,
                    descriptor.id};
        }

        if (index < core_atom_species_count_v1) {
            const auto core = static_cast<core_atom_species_v1>(index + 1);
            const auto expected = core_atom_species_descriptor_v1(core);
            if (descriptor.id != expected.id) {
                return {atom_species_validation_code_v1::missing_core_species,
                        index, descriptor.id};
            }
            if (!detail::atom_species_name_equal_v1(
                    descriptor.stable_name, descriptor.stable_name_size,
                    expected.stable_name, expected.stable_name_size)) {
                return {atom_species_validation_code_v1::core_name_mismatch,
                        index, descriptor.id};
            }
        } else if (descriptor.id.provider_namespace
                   == core_atom_species_provider_v1) {
            return {atom_species_validation_code_v1::invalid_id, index,
                    descriptor.id};
        }
        previous = descriptor.id;
    }
    return {atom_species_validation_code_v1::valid, registry.species_count, {}};
}

// Binary lookup is valid only for a successfully validated sorted registry.
[[nodiscard]] constexpr const atom_species_descriptor_v1 *find_atom_species_v1(
    atom_species_registry_view_v1 registry, atom_species_id_v1 id) noexcept {
    std::size_t begin = 0;
    std::size_t end = registry.species_count;
    while (begin < end) {
        const auto middle = begin + (end - begin) / 2;
        const auto candidate = registry.species[middle].id;
        if (candidate == id) {
            return registry.species + middle;
        }
        if (atom_species_id_less_v1(candidate, id)) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return nullptr;
}

} // namespace cellshard::compiler::atom
