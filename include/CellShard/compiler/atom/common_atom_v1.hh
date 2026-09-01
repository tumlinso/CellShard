#pragma once

#include <CellShard/compiler/atom/atomicity_v1.hh>
#include <CellShard/compiler/atom/dependency_invalidation_plane_v1.hh>
#include <CellShard/compiler/atom/evidence_plane_v1.hh>
#include <CellShard/compiler/atom/executable_affordance_plane_v1.hh>
#include <CellShard/compiler/atom/identity_classes_v1.hh>
#include <CellShard/compiler/atom/level_v1.hh>
#include <CellShard/compiler/atom/logical_coverage_v1.hh>
#include <CellShard/compiler/atom/overlap_role_v1.hh>
#include <CellShard/compiler/atom/plane_directory_v1.hh>
#include <CellShard/compiler/atom/port_v1.hh>
#include <CellShard/compiler/atom/species_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t common_atom_schema_version_v1 = 1;

struct atom_parent_ref_v1 {
    atom_semantic_family_id_v1 parent_semantic_family{};
    atom_persistent_identity_v1 relation_identity{};
    std::uint64_t parent_generation = 0;
};

// The common envelope is a nonowning pointer-first view. Every constituent
// retains its own identity/generation and may outlive or invalidate separately.
struct common_atom_view_v1 {
    const atom_level_v1 *levels = nullptr;
    std::size_t level_count = 0;
    const atom_parent_ref_v1 *parents = nullptr;
    std::uint64_t parent_count = 0;
    atom_identity_binding_v1 identities{};
    atom_species_id_v1 species{};
    atom_atomicity_set_v1 atomicity{};
    atom_logical_coverage_ref_v1 exact_coverage{};
    atom_port_table_view_v1 ports{};
    atom_plane_directory_view_v1 planes{};
    atom_dependency_invalidation_plane_v1 dependencies{};
    atom_evidence_plane_v1 evidence{};
    atom_executable_affordance_plane_v1 affordances{};
    atom_overlap_role_table_v1 overlap_roles{};
    atom_persistent_identity_v1 lineage_identity{};
    std::uint64_t lineage_generation = 0;
};

enum class common_atom_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_levels,
    invalid_species,
    invalid_atomicity,
    invalid_identities,
    invalid_exact_coverage,
    invalid_ports,
    invalid_planes,
    inconsistent_parent_pointer,
    invalid_parent_identity,
    invalid_parent_relation,
    missing_parent_generation,
    self_parent,
    unordered_or_duplicate_parent,
    invalid_dependencies,
    invalid_evidence,
    invalid_affordances,
    affordance_port_table_mismatch,
    invalid_overlap_roles,
    invalid_lineage_identity,
    missing_lineage_generation,
};

struct common_atom_validation_v1 {
    common_atom_validation_code_v1 code =
        common_atom_validation_code_v1::valid;
    std::uint64_t index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == common_atom_validation_code_v1::valid;
    }
};

static_assert(offsetof(common_atom_view_v1, levels) == 0,
              "common atom views must remain pointer-first");
static_assert(std::is_standard_layout<atom_parent_ref_v1>::value);
static_assert(std::is_trivially_copyable<atom_parent_ref_v1>::value);
static_assert(std::is_standard_layout<common_atom_view_v1>::value);
static_assert(std::is_trivially_copyable<common_atom_view_v1>::value);

[[nodiscard]] constexpr bool atom_parent_ref_less_v1(
    const atom_parent_ref_v1 &lhs,
    const atom_parent_ref_v1 &rhs) noexcept {
    return atom_persistent_identity_less_v1(
               lhs.parent_semantic_family.persistent,
               rhs.parent_semantic_family.persistent)
        || (lhs.parent_semantic_family == rhs.parent_semantic_family
            && atom_persistent_identity_less_v1(
                lhs.relation_identity, rhs.relation_identity));
}

// Validation is linear except for the bounded binary searches performed by the
// affordance plane. It allocates nothing and never parses plane/evidence bytes.
[[nodiscard]] inline common_atom_validation_v1 validate_common_atom_v1(
    const common_atom_view_v1 &atom,
    std::uint32_t coverage_source_validation) noexcept {
    const auto level_result = validate_atom_level_path_v1(
        {atom.levels, atom.level_count});
    if (!level_result.valid()) {
        return {common_atom_validation_code_v1::invalid_levels,
                level_result.index,
                static_cast<std::uint32_t>(level_result.code)};
    }
    if (!valid_atom_species_id_v1(atom.species)) {
        return {common_atom_validation_code_v1::invalid_species, 0, 0};
    }
    const auto atomicity_result = validate_atom_atomicity_v1(atom.atomicity);
    if (!atomicity_result.valid()) {
        return {common_atom_validation_code_v1::invalid_atomicity,
                atomicity_result.unknown_capabilities,
                static_cast<std::uint32_t>(atomicity_result.code)};
    }
    const auto identity_result = validate_atom_identity_binding_v1(
        atom.identities);
    if (!identity_result.valid()) {
        return {common_atom_validation_code_v1::invalid_identities,
                static_cast<std::uint64_t>(identity_result.field),
                static_cast<std::uint32_t>(identity_result.code)};
    }
    const auto coverage_result = validate_atom_logical_coverage_ref_v1(
        atom.exact_coverage, coverage_source_validation);
    if (!coverage_result.valid()) {
        return {common_atom_validation_code_v1::invalid_exact_coverage,
                0, static_cast<std::uint32_t>(coverage_result.code)};
    }
    const auto port_result = validate_atom_port_table_v1(atom.ports);
    if (!port_result.valid()) {
        return {common_atom_validation_code_v1::invalid_ports,
                port_result.index,
                static_cast<std::uint32_t>(port_result.code)};
    }
    const auto plane_result = validate_atom_plane_directory_v1(atom.planes);
    if (!plane_result.valid()) {
        return {common_atom_validation_code_v1::invalid_planes,
                plane_result.index,
                static_cast<std::uint32_t>(plane_result.code)};
    }
    if ((atom.parent_count == 0) != (atom.parents == nullptr)) {
        return {common_atom_validation_code_v1::inconsistent_parent_pointer,
                0, 0};
    }
    for (std::uint64_t index = 0; index < atom.parent_count; ++index) {
        const auto &parent = atom.parents[index];
        if (!validate_atom_persistent_identity_v1(
                 parent.parent_semantic_family.persistent)
                 .valid()) {
            return {common_atom_validation_code_v1::invalid_parent_identity,
                    index, 0};
        }
        if (!validate_atom_persistent_identity_v1(parent.relation_identity)
                 .valid()) {
            return {common_atom_validation_code_v1::invalid_parent_relation,
                    index, 0};
        }
        if (parent.parent_generation == 0) {
            return {common_atom_validation_code_v1::missing_parent_generation,
                    index, 0};
        }
        if (parent.parent_semantic_family == atom.identities.semantic_family) {
            return {common_atom_validation_code_v1::self_parent, index, 0};
        }
        if (index != 0
            && !atom_parent_ref_less_v1(atom.parents[index - 1], parent)) {
            return {common_atom_validation_code_v1::
                        unordered_or_duplicate_parent,
                    index, 0};
        }
    }
    const auto dependency_result =
        validate_atom_dependency_invalidation_plane_v1(atom.dependencies);
    if (!dependency_result.valid()) {
        return {common_atom_validation_code_v1::invalid_dependencies,
                dependency_result.index,
                static_cast<std::uint32_t>(dependency_result.code)};
    }
    const auto evidence_result = validate_atom_evidence_plane_v1(atom.evidence);
    if (!evidence_result.valid()) {
        return {common_atom_validation_code_v1::invalid_evidence,
                evidence_result.index,
                static_cast<std::uint32_t>(evidence_result.code)};
    }
    const auto affordance_result =
        validate_atom_executable_affordance_plane_v1(atom.affordances);
    if (!affordance_result.valid()) {
        return {common_atom_validation_code_v1::invalid_affordances,
                affordance_result.affordance_index,
                static_cast<std::uint32_t>(affordance_result.code)};
    }
    if (atom.affordances.ports.ports != atom.ports.ports
        || atom.affordances.ports.port_count != atom.ports.port_count) {
        return {common_atom_validation_code_v1::
                    affordance_port_table_mismatch,
                0, 0};
    }
    const auto overlap_result = validate_atom_overlap_role_table_v1(
        atom.overlap_roles);
    if (!overlap_result.valid()) {
        return {common_atom_validation_code_v1::invalid_overlap_roles,
                overlap_result.index,
                static_cast<std::uint32_t>(overlap_result.code)};
    }
    if (!validate_atom_persistent_identity_v1(atom.lineage_identity).valid()) {
        return {common_atom_validation_code_v1::invalid_lineage_identity,
                0, 0};
    }
    if (atom.lineage_generation == 0) {
        return {common_atom_validation_code_v1::missing_lineage_generation,
                0, 0};
    }
    return {common_atom_validation_code_v1::valid, atom.parent_count, 0};
}

enum class common_atom_build_code_v1 : std::uint32_t {
    built = 0,
    invalid_input,
    capacity_overflow,
    allocation_failure,
    invalid_built_view,
};

struct common_atom_build_result_v1 {
    common_atom_build_code_v1 code = common_atom_build_code_v1::built;
    common_atom_validation_v1 validation{};

    [[nodiscard]] constexpr bool built() const noexcept {
        return code == common_atom_build_code_v1::built;
    }
};

// Cold owning builder. Peak memory is O(levels + parents + ports + planes +
// dependencies + evidence + affordances + required-port refs + overlap roles).
// Referenced Cellerator coverage, plane descriptors, and evidence payload bytes
// remain explicitly nonowned because their lifetimes are separate contracts.
class common_atom_builder_v1 {
public:
    common_atom_builder_v1() = default;

    [[nodiscard]] common_atom_build_result_v1 build(
        const common_atom_view_v1 &source,
        std::uint32_t coverage_source_validation) noexcept;

    void reset() noexcept;

    [[nodiscard]] const common_atom_view_v1 &view() const noexcept {
        return view_;
    }

private:
    void rebind() noexcept;

    common_atom_view_v1 view_{};
    std::vector<atom_level_v1> levels_{};
    std::vector<atom_parent_ref_v1> parents_{};
    std::vector<atom_port_v1> ports_{};
    std::vector<atom_plane_descriptor_v1> planes_{};
    std::vector<atom_dependency_requirement_v1> dependencies_{};
    std::vector<atom_evidence_record_ref_v1> evidence_{};
    std::vector<atom_executable_affordance_v1> affordances_{};
    std::vector<std::vector<atom_persistent_identity_v1>>
        required_mutable_ports_{};
    std::vector<atom_overlap_role_record_v1> overlap_roles_{};
};

} // namespace cellshard::compiler::atom
