#pragma once

#include <CellShard/compiler/atom/common_atom_v1.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t atom_certification_contract_version_v1 = 1;
inline constexpr std::uint64_t no_failed_certification_index_v1 =
    std::numeric_limits<std::uint64_t>::max();

// Global biological identity and all externally visible counts remain u64.
// This width is only a bound for transient/local ordinal maps produced during
// certification; selecting it never changes canonical identity.
enum class certification_local_index_width_v1 : std::uint8_t {
    u8 = 1,
    u16 = 2,
    u32 = 4,
    u64 = 8,
};

enum class atom_certification_outcome_v1 : std::uint32_t {
    not_run = 0,
    certified = 1,
    rejected = 2,
    invalid_request = 3,
    insufficient_workspace = 4,
};

enum class atom_certification_stage_v1 : std::uint32_t {
    none = 0,
    request = 1,
    canonical_domains = 2,
    entity_coverage = 3,
    relation_edge_coverage = 4,
    duplicate_detection = 5,
    local_maps = 6,
    read_only_halos = 7,
    physical_replicas = 8,
    contribution_owners = 9,
    residual_coverage = 10,
    multimodal_identity = 11,
    trajectory_lineage = 12,
    partial_result_algebra = 13,
    dependency_closure = 14,
    certificate_emission = 15,
    independent_verification = 16,
};

// A proposal provider supplies candidates and evidence; a distinct certification
// authority checks them against their source-linked canonical coverage. The
// workspace is caller-owned so exact certification has an explicit memory bound.
// Certification never calls back into a proposal builder.
struct atom_certification_request_v1 {
    const atom::common_atom_view_v1 *proposed_atoms = nullptr;
    void *workspace = nullptr;
    std::uint64_t proposed_atom_count = 0;
    std::uint64_t workspace_bytes = 0;
    atom::atom_persistent_identity_v1 request_identity{};
    atom::atom_persistent_identity_v1 proposal_provider_identity{};
    atom::atom_persistent_identity_v1 certification_authority_identity{};
    atom::atom_persistent_identity_v1 canonical_source_identity{};
    std::uint64_t canonical_source_generation = 0;
    certification_local_index_width_v1 maximum_local_index_width =
        certification_local_index_width_v1::u32;
    std::uint8_t reserved[7]{};
};

// This is an execution result, not the durable exact certificate introduced by
// the certificate-emission stage. Counts and failure positions are global u64
// values so atlas-scale inputs are not silently truncated to a local width.
struct atom_certification_result_v1 {
    atom::atom_persistent_identity_v1 request_identity{};
    atom::atom_persistent_identity_v1 certification_authority_identity{};
    std::uint64_t proposed_atom_count = 0;
    std::uint64_t certified_atom_count = 0;
    std::uint64_t certified_entity_count = 0;
    std::uint64_t certified_relation_edge_count = 0;
    std::uint64_t required_workspace_bytes = 0;
    std::uint64_t failed_atom_index = no_failed_certification_index_v1;
    std::uint64_t failed_member_index = no_failed_certification_index_v1;
    atom_certification_outcome_v1 outcome =
        atom_certification_outcome_v1::not_run;
    atom_certification_stage_v1 stage = atom_certification_stage_v1::none;
    std::uint32_t detail_code = 0;
    certification_local_index_width_v1 local_index_width =
        certification_local_index_width_v1::u32;
    std::uint8_t reserved[3]{};

    [[nodiscard]] constexpr bool certified() const noexcept {
        return outcome == atom_certification_outcome_v1::certified;
    }
};

enum class atom_certification_request_validation_code_v1 : std::uint32_t {
    valid = 0,
    empty_proposal,
    missing_proposal,
    missing_workspace,
    misaligned_workspace,
    invalid_request_identity,
    invalid_proposal_provider_identity,
    invalid_certification_authority_identity,
    provider_self_certification,
    invalid_canonical_source_identity,
    missing_canonical_source_generation,
    invalid_local_index_width,
    nonzero_reserved,
};

struct atom_certification_request_validation_v1 {
    atom_certification_request_validation_code_v1 code =
        atom_certification_request_validation_code_v1::valid;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_certification_request_validation_code_v1::valid;
    }
};

static_assert(offsetof(atom_certification_request_v1, proposed_atoms) == 0,
              "certification requests must remain pointer-first");
static_assert(std::is_standard_layout<atom_certification_request_v1>::value);
static_assert(std::is_trivially_copyable<atom_certification_request_v1>::value);
static_assert(std::is_standard_layout<atom_certification_result_v1>::value);
static_assert(std::is_trivially_copyable<atom_certification_result_v1>::value);

[[nodiscard]] constexpr bool valid_certification_local_index_width_v1(
    certification_local_index_width_v1 width) noexcept {
    const auto bytes = static_cast<std::uint8_t>(width);
    return bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8;
}

// O(1), allocation-free structural validation. Exact atom/member/edge scans
// belong to the later certification stages and remain independent of proposal
// construction.
[[nodiscard]] inline atom_certification_request_validation_v1
validate_atom_certification_request_v1(
    const atom_certification_request_v1 &request) noexcept {
    if (request.proposed_atom_count == 0) {
        return {atom_certification_request_validation_code_v1::empty_proposal};
    }
    if (request.proposed_atoms == nullptr) {
        return {atom_certification_request_validation_code_v1::missing_proposal};
    }
    if (request.workspace_bytes != 0 && request.workspace == nullptr) {
        return {atom_certification_request_validation_code_v1::missing_workspace};
    }
    if (request.workspace != nullptr
        && reinterpret_cast<std::uintptr_t>(request.workspace)
               % alignof(std::uint64_t)
            != 0) {
        return {
            atom_certification_request_validation_code_v1::misaligned_workspace};
    }
    if (!atom::validate_atom_persistent_identity_v1(request.request_identity)
             .valid()) {
        return {atom_certification_request_validation_code_v1::
                    invalid_request_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             request.proposal_provider_identity)
             .valid()) {
        return {atom_certification_request_validation_code_v1::
                    invalid_proposal_provider_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             request.certification_authority_identity)
             .valid()) {
        return {atom_certification_request_validation_code_v1::
                    invalid_certification_authority_identity};
    }
    if (request.proposal_provider_identity
        == request.certification_authority_identity) {
        return {atom_certification_request_validation_code_v1::
                    provider_self_certification};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             request.canonical_source_identity)
             .valid()) {
        return {atom_certification_request_validation_code_v1::
                    invalid_canonical_source_identity};
    }
    if (request.canonical_source_generation == 0) {
        return {atom_certification_request_validation_code_v1::
                    missing_canonical_source_generation};
    }
    if (!valid_certification_local_index_width_v1(
            request.maximum_local_index_width)) {
        return {atom_certification_request_validation_code_v1::
                    invalid_local_index_width};
    }
    for (const auto value : request.reserved) {
        if (value != 0) {
            return {atom_certification_request_validation_code_v1::
                        nonzero_reserved};
        }
    }
    return {};
}

} // namespace cellshard::compiler::certification
