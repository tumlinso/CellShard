#pragma once

#include <CellShard/compiler/certification/atom_certification_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t canonical_domain_contract_version_v1 = 1;

// One canonical biological domain has one exact axis and order identity for a
// source generation. Cardinality is global and therefore remains u64.
struct canonical_domain_v1 {
    atom::atom_persistent_identity_v1 domain_identity{};
    atom::atom_persistent_identity_v1 axis_identity{};
    atom::atom_persistent_identity_v1 order_identity{};
    std::uint64_t source_generation = 0;
    std::uint64_t entity_count = 0;
};

struct canonical_domain_table_view_v1 {
    const canonical_domain_v1 *domains = nullptr;
    std::uint64_t domain_count = 0;
};

enum class canonical_domain_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_request,
    empty_domain_table,
    missing_domain_table,
    invalid_domain_identity,
    invalid_axis_identity,
    invalid_order_identity,
    missing_source_generation,
    empty_domain,
    unordered_or_duplicate_domain,
    missing_atom_ports,
    unknown_port_domain,
    port_axis_mismatch,
    port_order_mismatch,
};

struct canonical_domain_validation_v1 {
    canonical_domain_validation_code_v1 code =
        canonical_domain_validation_code_v1::valid;
    std::uint64_t atom_index = no_failed_certification_index_v1;
    std::uint64_t port_index = no_failed_certification_index_v1;
    std::uint64_t domain_index = no_failed_certification_index_v1;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == canonical_domain_validation_code_v1::valid;
    }
};

static_assert(offsetof(canonical_domain_table_view_v1, domains) == 0,
              "canonical domain tables must remain pointer-first");
static_assert(std::is_standard_layout<canonical_domain_v1>::value);
static_assert(std::is_trivially_copyable<canonical_domain_v1>::value);
static_assert(std::is_standard_layout<canonical_domain_table_view_v1>::value);
static_assert(
    std::is_trivially_copyable<canonical_domain_table_view_v1>::value);

[[nodiscard]] constexpr bool canonical_domain_less_v1(
    const canonical_domain_v1 &lhs,
    const canonical_domain_v1 &rhs) noexcept {
    return atom::atom_persistent_identity_less_v1(
        lhs.domain_identity, rhs.domain_identity);
}

// The table is validated once in O(domain_count). Each port then uses a
// bounded binary search, making total work O(D + P log D), with O(1) storage.
// Shape, cardinality, and ordinal position never substitute for identity.
[[nodiscard]] inline canonical_domain_validation_v1
validate_canonical_domain_identities_v1(
    const atom_certification_request_v1 &request,
    canonical_domain_table_view_v1 table) noexcept {
    const auto request_result = validate_atom_certification_request_v1(request);
    if (!request_result.valid()) {
        return {canonical_domain_validation_code_v1::invalid_request,
                no_failed_certification_index_v1,
                no_failed_certification_index_v1,
                no_failed_certification_index_v1,
                static_cast<std::uint32_t>(request_result.code)};
    }
    if (table.domain_count == 0) {
        return {canonical_domain_validation_code_v1::empty_domain_table};
    }
    if (table.domains == nullptr) {
        return {canonical_domain_validation_code_v1::missing_domain_table};
    }
    for (std::uint64_t domain_index = 0;
         domain_index < table.domain_count;
         ++domain_index) {
        const auto &domain = table.domains[domain_index];
        if (!atom::validate_atom_persistent_identity_v1(
                 domain.domain_identity)
                 .valid()) {
            return {canonical_domain_validation_code_v1::
                        invalid_domain_identity,
                    no_failed_certification_index_v1,
                    no_failed_certification_index_v1,
                    domain_index};
        }
        if (!atom::validate_atom_persistent_identity_v1(domain.axis_identity)
                 .valid()) {
            return {canonical_domain_validation_code_v1::invalid_axis_identity,
                    no_failed_certification_index_v1,
                    no_failed_certification_index_v1,
                    domain_index};
        }
        if (!atom::validate_atom_persistent_identity_v1(domain.order_identity)
                 .valid()) {
            return {canonical_domain_validation_code_v1::
                        invalid_order_identity,
                    no_failed_certification_index_v1,
                    no_failed_certification_index_v1,
                    domain_index};
        }
        if (domain.source_generation == 0) {
            return {canonical_domain_validation_code_v1::
                        missing_source_generation,
                    no_failed_certification_index_v1,
                    no_failed_certification_index_v1,
                    domain_index};
        }
        if (domain.entity_count == 0) {
            return {canonical_domain_validation_code_v1::empty_domain,
                    no_failed_certification_index_v1,
                    no_failed_certification_index_v1,
                    domain_index};
        }
        if (domain_index != 0
            && !canonical_domain_less_v1(
                table.domains[domain_index - 1], domain)) {
            return {canonical_domain_validation_code_v1::
                        unordered_or_duplicate_domain,
                    no_failed_certification_index_v1,
                    no_failed_certification_index_v1,
                    domain_index};
        }
    }

    for (std::uint64_t atom_index = 0;
         atom_index < request.proposed_atom_count;
         ++atom_index) {
        const auto &ports = request.proposed_atoms[atom_index].ports;
        if (ports.port_count == 0 || ports.ports == nullptr) {
            return {canonical_domain_validation_code_v1::missing_atom_ports,
                    atom_index};
        }
        for (std::uint64_t port_index = 0;
             port_index < ports.port_count;
             ++port_index) {
            const auto &port = ports.ports[port_index];
            std::uint64_t begin = 0;
            std::uint64_t end = table.domain_count;
            while (begin < end) {
                const auto middle = begin + (end - begin) / 2;
                if (atom::atom_persistent_identity_less_v1(
                        table.domains[middle].domain_identity,
                        port.domain_identity)) {
                    begin = middle + 1;
                } else {
                    end = middle;
                }
            }
            if (begin == table.domain_count
                || table.domains[begin].domain_identity
                       != port.domain_identity) {
                return {canonical_domain_validation_code_v1::
                            unknown_port_domain,
                        atom_index,
                        port_index,
                        begin};
            }
            if (table.domains[begin].axis_identity != port.axis_identity) {
                return {canonical_domain_validation_code_v1::
                            port_axis_mismatch,
                        atom_index,
                        port_index,
                        begin};
            }
            if (table.domains[begin].order_identity != port.order_identity) {
                return {canonical_domain_validation_code_v1::
                            port_order_mismatch,
                        atom_index,
                        port_index,
                        begin};
            }
        }
    }
    return {canonical_domain_validation_code_v1::valid,
            request.proposed_atom_count,
            0,
            table.domain_count,
            0};
}

} // namespace cellshard::compiler::certification
