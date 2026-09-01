#pragma once

#include <CellShard/compiler/discovery/sequence_compat/long_range_relation_bridge_v1.hh>
#include <CellShard/compiler/discovery/sequence_compat/owned_halo_intervals_v1.hh>

#include <array>
#include <cstdint>

namespace cellshard::compiler::discovery::sequence_compat {

enum class mock_sequence_provider_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_coordinate_coverage,
    invalid_owned_halos,
    invalid_hierarchy,
    invalid_relation_bridge,
};

struct mock_sequence_provider_validation_v1 {
    mock_sequence_provider_validation_code_v1 code =
        mock_sequence_provider_validation_code_v1::valid;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == mock_sequence_provider_validation_code_v1::valid;
    }
};

// A fixed, cold fixture shaped like a possible packed-sequence provider. It
// implements no sequence algorithms and includes no Baseplane headers. Its
// only job is to prove the provider-neutral interval/halo/DAG/relation seam.
class mock_baseplane_shaped_provider_v1 {
public:
    mock_baseplane_shaped_provider_v1() noexcept {
        build(strand_identity_v1::forward);
    }

    mock_baseplane_shaped_provider_v1(
        const mock_baseplane_shaped_provider_v1 &) = delete;
    mock_baseplane_shaped_provider_v1 &operator=(
        const mock_baseplane_shaped_provider_v1 &) = delete;

    void build(strand_identity_v1 strand) noexcept {
        coordinate_ = {};
        coordinate_.provider_payload = &provider_payload_token_;
        coordinate_.exact_coverage.cellerator_coverage =
            &coordinate_coverage_token_;
        coordinate_.exact_coverage.coverage_identity = {100u, 1u};
        coordinate_.exact_coverage.logical_count = 10u;
        coordinate_.exact_coverage.source_schema_version =
            atom::cellerator_logical_coverage_schema_version_v1;
        coordinate_.exact_coverage.source_record_bytes =
            atom::cellerator_logical_coverage_record_bytes_v1;
        coordinate_.exact_coverage.role_flags =
            atom::atom_certified_exact_coverage_role_v1;
        coordinate_.exact_coverage.kind =
            atom::atom_logical_coverage_kind_v1::provider_defined;
        coordinate_.reference_identity = {101u, 1u};
        coordinate_.payload_schema = {102u, 1u};
        coordinate_.coordinate_begin = 5u;
        coordinate_.coordinate_end = 25u;
        coordinate_.owned_count = 10u;

        reference_ = {};
        reference_.assembly_identity = {103u, 1u};
        reference_.sequence_identity = {103u, 2u};
        reference_.strand = strand;

        owned_halo_records_ = {{
            {5u, 10u, coordinate_interval_role_v1::read_only_halo, false, {}},
            {10u, 20u, coordinate_interval_role_v1::owned, true, {}},
            {20u, 25u, coordinate_interval_role_v1::read_only_halo, false, {}},
        }};
        owned_halos_ = {};
        owned_halos_.intervals = owned_halo_records_.data();
        owned_halos_.interval_count = owned_halo_records_.size();
        owned_halos_.coordinate_coverage = &coordinate_;
        owned_halos_.required_left_halo = 5u;
        owned_halos_.required_right_halo = 5u;

        child_parent_[0] = {{104u, 1u}, 0u};
        hierarchy_records_[0] = {{104u, 1u}, 5u, 25u, nullptr, 0u};
        hierarchy_records_[1] = {
            {104u, 2u}, 10u, 15u, child_parent_.data(), 1u};
        hierarchy_records_[2] = {
            {104u, 3u}, 15u, 20u, child_parent_.data(), 1u};
        hierarchy_ = {};
        hierarchy_.intervals = hierarchy_records_.data();
        hierarchy_.interval_count = hierarchy_records_.size();
        hierarchy_.coordinate_coverage = &coordinate_;
        hierarchy_.reference = &reference_;

        mappings_[0] = {{104u, 2u}, {105u, 1u}, {106u, 1u},
            sequence_endpoint_kind_v1::enhancer, {}};
        mappings_[1] = {{104u, 3u}, {105u, 2u}, {106u, 2u},
            sequence_endpoint_kind_v1::gene, {}};
        productions_[0] = {{107u, 1u}, 0u, 1u,
            long_range_relation_kind_v1::enhancer_to_gene, {}};

        output_relation_coverage_ = {};
        output_relation_coverage_.cellerator_coverage =
            &relation_coverage_token_;
        output_relation_coverage_.coverage_identity = {108u, 1u};
        output_relation_coverage_.logical_count = 1u;
        output_relation_coverage_.source_schema_version =
            atom::cellerator_logical_coverage_schema_version_v1;
        output_relation_coverage_.source_record_bytes =
            atom::cellerator_logical_coverage_record_bytes_v1;
        output_relation_coverage_.role_flags =
            atom::atom_certified_exact_coverage_role_v1;
        output_relation_coverage_.kind =
            atom::atom_logical_coverage_kind_v1::relation_edge_ids;

        bridge_ = {};
        bridge_.mappings = mappings_.data();
        bridge_.mapping_count = mappings_.size();
        bridge_.productions = productions_.data();
        bridge_.production_count = productions_.size();
        bridge_.intervals = &hierarchy_;
        bridge_.output_relation_coverage = &output_relation_coverage_;
        bridge_.relation_species_identity = {109u, 1u};
    }

    [[nodiscard]] const provider_coordinate_coverage_v1 &coordinate() const
        noexcept {
        return coordinate_;
    }

    [[nodiscard]] const reference_strand_identity_v1 &reference() const
        noexcept {
        return reference_;
    }

    [[nodiscard]] const owned_halo_intervals_v1 &owned_halos() const noexcept {
        return owned_halos_;
    }

    [[nodiscard]] const hierarchical_interval_dag_v1 &hierarchy() const
        noexcept {
        return hierarchy_;
    }

    [[nodiscard]] const long_range_relation_bridge_v1 &bridge() const noexcept {
        return bridge_;
    }

private:
    std::uint64_t provider_payload_token_ = 1u;
    std::uint64_t coordinate_coverage_token_ = 2u;
    std::uint64_t relation_coverage_token_ = 3u;
    provider_coordinate_coverage_v1 coordinate_{};
    reference_strand_identity_v1 reference_{};
    std::array<coordinate_interval_role_record_v1, 3> owned_halo_records_{};
    owned_halo_intervals_v1 owned_halos_{};
    std::array<hierarchical_interval_parent_v1, 1> child_parent_{};
    std::array<hierarchical_interval_v1, 3> hierarchy_records_{};
    hierarchical_interval_dag_v1 hierarchy_{};
    std::array<sequence_entity_identity_map_v1, 2> mappings_{};
    std::array<long_range_relation_production_v1, 1> productions_{};
    atom::atom_logical_coverage_ref_v1 output_relation_coverage_{};
    long_range_relation_bridge_v1 bridge_{};
};

[[nodiscard]] inline mock_sequence_provider_validation_v1
validate_mock_baseplane_shaped_provider_v1(
    const mock_baseplane_shaped_provider_v1 &provider) noexcept {
    const auto coordinate =
        validate_provider_coordinate_coverage_v1(provider.coordinate(), 0);
    if (!coordinate.valid()) {
        return {mock_sequence_provider_validation_code_v1::
                    invalid_coordinate_coverage,
                static_cast<std::uint32_t>(coordinate.code)};
    }
    const auto halos = validate_owned_halo_intervals_v1(
        provider.owned_halos(), 0);
    if (!halos.valid()) {
        return {mock_sequence_provider_validation_code_v1::invalid_owned_halos,
                static_cast<std::uint32_t>(halos.code)};
    }
    const auto hierarchy = validate_hierarchical_interval_dag_v1(
        provider.hierarchy(), 0);
    if (!hierarchy.valid()) {
        return {mock_sequence_provider_validation_code_v1::invalid_hierarchy,
                static_cast<std::uint32_t>(hierarchy.code)};
    }
    const auto bridge = validate_long_range_relation_bridge_v1(
        provider.bridge(), 0, 0);
    if (!bridge.valid()) {
        return {mock_sequence_provider_validation_code_v1::
                    invalid_relation_bridge,
                static_cast<std::uint32_t>(bridge.code)};
    }
    return {};
}

} // namespace cellshard::compiler::discovery::sequence_compat
