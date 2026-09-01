#include <CellShard/compiler/atom/common_atom_v1.hh>

#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>

namespace cellshard::compiler::atom {

namespace {

[[nodiscard]] bool count_fits_pointer_range(std::uint64_t count) noexcept {
    return count
        <= static_cast<std::uint64_t>(
            std::numeric_limits<std::ptrdiff_t>::max());
}

} // namespace

void common_atom_builder_v1::reset() noexcept {
    view_ = {};
    levels_.clear();
    parents_.clear();
    ports_.clear();
    planes_.clear();
    dependencies_.clear();
    evidence_.clear();
    affordances_.clear();
    required_mutable_ports_.clear();
    overlap_roles_.clear();
}

void common_atom_builder_v1::rebind() noexcept {
    view_.levels = levels_.empty() ? nullptr : levels_.data();
    view_.level_count = levels_.size();
    view_.parents = parents_.empty() ? nullptr : parents_.data();
    view_.parent_count = parents_.size();
    view_.ports = {ports_.data(), ports_.size()};
    view_.planes = {planes_.data(), planes_.size()};
    view_.dependencies.dependencies = dependencies_.data();
    view_.dependencies.dependency_count = dependencies_.size();
    view_.evidence.records = evidence_.data();
    view_.evidence.record_count = evidence_.size();
    for (std::size_t index = 0; index < affordances_.size(); ++index) {
        affordances_[index].required_mutable_ports =
            required_mutable_ports_[index].data();
        affordances_[index].required_mutable_port_count =
            required_mutable_ports_[index].size();
    }
    view_.affordances.affordances = affordances_.data();
    view_.affordances.affordance_count = affordances_.size();
    view_.affordances.ports = view_.ports;
    view_.overlap_roles.records = overlap_roles_.data();
    view_.overlap_roles.record_count = overlap_roles_.size();
}

common_atom_build_result_v1 common_atom_builder_v1::build(
    const common_atom_view_v1 &source,
    std::uint32_t coverage_source_validation) noexcept {
    const auto input_validation = validate_common_atom_v1(
        source, coverage_source_validation);
    if (!input_validation.valid()) {
        return {common_atom_build_code_v1::invalid_input, input_validation};
    }
    if (!count_fits_pointer_range(source.parent_count)
        || !count_fits_pointer_range(source.ports.port_count)
        || !count_fits_pointer_range(source.planes.plane_count)
        || !count_fits_pointer_range(source.dependencies.dependency_count)
        || !count_fits_pointer_range(source.evidence.record_count)
        || !count_fits_pointer_range(source.affordances.affordance_count)
        || !count_fits_pointer_range(source.overlap_roles.record_count)) {
        return {common_atom_build_code_v1::capacity_overflow, {}};
    }
    for (std::uint64_t index = 0;
         index < source.affordances.affordance_count;
         ++index) {
        if (!count_fits_pointer_range(
                source.affordances.affordances[index]
                    .required_mutable_port_count)) {
            return {common_atom_build_code_v1::capacity_overflow, {}};
        }
    }

    reset();
    try {
        view_ = source;
        levels_.assign(source.levels, source.levels + source.level_count);
        if (source.parent_count != 0) {
            parents_.assign(
                source.parents, source.parents + source.parent_count);
        }
        ports_.assign(
            source.ports.ports,
            source.ports.ports + source.ports.port_count);
        planes_.assign(
            source.planes.planes,
            source.planes.planes + source.planes.plane_count);
        dependencies_.assign(
            source.dependencies.dependencies,
            source.dependencies.dependencies
                + source.dependencies.dependency_count);
        evidence_.assign(
            source.evidence.records,
            source.evidence.records + source.evidence.record_count);
        affordances_.assign(
            source.affordances.affordances,
            source.affordances.affordances
                + source.affordances.affordance_count);
        required_mutable_ports_.reserve(affordances_.size());
        for (const auto &affordance : affordances_) {
            required_mutable_ports_.emplace_back(
                affordance.required_mutable_ports,
                affordance.required_mutable_ports
                    + affordance.required_mutable_port_count);
        }
        overlap_roles_.assign(
            source.overlap_roles.records,
            source.overlap_roles.records + source.overlap_roles.record_count);
    } catch (const std::bad_alloc &) {
        reset();
        return {common_atom_build_code_v1::allocation_failure, {}};
    } catch (const std::length_error &) {
        reset();
        return {common_atom_build_code_v1::capacity_overflow, {}};
    }
    rebind();
    const auto built_validation = validate_common_atom_v1(
        view_, coverage_source_validation);
    if (!built_validation.valid()) {
        reset();
        return {common_atom_build_code_v1::invalid_built_view,
                built_validation};
    }
    return {common_atom_build_code_v1::built, built_validation};
}

} // namespace cellshard::compiler::atom
