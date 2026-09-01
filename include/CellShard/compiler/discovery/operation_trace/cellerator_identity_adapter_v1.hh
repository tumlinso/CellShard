#pragma once

#include <CellShard/compiler/discovery/operation_trace/atom_access_event_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::operation_trace {

inline constexpr std::uint32_t cellerator_identity_adapter_version_v1 = 1;

// Cellerator operation-core stable_id is two explicit u64 fields. Stage IDs in
// current prepared-program contracts are one u64, so the caller must supply
// their stable producer namespace. This scalar bridge intentionally imports no
// Cellerator headers and performs no hashing or process-local registration.
struct cellerator_operation_stage_source_v1 {
    std::uint64_t operation_low = 0;
    std::uint64_t operation_high = 0;
    std::uint64_t stage_producer_namespace = 0;
    std::uint64_t stage_identity = 0;
    std::uint64_t operation_generation = 0;
    std::uint64_t stage_generation = 0;
};

struct operation_stage_identity_binding_v1 {
    atom::atom_persistent_identity_v1 operation_identity{};
    atom::atom_persistent_identity_v1 stage_identity{};
    std::uint64_t operation_generation = 0;
    std::uint64_t stage_generation = 0;
};

enum class cellerator_identity_adapter_code_v1 : std::uint32_t {
    adapted = 0,
    missing_destination,
    invalid_operation_identity,
    missing_stage_namespace,
    missing_stage_identity,
    missing_operation_generation,
    missing_stage_generation,
};

struct cellerator_identity_adapter_result_v1 {
    cellerator_identity_adapter_code_v1 code =
        cellerator_identity_adapter_code_v1::adapted;

    [[nodiscard]] constexpr bool adapted() const noexcept {
        return code == cellerator_identity_adapter_code_v1::adapted;
    }
};

// Exact field mapping: operation high becomes the producer namespace and low
// becomes the local identity. A partially-qualified Cellerator ID is rejected
// rather than silently namespaced or digested.
[[nodiscard]] constexpr cellerator_identity_adapter_result_v1
adapt_cellerator_operation_stage_identities_v1(
    cellerator_operation_stage_source_v1 source,
    operation_stage_identity_binding_v1 *destination) noexcept {
    if (destination == nullptr) {
        return {cellerator_identity_adapter_code_v1::missing_destination};
    }
    *destination = {};
    if (source.operation_low == 0 || source.operation_high == 0) {
        return {cellerator_identity_adapter_code_v1::
                    invalid_operation_identity};
    }
    if (source.stage_producer_namespace == 0) {
        return {cellerator_identity_adapter_code_v1::missing_stage_namespace};
    }
    if (source.stage_identity == 0) {
        return {cellerator_identity_adapter_code_v1::missing_stage_identity};
    }
    if (source.operation_generation == 0) {
        return {cellerator_identity_adapter_code_v1::
                    missing_operation_generation};
    }
    if (source.stage_generation == 0) {
        return {cellerator_identity_adapter_code_v1::missing_stage_generation};
    }
    destination->operation_identity = {
        source.operation_high, source.operation_low};
    destination->stage_identity = {
        source.stage_producer_namespace, source.stage_identity};
    destination->operation_generation = source.operation_generation;
    destination->stage_generation = source.stage_generation;
    return {};
}

enum class bind_operation_stage_code_v1 : std::uint32_t {
    bound = 0,
    invalid_binding,
    missing_event,
};

// Binding changes only the operation/stage identity plane of an event. The
// caller still supplies trace, atom, graph, port, and access facts.
[[nodiscard]] constexpr bind_operation_stage_code_v1
bind_operation_stage_identity_v1(
    operation_stage_identity_binding_v1 binding,
    atom_access_event_v1 *event) noexcept {
    if (!atom::validate_atom_persistent_identity_v1(
             binding.operation_identity)
             .valid()
        || !atom::validate_atom_persistent_identity_v1(binding.stage_identity)
                .valid()
        || binding.operation_generation == 0
        || binding.stage_generation == 0) {
        return bind_operation_stage_code_v1::invalid_binding;
    }
    if (event == nullptr) {
        return bind_operation_stage_code_v1::missing_event;
    }
    event->operation_identity = binding.operation_identity;
    event->stage_identity = binding.stage_identity;
    event->operation_generation = binding.operation_generation;
    event->stage_generation = binding.stage_generation;
    return bind_operation_stage_code_v1::bound;
}

static_assert(std::is_standard_layout<
                  cellerator_operation_stage_source_v1>::value);
static_assert(std::is_trivially_copyable<
                  cellerator_operation_stage_source_v1>::value);
static_assert(std::is_standard_layout<
                  operation_stage_identity_binding_v1>::value);
static_assert(std::is_trivially_copyable<
                  operation_stage_identity_binding_v1>::value);

} // namespace cellshard::compiler::discovery::operation_trace
