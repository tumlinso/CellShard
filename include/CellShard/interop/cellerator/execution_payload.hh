#pragma once

#include <CellShard/io/pack/execution_payload.cuh>

#include <Cellerator/execution/opaque_artifact.hh>

namespace cellshard::interop::cellerator {

struct execution_artifact_expected {
    execution_payload_identity transport{};
    ::cellerator::execution::opaque_execution_artifact_expected image{};
};

struct validated_execution_artifact {
    execution_payload_identity transport{};
    ::cellerator::execution::validated_opaque_execution_artifact image{};
};

inline ::cellerator::execution::opaque_artifact_status validate_execution_artifact_host(
    const execution_payload_host &host,
    const execution_artifact_expected &expected,
    validated_execution_artifact *out) noexcept {
    namespace execution = ::cellerator::execution;
    namespace persistence = ::cellpack::persistence;
    if (out == nullptr) {
        return {execution::opaque_artifact_code::invalid_argument,
                "CellShard Cellerator artifact output is null"};
    }
    *out = {};
    if (host.storage == nullptr || host.payload == nullptr || host.payload_bytes == 0u) {
        return {execution::opaque_artifact_code::invalid_argument,
                "CellShard execution payload host residency is empty"};
    }
    if (!execution_payload_identity_matches(host.identity, expected.transport)) {
        return {execution::opaque_artifact_code::transport_identity_mismatch,
                "CellShard execution payload identity mismatches"};
    }
    if (host.identity.payload_kind != persistence::execution_image_v2_payload_kind
        || host.identity.payload_schema_version != persistence::execution_image_v2_schema_version
        || host.identity.payload_identity != expected.image.image.image_identity) {
        return {execution::opaque_artifact_code::unsupported_payload,
                "CellShard payload is not the expected CPE2 image"};
    }
    const execution::resident_execution_image resident{
        host.payload, (std::uint64_t) host.payload_bytes};
    const execution::opaque_artifact_status status =
        execution::validate_opaque_execution_artifact_host(resident, expected.image, &out->image);
    if (!status) return status;
    out->transport = host.identity;
    return {};
}

#if CELLSHARD_ENABLE_CUDA
using bound_execution_artifact = ::cellerator::execution::bound_opaque_execution_artifact;

inline ::cellerator::execution::opaque_artifact_status bind_execution_artifact_device(
    const validated_execution_artifact &validated,
    const execution_payload_device &device,
    bound_execution_artifact *out) noexcept {
    namespace execution = ::cellerator::execution;
    if (device.storage == nullptr || device.payload == nullptr
        || device.payload_bytes == 0u || device.device_id < 0) {
        return {execution::opaque_artifact_code::invalid_argument,
                "CellShard execution payload device residency is empty"};
    }
    if (!execution_payload_identity_matches(device.identity, validated.transport)) {
        return {execution::opaque_artifact_code::device_binding_mismatch,
                "CellShard device payload identity mismatches validated host state"};
    }
    const execution::resident_device_execution_image resident{
        device.payload, (std::uint64_t) device.payload_bytes, device.device_id};
    return execution::bind_opaque_execution_artifact_device(validated.image, resident, out);
}
#endif

} // namespace cellshard::interop::cellerator
