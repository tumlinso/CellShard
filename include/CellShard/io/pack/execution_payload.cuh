#pragma once

#include "../common/generation.hh"

#include <cstddef>
#include <cstdint>

#if CELLSHARD_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace cellshard {

inline constexpr std::uint32_t execution_payload_envelope_schema_version = 1u;
inline constexpr std::uint32_t execution_payload_endian_marker = 0x01020304u;

// CellShard owns this compatibility envelope, not the caller payload semantics.
// Every identity is explicit so a generation or canonical-axis mismatch is
// rejected before opaque execution bytes are exposed to Cellerator.
struct execution_payload_identity {
    std::uint64_t dataset_identity = 0u;
    dataset_generation_ref generation{};
    std::uint64_t partition_identity = 0u;
    std::uint64_t global_row_begin = 0u;
    std::uint32_t row_count = 0u;
    std::uint32_t feature_count = 0u;
    std::uint64_t feature_axis_fingerprint = 0u;
    std::uint32_t feature_axis_fingerprint_version = 0u;
    std::uint32_t payload_kind = 0u;
    std::uint32_t payload_schema_version = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t row_domain_identity = 0u;
    std::uint64_t payload_identity = 0u;
};

struct execution_payload_source {
    execution_payload_identity identity{};
    const void *payload = nullptr;
    std::size_t payload_bytes = 0u;
};

// Owns exactly one fetched partition payload in one contiguous host allocation.
struct execution_payload_host {
    execution_payload_identity identity{};
    void *storage = nullptr;
    const unsigned char *payload = nullptr;
    std::size_t payload_bytes = 0u;
};

bool valid_execution_payload_identity(
    const execution_payload_identity &identity) noexcept;

bool execution_payload_identity_matches(
    const execution_payload_identity &actual,
    const execution_payload_identity &expected) noexcept;

// Atomically publishes one CSPACK01 shard file. Each table entry contains one
// versioned/checksummed opaque execution envelope and caller-owned payload.
int store_execution_cspack(
    const char *path,
    std::uint64_t shard_id,
    const execution_payload_source *partitions,
    std::uint64_t partition_count);

// Fetches one partition by CSPACK table index and enforces the complete expected
// identity. Load performs no semantic interpretation of the caller payload.
int load_execution_cspack_partition(
    const char *path,
    std::uint64_t expected_shard_id,
    std::uint64_t partition_index,
    const execution_payload_identity &expected,
    execution_payload_host *out);

void clear_execution_payload_host(execution_payload_host *payload) noexcept;

#if CELLSHARD_ENABLE_CUDA
// Device staging owns one contiguous device allocation. Allocation is visible
// host work; the single H2D payload copy is enqueued on caller_stream and this
// function does not synchronize it.
struct execution_payload_device {
    execution_payload_identity identity{};
    void *storage = nullptr;
    const unsigned char *payload = nullptr;
    std::size_t payload_bytes = 0u;
    int device_id = -1;
};

cudaError_t upload_execution_payload_async(
    const execution_payload_host &host,
    int device_id,
    cudaStream_t caller_stream,
    execution_payload_device *out);

cudaError_t clear_execution_payload_device(
    execution_payload_device *payload) noexcept;
#endif

} // namespace cellshard
