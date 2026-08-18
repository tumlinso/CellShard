#include <CellShard/io/pack/execution_payload.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

namespace cellshard {
namespace {

constexpr unsigned char cspack_magic[8] = {'C','S','P','A','C','K','0','1'};
constexpr unsigned char execution_magic[8] = {'C','P','E','X','E','C','0','1'};
constexpr std::uint64_t fnv1a_offset = 1469598103934665603ull;
constexpr std::uint64_t fnv1a_prime = 1099511628211ull;

struct execution_payload_disk_header {
    unsigned char magic[8];
    std::uint32_t schema_version;
    std::uint32_t header_bytes;
    std::uint32_t endian;
    std::uint32_t reserved;
    std::uint64_t payload_bytes;
    std::uint64_t envelope_checksum;
    execution_payload_identity identity;
};

static_assert(sizeof(execution_payload_disk_header) % 8u == 0u,
    "execution envelope header must preserve payload alignment");

std::uint64_t hash_bytes(
    std::uint64_t hash, const void *data, std::size_t bytes) noexcept {
    const auto *cursor = static_cast<const unsigned char *>(data);
    for (std::size_t index = 0u; index < bytes; ++index) {
        hash ^= cursor[index];
        hash *= fnv1a_prime;
    }
    return hash;
}

std::uint64_t envelope_checksum(
    execution_payload_disk_header header,
    const void *payload,
    std::size_t payload_bytes) noexcept {
    header.envelope_checksum = 0u;
    std::uint64_t hash = hash_bytes(fnv1a_offset, &header, sizeof(header));
    hash = hash_bytes(hash, payload, payload_bytes);
    return hash == 0u ? 1u : hash;
}

bool write_exact(std::FILE *file, const void *data, std::size_t bytes) {
    return bytes == 0u || std::fwrite(data, 1u, bytes, file) == bytes;
}

bool read_exact(std::FILE *file, void *data, std::size_t bytes) {
    return bytes == 0u || std::fread(data, 1u, bytes, file) == bytes;
}

bool generation_valid(const dataset_generation_ref &generation) noexcept {
    return generation.canonical_generation != 0u
        && generation.execution_plan_generation != 0u
        && generation.pack_generation != 0u
        && generation.service_epoch != 0u;
}

bool generation_equal(
    const dataset_generation_ref &left,
    const dataset_generation_ref &right) noexcept {
    return left.canonical_generation == right.canonical_generation
        && left.execution_plan_generation == right.execution_plan_generation
        && left.pack_generation == right.pack_generation
        && left.service_epoch == right.service_epoch;
}

bool file_size(std::FILE *file, std::uint64_t *out) {
    struct stat status{};
    if (file == nullptr || out == nullptr || ::fstat(::fileno(file), &status) != 0
        || status.st_size < 0) return false;
    *out = static_cast<std::uint64_t>(status.st_size);
    return true;
}

} // namespace

bool valid_execution_payload_identity(
    const execution_payload_identity &identity) noexcept {
    return identity.dataset_identity != 0u
        && generation_valid(identity.generation)
        && identity.partition_identity != 0u
        && identity.row_count != 0u
        && identity.feature_count != 0u
        && identity.feature_axis_fingerprint != 0u
        && identity.feature_axis_fingerprint_version != 0u
        && identity.payload_kind != 0u
        && identity.payload_schema_version != 0u
        && identity.reserved == 0u
        && identity.row_domain_identity != 0u
        && identity.payload_identity != 0u;
}

bool execution_payload_identity_matches(
    const execution_payload_identity &actual,
    const execution_payload_identity &expected) noexcept {
    return valid_execution_payload_identity(actual)
        && valid_execution_payload_identity(expected)
        && actual.dataset_identity == expected.dataset_identity
        && generation_equal(actual.generation, expected.generation)
        && actual.partition_identity == expected.partition_identity
        && actual.global_row_begin == expected.global_row_begin
        && actual.row_count == expected.row_count
        && actual.feature_count == expected.feature_count
        && actual.feature_axis_fingerprint == expected.feature_axis_fingerprint
        && actual.feature_axis_fingerprint_version
            == expected.feature_axis_fingerprint_version
        && actual.payload_kind == expected.payload_kind
        && actual.payload_schema_version == expected.payload_schema_version
        && actual.row_domain_identity == expected.row_domain_identity
        && actual.payload_identity == expected.payload_identity;
}

int store_execution_cspack(
    const char *path,
    std::uint64_t shard_id,
    const execution_payload_source *partitions,
    std::uint64_t partition_count) {
    if (path == nullptr || path[0] == '\0' || shard_id == 0u
        || partitions == nullptr || partition_count == 0u
        || partition_count > std::numeric_limits<std::size_t>::max()
            / sizeof(std::uint64_t)) return 0;
    for (std::uint64_t index = 0u; index < partition_count; ++index) {
        const execution_payload_source &source = partitions[index];
        if (!valid_execution_payload_identity(source.identity)
            || source.payload == nullptr || source.payload_bytes == 0u) return 0;
        for (std::uint64_t prior = 0u; prior < index; ++prior) {
            if (partitions[prior].identity.partition_identity
                == source.identity.partition_identity) return 0;
        }
    }

    const std::string temporary_path = std::string(path) + ".tmp";
    std::remove(temporary_path.c_str());
    std::FILE *file = std::fopen(temporary_path.c_str(), "wb+");
    if (file == nullptr) return 0;
    auto *offsets = static_cast<std::uint64_t *>(
        std::calloc(static_cast<std::size_t>(partition_count), sizeof(std::uint64_t)));
    bool ok = offsets != nullptr;
    if (ok) ok = write_exact(file, cspack_magic, sizeof(cspack_magic))
        && write_exact(file, &shard_id, sizeof(shard_id))
        && write_exact(file, &partition_count, sizeof(partition_count))
        && write_exact(file, offsets,
            static_cast<std::size_t>(partition_count) * sizeof(std::uint64_t));
    for (std::uint64_t index = 0u; ok && index < partition_count; ++index) {
        const off_t position = ::ftello(file);
        if (position < 0) { ok = false; break; }
        offsets[index] = static_cast<std::uint64_t>(position);
        execution_payload_disk_header header{};
        std::memcpy(header.magic, execution_magic, sizeof(execution_magic));
        header.schema_version = execution_payload_envelope_schema_version;
        header.header_bytes = sizeof(header);
        header.endian = execution_payload_endian_marker;
        header.payload_bytes = partitions[index].payload_bytes;
        header.identity = partitions[index].identity;
        header.envelope_checksum = envelope_checksum(
            header, partitions[index].payload, partitions[index].payload_bytes);
        ok = write_exact(file, &header, sizeof(header))
            && write_exact(file, partitions[index].payload,
                partitions[index].payload_bytes);
    }
    if (ok) ok = ::fseeko(file,
        static_cast<off_t>(sizeof(cspack_magic) + sizeof(std::uint64_t) * 2u),
        SEEK_SET) == 0
        && write_exact(file, offsets,
            static_cast<std::size_t>(partition_count) * sizeof(std::uint64_t));
    if (ok) ok = std::fflush(file) == 0 && ::fsync(::fileno(file)) == 0;
    if (std::fclose(file) != 0) ok = false;
    std::free(offsets);
    if (ok) ok = std::rename(temporary_path.c_str(), path) == 0;
    if (!ok) std::remove(temporary_path.c_str());
    return ok ? 1 : 0;
}

int load_execution_cspack_partition(
    const char *path,
    std::uint64_t expected_shard_id,
    std::uint64_t partition_index,
    const execution_payload_identity &expected,
    execution_payload_host *out) {
    if (path == nullptr || expected_shard_id == 0u
        || !valid_execution_payload_identity(expected) || out == nullptr) return 0;
    clear_execution_payload_host(out);
    std::FILE *file = std::fopen(path, "rb");
    if (file == nullptr) return 0;
    unsigned char magic[8]{};
    std::uint64_t shard_id = 0u, partition_count = 0u, bytes = 0u;
    std::uint64_t *offsets = nullptr;
    bool ok = file_size(file, &bytes)
        && read_exact(file, magic, sizeof(magic))
        && std::memcmp(magic, cspack_magic, sizeof(magic)) == 0
        && read_exact(file, &shard_id, sizeof(shard_id))
        && read_exact(file, &partition_count, sizeof(partition_count))
        && shard_id == expected_shard_id
        && partition_count != 0u
        && partition_index < partition_count
        && partition_count <= std::numeric_limits<std::size_t>::max()
            / sizeof(std::uint64_t)
        && bytes >= sizeof(cspack_magic) + sizeof(std::uint64_t) * 2u
        && partition_count <= (bytes - sizeof(cspack_magic)
            - sizeof(std::uint64_t) * 2u) / sizeof(std::uint64_t);
    if (ok) {
        offsets = static_cast<std::uint64_t *>(
            std::malloc(static_cast<std::size_t>(partition_count)
                * sizeof(std::uint64_t)));
        ok = offsets != nullptr && read_exact(file, offsets,
            static_cast<std::size_t>(partition_count) * sizeof(std::uint64_t));
    }
    const std::uint64_t table_end = sizeof(cspack_magic)
        + sizeof(std::uint64_t) * (2u + partition_count);
    for (std::uint64_t index = 0u; ok && index < partition_count; ++index) {
        const std::uint64_t end = index + 1u < partition_count
            ? offsets[index + 1u] : bytes;
        if (offsets[index] < table_end || offsets[index] >= end || end > bytes) ok = false;
    }
    execution_payload_disk_header header{};
    std::uint64_t partition_end = 0u;
    if (ok) {
        partition_end = partition_index + 1u < partition_count
            ? offsets[partition_index + 1u] : bytes;
        ok = ::fseeko(file, static_cast<off_t>(offsets[partition_index]), SEEK_SET) == 0
            && read_exact(file, &header, sizeof(header))
            && std::memcmp(header.magic, execution_magic, sizeof(header.magic)) == 0
            && header.schema_version == execution_payload_envelope_schema_version
            && header.header_bytes == sizeof(header)
            && header.endian == execution_payload_endian_marker
            && header.reserved == 0u
            && header.payload_bytes != 0u
            && header.payload_bytes <= std::numeric_limits<std::size_t>::max()
            && offsets[partition_index] <= partition_end
            && sizeof(header) <= partition_end - offsets[partition_index]
            && header.payload_bytes == partition_end - offsets[partition_index]
                - sizeof(header)
            && execution_payload_identity_matches(header.identity, expected);
    }
    void *storage = nullptr;
    if (ok) {
        storage = std::malloc(static_cast<std::size_t>(header.payload_bytes));
        ok = storage != nullptr
            && read_exact(file, storage, static_cast<std::size_t>(header.payload_bytes))
            && header.envelope_checksum == envelope_checksum(
                header, storage, static_cast<std::size_t>(header.payload_bytes));
    }
    std::free(offsets);
    std::fclose(file);
    if (!ok) {
        std::free(storage);
        return 0;
    }
    out->identity = header.identity;
    out->storage = storage;
    out->payload = static_cast<const unsigned char *>(storage);
    out->payload_bytes = static_cast<std::size_t>(header.payload_bytes);
    return 1;
}

void clear_execution_payload_host(execution_payload_host *payload) noexcept {
    if (payload == nullptr) return;
    std::free(payload->storage);
    *payload = execution_payload_host{};
}

#if CELLSHARD_ENABLE_CUDA
cudaError_t upload_execution_payload_async(
    const execution_payload_host &host,
    int device_id,
    cudaStream_t caller_stream,
    execution_payload_device *out) {
    if (out == nullptr || host.storage == nullptr || host.payload == nullptr
        || host.payload_bytes == 0u || !valid_execution_payload_identity(host.identity)
        || device_id < 0 || out->storage != nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) return status;
    void *storage = nullptr;
    status = cudaMalloc(&storage, host.payload_bytes);
    if (status != cudaSuccess) return status;
    status = cudaMemcpyAsync(storage, host.payload, host.payload_bytes,
        cudaMemcpyHostToDevice, caller_stream);
    if (status != cudaSuccess) {
        cudaFree(storage);
        return status;
    }
    out->identity = host.identity;
    out->storage = storage;
    out->payload = static_cast<const unsigned char *>(storage);
    out->payload_bytes = host.payload_bytes;
    out->device_id = device_id;
    return cudaSuccess;
}

cudaError_t clear_execution_payload_device(
    execution_payload_device *payload) noexcept {
    if (payload == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaSuccess;
    if (payload->storage != nullptr) {
        if (payload->device_id >= 0) status = cudaSetDevice(payload->device_id);
        if (status == cudaSuccess) status = cudaFree(payload->storage);
    }
    *payload = execution_payload_device{};
    return status;
}
#endif

} // namespace cellshard
