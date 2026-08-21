#include <CellShard/io/pack/execution_payload.cuh>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

namespace cs = cellshard;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "cellShardExecutionPayloadTest: %s\n", message);
        std::exit(1);
    }
}

std::string temporary_path() {
    std::string value = "/tmp/cellshard_execution_payloadXXXXXX";
    const int descriptor = ::mkstemp(value.data());
    require(descriptor >= 0, "mkstemp");
    ::close(descriptor);
    ::unlink(value.c_str());
    return value + ".cspack";
}

cs::execution_payload_identity identity(std::uint64_t partition, std::uint64_t row) {
    cs::execution_payload_identity result;
    result.dataset_identity = 0x1100u;
    result.generation = {1u, 2u, 3u, 4u};
    result.partition_identity = partition;
    result.global_row_begin = row;
    result.row_count = 3u;
    result.feature_count = 7u;
    result.feature_axis_fingerprint = 0x2200u;
    result.feature_axis_fingerprint_version = 1u;
    result.payload_kind = 0x43504b31u;
    result.payload_schema_version = 1u;
    result.row_domain_identity = 0x3300u;
    result.payload_identity = 0x4400u + partition;
    return result;
}

void flip_last_byte(const std::string &path) {
    std::FILE *file = std::fopen(path.c_str(), "rb+");
    require(file != nullptr, "open tamper file");
    require(::fseeko(file, -1, SEEK_END) == 0, "seek tamper byte");
    unsigned char value = 0u;
    require(std::fread(&value, 1u, 1u, file) == 1u, "read tamper byte");
    value ^= 0x80u;
    require(::fseeko(file, -1, SEEK_END) == 0, "reseak tamper byte");
    require(std::fwrite(&value, 1u, 1u, file) == 1u, "write tamper byte");
    std::fclose(file);
}

} // namespace

int main() {
    const std::string path = temporary_path();
    const unsigned char first[] = {1u, 2u, 3u, 4u};
    const unsigned char second[] = {8u, 7u, 6u, 5u, 4u, 3u};
    cs::execution_payload_source sources[2];
    sources[0] = {identity(0x100u, 0u), first, sizeof(first)};
    sources[1] = {identity(0x101u, 3u), second, sizeof(second)};
    require(cs::store_execution_cspack(path.c_str(), 9u, sources, 2u) != 0,
        "store two-part CSPACK");
    require(::access((path + ".tmp").c_str(), F_OK) != 0,
        "temporary publication file must be gone");

    cs::execution_payload_host loaded;
    require(cs::load_execution_cspack_partition(path.c_str(), 9u, 1u,
        sources[1].identity, &loaded) != 0, "load second partition");
    require(loaded.payload_bytes == sizeof(second)
        && std::memcmp(loaded.payload, second, sizeof(second)) == 0,
        "loaded payload bytes");

    auto wrong = sources[1].identity;
    ++wrong.generation.pack_generation;
    cs::execution_payload_host rejected;
    require(cs::load_execution_cspack_partition(path.c_str(), 9u, 1u,
        wrong, &rejected) == 0, "generation mismatch rejection");
    wrong = sources[1].identity;
    wrong.feature_axis_fingerprint ^= 1u;
    require(cs::load_execution_cspack_partition(path.c_str(), 9u, 1u,
        wrong, &rejected) == 0, "feature identity mismatch rejection");
    require(cs::load_execution_cspack_partition(path.c_str(), 10u, 1u,
        sources[1].identity, &rejected) == 0, "shard mismatch rejection");

    int device_count = 0;
    require(cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0,
        "CUDA device required");
    cudaStream_t stream = nullptr;
    require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess,
        "create stream");
    cs::execution_payload_device invalid_device;
    require(cs::upload_execution_payload_async(loaded, -1, stream, &invalid_device)
        == cudaErrorInvalidValue, "invalid device rejection");
    require(invalid_device.storage == nullptr && invalid_device.payload == nullptr,
        "invalid upload leaves no allocation");
    cs::execution_payload_device device;
    require(cs::upload_execution_payload_async(loaded, 0, stream, &device)
        == cudaSuccess, "async payload upload");
    require(cs::upload_execution_payload_async(loaded, 0, stream, &device)
        == cudaErrorInvalidValue, "occupied device output rejection");
    unsigned char round_trip[sizeof(second)]{};
    require(cudaMemcpyAsync(round_trip, device.payload, sizeof(round_trip),
        cudaMemcpyDeviceToHost, stream) == cudaSuccess, "download staged payload");
    require(cudaStreamSynchronize(stream) == cudaSuccess, "synchronize payload copy");
    require(std::memcmp(round_trip, second, sizeof(second)) == 0,
        "device payload bytes");
    require(cs::clear_execution_payload_device(&device) == cudaSuccess,
        "release device payload");
    require(device.storage == nullptr && device.payload == nullptr
        && device.payload_bytes == 0u && device.device_id == -1,
        "device cleanup clears ownership state");
    require(cs::clear_execution_payload_device(&device) == cudaSuccess,
        "repeat device cleanup");
    require(cudaStreamDestroy(stream) == cudaSuccess, "destroy stream");
    cs::clear_execution_payload_host(&loaded);

    flip_last_byte(path);
    require(cs::load_execution_cspack_partition(path.c_str(), 9u, 1u,
        sources[1].identity, &rejected) == 0, "checksum tamper rejection");
    require(cs::store_execution_cspack(path.c_str(), 9u, sources, 2u) != 0,
        "restore CSPACK");

    sources[1].identity.partition_identity = sources[0].identity.partition_identity;
    require(cs::store_execution_cspack(path.c_str(), 9u, sources, 2u) == 0,
        "duplicate partition identity rejection");
    ::unlink(path.c_str());
    std::puts("cellShardExecutionPayloadTest: passed");
    return 0;
}
