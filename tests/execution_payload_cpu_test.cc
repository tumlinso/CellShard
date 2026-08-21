#include <CellShard/io/pack/execution_payload.cuh>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace cs = cellshard;

namespace {

constexpr unsigned char cspack_magic[8] = {'C','S','P','A','C','K','0','1'};
constexpr std::size_t shard_offset = sizeof(cspack_magic);
constexpr std::size_t count_offset = shard_offset + sizeof(std::uint64_t);
constexpr std::size_t offsets_offset = count_offset + sizeof(std::uint64_t);

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "cellShardExecutionPayloadCpuTest: %s\n", message);
        std::exit(1);
    }
}

std::string temporary_path() {
    std::string value = "/tmp/cellshard_execution_payload_cpuXXXXXX";
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

std::vector<unsigned char> read_file(const std::string &path) {
    std::FILE *file = std::fopen(path.c_str(), "rb");
    require(file != nullptr, "open file for read");
    require(::fseeko(file, 0, SEEK_END) == 0, "seek file end");
    const off_t end = ::ftello(file);
    require(end >= 0, "measure file");
    require(::fseeko(file, 0, SEEK_SET) == 0, "rewind file");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
    require(bytes.empty() || std::fread(bytes.data(), 1u, bytes.size(), file) == bytes.size(),
        "read file bytes");
    require(std::fclose(file) == 0, "close read file");
    return bytes;
}

void write_file(const std::string &path, const std::vector<unsigned char> &bytes) {
    std::FILE *file = std::fopen(path.c_str(), "wb");
    require(file != nullptr, "open file for write");
    require(bytes.empty() || std::fwrite(bytes.data(), 1u, bytes.size(), file) == bytes.size(),
        "write file bytes");
    require(std::fclose(file) == 0, "close write file");
}

std::uint64_t native_u64(const std::vector<unsigned char> &bytes, std::size_t offset) {
    require(offset <= bytes.size() && sizeof(std::uint64_t) <= bytes.size() - offset,
        "native uint64 bounds");
    std::uint64_t value = 0u;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

void set_native_u64(
    std::vector<unsigned char> &bytes, std::size_t offset, std::uint64_t value) {
    require(offset <= bytes.size() && sizeof(value) <= bytes.size() - offset,
        "set native uint64 bounds");
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void require_rejected(
    const std::string &path,
    std::uint64_t shard,
    std::uint64_t partition,
    const cs::execution_payload_identity &expected,
    const char *message) {
    cs::execution_payload_host rejected;
    require(cs::load_execution_cspack_partition(
        path.c_str(), shard, partition, expected, &rejected) == 0, message);
    require(rejected.storage == nullptr && rejected.payload == nullptr
        && rejected.payload_bytes == 0u, "rejected load remains empty");
}

template <typename Mutator>
void require_identity_mismatch(
    const std::string &path,
    const cs::execution_payload_identity &actual,
    Mutator mutate,
    const char *message) {
    cs::execution_payload_identity wrong = actual;
    mutate(wrong);
    require_rejected(path, 9u, 1u, wrong, message);
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
        "temporary publication file removed after success");

    const std::vector<unsigned char> original = read_file(path);
    const std::size_t table_end = offsets_offset + 2u * sizeof(std::uint64_t);
    require(original.size() > table_end, "payload follows CSPACK table");
    require(std::memcmp(original.data(), cspack_magic, sizeof(cspack_magic)) == 0,
        "CSPACK01 magic bytes");
    require(native_u64(original, shard_offset) == 9u, "top-level shard id");
    require(native_u64(original, count_offset) == 2u, "top-level partition count");
    const std::uint64_t first_offset = native_u64(original, offsets_offset);
    const std::uint64_t second_offset = native_u64(
        original, offsets_offset + sizeof(std::uint64_t));
    require(first_offset == table_end, "first partition begins after table");
    require(first_offset < second_offset && second_offset < original.size(),
        "partition offsets are increasing and bounded");

    cs::execution_payload_host loaded;
    require(cs::load_execution_cspack_partition(path.c_str(), 9u, 1u,
        sources[1].identity, &loaded) != 0, "CPEXEC01 round trip");
    require(loaded.storage != nullptr && loaded.payload == loaded.storage,
        "host payload owns one exposed allocation");
    require(loaded.payload_bytes == sizeof(second)
        && std::memcmp(loaded.payload, second, sizeof(second)) == 0,
        "round-trip payload bytes");
    cs::clear_execution_payload_host(&loaded);
    require(loaded.storage == nullptr && loaded.payload == nullptr
        && loaded.payload_bytes == 0u && loaded.identity.dataset_identity == 0u,
        "host cleanup clears ownership and identity");
    cs::clear_execution_payload_host(&loaded);
    require(loaded.storage == nullptr, "host cleanup is repeatable");

    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.dataset_identity; }, "dataset mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.generation.canonical_generation; },
        "canonical generation mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.generation.execution_plan_generation; },
        "execution-plan generation mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.generation.pack_generation; },
        "pack generation mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.generation.service_epoch; },
        "service epoch mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.partition_identity; }, "partition mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.global_row_begin; }, "row begin mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.row_count; }, "row count mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.feature_count; }, "feature count mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { value.feature_axis_fingerprint ^= 1u; },
        "feature fingerprint mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.feature_axis_fingerprint_version; },
        "feature fingerprint version mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.payload_kind; }, "payload kind mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.payload_schema_version; },
        "payload schema mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.row_domain_identity; },
        "row-domain mismatch rejection");
    require_identity_mismatch(path, sources[1].identity,
        [](auto &value) { ++value.payload_identity; }, "payload identity mismatch rejection");
    require_rejected(path, 10u, 1u, sources[1].identity, "shard mismatch rejection");
    require_rejected(path, 9u, 2u, sources[1].identity, "partition index rejection");

    auto malformed = original;
    malformed[0] ^= 0x80u;
    write_file(path, malformed);
    require_rejected(path, 9u, 1u, sources[1].identity, "wrong CSPACK magic rejection");

    malformed = original;
    set_native_u64(malformed, count_offset, 0u);
    write_file(path, malformed);
    require_rejected(path, 9u, 0u, sources[0].identity, "zero partition count rejection");

    malformed = original;
    set_native_u64(malformed, count_offset, std::numeric_limits<std::uint64_t>::max());
    write_file(path, malformed);
    require_rejected(path, 9u, 0u, sources[0].identity, "oversized partition count rejection");

    malformed = original;
    set_native_u64(malformed, offsets_offset, table_end - 1u);
    write_file(path, malformed);
    require_rejected(path, 9u, 0u, sources[0].identity, "offset before table rejection");

    malformed = original;
    set_native_u64(malformed, offsets_offset + sizeof(std::uint64_t), first_offset);
    write_file(path, malformed);
    require_rejected(path, 9u, 0u, sources[0].identity, "equal offsets rejection");

    malformed = original;
    set_native_u64(malformed, offsets_offset + sizeof(std::uint64_t), first_offset - 1u);
    write_file(path, malformed);
    require_rejected(path, 9u, 0u, sources[0].identity, "descending offsets rejection");

    malformed = original;
    set_native_u64(malformed, offsets_offset + sizeof(std::uint64_t), original.size());
    write_file(path, malformed);
    require_rejected(path, 9u, 1u, sources[1].identity, "offset at end rejection");

    malformed.assign(original.begin(), original.begin() + offsets_offset + sizeof(std::uint64_t));
    write_file(path, malformed);
    require_rejected(path, 9u, 0u, sources[0].identity, "truncated offset table rejection");

    malformed = original;
    malformed.resize(malformed.size() - 1u);
    write_file(path, malformed);
    require_rejected(path, 9u, 1u, sources[1].identity, "truncated payload rejection");

    malformed = original;
    malformed[static_cast<std::size_t>(first_offset)] ^= 0x40u;
    write_file(path, malformed);
    require_rejected(path, 9u, 0u, sources[0].identity, "wrong CPEXEC01 magic rejection");

    malformed = original;
    malformed.back() ^= 0x80u;
    write_file(path, malformed);
    require_rejected(path, 9u, 1u, sources[1].identity, "checksum tamper rejection");
    write_file(path, original);

    cs::execution_payload_source duplicates[2] = {sources[0], sources[1]};
    duplicates[1].identity.partition_identity = duplicates[0].identity.partition_identity;
    require(cs::store_execution_cspack(path.c_str(), 9u, duplicates, 2u) == 0,
        "duplicate partition identity rejection");
    require(::access((path + ".tmp").c_str(), F_OK) != 0,
        "duplicate rejection leaves no temporary file");

    const std::string blocked_destination = path + ".directory";
    require(::mkdir(blocked_destination.c_str(), 0700) == 0,
        "create publication-blocking directory");
    require(cs::store_execution_cspack(
        blocked_destination.c_str(), 9u, sources, 2u) == 0,
        "rename failure rejects publication");
    require(::access((blocked_destination + ".tmp").c_str(), F_OK) != 0,
        "failed publication removes temporary file");
    require(::rmdir(blocked_destination.c_str()) == 0,
        "remove publication-blocking directory");

    ::unlink(path.c_str());
    std::puts("cellShardExecutionPayloadCpuTest: passed");
    return 0;
}
