#include <CellShard/runtime/source.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

int fail(const char *message) {
    std::fprintf(stderr, "cellShardPayloadSourceTest: %s\n", message);
    return 1;
}

cellshard::content_digest digest(std::byte first) {
    cellshard::content_digest result{};
    result.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    result.used_bytes = sizeof(std::uint64_t);
    result.bytes[0] = first;
    return result;
}

struct memory_source {
    cellshard::storage_object_id object{};
    const std::byte *bytes = nullptr;
    std::size_t size = 0;
    std::size_t reads = 0;
};

cellshard::status_code memory_read_exact(
    void *opaque, const cellshard::exact_read_request &request) noexcept {
    auto &source = *static_cast<memory_source *>(opaque);
    if (request.object != source.object) {
        return cellshard::status_code::missing_object;
    }
    if (request.object_offset > source.size
        || request.byte_count > source.size - request.object_offset) {
        return cellshard::status_code::short_read;
    }
    std::memcpy(request.destination, source.bytes + request.object_offset,
                static_cast<std::size_t>(request.byte_count));
    ++source.reads;
    return cellshard::status_code::success;
}

constexpr cellshard::payload_source_ops memory_ops{memory_read_exact};

} // namespace

int main() {
    using namespace cellshard;

    const storage_object_descriptor object_a{
        storage_object_id{1}, 16, digest(std::byte{0x11})};
    const storage_object_descriptor object_b{
        storage_object_id{2}, 32, digest(std::byte{0x22})};
    const extent_descriptor extent_a{
        extent_id{11}, object_a.id, 4, 6, 4, digest(std::byte{0x33})};
    const extent_descriptor extent_b{
        extent_id{12}, object_b.id, 8, 8, 8, digest(std::byte{0x44})};
    if (!valid_extent_descriptor(extent_a, object_a)
        || !valid_extent_descriptor(extent_b, object_b)) {
        return fail("independently addressable extents were rejected");
    }

    source_location_descriptor location_a{
        source_location_id{21}, source_provider_id{22}, object_a.id,
        capability_bit(source_capability::exact_range_read)
            | capability_bit(source_capability::stable_size),
        "memory://replica-a"};
    auto replica_a = location_a;
    replica_a.id = source_location_id{23};
    replica_a.locator = "memory://replica-b";
    if (!valid_source_location_descriptor(location_a, object_a)
        || !valid_source_location_descriptor(replica_a, object_a)
        || extent_a.object != object_a.id || extent_a.id != extent_id{11}) {
        return fail("mutable replica locations changed immutable extent identity");
    }

    const std::array<std::byte, 16> bytes{{
        std::byte{0}, std::byte{1}, std::byte{2}, std::byte{3},
        std::byte{4}, std::byte{5}, std::byte{6}, std::byte{7},
        std::byte{8}, std::byte{9}, std::byte{10}, std::byte{11},
        std::byte{12}, std::byte{13}, std::byte{14}, std::byte{15},
    }};
    memory_source memory{object_a.id, bytes.data(), bytes.size(), 0};
    const payload_source_ref source{
        &memory, &memory_ops, location_a.provider, location_a.id, object_a.id,
        object_a.byte_count, location_a.capabilities};
    std::array<std::byte, 6> destination{};
    if (read_extent_exact(source, extent_a, object_a, destination.data(),
                          destination.size()) != status_code::success
        || memory.reads != 1 || destination.front() != std::byte{4}
        || destination.back() != std::byte{9}) {
        return fail("exact-range source did not preserve requested bytes");
    }

    auto wrong_request = exact_read_request{
        object_b.id, 0, 1, destination.data(), destination.size()};
    if (read_exact(source, wrong_request) != status_code::invalid_input) {
        return fail("cross-object read was accepted");
    }
    wrong_request = {object_a.id, 15, 2, destination.data(), destination.size()};
    if (read_exact(source, wrong_request) != status_code::invalid_input) {
        return fail("out-of-bounds read was accepted");
    }
    wrong_request = {object_a.id, 0, 7, destination.data(), destination.size()};
    if (read_exact(source, wrong_request) != status_code::invalid_input) {
        return fail("undersized destination was accepted");
    }

    auto malformed_extent = extent_a;
    malformed_extent.object_offset = 14;
    malformed_extent.byte_count = 3;
    if (valid_extent_descriptor(malformed_extent, object_a)) {
        return fail("extent outside its storage object was accepted");
    }
    malformed_extent = extent_a;
    malformed_extent.required_alignment = 3;
    if (valid_extent_descriptor(malformed_extent, object_a)) {
        return fail("invalid extent alignment was accepted");
    }

    auto malformed_location = location_a;
    malformed_location.object = object_b.id;
    if (valid_source_location_descriptor(malformed_location, object_a)) {
        return fail("cross-object source location was accepted");
    }

    std::puts("cellShardPayloadSourceTest: passed");
    return 0;
}
