#include <CellShard/runtime/residency.hh>
#include <CellShard/runtime/source/local_file_source.hh>

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <unistd.h>

namespace {
void require(bool value, const char *message) {
    if (!value) {
        std::fprintf(stderr, "cellShardHostResidencyTest: %s\n", message);
        std::exit(1);
    }
}
cellshard::content_digest digest(const std::byte *bytes, std::size_t count) {
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t i = 0; i < count; ++i) {
        hash ^= std::to_integer<unsigned char>(bytes[i]);
        hash *= 1099511628211ull;
    }
    if (hash == 0) hash = 1;
    cellshard::content_digest result{};
    result.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    result.used_bytes = 8;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        result.bytes[shift / 8] = std::byte((hash >> shift) & 0xffu);
    }
    return result;
}
struct counters { int allocations = 0; int releases = 0; };
void *count_allocate(void *context, std::size_t bytes,
                     std::size_t alignment) noexcept {
    auto *state = static_cast<counters *>(context);
    ++state->allocations;
    void *result = nullptr;
    return ::posix_memalign(&result, alignment, bytes) == 0 ? result : nullptr;
}
void count_release(void *context, void *allocation) noexcept {
    ++static_cast<counters *>(context)->releases;
    std::free(allocation);
}
const cellshard::host_allocator_ops counting_ops{&count_allocate, &count_release};
}

int main() {
    using namespace cellshard;
    static_assert(!std::is_copy_constructible<local_file_source>::value);
    static_assert(!std::is_copy_constructible<host_residency>::value);
    const std::array<std::byte, 12> payload{{
        std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
        std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8},
        std::byte{9}, std::byte{10}, std::byte{11}, std::byte{12}}};
    const std::string path = "/tmp/cellshard-host-residency-"
        + std::to_string(static_cast<unsigned long long>(::getpid()));
    std::FILE *file = std::fopen(path.c_str(), "wb");
    require(file != nullptr, "create source file");
    const std::array<std::byte, 256> prefix{};
    require(std::fwrite(prefix.data(), 1, prefix.size(), file) == prefix.size()
            && std::fwrite(payload.data(), 1, payload.size(), file) == payload.size()
            && std::fclose(file) == 0, "write source file");

    storage_object_descriptor object{storage_object_id{1},
        prefix.size() + payload.size(), digest(prefix.data(), prefix.size())};
    const auto payload_digest = digest(payload.data(), payload.size());
    extent_descriptor extent{extent_id{2}, object.id, prefix.size(), payload.size(),
                             256, payload_digest};
    image_descriptor image{};
    image.id = image_id{3};
    image.projection = {producer_abi_id{4}, structure_id{5}, geometry_id{6},
                        operator_class_id{7}, scalar_encoding_id{8},
                        {execution_backend::cpu, 0, 0, 0}};
    image.stored_bytes = payload.size();
    image.device_bytes = payload.size();
    image.required_alignment = 256;
    image.reuse = image_reuse_class::bounded_reuse;
    image.payload_digest = payload_digest;
    image.domains = {{domain_binding_role::primary, domain_id{9},
                      partition_map_id{10}, partition_id{11}, order_id{12}}};

    local_file_source source{};
    require(open_local_file_source("/tmp/does-not-exist-cellshard",
                source_provider_id{20}, source_location_id{21}, object, &source)
                == status_code::missing_object, "missing source rejected");
    require(open_local_file_source(path.c_str(), source_provider_id{20},
                source_location_id{21}, object, &source) == status_code::success,
            "open local source");
    require(valid_payload_source_ref(source.ref()), "valid source reference");
    std::array<std::byte, 4> scratch{};
    require(read_exact(source.ref(), {object.id, object.byte_count - 1, 4,
                                      scratch.data(), scratch.size()})
                == status_code::invalid_input, "out of bounds read rejected");

    counters counts{};
    host_residency residency{};
    require(load_host_residency(source.ref(), object, extent, view_of(image),
                                {&counts, &counting_ops}, &residency)
                == status_code::success, "load host residency");
    const auto view = residency.view();
    require(counts.allocations == 1 && counts.releases == 0
            && view.image == image.id && view.payload_bytes == payload.size()
            && reinterpret_cast<std::uintptr_t>(view.payload) % 256 == 0
            && std::equal(payload.begin(), payload.end(), view.payload),
            "one allocation direct opaque read");
    host_residency moved(std::move(residency));
    require(!residency.valid() && moved.valid(), "move-only residency ownership");
    moved.reset();
    require(counts.releases == 1, "residency releases owning allocation");

    auto bad_image = image;
    bad_image.payload_digest.bytes[0] ^= std::byte{1};
    auto bad_extent = extent;
    bad_extent.payload_digest = bad_image.payload_digest;
    require(load_host_residency(source.ref(), object, bad_extent,
                                view_of(bad_image), {&counts, &counting_ops},
                                &residency) == status_code::corruption,
            "payload digest mismatch rejected");
    require(counts.allocations == 2 && counts.releases == 2 && !residency.valid(),
            "checksum failure cleanup");

    require(::truncate(path.c_str(), prefix.size() + payload.size() - 1) == 0,
            "truncate open source");
    require(read_extent_exact(source.ref(), extent, object, scratch.data(),
                              scratch.size()) == status_code::invalid_input,
            "undersized destination rejected before read");
    std::array<std::byte, 12> full_scratch{};
    require(read_extent_exact(source.ref(), extent, object, full_scratch.data(),
                              full_scratch.size()) == status_code::short_read,
            "short pread reported");
    source.reset();
    require(!source.valid(), "source cleanup");
    std::remove(path.c_str());
    std::puts("cellShardHostResidencyTest: passed");
    return 0;
}
