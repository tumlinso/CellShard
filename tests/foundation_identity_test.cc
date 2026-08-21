#include <CellShard/identity.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <unordered_map>

namespace {

template<typename Id>
constexpr bool valid_id_representation =
    sizeof(Id) == sizeof(std::uint64_t)
    && std::is_standard_layout<Id>::value
    && std::is_trivially_copyable<Id>::value;

#define CELLSHARD_ASSERT_ID(name) static_assert(valid_id_representation<cellshard::name>, #name)

CELLSHARD_ASSERT_ID(dataset_id);
CELLSHARD_ASSERT_ID(archive_generation_id);
CELLSHARD_ASSERT_ID(catalog_generation_id);
CELLSHARD_ASSERT_ID(pack_generation_id);
CELLSHARD_ASSERT_ID(domain_id);
CELLSHARD_ASSERT_ID(partition_map_id);
CELLSHARD_ASSERT_ID(partition_id);
CELLSHARD_ASSERT_ID(structure_id);
CELLSHARD_ASSERT_ID(order_id);
CELLSHARD_ASSERT_ID(geometry_id);
CELLSHARD_ASSERT_ID(operator_class_id);
CELLSHARD_ASSERT_ID(scalar_encoding_id);
CELLSHARD_ASSERT_ID(producer_abi_id);
CELLSHARD_ASSERT_ID(image_id);
CELLSHARD_ASSERT_ID(route_table_id);
CELLSHARD_ASSERT_ID(storage_object_id);
CELLSHARD_ASSERT_ID(extent_id);
CELLSHARD_ASSERT_ID(source_provider_id);
CELLSHARD_ASSERT_ID(source_location_id);
CELLSHARD_ASSERT_ID(snapshot_id);
CELLSHARD_ASSERT_ID(placement_epoch_id);
CELLSHARD_ASSERT_ID(residency_id);

#undef CELLSHARD_ASSERT_ID

static_assert(std::is_constructible<cellshard::dataset_id, std::uint64_t>::value,
              "strong IDs require explicit construction from uint64_t");
static_assert(!std::is_convertible<std::uint64_t, cellshard::dataset_id>::value,
              "uint64_t must not implicitly convert to a strong ID");
static_assert(!std::is_convertible<cellshard::dataset_id, std::uint64_t>::value,
              "strong IDs must not implicitly convert to uint64_t");
static_assert(!std::is_constructible<cellshard::image_id, cellshard::extent_id>::value,
              "distinct ID categories must not mix");
static_assert(std::is_trivially_copyable<cellshard::content_digest>::value,
              "content_digest remains codec-friendly metadata");
static_assert(std::is_standard_layout<cellshard::content_digest>::value,
              "content_digest remains codec-friendly metadata");

int fail(const char *message) {
    std::fprintf(stderr, "cellShardFoundationIdentityTest: %s\n", message);
    return 1;
}

} // namespace

int main() {
    const cellshard::dataset_id invalid{};
    const cellshard::dataset_id dataset{41};
    const cellshard::dataset_id same_dataset{41};
    const cellshard::dataset_id later_dataset{42};
    if (invalid.valid() || static_cast<bool>(invalid)) {
        return fail("zero strong ID was accepted as valid");
    }
    if (!dataset.valid() || !static_cast<bool>(dataset) || dataset.value() != 41) {
        return fail("explicit strong ID construction failed");
    }
    if (dataset != same_dataset || !(dataset < later_dataset)
        || !(dataset <= same_dataset) || !(later_dataset > dataset)
        || !(later_dataset >= dataset)) {
        return fail("strong ID comparison failed");
    }

    std::unordered_map<cellshard::dataset_id, unsigned> cold_metadata;
    cold_metadata.emplace(dataset, 7);
    if (cold_metadata.at(same_dataset) != 7) {
        return fail("strong ID hashing failed");
    }

    const std::array<std::uint32_t, 3> values{{2, 3, 5}};
    const cellshard::array_view<std::uint32_t> view{values.data(), values.size()};
    if (view.empty() || view.size != 3 || view[2] != 5
        || view.begin() != values.data() || view.end() != values.data() + values.size()) {
        return fail("array_view did not preserve pointer-plus-count semantics");
    }
    const cellshard::array_view<std::uint32_t> empty_view{};
    if (!empty_view.empty() || empty_view.begin() != empty_view.end()) {
        return fail("empty array_view is malformed");
    }

    const cellshard::content_digest empty_digest{};
    if (!cellshard::valid_content_digest(empty_digest)) {
        return fail("default digest was rejected");
    }

    cellshard::content_digest legacy_digest{};
    legacy_digest.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    legacy_digest.used_bytes = sizeof(std::uint64_t);
    legacy_digest.bytes[0] = std::byte{0x9d};
    if (!cellshard::valid_content_digest(legacy_digest)) {
        return fail("tagged legacy digest was rejected");
    }

    auto malformed_digest = legacy_digest;
    malformed_digest.used_bytes = 7;
    if (cellshard::valid_content_digest(malformed_digest)) {
        return fail("wrong FNV-1a digest length was accepted");
    }
    malformed_digest = empty_digest;
    malformed_digest.bytes[0] = std::byte{1};
    if (cellshard::valid_content_digest(malformed_digest)) {
        return fail("untagged digest bytes were accepted");
    }
    malformed_digest = legacy_digest;
    malformed_digest.bytes[31] = std::byte{1};
    if (cellshard::valid_content_digest(malformed_digest)) {
        return fail("non-deterministic digest padding was accepted");
    }
    malformed_digest = empty_digest;
    malformed_digest.algorithm = static_cast<cellshard::digest_algorithm>(99);
    if (cellshard::valid_content_digest(malformed_digest)) {
        return fail("unsupported digest algorithm was accepted");
    }

    if (!cellshard::status_ok(cellshard::status_code::success)
        || cellshard::status_ok(cellshard::status_code::invalid_input)
        || cellshard::status_ok(cellshard::status_code::missing_object)
        || cellshard::status_ok(cellshard::status_code::short_read)
        || cellshard::status_ok(cellshard::status_code::corruption)
        || cellshard::status_ok(cellshard::status_code::incompatible_image)
        || cellshard::status_ok(cellshard::status_code::unsupported_capability)
        || cellshard::status_ok(cellshard::status_code::allocation_failure)
        || cellshard::status_ok(cellshard::status_code::cuda_failure)) {
        return fail("typed status success classification failed");
    }

    std::puts("cellShardFoundationIdentityTest: passed");
    return 0;
}
