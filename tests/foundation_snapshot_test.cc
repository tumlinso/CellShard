#include <CellShard/artifact/snapshot.hh>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

int fail(const char *message) {
    std::fprintf(stderr, "cellShardFoundationSnapshotTest: %s\n", message);
    return 1;
}

cellshard::content_digest digest(std::byte first) {
    cellshard::content_digest result{};
    result.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    result.used_bytes = sizeof(std::uint64_t);
    result.bytes[0] = first;
    return result;
}

cellshard::artifact_catalog make_catalog() {
    using namespace cellshard;
    artifact_catalog catalog{};
    catalog.generation = catalog_generation_id{1};
    catalog.domains = {
        {domain_id{10}, domain_kind::cells, archive_generation_id{2}, 100},
        {domain_id{11}, domain_kind::genes, archive_generation_id{2}, 50},
    };
    catalog.partition_maps = {
        {partition_map_id{20}, domain_id{10}, archive_generation_id{2}, 100, 1},
        {partition_map_id{21}, domain_id{11}, archive_generation_id{2}, 50, 1},
    };
    catalog.partitions = {
        {partition_id{30}, partition_map_id{20}, domain_id{10},
         archive_generation_id{2}, 0, partition_selection::contiguous(0, 100)},
        {partition_id{31}, partition_map_id{21}, domain_id{11},
         archive_generation_id{2}, 0, partition_selection::contiguous(0, 50)},
    };

    image_descriptor dependency{};
    dependency.id = image_id{40};
    dependency.projection = {producer_abi_id{41}, structure_id{42}, geometry_id{43},
                             operator_class_id{44}, scalar_encoding_id{45},
                             {execution_backend::cuda, 7, 0, 1}};
    dependency.stored_bytes = 4;
    dependency.device_bytes = 4;
    dependency.required_alignment = 4;
    dependency.reuse = image_reuse_class::bounded_reuse;
    dependency.payload_digest = digest(std::byte{0x40});
    dependency.domains = {{domain_binding_role::primary, domain_id{10},
                           partition_map_id{20}, partition_id{30}, order_id{46}}};

    auto image = dependency;
    image.id = image_id{50};
    image.stored_bytes = 8;
    image.device_bytes = 16;
    image.required_alignment = 8;
    image.payload_digest = digest(std::byte{0x50});
    image.domains.push_back({domain_binding_role::secondary, domain_id{11},
                             partition_map_id{21}, partition_id{31}, order_id{47}});
    image.dependencies = {dependency.id};
    image.routes = {route_table_id{60}};
    catalog.images = {dependency, image};

    catalog.storage_objects = {
        {storage_object_id{70}, 16, digest(std::byte{0x70})},
        {storage_object_id{71}, 16, digest(std::byte{0x71})},
    };
    catalog.extents = {
        {extent_id{80}, storage_object_id{70}, 0, 4, 4, digest(std::byte{0x80})},
        {extent_id{81}, storage_object_id{70}, 4, 4, 8, digest(std::byte{0x81})},
        {extent_id{82}, storage_object_id{71}, 0, 4, 8, digest(std::byte{0x82})},
    };
    catalog.image_extents = {
        {dependency.id, {extent_id{80}}},
        {image.id, {extent_id{81}, extent_id{82}}},
    };
    return catalog;
}

cellshard::source_catalog make_sources() {
    using namespace cellshard;
    const auto capabilities = capability_bit(source_capability::exact_range_read);
    return {{{source_location_id{90}, source_provider_id{91}, storage_object_id{70},
              capabilities, "memory://object-70-a"},
             {source_location_id{92}, source_provider_id{91}, storage_object_id{70},
              capabilities, "memory://object-70-b"},
             {source_location_id{93}, source_provider_id{91}, storage_object_id{71},
              capabilities, "memory://object-71"}}};
}

cellshard::snapshot_manifest make_snapshot() {
    using namespace cellshard;
    return {snapshot_id{100}, catalog_generation_id{1}, archive_generation_id{2},
            {domain_id{10}, domain_id{11}},
            {partition_map_id{20}, partition_map_id{21}},
            {image_id{40}, image_id{50}}, {route_table_id{60}}};
}

} // namespace

int main() {
    using namespace cellshard;
    const auto catalog = make_catalog();
    auto sources = make_sources();
    const auto snapshot = make_snapshot();
    if (!valid_artifact_catalog(catalog)
        || !valid_source_catalog(sources, catalog)
        || !valid_snapshot_manifest(snapshot, catalog, sources)) {
        return fail("valid catalog and snapshot fan-in was rejected");
    }

    const auto pinned_snapshot = snapshot;
    sources.locations[0].locator = "memory://moved-object-70";
    sources.locations.push_back({source_location_id{94}, source_provider_id{95},
                                 storage_object_id{71},
                                 capability_bit(source_capability::exact_range_read),
                                 "memory://object-71-replica"});
    if (!valid_snapshot_manifest(pinned_snapshot, catalog, sources)
        || pinned_snapshot.id != snapshot.id
        || pinned_snapshot.images != snapshot.images) {
        return fail("source mutation changed an immutable snapshot");
    }

    auto malformed_catalog = catalog;
    malformed_catalog.images.push_back(malformed_catalog.images.front());
    if (valid_artifact_catalog(malformed_catalog)) {
        return fail("duplicate image identity was accepted");
    }
    malformed_catalog = catalog;
    malformed_catalog.images[1].domains[1].map = partition_map_id{20};
    if (valid_artifact_catalog(malformed_catalog)) {
        return fail("mixed-domain binding was accepted");
    }
    malformed_catalog = catalog;
    malformed_catalog.image_extents[1].extents.pop_back();
    if (valid_artifact_catalog(malformed_catalog)) {
        return fail("image byte total mismatch was accepted");
    }
    malformed_catalog = catalog;
    malformed_catalog.images[0].dependencies = {image_id{50}};
    if (valid_artifact_catalog(malformed_catalog)) {
        return fail("image dependency cycle was accepted");
    }

    auto malformed_snapshot = snapshot;
    malformed_snapshot.images.erase(malformed_snapshot.images.begin());
    if (valid_snapshot_manifest(malformed_snapshot, catalog, sources)) {
        return fail("snapshot missing an image dependency was accepted");
    }
    malformed_snapshot = snapshot;
    malformed_snapshot.routes.clear();
    if (valid_snapshot_manifest(malformed_snapshot, catalog, sources)) {
        return fail("snapshot missing a route identity was accepted");
    }
    malformed_snapshot = snapshot;
    malformed_snapshot.archive_generation = archive_generation_id{3};
    if (valid_snapshot_manifest(malformed_snapshot, catalog, sources)) {
        return fail("mixed archive generation was accepted");
    }

    auto missing_source = sources;
    for (auto iterator = missing_source.locations.begin();
         iterator != missing_source.locations.end();) {
        if (iterator->object == storage_object_id{71}) {
            iterator = missing_source.locations.erase(iterator);
        } else {
            ++iterator;
        }
    }
    if (valid_snapshot_manifest(snapshot, catalog, missing_source)) {
        return fail("snapshot with missing extent source was accepted");
    }

    std::puts("cellShardFoundationSnapshotTest: passed");
    return 0;
}
