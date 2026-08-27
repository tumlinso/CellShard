#pragma once

#include <cstddef>
#include <vector>

#include <CellShard/artifact/extent.hh>
#include <CellShard/artifact/image.hh>

namespace cellshard {

struct image_extent_set {
    image_id image{};
    std::vector<extent_id> extents{};
};

struct artifact_catalog {
    catalog_generation_id generation{};
    std::vector<domain_descriptor> domains{};
    std::vector<partition_map_descriptor> partition_maps{};
    std::vector<partition_descriptor> partitions{};
    std::vector<image_descriptor> images{};
    std::vector<storage_object_descriptor> storage_objects{};
    std::vector<extent_descriptor> extents{};
    std::vector<image_extent_set> image_extents{};
};

// Locations are operational state and deliberately live outside the immutable
// artifact catalog and snapshot identity.
struct source_catalog {
    std::vector<source_location_descriptor> locations{};
};

template<typename T, typename Id, typename IdOf>
[[nodiscard]] inline const T *catalog_find(const std::vector<T> &items,
                                           Id id, IdOf id_of) noexcept {
    for (const auto &item : items) {
        if (id_of(item) == id) {
            return &item;
        }
    }
    return nullptr;
}

[[nodiscard]] inline const domain_descriptor *find_domain(
    const artifact_catalog &catalog, domain_id id) noexcept {
    return catalog_find(catalog.domains, id,
                        [](const auto &item) { return item.id; });
}

[[nodiscard]] inline const partition_map_descriptor *find_partition_map(
    const artifact_catalog &catalog, partition_map_id id) noexcept {
    return catalog_find(catalog.partition_maps, id,
                        [](const auto &item) { return item.id; });
}

[[nodiscard]] inline const partition_descriptor *find_partition(
    const artifact_catalog &catalog, partition_id id) noexcept {
    return catalog_find(catalog.partitions, id,
                        [](const auto &item) { return item.id; });
}

[[nodiscard]] inline const image_descriptor *find_image(
    const artifact_catalog &catalog, image_id id) noexcept {
    return catalog_find(catalog.images, id,
                        [](const auto &item) { return item.id; });
}

[[nodiscard]] inline const storage_object_descriptor *find_storage_object(
    const artifact_catalog &catalog, storage_object_id id) noexcept {
    return catalog_find(catalog.storage_objects, id,
                        [](const auto &item) { return item.id; });
}

[[nodiscard]] inline const extent_descriptor *find_extent(
    const artifact_catalog &catalog, extent_id id) noexcept {
    return catalog_find(catalog.extents, id,
                        [](const auto &item) { return item.id; });
}

[[nodiscard]] inline const image_extent_set *find_image_extents(
    const artifact_catalog &catalog, image_id id) noexcept {
    return catalog_find(catalog.image_extents, id,
                        [](const auto &item) { return item.image; });
}

[[nodiscard]] inline bool source_catalog_has_object(
    const source_catalog &catalog, storage_object_id object) noexcept {
    for (const auto &location : catalog.locations) {
        if (location.object == object) {
            return true;
        }
    }
    return false;
}

} // namespace cellshard
