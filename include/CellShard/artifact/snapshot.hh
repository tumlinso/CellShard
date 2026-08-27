#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <CellShard/artifact/catalog.hh>

namespace cellshard {

struct snapshot_manifest {
    snapshot_id id{};
    catalog_generation_id catalog_generation{};
    archive_generation_id archive_generation{};
    std::vector<domain_id> domains{};
    std::vector<partition_map_id> partition_maps{};
    std::vector<image_id> images{};
    std::vector<route_table_id> routes{};
};

namespace detail {

template<typename T>
[[nodiscard]] inline bool unique_valid_snapshot_ids(
    const std::vector<T> &ids) noexcept {
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (!ids[i].valid()) {
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (ids[i] == ids[j]) {
                return false;
            }
        }
    }
    return true;
}

template<typename T, typename IdOf>
[[nodiscard]] inline bool unique_catalog_ids(const std::vector<T> &items,
                                             IdOf id_of) noexcept {
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (!id_of(items[i]).valid()) {
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (id_of(items[i]) == id_of(items[j])) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] inline bool image_dependency_cycle_visit(
    const artifact_catalog &catalog, std::size_t index,
    std::vector<std::uint8_t> &colors) noexcept {
    if (colors[index] == 1) {
        return false;
    }
    if (colors[index] == 2) {
        return true;
    }
    colors[index] = 1;
    for (const auto dependency : catalog.images[index].dependencies) {
        std::size_t dependency_index = 0;
        while (dependency_index < catalog.images.size()
               && catalog.images[dependency_index].id != dependency) {
            ++dependency_index;
        }
        if (dependency_index == catalog.images.size()
            || !image_dependency_cycle_visit(catalog, dependency_index, colors)) {
            return false;
        }
    }
    colors[index] = 2;
    return true;
}

template<typename T>
[[nodiscard]] inline bool contains_id(const std::vector<T> &ids, T id) noexcept {
    for (const auto candidate : ids) {
        if (candidate == id) {
            return true;
        }
    }
    return false;
}

} // namespace detail

[[nodiscard]] inline bool valid_artifact_catalog(
    const artifact_catalog &catalog) noexcept {
    if (!catalog.generation.valid()
        || !detail::unique_catalog_ids(catalog.domains,
                                      [](const auto &item) { return item.id; })
        || !detail::unique_catalog_ids(catalog.partition_maps,
                                      [](const auto &item) { return item.id; })
        || !detail::unique_catalog_ids(catalog.partitions,
                                      [](const auto &item) { return item.id; })
        || !detail::unique_catalog_ids(catalog.images,
                                      [](const auto &item) { return item.id; })
        || !detail::unique_catalog_ids(catalog.storage_objects,
                                      [](const auto &item) { return item.id; })
        || !detail::unique_catalog_ids(catalog.extents,
                                      [](const auto &item) { return item.id; })
        || !detail::unique_catalog_ids(catalog.image_extents,
                                      [](const auto &item) { return item.image; })) {
        return false;
    }

    for (const auto &domain : catalog.domains) {
        if (!valid_domain_descriptor(domain)) {
            return false;
        }
    }
    for (const auto &map : catalog.partition_maps) {
        const auto *domain = find_domain(catalog, map.domain);
        if (domain == nullptr || !valid_partition_map_descriptor(map, *domain)) {
            return false;
        }
    }
    for (const auto &partition : catalog.partitions) {
        const auto *map = find_partition_map(catalog, partition.map);
        const auto *domain = find_domain(catalog, partition.domain);
        if (map == nullptr || domain == nullptr
            || !valid_partition_descriptor(partition, *map, *domain)) {
            return false;
        }
    }
    for (const auto &object : catalog.storage_objects) {
        if (!valid_storage_object_descriptor(object)) {
            return false;
        }
    }
    for (const auto &extent : catalog.extents) {
        const auto *object = find_storage_object(catalog, extent.object);
        if (object == nullptr || !valid_extent_descriptor(extent, *object)) {
            return false;
        }
    }
    for (const auto &image : catalog.images) {
        if (!valid_image_descriptor(image)) {
            return false;
        }
        for (const auto &binding : image.domains) {
            const auto *partition = find_partition(catalog, binding.partition);
            const auto *map = find_partition_map(catalog, binding.map);
            const auto *domain = find_domain(catalog, binding.domain);
            if (partition == nullptr || map == nullptr || domain == nullptr
                || !valid_domain_binding(binding, *partition, *map, *domain)) {
                return false;
            }
        }
        const auto *binding = find_image_extents(catalog, image.id);
        if (binding == nullptr || binding->extents.empty()) {
            return false;
        }
        std::uint64_t total_bytes = 0;
        for (const auto extent_id_value : binding->extents) {
            const auto *extent = find_extent(catalog, extent_id_value);
            if (extent == nullptr
                || extent->required_alignment < image.required_alignment
                || total_bytes > std::numeric_limits<std::uint64_t>::max()
                                      - extent->byte_count) {
                return false;
            }
            total_bytes += extent->byte_count;
        }
        if (total_bytes != image.stored_bytes) {
            return false;
        }
    }

    std::vector<std::uint8_t> colors(catalog.images.size(), 0);
    for (std::size_t index = 0; index < catalog.images.size(); ++index) {
        if (!detail::image_dependency_cycle_visit(catalog, index, colors)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline bool valid_source_catalog(
    const source_catalog &sources,
    const artifact_catalog &artifacts) noexcept {
    if (!detail::unique_catalog_ids(sources.locations,
                                   [](const auto &item) { return item.id; })) {
        return false;
    }
    for (const auto &location : sources.locations) {
        const auto *object = find_storage_object(artifacts, location.object);
        if (object == nullptr
            || !valid_source_location_descriptor(location, *object)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline bool valid_snapshot_manifest(
    const snapshot_manifest &snapshot,
    const artifact_catalog &artifacts,
    const source_catalog &sources) noexcept {
    if (!snapshot.id.valid() || snapshot.catalog_generation != artifacts.generation
        || !snapshot.archive_generation.valid()
        || snapshot.domains.empty() || snapshot.partition_maps.empty()
        || snapshot.images.empty()
        || !detail::unique_valid_snapshot_ids(snapshot.domains)
        || !detail::unique_valid_snapshot_ids(snapshot.partition_maps)
        || !detail::unique_valid_snapshot_ids(snapshot.images)
        || !detail::unique_valid_snapshot_ids(snapshot.routes)
        || !valid_artifact_catalog(artifacts)
        || !valid_source_catalog(sources, artifacts)) {
        return false;
    }

    for (const auto domain_id_value : snapshot.domains) {
        const auto *domain = find_domain(artifacts, domain_id_value);
        if (domain == nullptr || domain->generation != snapshot.archive_generation) {
            return false;
        }
    }
    for (const auto map_id_value : snapshot.partition_maps) {
        const auto *map = find_partition_map(artifacts, map_id_value);
        if (map == nullptr || map->generation != snapshot.archive_generation
            || !detail::contains_id(snapshot.domains, map->domain)) {
            return false;
        }
    }
    for (const auto image_id_value : snapshot.images) {
        const auto *image = find_image(artifacts, image_id_value);
        const auto *extent_set = find_image_extents(artifacts, image_id_value);
        if (image == nullptr || extent_set == nullptr) {
            return false;
        }
        for (const auto &binding : image->domains) {
            if (!detail::contains_id(snapshot.domains, binding.domain)
                || !detail::contains_id(snapshot.partition_maps, binding.map)) {
                return false;
            }
        }
        for (const auto dependency : image->dependencies) {
            if (!detail::contains_id(snapshot.images, dependency)) {
                return false;
            }
        }
        for (const auto route : image->routes) {
            if (!detail::contains_id(snapshot.routes, route)) {
                return false;
            }
        }
        for (const auto extent_id_value : extent_set->extents) {
            const auto *extent = find_extent(artifacts, extent_id_value);
            if (extent == nullptr || !source_catalog_has_object(sources, extent->object)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace cellshard
