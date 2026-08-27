#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <CellShard/domain/descriptor.hh>

namespace cellshard {

struct partition_map_descriptor {
    partition_map_id id{};
    domain_id domain{};
    archive_generation_id generation{};
    std::uint64_t domain_element_count = 0;
    std::uint64_t partition_count = 0;
};

struct domain_extent {
    std::uint64_t begin = 0;
    std::uint64_t count = 0;
};

enum class partition_selection_kind : std::uint32_t {
    invalid = 0,
    contiguous,
    explicit_extents,
    opaque,
};

struct partition_selection {
    partition_selection_kind kind = partition_selection_kind::invalid;
    std::uint64_t element_count = 0;
    std::uint64_t contiguous_begin = 0;
    std::vector<domain_extent> extents{};
    content_digest opaque_identity{};

    [[nodiscard]] static partition_selection contiguous(
        std::uint64_t begin, std::uint64_t count) {
        partition_selection selection;
        selection.kind = partition_selection_kind::contiguous;
        selection.element_count = count;
        selection.contiguous_begin = begin;
        return selection;
    }

    [[nodiscard]] static partition_selection explicit_ranges(
        std::vector<domain_extent> ranges) {
        partition_selection selection;
        selection.kind = partition_selection_kind::explicit_extents;
        selection.extents = std::move(ranges);
        for (const auto &range : selection.extents) {
            selection.element_count += range.count;
        }
        return selection;
    }

    [[nodiscard]] static partition_selection opaque(
        std::uint64_t count, content_digest identity) {
        partition_selection selection;
        selection.kind = partition_selection_kind::opaque;
        selection.element_count = count;
        selection.opaque_identity = identity;
        return selection;
    }
};

struct partition_descriptor {
    partition_id id{};
    partition_map_id map{};
    domain_id domain{};
    archive_generation_id generation{};
    std::uint64_t ordinal = 0;
    partition_selection owned{};
};

enum class domain_binding_role : std::uint32_t {
    unspecified = 0,
    primary,
    secondary,
    source,
    destination,
};

struct domain_binding {
    domain_binding_role role = domain_binding_role::unspecified;
    domain_id domain{};
    partition_map_id map{};
    partition_id partition{};
    order_id order{};
};

[[nodiscard]] constexpr bool valid_partition_map_descriptor(
    const partition_map_descriptor &map,
    const domain_descriptor &domain) noexcept {
    return valid_domain_descriptor(domain) && map.id.valid()
        && map.domain == domain.id && map.generation == domain.generation
        && map.domain_element_count == domain.element_count
        && map.partition_count != 0;
}

namespace detail {

[[nodiscard]] constexpr bool valid_bounded_range(
    std::uint64_t begin, std::uint64_t count,
    std::uint64_t domain_element_count) noexcept {
    return count != 0 && begin <= domain_element_count
        && count <= domain_element_count - begin;
}

} // namespace detail

[[nodiscard]] inline bool valid_partition_selection(
    const partition_selection &selection,
    std::uint64_t domain_element_count) noexcept {
    if (domain_element_count == 0 || selection.element_count == 0
        || selection.element_count > domain_element_count) {
        return false;
    }

    switch (selection.kind) {
    case partition_selection_kind::contiguous:
        return selection.extents.empty()
            && selection.opaque_identity.algorithm == digest_algorithm::none
            && detail::valid_bounded_range(selection.contiguous_begin,
                                           selection.element_count,
                                           domain_element_count);
    case partition_selection_kind::explicit_extents: {
        if (selection.contiguous_begin != 0 || selection.extents.empty()
            || selection.opaque_identity.algorithm != digest_algorithm::none) {
            return false;
        }
        std::uint64_t selected = 0;
        std::uint64_t previous_end = 0;
        bool first = true;
        for (const auto &extent : selection.extents) {
            if (!detail::valid_bounded_range(extent.begin, extent.count,
                                             domain_element_count)) {
                return false;
            }
            if (!first && extent.begin < previous_end) {
                return false;
            }
            if (selected > std::numeric_limits<std::uint64_t>::max() - extent.count) {
                return false;
            }
            selected += extent.count;
            previous_end = extent.begin + extent.count;
            first = false;
        }
        return selected == selection.element_count;
    }
    case partition_selection_kind::opaque:
        return selection.contiguous_begin == 0 && selection.extents.empty()
            && selection.opaque_identity.algorithm != digest_algorithm::none
            && valid_content_digest(selection.opaque_identity);
    case partition_selection_kind::invalid:
        return false;
    }
    return false;
}

[[nodiscard]] inline bool valid_partition_descriptor(
    const partition_descriptor &partition,
    const partition_map_descriptor &map,
    const domain_descriptor &domain) noexcept {
    return valid_partition_map_descriptor(map, domain) && partition.id.valid()
        && partition.map == map.id && partition.domain == domain.id
        && partition.generation == domain.generation
        && partition.ordinal < map.partition_count
        && valid_partition_selection(partition.owned, domain.element_count);
}

[[nodiscard]] constexpr bool valid_domain_binding_role(
    domain_binding_role role) noexcept {
    switch (role) {
    case domain_binding_role::primary:
    case domain_binding_role::secondary:
    case domain_binding_role::source:
    case domain_binding_role::destination:
        return true;
    case domain_binding_role::unspecified:
        return false;
    }
    return false;
}

[[nodiscard]] inline bool valid_domain_binding(
    const domain_binding &binding,
    const partition_descriptor &partition,
    const partition_map_descriptor &map,
    const domain_descriptor &domain) noexcept {
    return valid_partition_descriptor(partition, map, domain)
        && valid_domain_binding_role(binding.role)
        && binding.domain == domain.id && binding.map == map.id
        && binding.partition == partition.id && binding.order.valid();
}

} // namespace cellshard
