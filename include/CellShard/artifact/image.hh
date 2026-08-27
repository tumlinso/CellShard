#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <CellShard/domain.hh>
#include <CellShard/identity.hh>

namespace cellshard {

enum class execution_backend : std::uint32_t {
    invalid = 0,
    cpu,
    cuda,
    producer_defined,
};

struct target_capabilities {
    execution_backend backend = execution_backend::invalid;
    std::uint32_t capability_major = 0;
    std::uint32_t capability_minor = 0;
    std::uint64_t capability_flags = 0;
};

struct projection_key {
    producer_abi_id producer{};
    structure_id structure{};
    geometry_id geometry{};
    operator_class_id operation{};
    scalar_encoding_id encoding{};
    target_capabilities target{};
};

enum class image_reuse_class : std::uint32_t {
    invalid = 0,
    single_use,
    bounded_reuse,
    durable_reuse,
};

struct image_descriptor {
    image_id id{};
    projection_key projection{};
    std::uint64_t stored_bytes = 0;
    std::uint64_t device_bytes = 0;
    std::uint32_t required_alignment = 0;
    image_reuse_class reuse = image_reuse_class::invalid;
    content_digest payload_digest{};
    std::vector<domain_binding> domains{};
    std::vector<image_id> dependencies{};
    std::vector<route_table_id> routes{};
};

struct image_descriptor_view {
    image_id id{};
    projection_key projection{};
    std::uint64_t stored_bytes = 0;
    std::uint64_t device_bytes = 0;
    std::uint32_t required_alignment = 0;
    image_reuse_class reuse = image_reuse_class::invalid;
    content_digest payload_digest{};
    array_view<domain_binding> domains{};
    array_view<image_id> dependencies{};
    array_view<route_table_id> routes{};
};

static_assert(std::is_trivially_copyable<image_descriptor_view>::value,
              "image descriptor views must stay allocation-free values");

[[nodiscard]] constexpr bool valid_execution_backend(
    execution_backend backend) noexcept {
    switch (backend) {
    case execution_backend::cpu:
    case execution_backend::cuda:
    case execution_backend::producer_defined:
        return true;
    case execution_backend::invalid:
        return false;
    }
    return false;
}

[[nodiscard]] constexpr bool valid_target_capabilities(
    const target_capabilities &target) noexcept {
    if (!valid_execution_backend(target.backend)) {
        return false;
    }
    if (target.backend == execution_backend::cpu) {
        return target.capability_major == 0 && target.capability_minor == 0;
    }
    if (target.backend == execution_backend::cuda) {
        return target.capability_major != 0;
    }
    return target.capability_flags != 0;
}

[[nodiscard]] constexpr bool valid_projection_key(
    const projection_key &projection) noexcept {
    return projection.producer.valid() && projection.structure.valid()
        && projection.geometry.valid() && projection.operation.valid()
        && projection.encoding.valid()
        && valid_target_capabilities(projection.target);
}

[[nodiscard]] constexpr bool valid_image_reuse_class(
    image_reuse_class reuse) noexcept {
    switch (reuse) {
    case image_reuse_class::single_use:
    case image_reuse_class::bounded_reuse:
    case image_reuse_class::durable_reuse:
        return true;
    case image_reuse_class::invalid:
        return false;
    }
    return false;
}

[[nodiscard]] constexpr bool valid_required_alignment(
    std::uint32_t alignment) noexcept {
    return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

namespace detail {

template<typename T>
[[nodiscard]] inline bool unique_valid_ids(array_view<T> ids) noexcept {
    for (std::size_t i = 0; i < ids.size; ++i) {
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

[[nodiscard]] inline bool valid_domain_bindings(
    array_view<domain_binding> bindings) noexcept {
    if (bindings.empty()) {
        return false;
    }
    for (std::size_t i = 0; i < bindings.size; ++i) {
        const auto &binding = bindings[i];
        if (!valid_domain_binding_role(binding.role) || !binding.domain.valid()
            || !binding.map.valid() || !binding.partition.valid()
            || !binding.order.valid()) {
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            const auto &previous = bindings[j];
            if (binding.role == previous.role && binding.domain == previous.domain
                && binding.map == previous.map
                && binding.partition == previous.partition
                && binding.order == previous.order) {
                return false;
            }
        }
    }
    return true;
}

} // namespace detail

[[nodiscard]] inline bool valid_image_descriptor(
    const image_descriptor_view &image) noexcept {
    if (!image.id.valid() || !valid_projection_key(image.projection)
        || image.stored_bytes == 0 || image.device_bytes == 0
        || !valid_required_alignment(image.required_alignment)
        || !valid_image_reuse_class(image.reuse)
        || image.payload_digest.algorithm == digest_algorithm::none
        || !valid_content_digest(image.payload_digest)
        || !detail::valid_domain_bindings(image.domains)
        || !detail::unique_valid_ids(image.dependencies)
        || !detail::unique_valid_ids(image.routes)) {
        return false;
    }
    for (std::size_t index = 0; index < image.dependencies.size; ++index) {
        if (image.dependencies[index] == image.id) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline image_descriptor_view view_of(
    const image_descriptor &image) noexcept {
    return image_descriptor_view{
        image.id,
        image.projection,
        image.stored_bytes,
        image.device_bytes,
        image.required_alignment,
        image.reuse,
        image.payload_digest,
        {image.domains.data(), image.domains.size()},
        {image.dependencies.data(), image.dependencies.size()},
        {image.routes.data(), image.routes.size()},
    };
}

[[nodiscard]] inline bool valid_image_descriptor(
    const image_descriptor &image) noexcept {
    return valid_image_descriptor(view_of(image));
}

} // namespace cellshard
