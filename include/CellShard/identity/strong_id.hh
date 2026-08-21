#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace cellshard {

template<typename Tag>
class strong_id {
public:
    constexpr strong_id() noexcept = default;
    explicit constexpr strong_id(std::uint64_t value) noexcept : value_(value) {}

    [[nodiscard]] constexpr bool valid() const noexcept { return value_ != 0; }
    explicit constexpr operator bool() const noexcept { return valid(); }
    [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }

    friend constexpr bool operator==(strong_id lhs, strong_id rhs) noexcept {
        return lhs.value_ == rhs.value_;
    }
    friend constexpr bool operator!=(strong_id lhs, strong_id rhs) noexcept {
        return !(lhs == rhs);
    }
    friend constexpr bool operator<(strong_id lhs, strong_id rhs) noexcept {
        return lhs.value_ < rhs.value_;
    }
    friend constexpr bool operator<=(strong_id lhs, strong_id rhs) noexcept {
        return !(rhs < lhs);
    }
    friend constexpr bool operator>(strong_id lhs, strong_id rhs) noexcept {
        return rhs < lhs;
    }
    friend constexpr bool operator>=(strong_id lhs, strong_id rhs) noexcept {
        return !(lhs < rhs);
    }

private:
    std::uint64_t value_ = 0;
};

namespace detail {

template<typename Id>
constexpr bool strong_id_layout_valid =
    sizeof(Id) == sizeof(std::uint64_t)
    && std::is_standard_layout<Id>::value
    && std::is_trivially_copyable<Id>::value;

#define CELLSHARD_DECLARE_STRONG_ID(name) \
    struct name##_tag {};                  \
    using name = strong_id<name##_tag>;    \
    static_assert(detail::strong_id_layout_valid<name>, \
                  #name " must be one trivial 64-bit field")

} // namespace detail

CELLSHARD_DECLARE_STRONG_ID(dataset_id);
CELLSHARD_DECLARE_STRONG_ID(archive_generation_id);
CELLSHARD_DECLARE_STRONG_ID(catalog_generation_id);
CELLSHARD_DECLARE_STRONG_ID(pack_generation_id);
CELLSHARD_DECLARE_STRONG_ID(domain_id);
CELLSHARD_DECLARE_STRONG_ID(partition_map_id);
CELLSHARD_DECLARE_STRONG_ID(partition_id);
CELLSHARD_DECLARE_STRONG_ID(structure_id);
CELLSHARD_DECLARE_STRONG_ID(order_id);
CELLSHARD_DECLARE_STRONG_ID(geometry_id);
CELLSHARD_DECLARE_STRONG_ID(operator_class_id);
CELLSHARD_DECLARE_STRONG_ID(scalar_encoding_id);
CELLSHARD_DECLARE_STRONG_ID(producer_abi_id);
CELLSHARD_DECLARE_STRONG_ID(image_id);
CELLSHARD_DECLARE_STRONG_ID(route_table_id);
CELLSHARD_DECLARE_STRONG_ID(storage_object_id);
CELLSHARD_DECLARE_STRONG_ID(extent_id);
CELLSHARD_DECLARE_STRONG_ID(source_provider_id);
CELLSHARD_DECLARE_STRONG_ID(source_location_id);
CELLSHARD_DECLARE_STRONG_ID(snapshot_id);
CELLSHARD_DECLARE_STRONG_ID(placement_epoch_id);
CELLSHARD_DECLARE_STRONG_ID(residency_id);

#undef CELLSHARD_DECLARE_STRONG_ID

} // namespace cellshard

namespace std {

template<typename Tag>
struct hash<cellshard::strong_id<Tag>> {
    std::size_t operator()(cellshard::strong_id<Tag> id) const noexcept {
        return std::hash<std::uint64_t>{}(id.value());
    }
};

} // namespace std
