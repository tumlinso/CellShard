#pragma once

#include <cstdint>

#include <CellShard/identity.hh>

namespace cellshard {

enum class domain_kind : std::uint32_t {
    invalid = 0,
    cells,
    genes,
    sequence_coordinates,
    opaque,
};

struct domain_descriptor {
    domain_id id{};
    domain_kind kind = domain_kind::invalid;
    archive_generation_id generation{};
    std::uint64_t element_count = 0;
};

[[nodiscard]] constexpr bool valid_domain_kind(domain_kind kind) noexcept {
    switch (kind) {
    case domain_kind::cells:
    case domain_kind::genes:
    case domain_kind::sequence_coordinates:
    case domain_kind::opaque:
        return true;
    case domain_kind::invalid:
        return false;
    }
    return false;
}

[[nodiscard]] constexpr bool valid_domain_descriptor(
    const domain_descriptor &domain) noexcept {
    return domain.id.valid() && valid_domain_kind(domain.kind)
        && domain.generation.valid() && domain.element_count != 0;
}

} // namespace cellshard
