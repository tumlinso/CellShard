#pragma once

#include <CellShard/identity.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::runtime_v2 {

struct eviction_candidate {
    residency_id residency{};
    std::uint64_t bytes = 0;
    std::uint64_t reconstruct_nanoseconds = 0;
    std::uint64_t last_use_epoch = 0;
    std::uint32_t active_pins = 0;
};

struct residency_admission_request {
    std::uint64_t capacity_bytes = 0;
    std::uint64_t resident_bytes = 0;
    std::uint64_t requested_bytes = 0;
    array_view<eviction_candidate> candidates{};
};

struct residency_admission_plan {
    array_view<residency_id> evictions{};
    std::uint64_t evicted_bytes = 0;
    std::uint64_t reconstruction_nanoseconds = 0;
};

[[nodiscard]] status_code plan_residency_admission(
    const residency_admission_request &request, residency_id *eviction_storage,
    std::size_t eviction_capacity, bool *selection_workspace,
    std::size_t selection_capacity, residency_admission_plan *out) noexcept;

static_assert(std::is_trivially_copyable_v<eviction_candidate>);
static_assert(std::is_trivially_copyable_v<residency_admission_request>);
static_assert(std::is_trivially_copyable_v<residency_admission_plan>);

} // namespace cellshard::runtime_v2
