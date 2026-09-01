#include <CellShard/runtime/v2/residency_admission.hh>

#include <algorithm>
#include <limits>

namespace cellshard::runtime_v2 {

status_code plan_residency_admission(
    const residency_admission_request &request, residency_id *evictions,
    std::size_t eviction_capacity, bool *selected,
    std::size_t selection_capacity, residency_admission_plan *out) noexcept {
    if (request.capacity_bytes == 0 || request.requested_bytes == 0
        || request.resident_bytes > request.capacity_bytes
        || request.requested_bytes > request.capacity_bytes || out == nullptr
        || (request.candidates.size != 0
            && (evictions == nullptr || selected == nullptr
                || selection_capacity < request.candidates.size))) {
        return status_code::invalid_input;
    }
    *out = {};
    if (request.requested_bytes
        <= request.capacity_bytes - request.resident_bytes) {
        return status_code::success;
    }
    const std::uint64_t required = request.requested_bytes
        - (request.capacity_bytes - request.resident_bytes);
    std::fill(selected, selected + request.candidates.size, false);
    std::size_t eviction_count = 0;
    std::uint64_t evicted_bytes = 0;
    std::uint64_t reconstruction_ns = 0;
    while (evicted_bytes < required) {
        std::size_t best = request.candidates.size;
        for (std::size_t i = 0; i < request.candidates.size; ++i) {
            const auto &candidate = request.candidates[i];
            if (selected[i] || !candidate.residency.valid()
                || candidate.bytes == 0 || candidate.active_pins != 0) {
                continue;
            }
            if (best == request.candidates.size
                || candidate.reconstruct_nanoseconds
                       < request.candidates[best].reconstruct_nanoseconds
                || (candidate.reconstruct_nanoseconds
                        == request.candidates[best].reconstruct_nanoseconds
                    && candidate.last_use_epoch
                           < request.candidates[best].last_use_epoch)
                || (candidate.reconstruct_nanoseconds
                        == request.candidates[best].reconstruct_nanoseconds
                    && candidate.last_use_epoch
                           == request.candidates[best].last_use_epoch
                    && candidate.residency <
                           request.candidates[best].residency)) {
                best = i;
            }
        }
        if (best == request.candidates.size || eviction_count == eviction_capacity) {
            return status_code::unsupported_capability;
        }
        const auto &victim = request.candidates[best];
        if (evicted_bytes > std::numeric_limits<std::uint64_t>::max()
                                - victim.bytes
            || reconstruction_ns > std::numeric_limits<std::uint64_t>::max()
                                        - victim.reconstruct_nanoseconds) {
            return status_code::invalid_input;
        }
        selected[best] = true;
        evictions[eviction_count++] = victim.residency;
        evicted_bytes += victim.bytes;
        reconstruction_ns += victim.reconstruct_nanoseconds;
    }
    *out = residency_admission_plan{{evictions, eviction_count}, evicted_bytes,
                                    reconstruction_ns};
    return status_code::success;
}

} // namespace cellshard::runtime_v2
