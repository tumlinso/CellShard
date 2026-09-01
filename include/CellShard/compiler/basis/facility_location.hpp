#pragma once

#include "CellShard/compiler/basis/greedy.hpp"

namespace cellshard::compiler::basis {

struct facility_edge { local_index client = 0; std::uint64_t benefit = 0; };
struct facility_input {
    global_id atom_id = 0;
    local_index first_edge = 0;
    local_index edge_count = 0;
    std::uint64_t opening_cost = 0;
};
struct facility_view {
    const facility_input* facilities = nullptr;
    local_index facility_count = 0;
    const facility_edge* edges = nullptr;
    local_index edge_count = 0;
    local_index client_count = 0;
};

inline std::uint64_t facility_gain(const facility_view& view, local_index facility,
                                   const std::uint64_t* current) noexcept {
    if (facility >= view.facility_count) return 0;
    const auto& item = view.facilities[facility];
    const std::uint64_t end = static_cast<std::uint64_t>(item.first_edge) + item.edge_count;
    if (end > view.edge_count) return 0;
    std::uint64_t gain = 0;
    for (std::uint64_t i = item.first_edge; i < end; ++i) {
        const auto& edge = view.edges[i];
        if (edge.client >= view.client_count || edge.benefit <= current[edge.client]) continue;
        const auto delta = edge.benefit - current[edge.client];
        gain = delta > UINT64_MAX - gain ? UINT64_MAX : gain + delta;
    }
    return gain > item.opening_cost ? gain - item.opening_cost : 0;
}

inline basis_solution facility_location_select(const facility_view& view,
                                               std::uint64_t* current,
                                               local_index* output,
                                               local_index capacity) noexcept {
    basis_solution result{output, capacity, 0, 0, false};
    if (view.facilities == nullptr || view.edges == nullptr || current == nullptr || output == nullptr) return result;
    while (result.count < capacity) {
        local_index best = invalid_local_index; std::uint64_t best_gain = 0;
        for (local_index i = 0; i < view.facility_count; ++i) {
            if (already_selected(result, i)) continue;
            const auto gain = facility_gain(view, i, current);
            if (gain > best_gain || (gain == best_gain && gain != 0 &&
                (best == invalid_local_index || view.facilities[i].atom_id < view.facilities[best].atom_id))) {
                best = i; best_gain = gain;
            }
        }
        if (best == invalid_local_index) break;
        output[result.count++] = best;
        const auto& chosen = view.facilities[best];
        const std::uint64_t end = static_cast<std::uint64_t>(chosen.first_edge) + chosen.edge_count;
        for (std::uint64_t i = chosen.first_edge; i < end; ++i) {
            const auto& edge = view.edges[i];
            if (edge.client < view.client_count && edge.benefit > current[edge.client]) current[edge.client] = edge.benefit;
        }
    }
    return result;
}

}  // namespace cellshard::compiler::basis
