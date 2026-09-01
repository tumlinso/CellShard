#pragma once

#include "CellShard/compiler/basis/budget.hpp"

namespace cellshard::compiler::basis {

inline constexpr local_index max_pareto_bases = 256;
struct basis_point { global_id basis_id = 0; std::uint64_t utility = 0; basis_usage usage{}; };

inline bool dominates(const basis_point& left, const basis_point& right) noexcept {
    const bool no_worse = left.utility >= right.utility &&
        left.usage.storage_bytes <= right.usage.storage_bytes &&
        left.usage.build_cost <= right.usage.build_cost &&
        left.usage.resident_bytes <= right.usage.resident_bytes &&
        left.usage.mutation_cost <= right.usage.mutation_cost;
    const bool strict = left.utility > right.utility ||
        left.usage.storage_bytes < right.usage.storage_bytes ||
        left.usage.build_cost < right.usage.build_cost ||
        left.usage.resident_bytes < right.usage.resident_bytes ||
        left.usage.mutation_cost < right.usage.mutation_cost;
    return no_worse && strict;
}

inline bool pareto_insert(basis_point candidate, basis_point* portfolio,
                          local_index capacity, local_index& count) noexcept {
    if (candidate.basis_id == 0 || portfolio == nullptr || capacity > max_pareto_bases || count > capacity) return false;
    for (local_index i = 0; i < count; ++i) {
        if (dominates(portfolio[i], candidate)) return false;
        const bool equal = !dominates(candidate, portfolio[i]) && !dominates(portfolio[i], candidate) &&
            candidate.utility == portfolio[i].utility && candidate.usage.storage_bytes == portfolio[i].usage.storage_bytes &&
            candidate.usage.build_cost == portfolio[i].usage.build_cost && candidate.usage.resident_bytes == portfolio[i].usage.resident_bytes &&
            candidate.usage.mutation_cost == portfolio[i].usage.mutation_cost;
        if (equal) {
            if (portfolio[i].basis_id <= candidate.basis_id) return false;
            portfolio[i] = candidate;
            return true;
        }
    }
    for (local_index i = 0; i < count;) {
        if (!dominates(candidate, portfolio[i])) { ++i; continue; }
        for (local_index j = i + 1; j < count; ++j) portfolio[j - 1] = portfolio[j];
        --count;
    }
    if (count == capacity) return false;
    portfolio[count++] = candidate;
    return true;
}

}  // namespace cellshard::compiler::basis
