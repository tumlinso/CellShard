#pragma once

#include "CellShard/compiler/basis/input.hpp"

namespace cellshard::compiler::basis {

struct basis_budget {
    std::uint64_t storage_bytes = 0;
    std::uint64_t build_cost = 0;
    std::uint64_t resident_bytes = 0;
    std::uint64_t mutation_cost = 0;
};

using basis_usage = basis_budget;

inline bool add_within(std::uint64_t used, std::uint64_t addition,
                       std::uint64_t limit, std::uint64_t& next) noexcept {
    if (addition > limit || used > limit - addition) return false;
    next = used + addition;
    return true;
}

inline bool try_consume(const atom_input& atom, const basis_budget& budget,
                        basis_usage& usage) noexcept {
    basis_usage next{};
    if (!add_within(usage.storage_bytes, atom.storage_bytes, budget.storage_bytes, next.storage_bytes) ||
        !add_within(usage.build_cost, atom.build_cost, budget.build_cost, next.build_cost) ||
        !add_within(usage.resident_bytes, atom.resident_bytes, budget.resident_bytes, next.resident_bytes) ||
        !add_within(usage.mutation_cost, atom.mutation_cost, budget.mutation_cost, next.mutation_cost)) {
        return false;
    }
    usage = next;
    return true;
}

}  // namespace cellshard::compiler::basis
