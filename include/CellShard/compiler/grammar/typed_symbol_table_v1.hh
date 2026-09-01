#pragma once

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::grammar {

struct grammar_identity_v1 {
    std::uint64_t producer_namespace = 0;
    std::uint64_t local_identity = 0;
};

[[nodiscard]] constexpr bool valid(grammar_identity_v1 id) noexcept {
    return id.producer_namespace != 0 && id.local_identity != 0;
}
[[nodiscard]] constexpr bool operator==(grammar_identity_v1 a,
                                        grammar_identity_v1 b) noexcept {
    return a.producer_namespace == b.producer_namespace
        && a.local_identity == b.local_identity;
}
[[nodiscard]] constexpr bool less(grammar_identity_v1 a,
                                  grammar_identity_v1 b) noexcept {
    return a.producer_namespace < b.producer_namespace
        || (a.producer_namespace == b.producer_namespace
            && a.local_identity < b.local_identity);
}

enum class grammar_symbol_kind_v1 : std::uint32_t {
    terminal_atom = 1,
    nonterminal = 2,
};
enum class grammar_value_kind_v1 : std::uint32_t {
    immutable_structure = 1,
    mutable_value = 2,
    mutable_state = 3,
    partial_result = 4,
    materialization = 5,
};

struct typed_grammar_symbol_v1 {
    grammar_identity_v1 identity{};
    grammar_identity_v1 domain_identity{};
    grammar_identity_v1 order_identity{};
    grammar_identity_v1 relation_identity{};
    grammar_identity_v1 scalar_encoding_identity{};
    std::uint64_t structure_generation = 0;
    std::uint64_t value_generation = 0;
    grammar_symbol_kind_v1 symbol_kind = grammar_symbol_kind_v1::terminal_atom;
    grammar_value_kind_v1 value_kind = grammar_value_kind_v1::immutable_structure;
};

struct typed_symbol_table_v1 {
    const typed_grammar_symbol_v1 *symbols = nullptr;
    std::uint64_t symbol_count = 0;
    std::uint64_t symbol_capacity = 0;
    grammar_identity_v1 table_identity{};
    std::uint64_t table_generation = 0;
};

enum class typed_symbol_table_code_v1 : std::uint32_t {
    valid = 0,
    invalid_table_identity,
    missing_table_generation,
    empty_table,
    missing_symbols,
    capacity_overflow,
    invalid_symbol_identity,
    unordered_or_duplicate_symbol,
    invalid_symbol_kind,
    invalid_value_kind,
    invalid_domain,
    invalid_order,
    invalid_relation,
    invalid_generation,
    invalid_encoding,
    unexpected_encoding,
};
struct typed_symbol_table_validation_v1 {
    typed_symbol_table_code_v1 code = typed_symbol_table_code_v1::valid;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == typed_symbol_table_code_v1::valid;
    }
};

[[nodiscard]] constexpr typed_symbol_table_validation_v1
validate_typed_symbol_table_v1(typed_symbol_table_v1 table) noexcept {
    if (!valid(table.table_identity))
        return {typed_symbol_table_code_v1::invalid_table_identity};
    if (table.table_generation == 0)
        return {typed_symbol_table_code_v1::missing_table_generation};
    if (table.symbol_count == 0)
        return {typed_symbol_table_code_v1::empty_table};
    if (table.symbols == nullptr)
        return {typed_symbol_table_code_v1::missing_symbols};
    if (table.symbol_count > table.symbol_capacity)
        return {typed_symbol_table_code_v1::capacity_overflow};
    for (std::uint64_t i = 0; i < table.symbol_count; ++i) {
        const auto &x = table.symbols[i];
        if (!valid(x.identity))
            return {typed_symbol_table_code_v1::invalid_symbol_identity, i};
        if (i != 0 && !less(table.symbols[i - 1].identity, x.identity))
            return {typed_symbol_table_code_v1::unordered_or_duplicate_symbol, i};
        if (x.symbol_kind != grammar_symbol_kind_v1::terminal_atom
            && x.symbol_kind != grammar_symbol_kind_v1::nonterminal)
            return {typed_symbol_table_code_v1::invalid_symbol_kind, i};
        const auto kind = static_cast<std::uint32_t>(x.value_kind);
        if (kind < 1 || kind > 5)
            return {typed_symbol_table_code_v1::invalid_value_kind, i};
        if (!valid(x.domain_identity))
            return {typed_symbol_table_code_v1::invalid_domain, i};
        if (!valid(x.order_identity))
            return {typed_symbol_table_code_v1::invalid_order, i};
        if (!valid(x.relation_identity))
            return {typed_symbol_table_code_v1::invalid_relation, i};
        if (x.structure_generation == 0)
            return {typed_symbol_table_code_v1::invalid_generation, i};
        const bool numerical = x.value_kind == grammar_value_kind_v1::mutable_value
            || x.value_kind == grammar_value_kind_v1::mutable_state
            || x.value_kind == grammar_value_kind_v1::partial_result;
        if (numerical && (!valid(x.scalar_encoding_identity)
                          || x.value_generation == 0))
            return {typed_symbol_table_code_v1::invalid_encoding, i};
        if (!numerical && (valid(x.scalar_encoding_identity)
                           || x.value_generation != 0))
            return {typed_symbol_table_code_v1::unexpected_encoding, i};
    }
    return {typed_symbol_table_code_v1::valid, table.symbol_count};
}

[[nodiscard]] constexpr const typed_grammar_symbol_v1 *find_symbol_v1(
    typed_symbol_table_v1 table, grammar_identity_v1 identity) noexcept {
    std::uint64_t first = 0;
    std::uint64_t last = table.symbol_count;
    while (first < last) {
        const auto middle = first + (last - first) / 2;
        if (less(table.symbols[middle].identity, identity)) first = middle + 1;
        else last = middle;
    }
    return first < table.symbol_count && table.symbols[first].identity == identity
        ? &table.symbols[first] : nullptr;
}

[[nodiscard]] constexpr bool authorizes_execution(typed_symbol_table_v1) noexcept {
    return false;
}

static_assert(std::is_trivially_copyable<typed_grammar_symbol_v1>::value);
static_assert(std::is_trivially_copyable<typed_symbol_table_v1>::value);
} // namespace cellshard::compiler::grammar
