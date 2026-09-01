#pragma once
#include <CellShard/artifact/atom_store/identity_v1.hh>
#include <cstdint>
#include <limits>
#include <type_traits>
namespace cellshard::artifact::atom_store {
struct exact_coverage_record_v1 {
    std::uint64_t domain_identity = 0;
    std::uint64_t item_begin = 0;
    std::uint64_t item_count = 0;
    semantic_identity_v1 atom_identity{};
    std::uint64_t contribution_owner = 0;
};
enum class coverage_index_code_v1 : std::uint32_t {
    valid = 0, missing_records, invalid_record, range_overflow,
    unordered_record, overlapping_coverage,
};
struct coverage_index_result_v1 {
    coverage_index_code_v1 code = coverage_index_code_v1::valid;
    std::uint64_t record_index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept { return code == coverage_index_code_v1::valid; }
};
[[nodiscard]] inline coverage_index_result_v1 validate_exact_coverage_index_v1(
    const exact_coverage_record_v1 *records, std::uint64_t count) noexcept {
    if (count != 0 && records == nullptr) return {coverage_index_code_v1::missing_records};
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto &r = records[i];
        if (r.domain_identity == 0 || r.item_count == 0 || !r.atom_identity.valid()
            || r.contribution_owner == 0) return {coverage_index_code_v1::invalid_record, i};
        if (r.item_begin > std::numeric_limits<std::uint64_t>::max() - r.item_count)
            return {coverage_index_code_v1::range_overflow, i};
        if (i == 0) continue;
        const auto &p = records[i - 1];
        if (r.domain_identity < p.domain_identity
            || (r.domain_identity == p.domain_identity && r.item_begin < p.item_begin))
            return {coverage_index_code_v1::unordered_record, i};
        if (r.domain_identity == p.domain_identity
            && r.item_begin < p.item_begin + p.item_count)
            return {coverage_index_code_v1::overlapping_coverage, i};
    }
    return {};
}
static_assert(std::is_trivially_copyable<exact_coverage_record_v1>::value);
} // namespace cellshard::artifact::atom_store
