#include <CellShard/compiler/discovery/co_support/relation_statistics_v1.hh>

#include <cassert>

namespace co_support = cellshard::compiler::discovery::co_support;

int main() {
    const std::uint64_t offsets[] = {0, 2, 3, 5};
    const std::uint32_t sources[] = {0, 2, 1, 0, 2};
    const co_support::support_relation_view_v1 relation{
        offsets, sources, nullptr, nullptr, 5, 3, 3, 10, 2};
    std::uint64_t prevalence[3]{};
    std::uint64_t degrees[3]{};
    auto result = co_support::compute_exact_relation_statistics_v1(
        relation, prevalence, 3, degrees, 3);
    assert(result.computed());
    assert(prevalence[0] == 2 && prevalence[1] == 1 && prevalence[2] == 2);
    assert(degrees[0] == 2 && degrees[1] == 1 && degrees[2] == 2);

    const std::uint32_t duplicate[] = {0, 0, 1, 0, 2};
    auto invalid = relation;
    invalid.source_ids = duplicate;
    result = co_support::compute_exact_relation_statistics_v1(
        invalid, prevalence, 3, degrees, 3);
    assert(result.code
           == co_support::relation_statistics_code_v1::
               unordered_or_duplicate_source);
    return 0;
}
