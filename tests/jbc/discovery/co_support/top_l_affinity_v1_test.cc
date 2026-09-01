#include <CellShard/compiler/discovery/co_support/top_l_affinity_v1.hh>

#include <cassert>

namespace co_support = cellshard::compiler::discovery::co_support;

int main() {
    const co_support::normalized_association_record_v1 associations[] = {
        {0, 1, 1, 2, 0, 1},
        {0, 2, 3, 4, 0, 1},
        {0, 3, 2, 3, 0, 1},
        {1, 2, 1, 4, 0, 1},
    };
    co_support::source_affinity_edge_v1 edges[8]{};
    auto result = co_support::build_top_l_affinity_v1(
        associations, 4, 4, 2, edges, 8);
    assert(result.built());
    assert(result.edge_count == 7);
    assert(edges[0].source_id == 0 && edges[0].neighbor_source_id == 2);
    assert(edges[1].source_id == 0 && edges[1].neighbor_source_id == 3);
    assert(edges[0].rank == 0 && edges[1].rank == 1);
    assert(edges[2].source_id == 1 && edges[2].neighbor_source_id == 0);

    assert(co_support::fraction_greater_v1(
        9'000'000'000'000'000'001ull, 9'000'000'000'000'000'003ull,
        9'000'000'000'000'000'000ull, 9'000'000'000'000'000'003ull));

    result = co_support::build_top_l_affinity_v1(
        associations, 4, 4, 2, edges, 7);
    assert(result.code
           == co_support::top_l_affinity_code_v1::insufficient_capacity);
    return 0;
}
