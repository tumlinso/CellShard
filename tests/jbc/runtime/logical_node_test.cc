#include <CellShard/runtime/v2/logical_node.hh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    std::array nodes{
        logical_node{1, {0, 0, 16, 64ULL << 30}},
        logical_node{2, {1, 16, 16, 64ULL << 30}},
    };
    assert(valid_logical_nodes({nodes.data(), nodes.size()}));

    nodes[1].id = 1;
    assert(!valid_logical_nodes({nodes.data(), nodes.size()}));
    nodes[1].id = 2;
    nodes[1].owner.first_cpu = 8;
    assert(!valid_logical_nodes({nodes.data(), nodes.size()}));
    nodes[1].owner.first_cpu = 16;
    nodes[1].owner.numa_id = 0;
    assert(!valid_logical_nodes({nodes.data(), nodes.size()}));
}
