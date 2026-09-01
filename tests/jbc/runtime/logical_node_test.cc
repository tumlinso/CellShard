#include <CellShard/runtime/v2/logical_node.hh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    content_digest first_cpu_set{};
    first_cpu_set.algorithm = digest_algorithm::legacy_fnv1a64;
    first_cpu_set.used_bytes = 8;
    first_cpu_set.bytes[0] = std::byte{1};
    auto second_cpu_set = first_cpu_set;
    second_cpu_set.bytes[0] = std::byte{2};
    std::array nodes{
        logical_node{1, {0, 16, 64ULL << 30, first_cpu_set}},
        logical_node{2, {1, 16, 64ULL << 30, second_cpu_set}},
    };
    assert(valid_logical_nodes({nodes.data(), nodes.size()}));

    nodes[1].id = 1;
    assert(!valid_logical_nodes({nodes.data(), nodes.size()}));
    nodes[1].id = 2;
    nodes[1].owner.cpu_set_identity = first_cpu_set;
    assert(!valid_logical_nodes({nodes.data(), nodes.size()}));
    nodes[1].owner.cpu_set_identity = second_cpu_set;
    nodes[1].owner.numa_id = 0;
    assert(!valid_logical_nodes({nodes.data(), nodes.size()}));
}
