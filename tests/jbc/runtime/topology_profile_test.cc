#include <CellShard/runtime/v2/topology_profile.hh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

namespace {
content_digest identity() {
    content_digest result{};
    result.algorithm = digest_algorithm::legacy_fnv1a64;
    result.used_bytes = 8;
    result.bytes[0] = std::byte{0x42};
    return result;
}
} // namespace

int main() {
    const std::array nodes{
        topology_node{1, topology_node_kind::host_numa, 0, -1, 1ULL << 30},
        topology_node{2, topology_node_kind::cuda_device, 0, 0, 16ULL << 30},
    };
    const std::array links{
        topology_link{1, 2, topology_link_kind::pcie, 12'000'000'000ULL, 800, false},
        topology_link{2, 1, topology_link_kind::pcie, 12'000'000'000ULL, 800, false},
    };
    topology_profile profile{identity(), {nodes.data(), nodes.size()},
                             {links.data(), links.size()}};
    assert(valid_topology_profile(profile));

    auto bad_nodes = nodes;
    bad_nodes[1].id = 1;
    profile.nodes = {bad_nodes.data(), bad_nodes.size()};
    assert(!valid_topology_profile(profile));

    profile.nodes = {nodes.data(), nodes.size()};
    auto bad_links = links;
    bad_links[0].destination = 99;
    profile.links = {bad_links.data(), bad_links.size()};
    assert(!valid_topology_profile(profile));

    profile.links = {links.data(), links.size()};
    profile.identity = {};
    assert(!valid_topology_profile(profile));
}
