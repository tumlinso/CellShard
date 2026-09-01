#include <CellShard/runtime/v2/numabraid_capability.hh>

#include <cassert>

using namespace cellshard::runtime_v2;

int main() {
    constexpr numabraid_capabilities capabilities =
        discover_numabraid_capabilities();
#if defined(CELLSHARD_RUNTIME_V2_HAS_NUMABRAID)
    static_assert(capabilities.package_available);
    static_assert(capabilities.version_major == 0);
    static_assert(!capabilities.topology_api);
    static_assert(!capabilities.forwarding_api);
#else
    static_assert(!capabilities.package_available);
#endif
    assert(!capabilities.nccl_provider);
}
