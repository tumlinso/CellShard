#include <CellShard/runtime/v2/numabraid_transport.hh>

#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

namespace {
content_digest identity() {
    content_digest value{};
    value.algorithm = digest_algorithm::legacy_fnv1a64;
    value.used_bytes = 8;
    return value;
}
status_code prepare(void *, const numabraid_plan_request &request,
                    numabraid_transport_plan *out) noexcept {
    *out = {identity(), request.source_device, 2, request.destination_device,
            request.bytes, 12U << 20, 2, 7};
    return status_code::success;
}
status_code launch(void *, const numabraid_transport_plan &, const void *, void *,
                   std::uint64_t) noexcept { return status_code::success; }
numabraid_transfer_state query(void *,
                               const numabraid_transport_plan &) noexcept {
    return numabraid_transfer_state::complete;
}
status_code synchronize(void *, const numabraid_transport_plan &) noexcept {
    return status_code::success;
}
} // namespace

int main() {
    int context = 0;
    const numabraid_transport_ops operations{prepare, launch, query, synchronize};
    const numabraid_transport_ref provider{&context, &operations, 1, identity()};
    numabraid_transport_plan plan{};
    assert(plan_numabraid_transport(provider, {0, 3, 24U << 20}, &plan)
           == status_code::success);
    int source = 1;
    int destination = 0;
    assert(launch_numabraid_transport(provider, plan, &source, &destination,
                                      24U << 20)
           == status_code::success);
    auto stale = provider;
    stale.topology_identity.bytes[0] = std::byte{1};
    assert(launch_numabraid_transport(stale, plan, &source, &destination,
                                      24U << 20)
           == status_code::invalid_input);
}
