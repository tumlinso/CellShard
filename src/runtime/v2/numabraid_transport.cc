#include <CellShard/runtime/v2/numabraid_transport.hh>

namespace cellshard::runtime_v2 {
namespace {
bool valid_provider(numabraid_transport_ref provider) noexcept {
    return provider.context != nullptr && provider.ops != nullptr
        && provider.abi_version == 1 && provider.ops->prepare != nullptr
        && provider.ops->launch != nullptr && provider.ops->query != nullptr
        && provider.ops->synchronize != nullptr
        && provider.topology_identity.algorithm != digest_algorithm::none
        && valid_content_digest(provider.topology_identity);
}
} // namespace

status_code plan_numabraid_transport(
    numabraid_transport_ref provider, const numabraid_plan_request &request,
    numabraid_transport_plan *out) noexcept {
    if (!valid_provider(provider) || out == nullptr || request.source_device < 0
        || request.destination_device < 0
        || request.source_device == request.destination_device
        || request.bytes == 0) {
        return status_code::invalid_input;
    }
    numabraid_transport_plan candidate{};
    const status_code status =
        provider.ops->prepare(provider.context, request, &candidate);
    if (!status_ok(status)) {
        return status;
    }
    if (!valid_numabraid_transport_plan(candidate)
        || candidate.topology_identity != provider.topology_identity
        || candidate.source_device != request.source_device
        || candidate.destination_device != request.destination_device
        || candidate.capacity_bytes < request.bytes) {
        return status_code::incompatible_image;
    }
    *out = candidate;
    return status_code::success;
}

status_code launch_numabraid_transport(
    numabraid_transport_ref provider, const numabraid_transport_plan &plan,
    const void *source, void *destination, std::uint64_t bytes) noexcept {
    if (!valid_provider(provider) || !valid_numabraid_transport_plan(plan)
        || plan.topology_identity != provider.topology_identity
        || source == nullptr || destination == nullptr || bytes == 0
        || bytes > plan.capacity_bytes) {
        return status_code::invalid_input;
    }
    return provider.ops->launch(provider.context, plan, source, destination,
                                bytes);
}

} // namespace cellshard::runtime_v2
