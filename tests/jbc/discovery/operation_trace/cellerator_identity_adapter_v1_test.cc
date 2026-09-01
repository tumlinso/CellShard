#include <CellShard/compiler/discovery/operation_trace/cellerator_identity_adapter_v1.hh>

#include <cassert>
#include <cstdint>

namespace trace = cellshard::compiler::discovery::operation_trace;

int main() {
    const trace::cellerator_operation_stage_source_v1 source{
        UINT64_C(0xfedcba9876543210),
        UINT64_C(0x0123456789abcdef),
        UINT64_C(0x53544147454e5301),
        UINT64_C(0xfffffffffffffff0),
        UINT64_C(0x100000001),
        UINT64_C(0x100000002)};

    trace::operation_stage_identity_binding_v1 binding{};
    assert(trace::adapt_cellerator_operation_stage_identities_v1(
               source, &binding)
               .adapted());
    assert(binding.operation_identity.producer_namespace
           == source.operation_high);
    assert(binding.operation_identity.local_identity == source.operation_low);
    assert(binding.stage_identity.producer_namespace
           == source.stage_producer_namespace);
    assert(binding.stage_identity.local_identity == source.stage_identity);

    trace::atom_access_event_v1 event{};
    assert(trace::bind_operation_stage_identity_v1(binding, &event)
           == trace::bind_operation_stage_code_v1::bound);
    assert(event.operation_identity == binding.operation_identity);
    assert(event.stage_identity == binding.stage_identity);
    assert(event.operation_generation == source.operation_generation);
    assert(event.stage_generation == source.stage_generation);

    auto malformed = source;
    malformed.operation_high = 0;
    binding = {{1, 2}, {3, 4}, 5, 6};
    assert(trace::adapt_cellerator_operation_stage_identities_v1(
               malformed, &binding)
               .code
           == trace::cellerator_identity_adapter_code_v1::
               invalid_operation_identity);
    assert(binding.operation_identity ==
           cellshard::compiler::atom::atom_persistent_identity_v1{});

    malformed = source;
    malformed.stage_producer_namespace = 0;
    assert(trace::adapt_cellerator_operation_stage_identities_v1(
               malformed, &binding)
               .code
           == trace::cellerator_identity_adapter_code_v1::
               missing_stage_namespace);
    malformed = source;
    malformed.stage_identity = 0;
    assert(trace::adapt_cellerator_operation_stage_identities_v1(
               malformed, &binding)
               .code
           == trace::cellerator_identity_adapter_code_v1::
               missing_stage_identity);
    assert(trace::adapt_cellerator_operation_stage_identities_v1(source, nullptr)
               .code
           == trace::cellerator_identity_adapter_code_v1::missing_destination);
    assert(trace::bind_operation_stage_identity_v1({}, &event)
           == trace::bind_operation_stage_code_v1::invalid_binding);
    assert(trace::adapt_cellerator_operation_stage_identities_v1(
               source, &binding)
               .adapted());
    assert(trace::bind_operation_stage_identity_v1(binding, nullptr)
           == trace::bind_operation_stage_code_v1::missing_event);
}
