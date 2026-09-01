#include <CellShard/compiler/graph/operation_provider.hh>

#include <cassert>

using namespace cellshard::compiler::graph;

int main() {
    operation_provider_descriptor descriptor{};
    descriptor.provider = cellshard::producer_abi_id{1};
    descriptor.operation = cellshard::operator_class_id{2};
    descriptor.source_content.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    descriptor.source_content.used_bytes = sizeof(std::uint64_t);
    descriptor.source_content.bytes[0] = std::byte{1};
    descriptor.source_revision = 3;
    descriptor.capability_flags = supports_cpu | supports_partial_lowering;
    descriptor.input_arity = 2;
    descriptor.output_arity = 1;
    assert(valid_operation_provider_descriptor(descriptor));
    descriptor.capability_flags = supports_partial_lowering;
    assert(!valid_operation_provider_descriptor(descriptor));
    descriptor.capability_flags = UINT64_C(1) << 63;
    assert(!valid_operation_provider_descriptor(descriptor));
}
