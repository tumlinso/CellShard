#include <CellShard/compiler/discovery/operation_trace/atom_access_event_v1.hh>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace trace = cellshard::compiler::discovery::operation_trace;

namespace {

constexpr cellshard::compiler::atom::atom_persistent_identity_v1 atom_id(
    std::uint64_t local_identity) noexcept {
    return {UINT64_C(0x43454c4c53484152), local_identity};
}

constexpr cellshard::compiler::evidence::evidence_identity_v1 evidence_id(
    std::uint64_t local_identity) noexcept {
    return {UINT64_C(0x5452414345455644), local_identity};
}

trace::atom_access_event_v1 valid_event() {
    trace::atom_access_event_v1 event{};
    event.event_identity = evidence_id(1);
    event.trace_identity = evidence_id(2);
    event.source_identity = evidence_id(3);
    event.workload_identity = atom_id(4);
    event.graph_identity = atom_id(5);
    event.operation_identity = atom_id(6);
    event.stage_identity = atom_id(7);
    event.atom_identity = atom_id(UINT64_C(0xfffffffffffffff0));
    event.port_identity = atom_id(9);
    event.trace_generation = 10;
    event.graph_generation = 11;
    event.operation_generation = 12;
    event.stage_generation = 13;
    event.atom_generation = 14;
    event.sequence_number = UINT64_C(0x100000001);
    event.logical_byte_count = UINT64_C(1) << 40;
    event.mode = trace::atom_access_mode_v1::read_write;
    event.role = trace::atom_access_role_v1::state;
    return event;
}

} // namespace

int main() {
    static_assert(std::is_standard_layout<trace::atom_access_event_v1>::value);
    static_assert(
        std::is_trivially_copyable<trace::atom_access_event_v1>::value);

    const auto base = valid_event();
    assert(trace::validate_atom_access_event_v1(base).valid());
    assert(!trace::authorizes_execution(base));
    assert(base.atom_identity.local_identity
           == UINT64_C(0xfffffffffffffff0));
    assert(base.sequence_number == UINT64_C(0x100000001));
    assert(base.logical_byte_count == (UINT64_C(1) << 40));

    auto malformed = base;
    malformed.schema_version = 2;
    assert(trace::validate_atom_access_event_v1(malformed).code
           == trace::atom_access_event_validation_code_v1::unsupported_schema);
    malformed = base;
    malformed.trace_identity = {};
    assert(trace::validate_atom_access_event_v1(malformed).code
           == trace::atom_access_event_validation_code_v1::invalid_trace_identity);
    malformed = base;
    malformed.atom_identity = {};
    assert(trace::validate_atom_access_event_v1(malformed).code
           == trace::atom_access_event_validation_code_v1::invalid_atom_identity);
    malformed = base;
    malformed.atom_generation = 0;
    assert(trace::validate_atom_access_event_v1(malformed).code
           == trace::atom_access_event_validation_code_v1::missing_atom_generation);
    malformed = base;
    malformed.sequence_number = 0;
    assert(trace::validate_atom_access_event_v1(malformed).code
           == trace::atom_access_event_validation_code_v1::missing_sequence_number);
    malformed = base;
    malformed.mode = static_cast<trace::atom_access_mode_v1>(0);
    assert(trace::validate_atom_access_event_v1(malformed).code
           == trace::atom_access_event_validation_code_v1::invalid_mode);
    malformed = base;
    malformed.reserved32 = 1;
    assert(trace::validate_atom_access_event_v1(malformed).code
           == trace::atom_access_event_validation_code_v1::nonzero_reserved);
}
