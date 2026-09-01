#pragma once
#include "metrics.hpp"
#include <array>
namespace cellshard::jbc::validation {
enum class slice_kind : std::uint8_t { support_signature_basis, cross_operation_support_family, stable_structure_mutable_value };
struct slice_evidence {
    slice_kind kind = slice_kind::support_signature_basis;
    global_id fixture_id = 0;
    global_id structure_id = 0;
    global_id value_generation = 0;
    global_id source_order_id = 0;
    global_id target_order_id = 0;
    std::uint32_t operation_count = 0;
    std::array<std::uint64_t, 4> expected_digest{};
    std::array<std::uint64_t, 4> observed_digest{};
    metric_record complete_cost{};
    bool independent_reference = false;
};
inline bool valid_slice(const slice_evidence& evidence) noexcept {
    return evidence.fixture_id != 0 && evidence.structure_id != 0 && evidence.value_generation != 0 &&
           evidence.source_order_id != 0 && evidence.target_order_id != 0 && evidence.operation_count != 0 &&
           evidence.expected_digest == evidence.observed_digest && evidence.independent_reference &&
           complete_metric(evidence.complete_cost) && evidence.complete_cost.fixture_id == evidence.fixture_id;
}

struct value_rebind_evidence {
    global_id structure_before = 0;
    global_id structure_after = 0;
    global_id generation_before = 0;
    global_id generation_after = 0;
    std::array<std::uint64_t, 4> independently_expected_digest{};
    std::array<std::uint64_t, 4> observed_digest{};
    bool rebuilt_structure = false;
};
inline bool valid_value_rebind(const value_rebind_evidence& evidence) noexcept {
    return evidence.structure_before != 0 && evidence.structure_before == evidence.structure_after &&
           evidence.generation_after > evidence.generation_before && !evidence.rebuilt_structure &&
           evidence.independently_expected_digest == evidence.observed_digest;
}
}  // namespace cellshard::jbc::validation
