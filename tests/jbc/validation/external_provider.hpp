#pragma once
#include "fixtures.hpp"
#include <array>
namespace cellshard::jbc::validation {
struct external_proposal {
    global_id provider_id = 0;
    global_id structure_id = 0;
    std::uint32_t abi_version = 0;
    std::array<std::uint64_t, 4> payload_digest{};
    bool provider_correctness_claim = false;
};
struct external_evidence {
    global_id validator_id = 0;
    global_id structure_id = 0;
    std::array<std::uint64_t, 4> observed_digest{};
    bool exact_reference_match = false;
};
inline bool accept_external(const external_proposal& proposal,
                            const external_evidence& evidence,
                            std::uint32_t supported_abi) noexcept {
    return proposal.provider_id != 0 && evidence.validator_id != 0 &&
           proposal.provider_id != evidence.validator_id && proposal.abi_version == supported_abi &&
           proposal.structure_id != 0 && proposal.structure_id == evidence.structure_id &&
           proposal.payload_digest == evidence.observed_digest && evidence.exact_reference_match;
}
}  // namespace cellshard::jbc::validation
