#pragma once

#include <CellShard/compiler/discovery/multimodal/multi_payload_atom_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::multimodal {

struct payload_exact_check_v1 {
    std::uint64_t payload_identity = 0;
    std::uint64_t checked_element_count = 0;
    std::uint64_t mismatch_count = 0;
    std::uint64_t expected_checksum = 0;
    std::uint64_t observed_checksum = 0;
};

struct multimodal_exact_certificate_v1 {
    std::uint64_t certificate_identity = 0;
    std::uint64_t atom_identity = 0;
    std::uint64_t evidence_identity = 0;
    std::uint64_t spine_identity = 0;
    std::uint64_t structure_epoch = 0;
    std::uint64_t checked_subject_count = 0;
    std::uint64_t checked_payload_count = 0;
    std::uint64_t checked_element_count = 0;
    std::uint64_t mismatch_count = 0;
    std::uint32_t certified = 0;
    std::uint32_t reserved = 0;
};

enum class multimodal_certification_code_v1 : std::uint32_t {
    certified = 0,
    invalid_atom,
    invalid_certificate_identity,
    missing_checks,
    check_count_mismatch,
    missing_payload_check,
    duplicate_payload_check,
    checked_count_mismatch,
    count_overflow,
    oracle_mismatch,
};

struct multimodal_certification_result_v1 {
    multimodal_certification_code_v1 code
        = multimodal_certification_code_v1::certified;
    std::uint32_t payload_index = 0;
    [[nodiscard]] constexpr bool certified() const noexcept {
        return code == multimodal_certification_code_v1::certified;
    }
};

[[nodiscard]] inline multimodal_certification_result_v1
certify_multimodal_atom_v1(
    multimodal_identity_spine_view_v1 spine,
    multi_payload_atom_v1 atom,
    const multimodal_payload_descriptor_v1 *payloads,
    std::uint64_t payload_descriptor_count,
    std::uint64_t payload_bytes,
    const payload_exact_check_v1 *checks,
    std::uint64_t check_count,
    std::uint64_t certificate_identity,
    multimodal_exact_certificate_v1 *certificate) noexcept {
    if (!validate_multi_payload_atom_v1(
            spine, atom, payloads, payload_descriptor_count, payload_bytes).valid()
        || certificate == nullptr)
        return {multimodal_certification_code_v1::invalid_atom};
    *certificate = {};
    if (certificate_identity == 0)
        return {multimodal_certification_code_v1::invalid_certificate_identity};
    if (checks == nullptr)
        return {multimodal_certification_code_v1::missing_checks};
    if (check_count != atom.payload_count)
        return {multimodal_certification_code_v1::check_count_mismatch};
    certificate->certificate_identity = certificate_identity;
    certificate->atom_identity = atom.atom_identity;
    certificate->evidence_identity = atom.evidence_identity;
    certificate->spine_identity = atom.spine_identity;
    certificate->structure_epoch = atom.structure_epoch;
    certificate->checked_subject_count = atom.subject_count;
    for (std::uint32_t payload_index = 0;
         payload_index < atom.payload_count; ++payload_index) {
        const auto &payload = payloads[atom.payload_offset + payload_index];
        std::uint64_t found = check_count;
        for (std::uint64_t check_index = 0; check_index < check_count;
             ++check_index)
            if (checks[check_index].payload_identity == payload.payload_identity) {
                if (found != check_count)
                    return {multimodal_certification_code_v1::
                        duplicate_payload_check, payload_index};
                found = check_index;
            }
        if (found == check_count)
            return {multimodal_certification_code_v1::missing_payload_check,
                    payload_index};
        const auto &check = checks[found];
        if (check.checked_element_count != payload.element_count)
            return {multimodal_certification_code_v1::checked_count_mismatch,
                    payload_index};
        if (check.checked_element_count
            > std::numeric_limits<std::uint64_t>::max()
                - certificate->checked_element_count
            || check.mismatch_count
                > std::numeric_limits<std::uint64_t>::max()
                    - certificate->mismatch_count)
            return {multimodal_certification_code_v1::count_overflow,
                    payload_index};
        certificate->checked_element_count += check.checked_element_count;
        certificate->mismatch_count += check.mismatch_count;
        ++certificate->checked_payload_count;
        if (check.mismatch_count != 0
            || check.expected_checksum != check.observed_checksum)
            return {multimodal_certification_code_v1::oracle_mismatch,
                    payload_index};
    }
    certificate->certified = 1;
    return {};
}

static_assert(std::is_standard_layout<payload_exact_check_v1>::value);
static_assert(std::is_trivially_copyable<payload_exact_check_v1>::value);
static_assert(std::is_standard_layout<multimodal_exact_certificate_v1>::value);
static_assert(std::is_trivially_copyable<multimodal_exact_certificate_v1>::value);

} // namespace cellshard::compiler::discovery::multimodal
