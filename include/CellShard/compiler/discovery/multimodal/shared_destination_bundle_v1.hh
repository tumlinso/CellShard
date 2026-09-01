#pragma once

#include <CellShard/compiler/discovery/multimodal/cross_modal_atom_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::multimodal {

struct shared_destination_bundle_v1 {
    std::uint64_t bundle_identity = 0;
    std::uint64_t destination_modality_identity = 0;
    std::uint64_t destination_entity_identity = 0;
    std::uint64_t member_offset = 0;
    std::uint32_t member_count = 0;
    std::uint32_t source_modality_count = 0;
};

enum class shared_destination_bundle_code_v1 : std::uint32_t {
    proposed = 0,
    missing_atoms,
    invalid_config,
    missing_bundles,
    missing_members,
    insufficient_bundle_capacity,
    insufficient_member_capacity,
    invalid_atom,
    work_limit_exceeded,
};

struct shared_destination_bundle_result_v1 {
    shared_destination_bundle_code_v1 code
        = shared_destination_bundle_code_v1::proposed;
    std::uint64_t bundle_count = 0;
    std::uint64_t member_count = 0;
    std::uint64_t work_items = 0;
    [[nodiscard]] constexpr bool proposed() const noexcept {
        return code == shared_destination_bundle_code_v1::proposed;
    }
};

[[nodiscard]] inline shared_destination_bundle_result_v1
propose_shared_destination_bundles_v1(
    const cross_modal_relation_atom_v1 *atoms,
    std::uint64_t atom_count,
    std::uint32_t minimum_source_modalities,
    std::uint64_t bundle_identity_seed,
    shared_destination_bundle_v1 *bundles,
    std::uint64_t bundle_capacity,
    std::uint64_t *member_atom_identities,
    std::uint64_t member_capacity,
    std::uint64_t maximum_work_items) noexcept {
    if (atom_count != 0 && atoms == nullptr)
        return {shared_destination_bundle_code_v1::missing_atoms};
    if (minimum_source_modalities < 2 || bundle_identity_seed == 0)
        return {shared_destination_bundle_code_v1::invalid_config};
    if (bundle_capacity != 0 && bundles == nullptr)
        return {shared_destination_bundle_code_v1::missing_bundles};
    if (member_capacity != 0 && member_atom_identities == nullptr)
        return {shared_destination_bundle_code_v1::missing_members};
    shared_destination_bundle_result_v1 result{};
    for (std::uint64_t anchor = 0; anchor < atom_count; ++anchor) {
        const auto &atom = atoms[anchor];
        if (atom.atom_identity == 0 || atom.destination_modality_identity == 0
            || atom.destination_entity_identity == 0)
            return {shared_destination_bundle_code_v1::invalid_atom,
                    result.bundle_count, result.member_count, result.work_items};
        bool seen_destination = false;
        for (std::uint64_t previous = 0; previous < anchor; ++previous) {
            if (atoms[previous].destination_modality_identity
                    == atom.destination_modality_identity
                && atoms[previous].destination_entity_identity
                    == atom.destination_entity_identity) {
                seen_destination = true;
                break;
            }
        }
        if (seen_destination) continue;
        std::uint32_t modality_count = 0;
        std::uint32_t member_count = 0;
        for (std::uint64_t candidate = anchor; candidate < atom_count;
             ++candidate) {
            if (result.work_items == maximum_work_items)
                return {shared_destination_bundle_code_v1::work_limit_exceeded,
                        result.bundle_count, result.member_count,
                        result.work_items};
            ++result.work_items;
            if (atoms[candidate].destination_modality_identity
                    != atom.destination_modality_identity
                || atoms[candidate].destination_entity_identity
                    != atom.destination_entity_identity)
                continue;
            ++member_count;
            bool seen_modality = false;
            for (std::uint64_t prior = anchor; prior < candidate; ++prior)
                if (atoms[prior].destination_modality_identity
                        == atom.destination_modality_identity
                    && atoms[prior].destination_entity_identity
                        == atom.destination_entity_identity
                    && atoms[prior].source_modality_identity
                        == atoms[candidate].source_modality_identity) {
                    seen_modality = true;
                    break;
                }
            if (!seen_modality) ++modality_count;
        }
        if (modality_count < minimum_source_modalities) continue;
        if (result.bundle_count == bundle_capacity)
            return {shared_destination_bundle_code_v1::
                insufficient_bundle_capacity, result.bundle_count,
                result.member_count, result.work_items};
        if (member_count > member_capacity - result.member_count)
            return {shared_destination_bundle_code_v1::
                insufficient_member_capacity, result.bundle_count,
                result.member_count, result.work_items};
        auto &bundle = bundles[result.bundle_count++];
        bundle = {bundle_identity_seed + result.bundle_count - 1,
                  atom.destination_modality_identity,
                  atom.destination_entity_identity,
                  result.member_count, member_count, modality_count};
        for (std::uint64_t candidate = anchor; candidate < atom_count;
             ++candidate)
            if (atoms[candidate].destination_modality_identity
                    == atom.destination_modality_identity
                && atoms[candidate].destination_entity_identity
                    == atom.destination_entity_identity)
                member_atom_identities[result.member_count++]
                    = atoms[candidate].atom_identity;
    }
    return result;
}

static_assert(std::is_standard_layout<shared_destination_bundle_v1>::value);
static_assert(std::is_trivially_copyable<shared_destination_bundle_v1>::value);

} // namespace cellshard::compiler::discovery::multimodal
