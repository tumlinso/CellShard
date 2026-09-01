#pragma once

#include <CellShard/compiler/discovery/multimodal/identity_spine_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::multimodal {

struct cross_modal_order_key_v1 {
    std::uint64_t modality_identity = 0;
    std::uint64_t canonical_entity_id = 0;
    std::uint64_t shared_destination_support = 0;
};

struct persistent_order_entry_v1 {
    std::uint64_t modality_identity = 0;
    std::uint64_t canonical_entity_id = 0;
    std::uint64_t execution_ordinal = 0;
    std::uint64_t shared_destination_support = 0;
};

struct persistent_order_proposal_v1 {
    std::uint64_t proposal_identity = 0;
    std::uint64_t evidence_identity = 0;
    std::uint64_t modality_identity = 0;
    std::uint64_t canonical_order_identity = 0;
    std::uint64_t entry_offset = 0;
    std::uint64_t entry_count = 0;
};

enum class persistent_order_code_v1 : std::uint32_t {
    proposed = 0,
    invalid_spine,
    invalid_identity,
    missing_keys,
    missing_entries,
    insufficient_entry_capacity,
    missing_proposals,
    insufficient_proposal_capacity,
    unknown_modality,
    duplicate_entity,
};

struct persistent_order_result_v1 {
    persistent_order_code_v1 code = persistent_order_code_v1::proposed;
    std::uint64_t entry_count = 0;
    std::uint32_t proposal_count = 0;
    std::uint32_t reserved = 0;
    [[nodiscard]] constexpr bool proposed() const noexcept {
        return code == persistent_order_code_v1::proposed;
    }
};

[[nodiscard]] inline persistent_order_result_v1
propose_persistent_orders_v1(
    multimodal_identity_spine_view_v1 spine,
    const cross_modal_order_key_v1 *keys,
    std::uint64_t key_count,
    std::uint64_t proposal_identity_seed,
    std::uint64_t evidence_identity,
    persistent_order_entry_v1 *entries,
    std::uint64_t entry_capacity,
    persistent_order_proposal_v1 *proposals,
    std::uint64_t proposal_capacity) noexcept {
    if (!validate_multimodal_identity_spine_v1(spine).valid())
        return {persistent_order_code_v1::invalid_spine};
    if (proposal_identity_seed == 0 || evidence_identity == 0)
        return {persistent_order_code_v1::invalid_identity};
    if (key_count != 0 && keys == nullptr)
        return {persistent_order_code_v1::missing_keys};
    if (entry_capacity != 0 && entries == nullptr)
        return {persistent_order_code_v1::missing_entries};
    if (entry_capacity < key_count)
        return {persistent_order_code_v1::insufficient_entry_capacity};
    if (proposals == nullptr)
        return {persistent_order_code_v1::missing_proposals};
    if (proposal_capacity < spine.modality_count)
        return {persistent_order_code_v1::insufficient_proposal_capacity};

    persistent_order_result_v1 result{};
    for (std::uint32_t modality_index = 0;
         modality_index < spine.modality_count; ++modality_index) {
        const auto modality_identity
            = spine.modalities[modality_index].modality_identity;
        const auto offset = result.entry_count;
        for (std::uint64_t key_index = 0; key_index < key_count; ++key_index) {
            const auto &key = keys[key_index];
            bool known = false;
            for (std::uint32_t candidate = 0; candidate < spine.modality_count;
                 ++candidate)
                if (spine.modalities[candidate].modality_identity
                    == key.modality_identity) {
                    known = true;
                    break;
                }
            if (!known)
                return {persistent_order_code_v1::unknown_modality,
                        result.entry_count, result.proposal_count};
            if (key.modality_identity != modality_identity) continue;
            for (auto previous = offset; previous < result.entry_count; ++previous)
                if (entries[previous].canonical_entity_id
                    == key.canonical_entity_id)
                    return {persistent_order_code_v1::duplicate_entity,
                            result.entry_count, result.proposal_count};
            std::uint64_t position = offset;
            while (position < result.entry_count
                   && (entries[position].shared_destination_support
                           > key.shared_destination_support
                       || (entries[position].shared_destination_support
                               == key.shared_destination_support
                           && entries[position].canonical_entity_id
                               < key.canonical_entity_id)))
                ++position;
            for (auto move = result.entry_count; move > position; --move)
                entries[move] = entries[move - 1];
            entries[position] = {modality_identity, key.canonical_entity_id, 0,
                                 key.shared_destination_support};
            ++result.entry_count;
        }
        for (auto index = offset; index < result.entry_count; ++index)
            entries[index].execution_ordinal = index - offset;
        proposals[result.proposal_count++] = {
            proposal_identity_seed + modality_index, evidence_identity,
            modality_identity, spine.modalities[modality_index].feature_order_identity,
            offset, result.entry_count - offset};
    }
    return result;
}

static_assert(std::is_standard_layout<cross_modal_order_key_v1>::value);
static_assert(std::is_trivially_copyable<cross_modal_order_key_v1>::value);
static_assert(std::is_standard_layout<persistent_order_entry_v1>::value);
static_assert(std::is_trivially_copyable<persistent_order_entry_v1>::value);

} // namespace cellshard::compiler::discovery::multimodal
