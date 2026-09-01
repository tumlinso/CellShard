#pragma once

#include <CellShard/compiler/discovery/co_support/relation_statistics_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::co_support {

struct source_group_proposals_view_v1 {
    const std::uint64_t *group_offsets = nullptr;
    const std::uint32_t *source_ids = nullptr;
    const std::uint64_t *proposal_identities = nullptr;
    std::uint64_t source_id_count = 0;
    std::uint32_t group_count = 0;
    std::uint32_t reserved = 0;
    std::uint64_t relation_identity = 0;
    std::uint64_t structure_epoch = 0;
};

struct destination_group_member_v1 {
    std::uint64_t proposal_identity = 0;
    std::uint32_t destination_id = 0;
    std::uint32_t reserved = 0;
};

struct exact_group_rescan_summary_v1 {
    std::uint64_t proposal_identity = 0;
    std::uint64_t visited_edge_count = 0;
    std::uint64_t assigned_edge_count = 0;
    std::uint64_t residual_edge_count = 0;
    std::uint32_t source_count = 0;
    std::uint32_t destination_count = 0;
};

enum class exact_group_rescan_code_v1 : std::uint32_t {
    rescanned = 0,
    invalid_relation,
    invalid_proposals,
    identity_mismatch,
    missing_summaries,
    insufficient_summary_capacity,
    missing_members,
    insufficient_member_capacity,
    work_limit_exceeded,
    count_overflow,
};

struct exact_group_rescan_result_v1 {
    exact_group_rescan_code_v1 code = exact_group_rescan_code_v1::rescanned;
    std::uint64_t member_count = 0;
    std::uint32_t summary_count = 0;
    std::uint32_t reserved = 0;
    std::uint64_t work_items = 0;
    [[nodiscard]] constexpr bool rescanned() const noexcept {
        return code == exact_group_rescan_code_v1::rescanned;
    }
};

[[nodiscard]] inline exact_group_rescan_result_v1
exact_rescan_group_proposals_v1(
    support_relation_view_v1 relation,
    source_group_proposals_view_v1 proposals,
    destination_group_member_v1 *members,
    std::uint64_t member_capacity,
    exact_group_rescan_summary_v1 *summaries,
    std::uint64_t summary_capacity,
    std::uint64_t maximum_work_items) noexcept {
    if (relation.relation_identity == 0 || relation.structure_epoch == 0
        || relation.destination_offsets == nullptr
        || (relation.edge_count != 0 && relation.source_ids == nullptr)
        || relation.destination_offsets[0] != 0
        || relation.destination_offsets[relation.destination_count]
            != relation.edge_count)
        return {exact_group_rescan_code_v1::invalid_relation};
    if (proposals.group_count == 0 || proposals.group_offsets == nullptr
        || proposals.proposal_identities == nullptr
        || proposals.group_offsets[0] != 0
        || proposals.group_offsets[proposals.group_count]
            != proposals.source_id_count
        || (proposals.source_id_count != 0 && proposals.source_ids == nullptr))
        return {exact_group_rescan_code_v1::invalid_proposals};
    if (proposals.relation_identity != relation.relation_identity
        || proposals.structure_epoch != relation.structure_epoch)
        return {exact_group_rescan_code_v1::identity_mismatch};
    if (summaries == nullptr)
        return {exact_group_rescan_code_v1::missing_summaries};
    if (summary_capacity < proposals.group_count)
        return {exact_group_rescan_code_v1::insufficient_summary_capacity};
    if (member_capacity != 0 && members == nullptr)
        return {exact_group_rescan_code_v1::missing_members};

    exact_group_rescan_result_v1 result{};
    for (std::uint32_t group = 0; group < proposals.group_count; ++group) {
        const auto group_begin = proposals.group_offsets[group];
        const auto group_end = proposals.group_offsets[group + 1];
        if (group_end <= group_begin || group_end > proposals.source_id_count
            || proposals.proposal_identities[group] == 0)
            return {exact_group_rescan_code_v1::invalid_proposals,
                    result.member_count, group, 0, result.work_items};
        for (auto source = group_begin; source < group_end; ++source) {
            if (proposals.source_ids[source] >= relation.source_count
                || (source != group_begin
                    && proposals.source_ids[source]
                        <= proposals.source_ids[source - 1]))
                return {exact_group_rescan_code_v1::invalid_proposals,
                        result.member_count, group, 0, result.work_items};
        }
        auto &summary = summaries[group];
        summary = {};
        summary.proposal_identity = proposals.proposal_identities[group];
        summary.source_count = static_cast<std::uint32_t>(group_end - group_begin);
        for (std::uint32_t destination = 0;
             destination < relation.destination_count; ++destination) {
            const auto begin = relation.destination_offsets[destination];
            const auto end = relation.destination_offsets[destination + 1];
            if (end < begin || end > relation.edge_count)
                return {exact_group_rescan_code_v1::invalid_relation,
                        result.member_count, group, 0, result.work_items};
            auto edge = begin;
            auto source = group_begin;
            while (edge < end && source < group_end) {
                if (result.work_items == maximum_work_items)
                    return {exact_group_rescan_code_v1::work_limit_exceeded,
                            result.member_count, group, 0, result.work_items};
                ++result.work_items;
                ++summary.visited_edge_count;
                if (relation.source_ids[edge] < proposals.source_ids[source]) {
                    ++edge;
                } else if (relation.source_ids[edge]
                           == proposals.source_ids[source]) {
                    ++edge;
                    ++source;
                } else {
                    break;
                }
            }
            if (source != group_end) continue;
            if (result.member_count == member_capacity)
                return {exact_group_rescan_code_v1::insufficient_member_capacity,
                        result.member_count, group, 0, result.work_items};
            members[result.member_count++] = {
                proposals.proposal_identities[group], destination, 0};
            ++summary.destination_count;
        }
        if (summary.destination_count != 0
            && summary.source_count > std::numeric_limits<std::uint64_t>::max()
                / summary.destination_count)
            return {exact_group_rescan_code_v1::count_overflow,
                    result.member_count, group, 0, result.work_items};
        summary.assigned_edge_count
            = static_cast<std::uint64_t>(summary.source_count)
                * summary.destination_count;
        summary.residual_edge_count = relation.edge_count
            >= summary.assigned_edge_count
            ? relation.edge_count - summary.assigned_edge_count : 0;
        ++result.summary_count;
    }
    return result;
}

static_assert(std::is_standard_layout<source_group_proposals_view_v1>::value);
static_assert(std::is_trivially_copyable<source_group_proposals_view_v1>::value);
static_assert(std::is_standard_layout<destination_group_member_v1>::value);
static_assert(std::is_trivially_copyable<destination_group_member_v1>::value);
static_assert(std::is_standard_layout<exact_group_rescan_summary_v1>::value);
static_assert(std::is_trivially_copyable<exact_group_rescan_summary_v1>::value);

} // namespace cellshard::compiler::discovery::co_support
