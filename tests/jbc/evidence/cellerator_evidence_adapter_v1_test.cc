#include <CellShard/interop/cellerator/evidence_adapter_v1.hh>

#include <array>
#include <cassert>

namespace adapter = cellshard::interop::cellerator;
namespace evidence = cellshard::compiler::evidence;
namespace geometry = cellerator::geometry;

int main() {
    std::array<geometry::co_support_record_v1, 2> co_support{};
    std::array<geometry::exact_rescan_summary_v1, 1> exact_rescan{};
    geometry::support_atlas_view_v1 source{};
    source.flags = geometry::support_atlas_flag_sampled;
    source.evidence_identity = 10;
    source.relation_identity = 11;
    source.structure_identity = 12;
    source.structure_epoch = 13;
    source.co_support = co_support.data();
    source.co_support_count = co_support.size();
    source.exact_rescans = exact_rescan.data();
    source.exact_rescan_count = exact_rescan.size();
    const adapter::support_atlas_adapter_identity_v1 identity{
        20, 30, {40, 50}, 60};
    const auto requirement = adapter::support_atlas_adapter_requirements_v1(
        source, identity);
    assert(requirement.ok() && requirement.required_records == 2);
    std::array<evidence::atom_evidence_record_v1, 2> output{};
    assert(adapter::adapt_support_atlas_v1(
        source, identity, output.data(), output.size()).ok());
    assert(output[0].kind == evidence::evidence_kind::co_support);
    assert(output[1].kind == evidence::evidence_kind::support_signature);
    assert(output[0].disposition == evidence::evidence_disposition_v1::proposal_only);
    assert(!adapter::adapted_exact_rescan_certifies_coverage_v1());

    source.flags |= geometry::support_atlas_flag_weighted;
    assert(adapter::adapt_support_atlas_v1(
        source, identity, output.data(), output.size()).ok());
    assert(output[0].kind == evidence::evidence_kind::weighted_co_support);

    source.exact_rescans = nullptr;
    assert(adapter::support_atlas_adapter_requirements_v1(source, identity).code
           == adapter::support_atlas_adapter_code_v1::inconsistent_section_pointer);
}
