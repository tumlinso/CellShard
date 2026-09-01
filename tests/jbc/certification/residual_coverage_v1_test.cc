#include <CellShard/compiler/certification/residual_coverage_v1.hh>

#include <cassert>
#include <cstdint>

using namespace cellshard::compiler;

int main() {
    const std::uint64_t canonical[]{10, 20, 30, 40, 50};
    certification::exact_contribution_owner_v1 owners[]{
        {{1, 1}, 10, 2, certification::certification_member_kind_v1::entity, {}},
        {{1, 1}, 30, 3, certification::certification_member_kind_v1::entity, {}},
        {{1, 1}, 50, 4, certification::certification_member_kind_v1::entity, {}}};
    std::uint64_t residual[2]{};
    const auto result = certification::build_exact_residual_coverage_v1(
        canonical,
        5,
        {1, 1},
        certification::certification_member_kind_v1::entity,
        owners,
        3,
        residual,
        2);
    assert(result.built());
    assert(result.residual_count == 2);
    assert(residual[0] == 20 && residual[1] == 40);

    assert(certification::build_exact_residual_coverage_v1(
               canonical,
               5,
               {1, 1},
               certification::certification_member_kind_v1::entity,
               owners,
               3,
               residual,
               1)
               .code
           == certification::residual_coverage_build_code_v1::
               insufficient_output);

    owners[1].global_identity = 31;
    assert(certification::build_exact_residual_coverage_v1(
               canonical,
               5,
               {1, 1},
               certification::certification_member_kind_v1::entity,
               owners,
               3,
               residual,
               2)
               .code
           == certification::residual_coverage_build_code_v1::
               owner_not_canonical);
}
