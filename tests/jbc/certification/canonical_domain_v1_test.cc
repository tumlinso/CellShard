#include <CellShard/compiler/certification/canonical_domain_v1.hh>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using namespace cellshard::compiler;

certification::atom_certification_request_v1 make_request(
    atom::common_atom_view_v1 *atom_view,
    std::uint64_t *workspace) {
    certification::atom_certification_request_v1 request{};
    request.proposed_atoms = atom_view;
    request.workspace = workspace;
    request.proposed_atom_count = 1;
    request.workspace_bytes = sizeof(*workspace);
    request.request_identity = {1, 1};
    request.proposal_provider_identity = {2, 1};
    request.certification_authority_identity = {3, 1};
    request.canonical_source_identity = {4, 1};
    request.canonical_source_generation = 1;
    return request;
}

atom::atom_port_v1 make_port(
    atom::atom_persistent_identity_v1 domain,
    atom::atom_persistent_identity_v1 axis,
    atom::atom_persistent_identity_v1 order) {
    atom::atom_port_v1 port{};
    port.domain_identity = domain;
    port.axis_identity = axis;
    port.order_identity = order;
    return port;
}

} // namespace

int main() {
    constexpr std::uint64_t domain_count = 1024;
    std::vector<certification::canonical_domain_v1> domains;
    domains.reserve(domain_count);
    for (std::uint64_t index = 0; index < domain_count; ++index) {
        domains.push_back({{10, index + 1},
                           {11, index + 1},
                           {12, index + 1},
                           7,
                           (UINT64_C(1) << 32) + index});
    }

    std::mt19937_64 generator(0x43454c4c53484152ULL);
    std::vector<atom::atom_port_v1> ports;
    ports.reserve(300);
    for (std::uint64_t index = 0; index < 300; ++index) {
        const auto domain_index = generator() % domain_count;
        const auto &domain = domains[domain_index];
        ports.push_back(make_port(domain.domain_identity,
                                  domain.axis_identity,
                                  domain.order_identity));
    }

    atom::common_atom_view_v1 atom_view{};
    atom_view.ports = {ports.data(), ports.size()};
    std::uint64_t workspace = 0;
    auto request = make_request(&atom_view, &workspace);
    certification::canonical_domain_table_view_v1 table{domains.data(),
                                                         domains.size()};
    assert(certification::validate_canonical_domain_identities_v1(
               request, table)
               .valid());

    auto corrupted = ports;
    corrupted[173].axis_identity = {99, 1};
    atom_view.ports = {corrupted.data(), corrupted.size()};
    const auto axis_result =
        certification::validate_canonical_domain_identities_v1(request, table);
    assert(axis_result.code
           == certification::canonical_domain_validation_code_v1::
               port_axis_mismatch);
    assert(axis_result.port_index == 173);

    corrupted = ports;
    corrupted[41].domain_identity = {10, domain_count + 1};
    atom_view.ports = {corrupted.data(), corrupted.size()};
    assert(certification::validate_canonical_domain_identities_v1(request,
                                                                   table)
               .code
           == certification::canonical_domain_validation_code_v1::
               unknown_port_domain);

    atom_view.ports = {ports.data(), ports.size()};
    auto duplicate_domains = domains;
    duplicate_domains[512].domain_identity =
        duplicate_domains[511].domain_identity;
    assert(certification::validate_canonical_domain_identities_v1(
               request,
               {duplicate_domains.data(), duplicate_domains.size()})
               .code
           == certification::canonical_domain_validation_code_v1::
               unordered_or_duplicate_domain);

    auto reordered_domains = domains;
    std::swap(reordered_domains[300], reordered_domains[301]);
    assert(certification::validate_canonical_domain_identities_v1(
               request,
               {reordered_domains.data(), reordered_domains.size()})
               .code
           == certification::canonical_domain_validation_code_v1::
               unordered_or_duplicate_domain);
}
