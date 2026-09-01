#include <CellShard/compiler/schedule/distributed_certificate.hh>
namespace cellshard::compiler::schedule {
bool valid_distributed_certificate(const distributed_certificate&c,const participant_certificate*p,std::size_t n)noexcept{
    if(!c.schedule.valid()||!c.partition_map.valid()||!c.routes.valid()||!valid_content_digest(c.logical_graph)||!valid_content_digest(c.exact_coverage)||c.logical_graph.algorithm==digest_algorithm::none||c.exact_coverage.algorithm==digest_algorithm::none||c.participant_count!=n||n==0||p==nullptr)return false;
    std::uint64_t atoms=0,contributions=0;for(std::size_t i=0;i<n;++i){const auto&x=p[i];if(!x.partition.valid()||!valid_content_digest(x.contribution_digest)||x.contribution_digest.algorithm==digest_algorithm::none||x.atom_count==0||x.contribution_count==0||atoms>UINT64_MAX-x.atom_count||contributions>UINT64_MAX-x.contribution_count)return false;for(std::size_t j=0;j<i;++j)if(p[j].partition==x.partition)return false;atoms+=x.atom_count;contributions+=x.contribution_count;}return atoms==c.atom_count&&contributions==c.contribution_count;
}
}
