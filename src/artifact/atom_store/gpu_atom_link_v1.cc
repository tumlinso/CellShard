#include <CellShard/artifact/atom_store/gpu_atom_link_v1.hh>
namespace cellshard::artifact::atom_store {
namespace {bool less(semantic_identity_v1 a,semantic_identity_v1 b){return a.high<b.high||(a.high==b.high&&a.low<b.low);}bool equal(semantic_identity_v1 a,semantic_identity_v1 b){return a.high==b.high&&a.low==b.low;}}
atom_link_status_v1 link_atoms_cpu_reference_v1(const semantic_identity_v1*d,std::size_t dn,const semantic_identity_v1*q,std::size_t qn,std::uint64_t*out)noexcept{
    if(d==nullptr||dn==0||(qn!=0&&(q==nullptr||out==nullptr)))return atom_link_status_v1::invalid_input;
    for(std::size_t i=1;i<dn;++i)if(!less(d[i-1],d[i]))return atom_link_status_v1::unsorted_dictionary;
    for(std::size_t i=0;i<qn;++i){std::size_t lo=0,hi=dn;while(lo<hi){const auto mid=lo+(hi-lo)/2;if(less(d[mid],q[i]))lo=mid+1;else hi=mid;}out[i]=lo<dn&&equal(d[lo],q[i])?lo:atom_link_missing_v1;}return atom_link_status_v1::success;
}
}
