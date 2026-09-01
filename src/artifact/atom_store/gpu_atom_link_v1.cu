#include <CellShard/artifact/atom_store/gpu_atom_link_v1.hh>
#include <cuda_runtime.h>
namespace cellshard::artifact::atom_store {
namespace {
__device__ bool less(semantic_identity_v1 a,semantic_identity_v1 b){return a.high<b.high||(a.high==b.high&&a.low<b.low);}
__global__ void link_kernel(const semantic_identity_v1*d,std::uint64_t dn,const semantic_identity_v1*q,std::uint64_t qn,std::uint64_t*out){const auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=qn)return;std::uint64_t lo=0,hi=dn;while(lo<hi){const auto mid=lo+(hi-lo)/2;if(less(d[mid],q[i]))lo=mid+1;else hi=mid;}out[i]=lo<dn&&d[lo].high==q[i].high&&d[lo].low==q[i].low?lo:atom_link_missing_v1;}
}
atom_link_status_v1 link_atoms_gpu_v1(const semantic_identity_v1*d,std::size_t dn,const semantic_identity_v1*q,std::size_t qn,std::uint64_t*out,void*stream)noexcept{if(d==nullptr||dn==0||(qn!=0&&(q==nullptr||out==nullptr)))return atom_link_status_v1::invalid_input;if(qn==0)return atom_link_status_v1::success;constexpr unsigned threads=256;const auto blocks=(qn+threads-1)/threads;if(blocks>static_cast<std::size_t>(UINT32_MAX))return atom_link_status_v1::invalid_input;link_kernel<<<static_cast<unsigned>(blocks),threads,0,static_cast<cudaStream_t>(stream)>>>(d,dn,q,qn,out);return cudaGetLastError()==cudaSuccess?atom_link_status_v1::success:atom_link_status_v1::cuda_failure;}
}
