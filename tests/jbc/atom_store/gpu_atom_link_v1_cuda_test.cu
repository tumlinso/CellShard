#include <CellShard/artifact/atom_store/gpu_atom_link_v1.hh>
#include <cuda_runtime.h>
#include <array>
#include <cassert>
using namespace cellshard::artifact::atom_store;
int main(){std::array<semantic_identity_v1,3>d{{{1,1},{1,3},{2,1}}};std::array<semantic_identity_v1,3>q{{{1,3},{1,2},{2,1}}};semantic_identity_v1 *dd=nullptr,*dq=nullptr;std::uint64_t*out=nullptr;assert(cudaMalloc(&dd,sizeof(d))==cudaSuccess);assert(cudaMalloc(&dq,sizeof(q))==cudaSuccess);assert(cudaMalloc(&out,sizeof(std::uint64_t)*q.size())==cudaSuccess);assert(cudaMemcpy(dd,d.data(),sizeof(d),cudaMemcpyHostToDevice)==cudaSuccess);assert(cudaMemcpy(dq,q.data(),sizeof(q),cudaMemcpyHostToDevice)==cudaSuccess);assert(link_atoms_gpu_v1(dd,d.size(),dq,q.size(),out,nullptr)==atom_link_status_v1::success);std::array<std::uint64_t,3>result{};assert(cudaMemcpy(result.data(),out,sizeof(result),cudaMemcpyDeviceToHost)==cudaSuccess);assert(result[0]==1&&result[1]==atom_link_missing_v1&&result[2]==2);assert(cudaFree(out)==cudaSuccess);assert(cudaFree(dq)==cudaSuccess);assert(cudaFree(dd)==cudaSuccess);}
