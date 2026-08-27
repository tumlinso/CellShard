#include <CellShard/runtime/residency/device.cuh>
#include <array>
#include <cstdio>
#include <cstdlib>

namespace {
void require(bool v,const char*m){if(!v){std::fprintf(stderr,"cellShardOpaqueImagePipelineCudaTest: %s\n",m);std::exit(1);}}
struct counts{int alloc=0;int free=0;};
cudaError_t alloc(void*c,int,std::size_t n,std::size_t,void**p)noexcept{++static_cast<counts*>(c)->alloc;return cudaMalloc(p,n);} cudaError_t release(void*c,int,void*p)noexcept{++static_cast<counts*>(c)->free;return cudaFree(p);} const cellshard::device_allocator_ops ops{&alloc,&release};
}
int main(){using namespace cellshard;int count=0;require(cudaGetDeviceCount(&count)==cudaSuccess&&count>0,"CUDA device");std::array<std::byte,8> bytes{{std::byte{1},std::byte{3},std::byte{5},std::byte{7},std::byte{9},std::byte{11},std::byte{13},std::byte{15}}};content_digest digest{};digest.algorithm=digest_algorithm::legacy_fnv1a64;digest.used_bytes=8;digest.bytes[0]=std::byte{1};host_residency_view host{image_id{40},bytes.data(),bytes.size(),64,digest};cudaStream_t stream=nullptr;require(cudaStreamCreate(&stream)==cudaSuccess,"caller stream");counts state{};device_residency device{};require(stage_host_residency_async(host,0,stream,{&state,&ops},&device)==cudaSuccess&&state.alloc==1,"caller allocation and one async stage");std::array<std::byte,8> consumed{};require(cudaMemcpyAsync(consumed.data(),device.view().payload,consumed.size(),cudaMemcpyDeviceToHost,stream)==cudaSuccess&&cudaStreamSynchronize(stream)==cudaSuccess&&consumed==bytes&&device.view().image==image_id{40},"fake CUDA consumer exact identity and bytes");require(device.reset()==cudaSuccess&&state.free==1&&cudaStreamDestroy(stream)==cudaSuccess,"device ownership cleanup");std::puts("cellShardOpaqueImagePipelineCudaTest: passed");return 0;}
