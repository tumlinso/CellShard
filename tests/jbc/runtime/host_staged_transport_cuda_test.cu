#include <CellShard/runtime/v2/host_staged_transport.cuh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    std::array<std::byte, 4096> input{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = std::byte((i * 7U) & 0xffU);
    }
    std::byte *source_data = nullptr;
    std::byte *destination_data = nullptr;
    cudaStream_t source_stream = nullptr;
    cudaStream_t destination_stream = nullptr;
    cudaEvent_t event = nullptr;
    assert(cudaSetDevice(0) == cudaSuccess);
    assert(cudaMalloc(&source_data, input.size()) == cudaSuccess);
    assert(cudaMemcpy(source_data, input.data(), input.size(),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaStreamCreateWithFlags(&source_stream, cudaStreamNonBlocking)
           == cudaSuccess);
    assert(cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess);
    pinned_staging_pool pool;
    assert(pool.initialize(0, input.size(), 1, 64,
                           cuda_pinned_staging_allocator())
           == status_code::success);
    pinned_staging_lease staging{};
    assert(pool.acquire(input.size(), &staging) == status_code::success);

    assert(cudaSetDevice(1) == cudaSuccess);
    assert(cudaMalloc(&destination_data, input.size()) == cudaSuccess);
    assert(cudaStreamCreateWithFlags(&destination_stream, cudaStreamNonBlocking)
           == cudaSuccess);
    assert(cuda_host_staged_copy_async(
               {source_data, input.size(), 0},
               {destination_data, input.size(), 1}, staging, input.size(),
               source_stream, destination_stream, event)
           == status_code::success);
    assert(cudaStreamSynchronize(destination_stream) == cudaSuccess);
    std::array<std::byte, 4096> output{};
    assert(cudaMemcpy(output.data(), destination_data, output.size(),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(output == input);
    assert(cudaStreamDestroy(destination_stream) == cudaSuccess);
    assert(cudaFree(destination_data) == cudaSuccess);
    assert(cudaSetDevice(0) == cudaSuccess);
    assert(cudaEventDestroy(event) == cudaSuccess);
    assert(cudaStreamDestroy(source_stream) == cudaSuccess);
    assert(cudaFree(source_data) == cudaSuccess);
    assert(pool.release(staging) == status_code::success);
}
