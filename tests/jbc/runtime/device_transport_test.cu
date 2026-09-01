#include <CellShard/runtime/v2/device_transport.cuh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    std::array<std::byte, 4096> input{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = std::byte(i & 0xffU);
    }
    std::byte *source_data = nullptr;
    std::byte *destination_data = nullptr;
    assert(cudaSetDevice(0) == cudaSuccess);
    assert(cudaMalloc(&source_data, input.size()) == cudaSuccess);
    assert(cudaMemcpy(source_data, input.data(), input.size(),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    const const_device_region source{source_data, input.size(), 0};
    const_device_region alias{};
    assert(same_device_alias(source, 0, &alias) == status_code::success);
    assert(alias.data == source.data && alias.bytes == source.bytes);
    assert(same_device_alias(source, 1, &alias) == status_code::invalid_input);

    assert(cudaSetDevice(1) == cudaSuccess);
    assert(cudaMalloc(&destination_data, input.size()) == cudaSuccess);
    cudaStream_t stream = nullptr;
    assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
           == cudaSuccess);
    assert(prepare_cuda_p2p(0, 1) == status_code::success);
    assert(cuda_p2p_copy_async(
               source, {destination_data, input.size(), 1}, input.size(), stream)
           == status_code::success);
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    std::array<std::byte, 4096> output{};
    assert(cudaMemcpy(output.data(), destination_data, output.size(),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(output == input);
    assert(cudaStreamDestroy(stream) == cudaSuccess);
    assert(cudaFree(destination_data) == cudaSuccess);
    assert(cudaSetDevice(0) == cudaSuccess);
    assert(cudaFree(source_data) == cudaSuccess);
}
