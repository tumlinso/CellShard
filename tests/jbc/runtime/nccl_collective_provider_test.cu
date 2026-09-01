#include <CellShard/runtime/v2/nccl_collective_provider.cuh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    const std::array devices{0, 1};
    nccl_collective_provider provider;
    assert(provider.initialize({devices.data(), devices.size()})
           == status_code::success);
    assert(provider.ranks() == 2);
    std::array<std::byte, 4096> input{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = std::byte((i * 11U) & 0xffU);
    }
    std::array<void *, 2> buffers{};
    std::array<cudaStream_t, 2> streams{};
    for (int rank = 0; rank < 2; ++rank) {
        assert(cudaSetDevice(rank) == cudaSuccess);
        assert(cudaMalloc(&buffers[rank], input.size()) == cudaSuccess);
        assert(cudaMemset(buffers[rank], 0, input.size()) == cudaSuccess);
        assert(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking)
               == cudaSuccess);
    }
    assert(cudaSetDevice(0) == cudaSuccess);
    assert(cudaMemcpy(buffers[0], input.data(), input.size(),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    const std::array<const void *, 2> send{buffers[0], buffers[1]};
    assert(provider.launch({collective_kind::broadcast, collective_scalar::byte,
                            {send.data(), send.size()},
                            {buffers.data(), buffers.size()},
                            {streams.data(), streams.size()}, input.size(), 0})
           == status_code::success);
    for (int rank = 0; rank < 2; ++rank) {
        assert(cudaSetDevice(rank) == cudaSuccess);
        assert(cudaStreamSynchronize(streams[rank]) == cudaSuccess);
    }
    std::array<std::byte, 4096> output{};
    assert(cudaSetDevice(1) == cudaSuccess);
    assert(cudaMemcpy(output.data(), buffers[1], output.size(),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(output == input);
    for (int rank = 0; rank < 2; ++rank) {
        assert(cudaSetDevice(rank) == cudaSuccess);
        assert(cudaStreamDestroy(streams[rank]) == cudaSuccess);
        assert(cudaFree(buffers[rank]) == cudaSuccess);
    }
    provider.reset();
}
