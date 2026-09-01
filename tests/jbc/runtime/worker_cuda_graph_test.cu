#include <CellShard/runtime/v2/worker_cuda_graph.cuh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    const std::array dependencies{0U};
    const std::array commands{
        runtime_command{1, runtime_command_kind::stage_host, 1, {}, {}, 256, 0, 0, 1},
        runtime_command{2, runtime_command_kind::transport, 1, {}, {}, 256, 0, 1, 2},
    };
    const runtime_command_program program{{commands.data(), commands.size()},
                                          {dependencies.data(), dependencies.size()}};
    std::byte *input = nullptr;
    std::byte *output = nullptr;
    std::byte *device = nullptr;
    assert(cudaSetDevice(0) == cudaSuccess);
    assert(cudaMallocHost(&input, 256) == cudaSuccess);
    assert(cudaMallocHost(&output, 256) == cudaSuccess);
    assert(cudaMalloc(&device, 256) == cudaSuccess);
    for (std::size_t i = 0; i < 256; ++i) {
        input[i] = std::byte(i);
        output[i] = std::byte{0};
    }
    std::array bindings{
        cuda_graph_copy_binding{0, input, device, 256, cudaMemcpyHostToDevice},
        cuda_graph_copy_binding{1, device, output, 256, cudaMemcpyDeviceToHost},
    };
    worker_cuda_graph graph;
    assert(graph.prepare(0, program, {bindings.data(), bindings.size()})
           == status_code::success);
    cudaStream_t stream = nullptr;
    assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
           == cudaSuccess);
    assert(graph.launch({bindings.data(), bindings.size()}, stream)
           == status_code::success);
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    for (std::size_t i = 0; i < 256; ++i) {
        assert(output[i] == input[i]);
        input[i] = std::byte(255U - i);
    }
    assert(graph.launch({bindings.data(), bindings.size()}, stream)
           == status_code::success);
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    for (std::size_t i = 0; i < 256; ++i) {
        assert(output[i] == input[i]);
    }
    graph.reset();
    assert(cudaStreamDestroy(stream) == cudaSuccess);
    assert(cudaFree(device) == cudaSuccess);
    assert(cudaFreeHost(output) == cudaSuccess);
    assert(cudaFreeHost(input) == cudaSuccess);
}
