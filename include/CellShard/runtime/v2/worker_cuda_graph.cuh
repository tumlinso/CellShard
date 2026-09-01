#pragma once

#include <CellShard/runtime/v2/command_ir.hh>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace cellshard::runtime_v2 {

struct cuda_graph_copy_binding {
    std::uint32_t command_index = 0;
    const void *source = nullptr;
    void *destination = nullptr;
    std::uint64_t bytes = 0;
    cudaMemcpyKind kind = cudaMemcpyDefault;
};

class worker_cuda_graph {
public:
    worker_cuda_graph() noexcept;
    ~worker_cuda_graph() noexcept;
    worker_cuda_graph(const worker_cuda_graph &) = delete;
    worker_cuda_graph &operator=(const worker_cuda_graph &) = delete;

    [[nodiscard]] status_code prepare(
        int device_id, const runtime_command_program &program,
        array_view<cuda_graph_copy_binding> template_bindings) noexcept;
    [[nodiscard]] status_code launch(
        array_view<cuda_graph_copy_binding> bindings,
        cudaStream_t caller_stream) noexcept;
    [[nodiscard]] bool valid() const noexcept;
    void reset() noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_{};
};

static_assert(std::is_trivially_copyable_v<cuda_graph_copy_binding>);

} // namespace cellshard::runtime_v2
