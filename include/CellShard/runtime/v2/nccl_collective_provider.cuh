#pragma once

#include <CellShard/identity.hh>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

namespace cellshard::runtime_v2 {

enum class collective_kind : std::uint8_t {
    invalid = 0,
    broadcast,
    all_gather,
    all_reduce_sum,
};

enum class collective_scalar : std::uint8_t {
    invalid = 0,
    byte,
    float32,
    float64,
};

struct nccl_collective_batch {
    collective_kind kind = collective_kind::invalid;
    collective_scalar scalar = collective_scalar::invalid;
    array_view<const void *> send_buffers{};
    array_view<void *> receive_buffers{};
    array_view<cudaStream_t> streams{};
    std::uint64_t element_count = 0;
    std::uint32_t root_rank = 0;
};

class nccl_collective_provider {
public:
    nccl_collective_provider() noexcept;
    ~nccl_collective_provider() noexcept;
    nccl_collective_provider(const nccl_collective_provider &) = delete;
    nccl_collective_provider &operator=(const nccl_collective_provider &) = delete;

    [[nodiscard]] status_code initialize(array_view<int> devices) noexcept;
    [[nodiscard]] status_code launch(
        const nccl_collective_batch &batch) noexcept;
    [[nodiscard]] std::uint32_t ranks() const noexcept;
    void reset() noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_{};
};

} // namespace cellshard::runtime_v2
