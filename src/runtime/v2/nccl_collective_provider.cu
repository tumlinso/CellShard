#include <CellShard/runtime/v2/nccl_collective_provider.cuh>

#include <nccl.h>

#include <limits>
#include <vector>

namespace cellshard::runtime_v2 {

struct nccl_collective_provider::impl {
    std::vector<int> devices;
    std::vector<ncclComm_t> communicators;
};

namespace {
ncclDataType_t data_type(collective_scalar scalar) noexcept {
    switch (scalar) {
    case collective_scalar::byte:
        return ncclUint8;
    case collective_scalar::float32:
        return ncclFloat32;
    case collective_scalar::float64:
        return ncclFloat64;
    case collective_scalar::invalid:
        return ncclNumTypes;
    }
    return ncclNumTypes;
}
} // namespace

nccl_collective_provider::nccl_collective_provider() noexcept = default;

nccl_collective_provider::~nccl_collective_provider() noexcept { reset(); }

status_code nccl_collective_provider::initialize(
    array_view<int> devices) noexcept {
    if (impl_ || devices.empty()
        || devices.size > static_cast<std::size_t>(
                              std::numeric_limits<int>::max())) {
        return status_code::invalid_input;
    }
    try {
        auto candidate = std::make_unique<impl>();
        candidate->devices.assign(devices.begin(), devices.end());
        for (std::size_t i = 0; i < candidate->devices.size(); ++i) {
            if (candidate->devices[i] < 0) {
                return status_code::invalid_input;
            }
            for (std::size_t j = 0; j < i; ++j) {
                if (candidate->devices[i] == candidate->devices[j]) {
                    return status_code::invalid_input;
                }
            }
        }
        candidate->communicators.resize(candidate->devices.size(), nullptr);
        if (ncclCommInitAll(candidate->communicators.data(),
                            static_cast<int>(candidate->devices.size()),
                            candidate->devices.data()) != ncclSuccess) {
            for (ncclComm_t communicator : candidate->communicators) {
                if (communicator != nullptr) {
                    (void)ncclCommDestroy(communicator);
                }
            }
            return status_code::unsupported_capability;
        }
        impl_ = std::move(candidate);
    } catch (...) {
        return status_code::allocation_failure;
    }
    return status_code::success;
}

status_code nccl_collective_provider::launch(
    const nccl_collective_batch &batch) noexcept {
    const std::size_t rank_count = impl_ ? impl_->communicators.size() : 0;
    const ncclDataType_t type = data_type(batch.scalar);
    if (rank_count == 0 || batch.kind == collective_kind::invalid
        || type == ncclNumTypes || batch.element_count == 0
        || batch.send_buffers.size != rank_count
        || batch.receive_buffers.size != rank_count
        || batch.streams.size != rank_count
        || (batch.kind == collective_kind::broadcast
            && batch.root_rank >= rank_count)) {
        return status_code::invalid_input;
    }
    if (ncclGroupStart() != ncclSuccess) {
        return status_code::cuda_failure;
    }
    ncclResult_t result = ncclSuccess;
    for (std::size_t rank = 0; rank < rank_count && result == ncclSuccess;
         ++rank) {
        if (batch.send_buffers[rank] == nullptr
            || batch.receive_buffers[rank] == nullptr
            || batch.streams[rank] == nullptr) {
            result = ncclInvalidArgument;
            break;
        }
        switch (batch.kind) {
        case collective_kind::broadcast:
            result = ncclBroadcast(batch.send_buffers[rank],
                                   batch.receive_buffers[rank],
                                   static_cast<std::size_t>(batch.element_count),
                                   type, static_cast<int>(batch.root_rank),
                                   impl_->communicators[rank], batch.streams[rank]);
            break;
        case collective_kind::all_gather:
            result = ncclAllGather(batch.send_buffers[rank],
                                   batch.receive_buffers[rank],
                                   static_cast<std::size_t>(batch.element_count),
                                   type, impl_->communicators[rank],
                                   batch.streams[rank]);
            break;
        case collective_kind::all_reduce_sum:
            result = ncclAllReduce(batch.send_buffers[rank],
                                   batch.receive_buffers[rank],
                                   static_cast<std::size_t>(batch.element_count),
                                   type, ncclSum, impl_->communicators[rank],
                                   batch.streams[rank]);
            break;
        case collective_kind::invalid:
            result = ncclInvalidArgument;
            break;
        }
    }
    const ncclResult_t group_result = ncclGroupEnd();
    return result == ncclSuccess && group_result == ncclSuccess
        ? status_code::success
        : status_code::cuda_failure;
}

std::uint32_t nccl_collective_provider::ranks() const noexcept {
    return impl_ ? static_cast<std::uint32_t>(impl_->communicators.size()) : 0;
}

void nccl_collective_provider::reset() noexcept {
    if (impl_) {
        for (ncclComm_t communicator : impl_->communicators) {
            if (communicator != nullptr) {
                (void)ncclCommDestroy(communicator);
            }
        }
    }
    impl_.reset();
}

} // namespace cellshard::runtime_v2
