#pragma once

#include <cuda_runtime.h>

namespace cellshard {
namespace runtime {

struct device_binding_view {
    const int *device_ids;
    const cudaStream_t *streams;
    unsigned int count;
};

inline int valid(const device_binding_view *bindings) {
    return bindings != nullptr
        && bindings->count != 0u
        && bindings->device_ids != nullptr;
}

inline cudaStream_t stream_for_slot(const device_binding_view *bindings, unsigned int slot) {
    return bindings != nullptr && bindings->streams != nullptr && slot < bindings->count
        ? bindings->streams[slot]
        : (cudaStream_t) 0;
}

} // namespace runtime
} // namespace cellshard
