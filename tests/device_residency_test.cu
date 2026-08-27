#include <CellShard/runtime/residency/device.cuh>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace {
void require(bool value, const char *message) {
    if (!value) { std::fprintf(stderr, "cellShardDeviceResidencyTest: %s\n", message); std::exit(1); }
}
struct state { int allocations = 0; int releases = 0; int device = -1; bool fail = false; };
cudaError_t allocate(void *context, int device, std::size_t bytes,
                     std::size_t, void **out) noexcept {
    auto *s = static_cast<state *>(context); ++s->allocations; s->device = device;
    if (s->fail) return cudaErrorMemoryAllocation;
    return cudaMalloc(out, bytes);
}
cudaError_t release(void *context, int device, void *allocation) noexcept {
    auto *s = static_cast<state *>(context); ++s->releases; s->device = device;
    return cudaFree(allocation);
}
const cellshard::device_allocator_ops ops{&allocate, &release};
}

int main() {
    using namespace cellshard;
    static_assert(!std::is_copy_constructible<device_residency>::value);
    int count = 0; require(cudaGetDeviceCount(&count) == cudaSuccess && count > 0, "CUDA device available");
    int original = 0; require(cudaGetDevice(&original) == cudaSuccess, "get original device");
    const int target = count > 1 ? (original + 1) % count : original;
    std::array<std::byte, 16> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = std::byte(i + 1);
    content_digest digest{}; digest.algorithm = digest_algorithm::legacy_fnv1a64; digest.used_bytes = 8; digest.bytes[0] = std::byte{1};
    host_residency_view host{image_id{1}, bytes.data(), bytes.size(), 16, digest};
    cudaStream_t stream = nullptr;
    require(cudaSetDevice(target) == cudaSuccess && cudaStreamCreate(&stream) == cudaSuccess, "create caller stream");
    require(cudaSetDevice(original) == cudaSuccess, "restore before staging");
    state allocator{}; device_residency resident{};
    require(stage_host_residency_async(host, target, stream, {&allocator, &ops}, &resident) == cudaSuccess, "stage through caller allocator");
    int after = -1; require(cudaGetDevice(&after) == cudaSuccess && after == original, "staging restores prior device");
    require(allocator.allocations == 1 && allocator.releases == 0 && allocator.device == target, "caller allocator owns one allocation");
    std::array<std::byte, 16> round_trip{};
    require(cudaSetDevice(target) == cudaSuccess
            && cudaMemcpyAsync(round_trip.data(), resident.view().payload, round_trip.size(), cudaMemcpyDeviceToHost, stream) == cudaSuccess
            && cudaStreamSynchronize(stream) == cudaSuccess
            && round_trip == bytes, "opaque bytes preserved on caller stream");
    require(cudaSetDevice(original) == cudaSuccess, "restore before release");
    device_residency moved(std::move(resident)); require(!resident.valid() && moved.valid(), "move-only device ownership");
    require(moved.reset() == cudaSuccess && allocator.releases == 1, "balanced caller release");
    require(cudaGetDevice(&after) == cudaSuccess && after == original, "release restores prior device");
    state failed{}; failed.fail = true;
    require(stage_host_residency_async(host, target, stream, {&failed, &ops}, &resident) == cudaErrorMemoryAllocation
            && failed.allocations == 1 && failed.releases == 0 && !resident.valid(), "allocation failure leak-free");
    state invalid{};
    require(stage_host_residency_async(host, count, stream, {&invalid, &ops},
                                       &resident) != cudaSuccess
            && invalid.allocations == 0 && invalid.releases == 0
            && !resident.valid(), "invalid device rejected before allocation");
    require(cudaSetDevice(target) == cudaSuccess && cudaStreamDestroy(stream) == cudaSuccess && cudaSetDevice(original) == cudaSuccess, "stream cleanup");
    std::puts("cellShardDeviceResidencyTest: passed"); return 0;
}
