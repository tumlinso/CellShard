#include <CellShard/runtime/v2/exact_read_baseline.hh>

#include <array>
#include <cassert>
#include <cstring>

using namespace cellshard;
using namespace cellshard::runtime_v2;

namespace {
struct context {
    const std::byte *data;
    std::size_t size;
};

status_code read(void *opaque, const exact_read_request &request) noexcept {
    auto &source = *static_cast<context *>(opaque);
    if (request.object_offset + request.byte_count > source.size) {
        return status_code::short_read;
    }
    std::memcpy(request.destination, source.data + request.object_offset,
                static_cast<std::size_t>(request.byte_count));
    return status_code::success;
}
} // namespace

int main() {
    const std::array source_bytes{std::byte{1}, std::byte{2}, std::byte{3},
                                  std::byte{4}};
    context state{source_bytes.data(), source_bytes.size()};
    const payload_source_ops ops{read};
    const payload_source_ref source{&state, &ops, source_provider_id{1},
                                    source_location_id{1}, storage_object_id{5},
                                    source_bytes.size(), 1};
    std::array<std::byte, 4> destination{};
    const std::array ranges{atom_range{storage_object_id{5}, 1, 2, 0}};
    const atom_source_request request{{ranges.data(), ranges.size()},
                                      destination.data(), destination.size()};
    assert(synchronous_read_exact(request, {&source, 1}) == status_code::success);
    assert(destination[0] == std::byte{2} && destination[1] == std::byte{3});
    assert(synchronous_read_exact(request, {}) == status_code::invalid_input);
}
