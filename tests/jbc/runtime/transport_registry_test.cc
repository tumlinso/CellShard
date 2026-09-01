#include <CellShard/runtime/v2/transport_registry.hh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

namespace {
status_code submit(void *, const atom_source_request &,
                   atom_request_token *) noexcept { return status_code::success; }
atom_request_state query(void *, atom_request_token) noexcept {
    return atom_request_state::complete;
}
status_code cancel(void *, atom_request_token) noexcept {
    return status_code::success;
}
} // namespace

int main() {
    const atom_source_ops operations{submit, query, cancel};
    int context = 0;
    const atom_source_ref source{&context, &operations};
    std::array<transport_provider, 2> storage{};
    transport_provider_registry registry(storage.data(), storage.size());
    const transport_provider baseline{source_provider_id{1},
                                      source_location_id{2}, 3, 10, source};
    const transport_provider preferred{source_provider_id{1},
                                       source_location_id{2}, 3, 20, source};
    assert(registry.add(baseline) == status_code::success);
    assert(registry.add(preferred) == status_code::success);
    assert(registry.resolve(source_provider_id{1}, source_location_id{2}, 3)
               ->priority == 20);
    assert(registry.add(preferred) == status_code::invalid_input);
    assert(registry.resolve(source_provider_id{9}, source_location_id{2}, 3)
           == nullptr);
}
