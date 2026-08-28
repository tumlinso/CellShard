#include <CellShard/runtime/device_bindings.cuh>
#include <CellShard/runtime/distributed/distributed.cuh>

#include <type_traits>

int main() {
    static_assert(std::is_trivially_copyable<cellshard::runtime::device_binding_view>::value,
                  "device bindings must remain a neutral borrowed descriptor");
    const int device_id = 0;
    const cellshard::runtime::device_binding_view bindings{&device_id, nullptr, 1u};
    return cellshard::runtime::valid(&bindings) ? 0 : 1;
}
