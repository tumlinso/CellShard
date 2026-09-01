#include <CellShard/runtime/v2/pinned_staging_pool.hh>

#include <cassert>
#include <cstdlib>

using namespace cellshard;
using namespace cellshard::runtime_v2;

namespace {
status_code allocate(void *, std::uint32_t, std::size_t bytes,
                     std::size_t alignment, void **out) noexcept {
    return ::posix_memalign(out, alignment, bytes) == 0
        ? status_code::success
        : status_code::allocation_failure;
}
void deallocate(void *, void *allocation) noexcept { std::free(allocation); }
} // namespace

int main() {
    const pinned_staging_allocator_ops operations{allocate, deallocate};
    pinned_staging_pool pool;
    assert(pool.initialize(3, 4096, 2, 64, {nullptr, &operations})
           == status_code::success);
    pinned_staging_lease first{};
    pinned_staging_lease second{};
    pinned_staging_lease unavailable{};
    assert(pool.acquire(1024, &first) == status_code::success);
    assert(pool.acquire(4096, &second) == status_code::success);
    assert(first.numa_id == 3 && second.slot != first.slot);
    assert(pool.acquire(1, &unavailable) == status_code::allocation_failure);
    assert(pool.release(first) == status_code::success);
    assert(pool.release(first) == status_code::invalid_input);
    assert(pool.acquire(4097, &unavailable) == status_code::invalid_input);
}
