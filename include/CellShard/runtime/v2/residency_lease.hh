#pragma once

#include <CellShard/runtime/v2/atom_residency.hh>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>

namespace cellshard::runtime_v2 {

struct residency_lease {
    atom_plane_resident_instance instance{};
    std::uint32_t slot = 0;
    std::uint64_t pin_mask = 0;
    std::uint64_t incarnation = 0;
};

class residency_lease_table {
public:
    residency_lease_table() noexcept = default;
    residency_lease_table(const residency_lease_table &) = delete;
    residency_lease_table &operator=(const residency_lease_table &) = delete;

    [[nodiscard]] status_code initialize(std::uint32_t capacity) noexcept;
    [[nodiscard]] status_code publish(
        atom_plane_resident_instance instance) noexcept;
    [[nodiscard]] status_code acquire(residency_id residency,
                                      residency_lease *out) noexcept;
    [[nodiscard]] status_code release(residency_lease lease) noexcept;
    [[nodiscard]] status_code evict(residency_id residency) noexcept;

private:
    struct entry {
        atom_plane_resident_instance instance{};
        std::atomic<std::uint64_t> pins{0};
        std::uint64_t incarnation = 0;
        bool occupied = false;
    };

    std::unique_ptr<entry[]> entries_{};
    std::uint32_t capacity_ = 0;
    std::uint64_t next_incarnation_ = 1;
    std::mutex mutation_mutex_{};
};

static_assert(std::is_trivially_copyable_v<residency_lease>);

} // namespace cellshard::runtime_v2
