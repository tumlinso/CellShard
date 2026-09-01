#pragma once

#include <CellShard/runtime/v2/atom_source.hh>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace cellshard::runtime_v2 {

class async_file_atom_source {
public:
    async_file_atom_source() noexcept = default;
    ~async_file_atom_source() noexcept;
    async_file_atom_source(const async_file_atom_source &) = delete;
    async_file_atom_source &operator=(const async_file_atom_source &) = delete;

    [[nodiscard]] atom_source_ref ref() noexcept;
    [[nodiscard]] bool valid() const noexcept { return descriptor_ >= 0; }
    void reset() noexcept;

private:
    int descriptor_ = -1;
    storage_object_id object_{};
    std::uint64_t object_bytes_ = 0;
    std::atomic<atom_request_state> state_{atom_request_state::invalid};
    std::atomic<bool> cancel_requested_{false};
    std::uint64_t next_token_ = 1;
    std::uint64_t active_token_ = 0;
    std::thread worker_{};
    std::vector<atom_range> ranges_{};
    std::byte *destination_ = nullptr;

    static status_code submit_impl(void *, const atom_source_request &,
                                   atom_request_token *) noexcept;
    static atom_request_state query_impl(void *, atom_request_token) noexcept;
    static status_code cancel_impl(void *, atom_request_token) noexcept;
    void run() noexcept;

    friend status_code open_async_file_atom_source(
        const char *, storage_object_id, async_file_atom_source *) noexcept;
};

[[nodiscard]] status_code open_async_file_atom_source(
    const char *path, storage_object_id object,
    async_file_atom_source *out) noexcept;

} // namespace cellshard::runtime_v2
