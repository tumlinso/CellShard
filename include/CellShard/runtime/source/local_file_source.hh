#pragma once

#include <CellShard/runtime/source/payload_source.hh>

namespace cellshard {

class local_file_source {
public:
    local_file_source() noexcept = default;
    ~local_file_source() noexcept;
    local_file_source(const local_file_source &) = delete;
    local_file_source &operator=(const local_file_source &) = delete;
    local_file_source(local_file_source &&other) noexcept;
    local_file_source &operator=(local_file_source &&other) noexcept;

    [[nodiscard]] payload_source_ref ref() noexcept;
    [[nodiscard]] bool valid() const noexcept { return descriptor_ >= 0; }
    void reset() noexcept;
    static status_code read_exact_impl(
        void *context, const exact_read_request &request) noexcept;

private:
    int descriptor_ = -1;
    source_provider_id provider_{};
    source_location_id location_{};
    storage_object_id object_{};
    std::uint64_t object_bytes_ = 0;

    friend status_code open_local_file_source(
        const char *, source_provider_id, source_location_id,
        const storage_object_descriptor &, local_file_source *) noexcept;
};

[[nodiscard]] status_code open_local_file_source(
    const char *path,
    source_provider_id provider,
    source_location_id location,
    const storage_object_descriptor &object,
    local_file_source *out) noexcept;

} // namespace cellshard
