#include <CellShard/runtime/v2/async_file_atom_source.hh>

#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

namespace cellshard::runtime_v2 {

async_file_atom_source::~async_file_atom_source() noexcept { reset(); }

atom_source_ref async_file_atom_source::ref() noexcept {
    static const atom_source_ops file_ops{submit_impl, query_impl, cancel_impl};
    return valid() ? atom_source_ref{this, &file_ops} : atom_source_ref{};
}

void async_file_atom_source::reset() noexcept {
    cancel_requested_.store(true, std::memory_order_release);
    if (worker_.joinable()) {
        worker_.join();
    }
    if (descriptor_ >= 0) {
        ::close(descriptor_);
    }
    descriptor_ = -1;
    object_ = {};
    object_bytes_ = 0;
    active_token_ = 0;
    ranges_.clear();
    destination_ = nullptr;
    state_.store(atom_request_state::invalid, std::memory_order_release);
}

status_code async_file_atom_source::submit_impl(
    void *opaque, const atom_source_request &request,
    atom_request_token *token) noexcept {
    if (opaque == nullptr || token == nullptr
        || !valid_atom_source_request(request)) {
        return status_code::invalid_input;
    }
    auto &source = *static_cast<async_file_atom_source *>(opaque);
    if (!source.valid()) {
        return status_code::invalid_input;
    }
    const atom_request_state previous =
        source.state_.load(std::memory_order_acquire);
    if (previous == atom_request_state::pending) {
        return status_code::unsupported_capability;
    }
    if (source.worker_.joinable()) {
        source.worker_.join();
    }
    for (const atom_range &range : request.ranges) {
        if (range.object != source.object_
            || range.object_offset > source.object_bytes_
            || range.bytes > source.object_bytes_ - range.object_offset) {
            return status_code::invalid_input;
        }
    }
    try {
        source.ranges_.assign(request.ranges.begin(), request.ranges.end());
        source.destination_ = request.destination;
        source.cancel_requested_.store(false, std::memory_order_release);
        source.active_token_ = source.next_token_++;
        if (source.active_token_ == 0) {
            source.active_token_ = source.next_token_++;
        }
        source.state_.store(atom_request_state::pending,
                            std::memory_order_release);
        source.worker_ = std::thread([&source] { source.run(); });
    } catch (...) {
        source.ranges_.clear();
        source.destination_ = nullptr;
        source.active_token_ = 0;
        source.state_.store(atom_request_state::failed,
                            std::memory_order_release);
        return status_code::allocation_failure;
    }
    *token = atom_request_token{source.active_token_};
    return status_code::success;
}

atom_request_state async_file_atom_source::query_impl(
    void *opaque, atom_request_token token) noexcept {
    if (opaque == nullptr || !token.valid()) {
        return atom_request_state::invalid;
    }
    auto &source = *static_cast<async_file_atom_source *>(opaque);
    return token.value == source.active_token_
        ? source.state_.load(std::memory_order_acquire)
        : atom_request_state::invalid;
}

status_code async_file_atom_source::cancel_impl(
    void *opaque, atom_request_token token) noexcept {
    if (opaque == nullptr || !token.valid()) {
        return status_code::invalid_input;
    }
    auto &source = *static_cast<async_file_atom_source *>(opaque);
    if (token.value != source.active_token_) {
        return status_code::invalid_input;
    }
    source.cancel_requested_.store(true, std::memory_order_release);
    return status_code::success;
}

void async_file_atom_source::run() noexcept {
    for (const atom_range &range : ranges_) {
        if (cancel_requested_.load(std::memory_order_acquire)) {
            state_.store(atom_request_state::cancelled,
                         std::memory_order_release);
            return;
        }
        std::uint64_t completed = 0;
        while (completed < range.bytes) {
            const std::uint64_t remaining = range.bytes - completed;
            const std::size_t chunk = remaining >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<ssize_t>::max())
                ? static_cast<std::size_t>(
                      std::numeric_limits<ssize_t>::max())
                : static_cast<std::size_t>(remaining);
            const ssize_t count = ::pread(
                descriptor_, destination_ + range.destination_offset + completed,
                chunk, static_cast<off_t>(range.object_offset + completed));
            if (count <= 0) {
                if (count < 0 && errno == EINTR) {
                    continue;
                }
                state_.store(atom_request_state::failed,
                             std::memory_order_release);
                return;
            }
            completed += static_cast<std::uint64_t>(count);
        }
    }
    state_.store(cancel_requested_.load(std::memory_order_acquire)
                     ? atom_request_state::cancelled
                     : atom_request_state::complete,
                 std::memory_order_release);
}

status_code open_async_file_atom_source(
    const char *path, storage_object_id object,
    async_file_atom_source *out) noexcept {
    if (path == nullptr || path[0] == '\0' || !object.valid() || out == nullptr
        || out->valid()) {
        return status_code::invalid_input;
    }
    const int descriptor = ::open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        return status_code::missing_object;
    }
    struct stat metadata {};
    if (::fstat(descriptor, &metadata) != 0 || metadata.st_size <= 0) {
        ::close(descriptor);
        return status_code::missing_object;
    }
    out->descriptor_ = descriptor;
    out->object_ = object;
    out->object_bytes_ = static_cast<std::uint64_t>(metadata.st_size);
    return status_code::success;
}

} // namespace cellshard::runtime_v2
