#include <CellShard/runtime/source/local_file_source.hh>

#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <utility>
#include <sys/stat.h>
#include <unistd.h>

namespace cellshard {
namespace {
const payload_source_ops local_file_ops{&local_file_source::read_exact_impl};
}

local_file_source::~local_file_source() noexcept { reset(); }

local_file_source::local_file_source(local_file_source &&other) noexcept {
    *this = std::move(other);
}

local_file_source &local_file_source::operator=(local_file_source &&other) noexcept {
    if (this != &other) {
        reset();
        descriptor_ = other.descriptor_;
        provider_ = other.provider_;
        location_ = other.location_;
        object_ = other.object_;
        object_bytes_ = other.object_bytes_;
        other.descriptor_ = -1;
        other.object_bytes_ = 0;
    }
    return *this;
}

void local_file_source::reset() noexcept {
    if (descriptor_ >= 0) {
        ::close(descriptor_);
    }
    descriptor_ = -1;
    object_bytes_ = 0;
}

payload_source_ref local_file_source::ref() noexcept {
    if (!valid()) {
        return {};
    }
    return {this, &local_file_ops, provider_, location_, object_, object_bytes_,
            capability_bit(source_capability::exact_range_read)
                | capability_bit(source_capability::stable_size)};
}

status_code local_file_source::read_exact_impl(
    void *context, const exact_read_request &request) noexcept {
    auto *source = static_cast<local_file_source *>(context);
    if (source == nullptr || !source->valid()
        || request.object_offset
            > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        return status_code::invalid_input;
    }
    std::size_t completed = 0;
    while (completed < request.byte_count) {
        const std::size_t remaining = static_cast<std::size_t>(
            request.byte_count - completed);
        const ssize_t count = ::pread(
            source->descriptor_, request.destination + completed, remaining,
            static_cast<off_t>(request.object_offset + completed));
        if (count == 0) {
            return status_code::short_read;
        }
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return status_code::short_read;
        }
        completed += static_cast<std::size_t>(count);
    }
    return status_code::success;
}

status_code open_local_file_source(
    const char *path, source_provider_id provider, source_location_id location,
    const storage_object_descriptor &object, local_file_source *out) noexcept {
    if (path == nullptr || path[0] == '\0' || !provider.valid()
        || !location.valid() || !valid_storage_object_descriptor(object)
        || out == nullptr) {
        return status_code::invalid_input;
    }
    out->reset();
    const int descriptor = ::open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        return status_code::missing_object;
    }
    struct stat value {};
    if (::fstat(descriptor, &value) != 0 || value.st_size < 0
        || static_cast<std::uint64_t>(value.st_size) != object.byte_count) {
        ::close(descriptor);
        return status_code::corruption;
    }
    out->descriptor_ = descriptor;
    out->provider_ = provider;
    out->location_ = location;
    out->object_ = object.id;
    out->object_bytes_ = object.byte_count;
    return status_code::success;
}

} // namespace cellshard
