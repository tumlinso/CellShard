#include "../internal/snapshot_codec.hh"

#include <algorithm>
#include <cstring>

namespace cellshard::exporting {

namespace {

using namespace detail;

template<typename Part>
bool load_owner_snapshot_impl(const char *path,
                              const char *format_label,
                              global_metadata_snapshot *out,
                              std::string *error) {
    cellshard::sharded<Part> matrix;
    cellshard::shard_storage storage;
    cellshard::dataset_execution_view execution{};
    cellshard::dataset_runtime_service_view runtime{};
    cellshard::init(&matrix);
    cellshard::init(&storage);

    const auto fail = [&](const std::string &message) {
        clear_loaded_view(&matrix, &storage);
        set_error(error, message);
        return false;
    };

    if (!cellshard::load_header(path, &matrix, &storage)
        || !cellshard::get_dataset_h5_execution_metadata(&storage, &execution)
        || !cellshard::get_dataset_h5_runtime_service(&storage, &runtime)) {
        return fail(std::string("failed to load ") + format_label + " owner metadata");
    }

    copy_execution_partition_metadata(out->summary, execution, &out->execution_partitions);
    copy_execution_shard_metadata(out->summary, execution, &out->execution_shards);
    copy_runtime_service_metadata(runtime, &out->runtime_service);
    clear_loaded_view(&matrix, &storage);
    return true;
}

} // namespace

bool load_dataset_global_metadata_snapshot(const char *path,
                                           global_metadata_snapshot *out,
                                           std::string *error) {
    if (out == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "dataset path is empty");
        return false;
    }
    *out = global_metadata_snapshot{};
    if (!load_dataset_summary(path, &out->summary, error)) return false;

    if (out->summary.matrix_format == "blocked_ell") {
        if (!load_owner_snapshot_impl<cellshard::sparse::blocked_ell>(path, "blocked-ELL", out, error)) return false;
    } else if (out->summary.matrix_format == "sliced_ell") {
        if (!load_owner_snapshot_impl<cellshard::sparse::sliced_ell>(path, "sliced-ELL", out, error)) return false;
    } else {
        set_error(error, "unsupported matrix_format for owner metadata bootstrap: " + out->summary.matrix_format);
        return false;
    }

    out->snapshot_id = compute_snapshot_id(*out);
    return true;
}

client_snapshot_ref make_client_snapshot_ref(const global_metadata_snapshot &snapshot) {
    return build_client_snapshot_ref(snapshot);
}

bool validate_client_snapshot_ref(const global_metadata_snapshot &owner_snapshot,
                                  const client_snapshot_ref &request,
                                  std::string *error) {
    const client_snapshot_ref expected = build_client_snapshot_ref(owner_snapshot);
    if (request.snapshot_id != expected.snapshot_id) {
        set_error(error, "client snapshot_id does not match the owner snapshot");
        return false;
    }
    if (request.generation.canonical_generation != expected.generation.canonical_generation) {
        set_error(error, "client canonical_generation is stale");
        return false;
    }
    if (request.generation.execution_plan_generation != expected.generation.execution_plan_generation) {
        set_error(error, "client execution_plan_generation is stale");
        return false;
    }
    if (request.generation.pack_generation != expected.generation.pack_generation) {
        set_error(error, "client pack_generation is stale");
        return false;
    }
    if (request.generation.service_epoch != expected.generation.service_epoch) {
        set_error(error, "client service_epoch is stale");
        return false;
    }
    return true;
}

bool stage_append_only_runtime_service(const runtime_service_metadata &current,
                                       runtime_service_metadata *staged,
                                       std::string *error) {
    if (staged == nullptr) {
        set_error(error, "staged runtime_service output is null");
        return false;
    }
    if (current.live_write_mode != cellshard::dataset_live_write_mode_append_only) {
        set_error(error, "runtime service is not in append-only mode");
        return false;
    }

    const std::uint64_t next_generation = std::max({
        current.runtime_generation.generation.canonical_generation,
        current.runtime_generation.generation.execution_plan_generation,
        current.runtime_generation.generation.pack_generation,
        current.runtime_generation.active_read_generation,
        current.runtime_generation.staged_write_generation
    }) + 1u;

    *staged = current;
    staged->runtime_generation.generation.canonical_generation = next_generation;
    staged->runtime_generation.generation.execution_plan_generation = next_generation;
    staged->runtime_generation.generation.pack_generation = next_generation;
    staged->runtime_generation.staged_write_generation = next_generation;
    return true;
}

bool publish_runtime_service_cutover(const runtime_service_metadata &current,
                                     const runtime_service_metadata &staged,
                                     runtime_service_metadata *published,
                                     std::string *error) {
    if (published == nullptr) {
        set_error(error, "published runtime_service output is null");
        return false;
    }
    if (current.live_write_mode != cellshard::dataset_live_write_mode_append_only
        || staged.live_write_mode != cellshard::dataset_live_write_mode_append_only) {
        set_error(error, "runtime service cutover requires append-only mode");
        return false;
    }
    if (staged.runtime_generation.staged_write_generation == 0u) {
        set_error(error, "staged runtime service does not define a staged_write_generation");
        return false;
    }
    if (staged.runtime_generation.staged_write_generation < current.runtime_generation.active_read_generation) {
        set_error(error, "staged runtime generation is older than the active read generation");
        return false;
    }

    *published = staged;
    published->runtime_generation.active_read_generation = staged.runtime_generation.staged_write_generation;
    published->runtime_generation.staged_write_generation = published->runtime_generation.active_read_generation;
    published->runtime_generation.generation.service_epoch =
        std::max(current.runtime_generation.generation.service_epoch, staged.runtime_generation.generation.service_epoch) + 1u;
    return true;
}

bool describe_pack_delivery(const global_metadata_snapshot &owner_snapshot,
                            const pack_delivery_request &request,
                            pack_delivery_descriptor *out,
                            std::string *error) {
    if (out == nullptr) {
        set_error(error, "pack delivery descriptor output is null");
        return false;
    }
    if (!validate_client_snapshot_ref(owner_snapshot, request.request, error)) return false;
    if (request.shard_id >= owner_snapshot.summary.shards.size()) {
        set_error(error, "pack delivery shard_id is out of range");
        return false;
    }

    *out = pack_delivery_descriptor{};
    out->snapshot_id = owner_snapshot.snapshot_id;
    out->shard_id = request.shard_id;
    out->generation = generation_ref(owner_snapshot.runtime_service);

    if (request.shard_id < owner_snapshot.execution_shards.size()) {
        const execution_shard_metadata &shard = owner_snapshot.execution_shards[(std::size_t) request.shard_id];
        out->owner_node_id = shard.owner_node_id;
        out->owner_rank_id = shard.owner_rank_id;
        out->execution_format = shard.execution_format;
    }

    out->relative_pack_path = build_cspack_relative_path(owner_snapshot.runtime_service, request.shard_id);
    return true;
}

bool serialize_global_metadata_snapshot(const global_metadata_snapshot &snapshot,
                                        std::vector<std::uint8_t> *out,
                                        std::string *error) {
    global_metadata_snapshot normalized = snapshot;
    normalized.snapshot_id = compute_snapshot_id(snapshot);
    return serialize_global_metadata_snapshot_payload(normalized, out, true, true, error);
}

bool deserialize_global_metadata_snapshot(const void *data,
                                          std::size_t bytes,
                                          global_metadata_snapshot *out,
                                          std::string *error) {
    static const char magic[] = {'C', 'S', 'G', 'M'};
    byte_reader reader{};
    std::uint32_t version = 0u;
    std::uint64_t count = 0u;
    std::size_t size = 0u;
    std::uint64_t encoded_snapshot_id = 0u;

    if (out == nullptr || data == nullptr) {
        set_error(error, "metadata snapshot input is null");
        return false;
    }
    *out = global_metadata_snapshot{};
    reader.cur = static_cast<const std::uint8_t *>(data);
    reader.end = reader.cur + bytes;
    if ((std::size_t) (reader.end - reader.cur) < sizeof(magic) || std::memcmp(reader.cur, magic, sizeof(magic)) != 0) {
        set_error(error, "metadata snapshot magic mismatch");
        return false;
    }
    reader.cur += sizeof(magic);
    if (!read_pod(&reader, &version, error, "snapshot_version")) return false;
    if (version != 2u) {
        set_error(error, "unsupported metadata snapshot version");
        return false;
    }
    if (!read_pod(&reader, &encoded_snapshot_id, error, "snapshot_id")) return false;
    if (!read_dataset_summary(&reader, &out->summary, error, true)) return false;
    if (!read_pod(&reader, &count, error, "execution_partition_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "execution_partition_count is too large");
        return false;
    }
    out->execution_partitions.assign(size, execution_partition_metadata{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_execution_partition_metadata(&reader, out->execution_partitions.data() + i, error)) return false;
    }
    if (!read_pod(&reader, &count, error, "execution_shard_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "execution_shard_count is too large");
        return false;
    }
    out->execution_shards.assign(size, execution_shard_metadata{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_execution_shard_metadata(&reader, out->execution_shards.data() + i, error)) return false;
    }
    if (!read_runtime_service_metadata(&reader, &out->runtime_service, error)) return false;
    if (reader.cur != reader.end) {
        set_error(error, "metadata snapshot contains trailing bytes");
        return false;
    }

    out->snapshot_id = compute_snapshot_id(*out);
    if (out->snapshot_id != encoded_snapshot_id) {
        set_error(error, "metadata snapshot_id does not match the payload contents");
        return false;
    }
    return true;
}

} // namespace cellshard::exporting
