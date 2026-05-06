#include <CellShard/io/csh5/api.cuh>
#include <CellShard/io/pack/packfile.cuh>
#include <CellShard/runtime/device/sharded_device.cuh>
#include <CellShard/runtime/host/sharded_host.cuh>
#include <CellShard/runtime/storage/disk.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

namespace cs = cellshard;

static void require(bool ok, const char *message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static std::string temp_path(const char *suffix) {
    std::string mutable_part = "/tmp/cellshard_dense_runtimeXXXXXX";
    char *buf = mutable_part.data();
    const int fd = ::mkstemp(buf);
    require(fd >= 0, "mkstemp failed");
    ::close(fd);
    ::unlink(buf);
    return std::string(buf) + suffix;
}

static void fill_dense(cs::dense *m, std::initializer_list<float> values) {
    std::size_t i = 0u;
    require(cs::dense_is_packed_row_major(m), "dense fill requires packed row-major matrix");
    require(values.size() == (std::size_t) m->rows * (std::size_t) m->cols, "dense fill value count mismatch");
    for (float value : values) m->val[i++] = __float2half(value);
}

static bool close_half(const real::storage_t &value, float expected) {
    return std::fabs(__half2float(value) - expected) < 0.001f;
}

static void test_raw_pack_and_reject_layouts() {
    const std::string path = temp_path(".cspackpart");
    cs::dense m;
    cs::dense loaded;
    cs::dense bad_stride;
    cs::dense bad_order;
    cs::init(&m, 2u, 3u, cs::dense_row_major, 3u);
    cs::init(&loaded);
    cs::init(&bad_stride, 2u, 3u, cs::dense_row_major, 4u);
    cs::init(&bad_order, 2u, 3u, cs::dense_col_major, 2u);
    require(cs::allocate(&m) != 0, "dense allocation failed");
    require(cs::allocate(&bad_stride) != 0, "bad-stride dense allocation failed");
    require(cs::allocate(&bad_order) != 0, "bad-order dense allocation failed");
    fill_dense(&m, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    require(cs::store(path.c_str(), &m) != 0, "packed dense raw store failed");
    require(cs::load(path.c_str(), &loaded) != 0, "packed dense raw load failed");
    require(loaded.rows == 2u && loaded.cols == 3u, "loaded dense shape mismatch");
    require(loaded.order == cs::dense_row_major && loaded.stride == loaded.cols, "loaded dense layout mismatch");
    require(close_half(*cs::at(&loaded, 1u, 2u), 6.0f), "loaded dense value mismatch");
    require(cs::store(path.c_str(), &bad_stride) == 0, "non-packed dense store unexpectedly succeeded");
    require(cs::store(path.c_str(), &bad_order) == 0, "column-major dense store unexpectedly succeeded");

    cs::clear(&bad_order);
    cs::clear(&bad_stride);
    cs::clear(&loaded);
    cs::clear(&m);
    ::unlink(path.c_str());
}

static void test_sharded_dense_metadata() {
    cs::sharded<cs::dense> matrix;
    cs::dense *part0 = new cs::dense;
    cs::dense *part1 = new cs::dense;
    cs::init(&matrix);
    cs::init(part0, 1u, 3u, cs::dense_row_major, 3u);
    cs::init(part1, 2u, 3u, cs::dense_row_major, 3u);
    require(cs::allocate(part0) != 0 && cs::allocate(part1) != 0, "sharded dense allocation failed");
    fill_dense(part0, {1.0f, 2.0f, 3.0f});
    fill_dense(part1, {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    require(cs::append_partition(&matrix, part0) != 0, "append dense partition 0 failed");
    require(cs::append_partition(&matrix, part1) != 0, "append dense partition 1 failed");
    part0 = nullptr;
    part1 = nullptr;

    require(matrix.rows == 3u && matrix.cols == 3u && matrix.nnz == 9u, "sharded dense metadata mismatch");
    require(cs::partition_bytes(&matrix, 1u) == cs::bytes(matrix.parts[1]), "sharded dense byte estimate mismatch");
    require(close_half(*cs::at(&matrix, 2u, 2u), 9.0f), "sharded dense at() mismatch");
    cs::clear(&matrix);
}

static void test_dense_csh5_fetch_and_stage() {
    const std::string path = temp_path(".csh5");
    const std::string cache = path + ".cache";
    const std::uint64_t partition_rows[] = {2u, 1u};
    const std::uint64_t partition_nnz[] = {6u, 3u};
    const std::uint64_t partition_aux[] = {0u, 0u};
    const std::uint32_t partition_axes[] = {0u, 0u};
    const std::uint64_t partition_row_offsets[] = {0u, 2u, 3u};
    const std::uint32_t partition_dataset_ids[] = {0u, 0u};
    const std::uint32_t partition_codec_ids[] = {0u, 0u};
    const std::uint64_t shard_offsets[] = {0u, 3u};
    cs::dataset_codec_descriptor codec{};
    codec.codec_id = 0u;
    codec.family = cs::dataset_codec_family_dense;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    const cs::dataset_layout_view layout{
        3u,
        3u,
        9u,
        2u,
        1u,
        partition_rows,
        partition_nnz,
        partition_axes,
        partition_aux,
        partition_row_offsets,
        partition_dataset_ids,
        partition_codec_ids,
        shard_offsets,
        &codec,
        1u
    };
    cs::dense part0;
    cs::dense part1;
    cs::sharded<cs::dense> matrix;
    cs::shard_storage storage;

    cs::init(&part0, 2u, 3u, cs::dense_row_major, 3u);
    cs::init(&part1, 1u, 3u, cs::dense_row_major, 3u);
    require(cs::allocate(&part0) != 0 && cs::allocate(&part1) != 0, "dense csh5 allocation failed");
    fill_dense(&part0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fill_dense(&part1, {7.0f, 8.0f, 9.0f});

    require(cs::create_dataset_dense_h5(path.c_str(), &layout, nullptr, nullptr) != 0, "create dense csh5 failed");
    require(cs::append_dense_partition_h5(path.c_str(), 0u, &part0) != 0, "append dense partition 0 failed");
    require(cs::append_dense_partition_h5(path.c_str(), 1u, &part1) != 0, "append dense partition 1 failed");

    cs::init(&matrix);
    cs::init(&storage);
    require(cs::load_header(path.c_str(), &matrix, &storage) != 0, "load dense header failed");
    require(cs::bind_dataset_h5_cache(&storage, cache.c_str()) != 0, "bind dense cache failed");
    require(matrix.rows == 3u && matrix.cols == 3u && matrix.nnz == 9u, "dense header metadata mismatch");
    require(cs::fetch_partition(&matrix, &storage, 1u) != 0, "fetch dense partition failed");
    require(close_half(*cs::at(&matrix, 2u, 2u), 9.0f), "fetched dense partition value mismatch");
    require(cs::fetch_shard(&matrix, &storage, 0u) != 0, "fetch dense shard failed");
    require(close_half(*cs::at(&matrix, 1u, 1u), 5.0f), "fetched dense shard value mismatch");

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        cs::device::sharded_device<cs::dense> device_state;
        cs::device::dense_view host_view{};
        real::storage_t payload[3] = {};
        cs::device::init(&device_state);
        require(cs::device::reserve(&device_state, matrix.num_partitions) != 0, "dense device reserve failed");
        require(cs::device::stage_partition(&device_state, &matrix, &storage, 1u, 0, 0) == cudaSuccess,
                "dense stage partition failed");
        require(cudaMemcpy(&host_view, device_state.parts[1].view, sizeof(host_view), cudaMemcpyDeviceToHost) == cudaSuccess,
                "dense descriptor copy failed");
        require(host_view.rows == 1u && host_view.cols == 3u, "dense device descriptor shape mismatch");
        require(host_view.order == cs::dense_row_major && host_view.stride == 3u, "dense device descriptor layout mismatch");
        require(cudaMemcpy(payload, host_view.val, sizeof(payload), cudaMemcpyDeviceToHost) == cudaSuccess,
                "dense payload copy failed");
        require(close_half(payload[2], 9.0f), "dense device payload mismatch");

        cs::dense bad;
        cs::device::partition_record<cs::dense> bad_record{};
        cs::init(&bad, 1u, 3u, cs::dense_row_major, 4u);
        require(cs::allocate(&bad) != 0, "bad dense device allocation failed");
        require(cs::device::upload(&bad, &bad_record) == cudaErrorInvalidValue,
                "non-packed dense device upload unexpectedly succeeded");
        cs::clear(&bad);
        cs::device::clear(&device_state);
    }

    cs::clear(&storage);
    cs::clear(&matrix);
    cs::clear(&part1);
    cs::clear(&part0);
    ::unlink(path.c_str());
}

int main() {
    test_raw_pack_and_reject_layouts();
    test_sharded_dense_metadata();
    test_dense_csh5_fetch_and_stage();
    return 0;
}
