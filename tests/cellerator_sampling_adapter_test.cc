#include <CellShard/CellShard.hh>
#include <CellShard/interop/cellerator/sampling.hh>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace cs = ::cellshard;
namespace sampling = ::cellerator::compute::sampling;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    char path[] = "/tmp/cellshard_cellerator_samplingXXXXXX.csh5";
    const int fd = ::mkstemps(path, 5);
    require(fd >= 0, "failed to allocate temporary .csh5 path");
    ::close(fd);
    std::remove(path);

    cs::sparse::blocked_ell part;
    cs::bucketed_blocked_ell_partition bucket;
    cs::bucketed_blocked_ell_shard shard;
    cs::sparse::init(&part, 2u, 4u, 4u, 2u, 4u);
    cs::init(&bucket);
    cs::init(&shard);
    require(cs::sparse::allocate(&part) != 0, "failed to allocate blocked-ELL fixture");
    part.blockColIdx[0] = 0u;
    part.blockColIdx[1] = 1u;
    part.val[0] = __float2half(1.0f);
    part.val[1] = __float2half(0.0f);
    part.val[2] = __float2half(2.0f);
    part.val[3] = __float2half(0.0f);
    part.val[4] = __float2half(0.0f);
    part.val[5] = __float2half(3.0f);
    part.val[6] = __float2half(0.0f);
    part.val[7] = __float2half(4.0f);

    const std::uint64_t partition_rows[] = {2u};
    const std::uint64_t partition_nnz[] = {4u};
    const std::uint32_t partition_axes[] = {0u};
    const std::uint64_t partition_aux[] = {
        (std::uint64_t) cs::sparse::pack_blocked_ell_aux(2u, 2ul)};
    const std::uint64_t partition_offsets[] = {0u, 2u};
    const std::uint64_t shard_offsets[] = {0u, 2u};
    const std::uint32_t dataset_ids[] = {0u};
    const std::uint32_t codec_ids[] = {0u};
    cs::dataset_codec_descriptor codec{};
    codec.codec_id = 0u;
    codec.family = cs::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    const cs::dataset_layout_view layout{
        2u, 4u, 4u, 1u, 1u,
        partition_rows, partition_nnz, partition_axes, partition_aux,
        partition_offsets, dataset_ids, codec_ids, shard_offsets, &codec, 1u};

    require(cs::build_bucketed_blocked_ell_partition(&bucket, &part, 1u, nullptr) != 0,
            "failed to build bucketed fixture");
    bucket.exec_to_canonical_cols =
        (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    bucket.canonical_to_exec_cols =
        (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    shard.rows = 2u;
    shard.cols = 4u;
    shard.nnz = 4u;
    shard.partition_count = 1u;
    shard.partitions = (cs::bucketed_blocked_ell_partition *)
        std::calloc(1u, sizeof(cs::bucketed_blocked_ell_partition));
    shard.partition_row_offsets = (std::uint32_t *) std::calloc(2u, sizeof(std::uint32_t));
    shard.exec_to_canonical_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    shard.canonical_to_exec_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    require(bucket.exec_to_canonical_cols != nullptr
                && bucket.canonical_to_exec_cols != nullptr
                && shard.partitions != nullptr && shard.partition_row_offsets != nullptr
                && shard.exec_to_canonical_cols != nullptr
                && shard.canonical_to_exec_cols != nullptr,
            "failed to allocate fixture maps");
    for (std::uint32_t column = 0u; column < 4u; ++column) {
        bucket.exec_to_canonical_cols[column] = column;
        bucket.canonical_to_exec_cols[column] = column;
        shard.exec_to_canonical_cols[column] = column;
        shard.canonical_to_exec_cols[column] = column;
    }
    shard.partitions[0] = bucket;
    cs::init(&bucket);
    shard.partition_row_offsets[0] = 0u;
    shard.partition_row_offsets[1] = 2u;
    require(cs::create_dataset_optimized_blocked_ell_h5(path, &layout, nullptr, nullptr) != 0,
            "failed to create fixture dataset");
    require(cs::append_bucketed_blocked_ell_shard_h5(path, 0u, &shard) != 0,
            "failed to append fixture shard");

    sampling::sample_spec spec;
    sampling::sample_plan plan;
    sampling::owned_sampled_csr_structure sampled;
    std::string error;
    spec.mode = sampling::selection_mode::exact_lowest_hash;
    spec.seed = 919u;
    spec.split_name = "cellshard-file-adapter";
    spec.requested_row_count = 64u;
    require(cs::interop::cellerator::build_sample_plan(path, spec, &plan, &error), error.c_str());
    require(cs::interop::cellerator::materialize_sampled_csr_structure(
                path, plan, &sampled, &error), error.c_str());
    const sampling::sampled_csr_structure_view view = sampled.view();
    require(view.sampled_row_count == 2u && view.gene_count == 4u && view.nnz == 4u,
            "sampled adapter shape mismatch");
    require(view.row_ptr[0] == 0u && view.row_ptr[1] == 2u && view.row_ptr[2] == 4u,
            "sampled adapter row pointers mismatch");
    require(view.column_indices[0] == 0u && view.column_indices[1] == 2u
                && view.column_indices[2] == 1u && view.column_indices[3] == 3u,
            "sampled adapter columns mismatch");

    cs::clear(&shard);
    cs::clear(&bucket);
    cs::sparse::clear(&part);
    std::remove(path);
    return 0;
}
