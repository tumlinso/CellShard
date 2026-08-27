#include <CellShard/io/pack/execution_payload.cuh>
#include <CellShard/runtime/layout/sharded.cuh>

#include <cstdio>
#include <cstdlib>

namespace {
void require(bool value, const char *message) {
    if (!value) { std::fprintf(stderr, "cellShardFoundationLegacyAdapterTest: %s\n", message); std::exit(1); }
}
cellshard::content_digest digest() {
    cellshard::content_digest value{};
    value.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    value.used_bytes = 8;
    value.bytes[0] = std::byte{1};
    return value;
}
}

int main() {
    using namespace cellshard;
    execution_payload_identity legacy{};
    legacy.dataset_identity = 11;
    legacy.generation = {12, 13, 14, 15};
    legacy.partition_identity = 16;
    legacy.global_row_begin = 17;
    legacy.row_count = 3;
    legacy.feature_count = 4;
    legacy.feature_axis_fingerprint = 18;
    legacy.feature_axis_fingerprint_version = 1;
    legacy.payload_kind = 19;
    legacy.payload_schema_version = 1;
    legacy.row_domain_identity = 20;
    legacy.payload_identity = 21;
    legacy_execution_image_context context{};
    context.legacy_partition_identity = 16;
    context.legacy_row_domain_identity = 20;
    context.canonical_generation = 12;
    context.image = image_id{1001};
    context.projection = {producer_abi_id{1002}, structure_id{1003},
        geometry_id{1004}, operator_class_id{1005}, scalar_encoding_id{1006},
        {execution_backend::cpu, 0, 0, 0}};
    context.binding = {domain_binding_role::primary, domain_id{2001},
        partition_map_id{2002}, partition_id{2003}, order_id{2004}};
    context.device_bytes = 64;
    context.required_alignment = 64;
    context.reuse = image_reuse_class::bounded_reuse;
    context.payload_digest = digest();
    image_descriptor image{};
    require(adapt_legacy_execution_image(legacy, 32, context, &image)
                == status_code::success,
            "explicit CPEXEC01 image adapter");
    require(image.id == context.image && image.domains[0].partition == partition_id{2003}
            && image.id.value() != legacy.payload_identity
            && image.domains[0].partition.value() != legacy.partition_identity,
            "new identities are explicit rather than legacy-derived");
    auto wrong = context; wrong.canonical_generation = 99;
    require(adapt_legacy_execution_image(legacy, 32, wrong, &image)
                == status_code::invalid_input,
            "canonical generation mismatch rejected");
    wrong = context; wrong.image = {};
    require(adapt_legacy_execution_image(legacy, 32, wrong, &image)
                == status_code::invalid_input,
            "missing image identity never manufactured");

    sharded<int> matrix{};
    unsigned long offsets[]{0, 5, 9};
    matrix.num_partitions = 2;
    matrix.partition_offsets = offsets;
    legacy_row_partition_binding binding{};
    require(adapt_legacy_row_partition(matrix, 1, context.binding, 12, 7,
                                       &binding),
            "explicit row partition adapter");
    require(binding.binding.partition == context.binding.partition
            && binding.global_row_begin == 5 && binding.row_count == 4
            && binding.physical_shard_group == 7,
            "row range and physical grouping stay compatibility metadata");
    require(!adapt_legacy_row_partition(matrix, 2, context.binding, 12, 7,
                                        &binding),
            "invalid legacy partition rejected");
    std::puts("cellShardFoundationLegacyAdapterTest: passed");
    return 0;
}
