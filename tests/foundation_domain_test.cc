#include <CellShard/domain.hh>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

int fail(const char *message) {
    std::fprintf(stderr, "cellShardFoundationDomainTest: %s\n", message);
    return 1;
}

cellshard::content_digest opaque_digest() {
    cellshard::content_digest digest{};
    digest.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    digest.used_bytes = sizeof(std::uint64_t);
    digest.bytes[0] = std::byte{0x5a};
    return digest;
}

} // namespace

int main() {
    using namespace cellshard;

    const archive_generation_id generation{7};
    const domain_descriptor cells{domain_id{11}, domain_kind::cells, generation, 100};
    const domain_descriptor genes{domain_id{12}, domain_kind::genes, generation, 80};
    const domain_descriptor sequence{
        domain_id{13}, domain_kind::sequence_coordinates, generation, 1000};
    const domain_descriptor opaque_domain{
        domain_id{14}, domain_kind::opaque, generation, 40};
    if (!valid_domain_descriptor(cells) || !valid_domain_descriptor(genes)
        || !valid_domain_descriptor(sequence)
        || !valid_domain_descriptor(opaque_domain)) {
        return fail("a supported domain kind was rejected");
    }

    const partition_map_descriptor sample_map{
        partition_map_id{21}, cells.id, generation, cells.element_count, 2};
    const partition_map_descriptor state_map{
        partition_map_id{22}, cells.id, generation, cells.element_count, 3};
    if (!valid_partition_map_descriptor(sample_map, cells)
        || !valid_partition_map_descriptor(state_map, cells)
        || sample_map.id == state_map.id) {
        return fail("independent maps for one domain did not validate");
    }

    partition_descriptor contiguous_partition{
        partition_id{31}, sample_map.id, cells.id, generation, 0,
        partition_selection::contiguous(0, 50)};
    if (!valid_partition_descriptor(contiguous_partition, sample_map, cells)) {
        return fail("contiguous partition was rejected");
    }

    partition_descriptor extent_partition{
        partition_id{32}, state_map.id, cells.id, generation, 1,
        partition_selection::explicit_ranges({{2, 4}, {20, 6}, {70, 5}})};
    if (!valid_partition_descriptor(extent_partition, state_map, cells)) {
        return fail("explicit-extent partition was rejected");
    }

    partition_descriptor opaque_partition{
        partition_id{33}, partition_map_id{23}, opaque_domain.id, generation, 0,
        partition_selection::opaque(9, opaque_digest())};
    const partition_map_descriptor opaque_map{
        opaque_partition.map, opaque_domain.id, generation,
        opaque_domain.element_count, 1};
    if (!valid_partition_descriptor(opaque_partition, opaque_map, opaque_domain)) {
        return fail("opaque partition selection was rejected");
    }

    const domain_binding binding{
        domain_binding_role::primary, cells.id, sample_map.id,
        contiguous_partition.id, order_id{41}};
    if (!valid_domain_binding(binding, contiguous_partition, sample_map, cells)) {
        return fail("explicit-order domain binding was rejected");
    }

    auto malformed_domain = cells;
    malformed_domain.kind = domain_kind::invalid;
    if (valid_domain_descriptor(malformed_domain)) {
        return fail("invalid domain kind was accepted");
    }
    malformed_domain = cells;
    malformed_domain.element_count = 0;
    if (valid_domain_descriptor(malformed_domain)) {
        return fail("empty domain was accepted");
    }

    auto wrong_map = sample_map;
    wrong_map.domain = genes.id;
    if (valid_partition_map_descriptor(wrong_map, cells)) {
        return fail("cross-domain partition map was accepted");
    }
    wrong_map = sample_map;
    wrong_map.generation = archive_generation_id{8};
    if (valid_partition_map_descriptor(wrong_map, cells)) {
        return fail("cross-generation partition map was accepted");
    }

    auto malformed_partition = contiguous_partition;
    malformed_partition.domain = genes.id;
    if (valid_partition_descriptor(malformed_partition, sample_map, cells)) {
        return fail("cross-domain partition was accepted");
    }
    malformed_partition = contiguous_partition;
    malformed_partition.generation = archive_generation_id{8};
    if (valid_partition_descriptor(malformed_partition, sample_map, cells)) {
        return fail("cross-generation partition was accepted");
    }
    malformed_partition = contiguous_partition;
    malformed_partition.ordinal = sample_map.partition_count;
    if (valid_partition_descriptor(malformed_partition, sample_map, cells)) {
        return fail("out-of-range partition ordinal was accepted");
    }
    malformed_partition = contiguous_partition;
    malformed_partition.owned = partition_selection::contiguous(90, 11);
    if (valid_partition_descriptor(malformed_partition, sample_map, cells)) {
        return fail("out-of-range contiguous selection was accepted");
    }
    malformed_partition = extent_partition;
    malformed_partition.owned = partition_selection::explicit_ranges({{20, 5}, {24, 3}});
    if (valid_partition_descriptor(malformed_partition, state_map, cells)) {
        return fail("overlapping explicit extents were accepted");
    }
    malformed_partition = opaque_partition;
    malformed_partition.owned = partition_selection::opaque(9, content_digest{});
    if (valid_partition_descriptor(malformed_partition, opaque_map, opaque_domain)) {
        return fail("unidentified opaque selection was accepted");
    }

    auto malformed_binding = binding;
    malformed_binding.order = order_id{};
    if (valid_domain_binding(malformed_binding, contiguous_partition, sample_map, cells)) {
        return fail("binding without explicit order was accepted");
    }
    malformed_binding = binding;
    malformed_binding.map = state_map.id;
    if (valid_domain_binding(malformed_binding, contiguous_partition, sample_map, cells)) {
        return fail("cross-map domain binding was accepted");
    }
    malformed_binding = binding;
    malformed_binding.role = domain_binding_role::unspecified;
    if (valid_domain_binding(malformed_binding, contiguous_partition, sample_map, cells)) {
        return fail("binding without an operational role was accepted");
    }

    std::puts("cellShardFoundationDomainTest: passed");
    return 0;
}
