#include <CellShard/compiler/partial/partial_image_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {
using namespace cellshard::compiler::atom;
using namespace cellshard::compiler::partial;
alignas(16) std::array<std::byte, 256> payload{};

partial_atom_view_v1 partial(std::uint64_t bytes) {
    partial_atom_view_v1 p{};
    p.payload = payload.data(); p.payload_bytes = bytes; p.payload_alignment = 16;
    p.header.partial_identity = {1, 1}; p.header.source_atom_semantic_identity = {1, 2};
    p.header.partial_kind_identity = {1, 3}; p.header.payload_schema_identity = {1, 4};
    p.header.contribution_coverage_identity = {1, 5};
    p.header.dependency_closure_identity = {1, 6};
    p.header.reconstruction_algebra_identity = {1, 7};
    p.header.numerical_policy_identity = {1, 8};
    p.header.complete_cost_evidence_identity = {1, 9};
    p.header.structure_generation = 2; p.header.value_generation = 3;
    p.header.state_generation = 4; p.header.materialization_generation = 5;
    p.header.cost_model_generation = 6;
    p.result.partial_layout.values = payload.data();
    p.result.partial_layout.value_bytes = bytes;
    p.result.partial_layout.value_alignment = 16;
    p.result.exact_contribution_coverage.coverage_identity = {1, 5};
    p.result.reconstruction_algebra_identity = {1, 7};
    p.result.numerical_policy_identity = {1, 8};
    p.result.status = atom_partial_result_status_v1::ready_to_merge;
    return p;
}

const std::array<partial_dependency_requirement_v1, 2> dependencies{{
    {{2, 1}, {3, 1}, 10, atom_dependency_generation_kind_v1::structure,
     partial_dependency_role_v1::direct},
    {{2, 2}, {3, 1}, 11, atom_dependency_generation_kind_v1::value,
     partial_dependency_role_v1::transitive}}};

partial_dependency_closure_view_v1 closure() {
    return {dependencies.data(), dependencies.size(), {1, 6}, {1, 10}, 1, 0};
}

void test_round_trip_and_corruption() {
    for (std::size_t index = 0; index < payload.size(); ++index)
        payload[index] = std::byte(index);
    std::array<std::byte, 1024> image{};
    const auto stored = serialize_partial_image_v1(
        partial(payload.size()), closure(), image.data(), image.size());
    assert(stored.valid());
    materialized_partial_image_v1 loaded{};
    assert(materialize_partial_image_v1(image.data(), stored.bytes, &loaded).valid());
    assert(loaded.header.partial_identity == partial(1).header.partial_identity);
    assert(loaded.dependency_count == dependencies.size());
    assert(partial_image_dependency_v1(loaded, 1).captured_generation == 11);
    assert(std::memcmp(loaded.payload, payload.data(), payload.size()) == 0);
    assert(materialize_partial_image_v1(image.data(), stored.bytes - 1, &loaded).code
           == partial_image_code_v1::truncated_image);
    image[stored.bytes - 1] ^= std::byte{1};
    assert(materialize_partial_image_v1(image.data(), stored.bytes, &loaded).code
           == partial_image_code_v1::checksum_mismatch);
}

void test_randomized_payload_round_trip() {
    std::mt19937_64 generator(0x24a19947b3916cf7ULL);
    for (std::uint32_t trial = 0; trial < 1024; ++trial) {
        const std::uint64_t bytes = 1 + generator() % payload.size();
        for (std::uint64_t index = 0; index < bytes; ++index)
            payload[index] = std::byte(generator() & 0xffU);
        std::array<std::byte, 1024> image{};
        const auto stored = serialize_partial_image_v1(
            partial(bytes), closure(), image.data(), image.size());
        assert(stored.valid());
        materialized_partial_image_v1 loaded{};
        assert(materialize_partial_image_v1(
                   image.data(), stored.bytes, &loaded).valid());
        assert(loaded.payload_bytes == bytes);
        assert(std::memcmp(loaded.payload, payload.data(), bytes) == 0);
    }
}

void test_randomized_corruption_rejected() {
    std::array<std::byte, 1024> image{};
    const auto stored = serialize_partial_image_v1(
        partial(payload.size()), closure(), image.data(), image.size());
    assert(stored.valid());
    std::mt19937_64 generator(0x9ba5d2dfe5b9c9d3ULL);
    for (std::uint32_t trial = 0; trial < 4096; ++trial) {
        auto corrupted = image;
        const std::uint64_t index = generator() % stored.bytes;
        corrupted[index] ^= std::byte{
            static_cast<std::uint8_t>(1U << (generator() % 8))};
        materialized_partial_image_v1 loaded{};
        assert(!materialize_partial_image_v1(
                    corrupted.data(), stored.bytes, &loaded).valid());
    }
}
}

int main() {
    test_round_trip_and_corruption();
    test_randomized_payload_round_trip();
    test_randomized_corruption_rejected();
    return 0;
}
