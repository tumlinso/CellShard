#include <CellShard/artifact/image.hh>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace {

int fail(const char *message) {
    std::fprintf(stderr, "cellShardFoundationArtifactTest: %s\n", message);
    return 1;
}

cellshard::content_digest payload_digest() {
    cellshard::content_digest digest{};
    digest.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    digest.used_bytes = sizeof(std::uint64_t);
    digest.bytes[0] = std::byte{0xa5};
    return digest;
}

cellshard::image_descriptor fake_image(cellshard::image_id id,
                                       cellshard::producer_abi_id producer,
                                       cellshard::geometry_id geometry,
                                       cellshard::order_id order) {
    using namespace cellshard;
    image_descriptor image{};
    image.id = id;
    image.projection = {
        producer,
        structure_id{101},
        geometry,
        operator_class_id{103},
        scalar_encoding_id{104},
        {execution_backend::cuda, 7, 0, 0x1},
    };
    image.stored_bytes = 4096;
    image.device_bytes = 8192;
    image.required_alignment = 256;
    image.reuse = image_reuse_class::bounded_reuse;
    image.payload_digest = payload_digest();
    image.domains = {
        {domain_binding_role::primary, domain_id{201}, partition_map_id{202},
         partition_id{203}, order},
        {domain_binding_role::secondary, domain_id{204}, partition_map_id{205},
         partition_id{206}, order_id{207}},
    };
    image.dependencies = {image_id{301}, image_id{302}};
    image.routes = {route_table_id{401}};
    return image;
}

} // namespace

int main() {
    using namespace cellshard;

    static_assert(std::is_trivially_copyable<image_descriptor_view>::value,
                  "image views must not own storage");
    static_assert(sizeof(image_descriptor_view) < sizeof(image_descriptor),
                  "view should remain pointer-count metadata");

    auto cell_image = fake_image(image_id{1}, producer_abi_id{11},
                                 geometry_id{12}, order_id{13});
    auto sequence_image = fake_image(image_id{2}, producer_abi_id{21},
                                     geometry_id{22}, order_id{23});
    if (!valid_image_descriptor(cell_image)
        || !valid_image_descriptor(sequence_image)) {
        return fail("valid fake producer image was rejected");
    }
    if (cell_image.projection.producer == sequence_image.projection.producer
        || cell_image.projection.geometry == sequence_image.projection.geometry
        || cell_image.domains[0].order == sequence_image.domains[0].order) {
        return fail("producer, geometry, or order identity was collapsed");
    }

    const auto view = view_of(cell_image);
    if (view.domains.data != cell_image.domains.data()
        || view.dependencies.data != cell_image.dependencies.data()
        || view.routes.data != cell_image.routes.data()
        || view.domains.size != cell_image.domains.size()) {
        return fail("allocation-free image view did not alias owning vectors");
    }

    auto malformed = cell_image;
    malformed.projection.producer = producer_abi_id{};
    if (valid_image_descriptor(malformed)) {
        return fail("image without producer ABI was accepted");
    }
    malformed = cell_image;
    malformed.projection.target = {execution_backend::cuda, 0, 0, 0};
    if (valid_image_descriptor(malformed)) {
        return fail("CUDA image without target capability was accepted");
    }
    malformed = cell_image;
    malformed.projection.target = {execution_backend::cpu, 1, 0, 0};
    if (valid_image_descriptor(malformed)) {
        return fail("CPU target with CUDA-style capability was accepted");
    }
    malformed = cell_image;
    malformed.required_alignment = 192;
    if (valid_image_descriptor(malformed)) {
        return fail("non-power-of-two alignment was accepted");
    }
    malformed = cell_image;
    malformed.payload_digest = content_digest{};
    if (valid_image_descriptor(malformed)) {
        return fail("image without payload digest was accepted");
    }
    malformed = cell_image;
    malformed.domains.clear();
    if (valid_image_descriptor(malformed)) {
        return fail("image without operational domain metadata was accepted");
    }
    malformed = cell_image;
    malformed.domains.push_back(malformed.domains.front());
    if (valid_image_descriptor(malformed)) {
        return fail("duplicate domain binding was accepted");
    }
    malformed = cell_image;
    malformed.dependencies.push_back(malformed.id);
    if (valid_image_descriptor(malformed)) {
        return fail("self-dependent image was accepted");
    }
    malformed = cell_image;
    malformed.dependencies.push_back(malformed.dependencies.front());
    if (valid_image_descriptor(malformed)) {
        return fail("duplicate image dependency was accepted");
    }
    malformed = cell_image;
    malformed.routes.push_back(malformed.routes.front());
    if (valid_image_descriptor(malformed)) {
        return fail("duplicate route identity was accepted");
    }

    std::puts("cellShardFoundationArtifactTest: passed");
    return 0;
}
