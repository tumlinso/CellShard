#include <CellShard/compiler/composition/production_identity_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>

namespace composition = cellshard::compiler::composition;

namespace {

cellshard::content_digest digest(std::uint8_t seed) {
    cellshard::content_digest value{};
    value.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    value.used_bytes = sizeof(std::uint64_t);
    for (std::size_t index = 0; index < value.used_bytes; ++index) {
        value.bytes[index] = std::byte{static_cast<std::uint8_t>(seed + index)};
    }
    return value;
}

composition::composition_production_identity_v1 identity() {
    composition::composition_production_identity_v1 value{};
    value.production = composition::composition_production_id{7};
    value.lineage = composition::composition_lineage_id{9};
    value.definition_digest = digest(3);
    return value;
}

} // namespace

int main() {
    const auto current = identity();
    assert(composition::validate_composition_production_identity_v1(current)
               .valid());

    composition::composition_production_identity_v1 next{};
    assert(composition::next_composition_revision_v1(
               current, digest(11), &next).valid());
    assert(composition::same_composition_production_v1(current, next));
    assert(!composition::same_composition_version_v1(current, next));
    assert(next.version.revision == 2);

    auto wrong_lineage = current;
    wrong_lineage.lineage = composition::composition_lineage_id{10};
    assert(!composition::same_composition_production_v1(
        current, wrong_lineage));

    auto reserved = current;
    reserved.version.reserved = 1;
    assert(composition::validate_composition_production_identity_v1(reserved)
               .code == composition::production_identity_code_v1::nonzero_reserved);

    auto overflow = current;
    overflow.version.revision = std::numeric_limits<std::uint64_t>::max();
    assert(composition::next_composition_revision_v1(
               overflow, digest(12), &next).code
           == composition::production_identity_code_v1::revision_overflow);
}
