#include <CellShard/compiler/certification/local_identity_map_v1.hh>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using namespace cellshard::compiler;

int main() {
    std::vector<std::uint64_t> canonical(300);
    for (std::uint64_t index = 0; index < canonical.size(); ++index) {
        canonical[index] = (UINT64_C(1) << 40) + index * 2 + 1;
    }
    std::vector<std::uint64_t> local = canonical;
    std::mt19937_64 generator(1234);
    std::shuffle(local.begin(), local.end(), generator);
    local.resize(200);

    std::vector<std::byte> canonical_to_local(canonical.size() * 2);
    std::vector<std::byte> local_to_canonical(local.size() * 2);
    std::vector<std::uint8_t> marks(canonical.size());
    certification::local_identity_map_buffers_v1 buffers{
        canonical_to_local.data(),
        canonical_to_local.size(),
        local_to_canonical.data(),
        local_to_canonical.size(),
        marks.data(),
        marks.size()};
    const auto result = certification::build_local_identity_maps_v1(
        canonical.data(),
        canonical.size(),
        local.data(),
        local.size(),
        certification::certification_local_index_width_v1::u16,
        buffers);
    assert(result.built());
    assert(result.map.index_width
           == certification::certification_local_index_width_v1::u16);
    for (std::uint64_t local_index = 0; local_index < local.size();
         ++local_index) {
        const auto canonical_index = certification::read_local_index_v1(
            result.map.local_to_canonical,
            local_index,
            result.map.index_width);
        assert(canonical[canonical_index] == local[local_index]);
        assert(certification::read_local_index_v1(
                   result.map.canonical_to_local,
                   canonical_index,
                   result.map.index_width)
               == local_index);
    }

    assert(certification::build_local_identity_maps_v1(
               canonical.data(),
               canonical.size(),
               local.data(),
               local.size(),
               certification::certification_local_index_width_v1::u8,
               buffers)
               .code
           == certification::local_identity_map_build_code_v1::width_exceeded);

    local[10] = local[9];
    assert(certification::build_local_identity_maps_v1(
               canonical.data(),
               canonical.size(),
               local.data(),
               local.size(),
               certification::certification_local_index_width_v1::u16,
               buffers)
               .code
           == certification::local_identity_map_build_code_v1::
               duplicate_local_identity);
}
