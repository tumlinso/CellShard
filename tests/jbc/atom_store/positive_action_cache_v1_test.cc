#include <CellShard/artifact/atom_store/positive_action_cache_v1.hh>

#include <array>
#include <cassert>

using namespace cellshard::artifact::atom_store;

static positive_action_cache_entry_v1 entry(std::uint64_t action_low,
                                            std::byte source_byte) {
    positive_action_cache_entry_v1 value{};
    value.action = {1, action_low};
    value.source_content.bytes[0] = source_byte;
    value.materialization = {2, action_low};
    value.output_content.bytes[0] = std::byte{3};
    value.structure_epoch = 4;
    value.evidence_generation = 5;
    value.certified = 1;
    return value;
}

int main() {
    std::array<positive_action_cache_entry_v1, 2> storage{};
    positive_action_cache_v1 cache(storage.data(), storage.size());
    const auto first = entry(10, std::byte{1});
    assert(cache.insert(first) == positive_action_cache_insert_result_v1::inserted);
    assert(cache.insert(first) == positive_action_cache_insert_result_v1::already_present);
    assert(cache.find(first.action, first.source_content, 4) != nullptr);
    assert(cache.find(first.action, first.source_content, 5) == nullptr);

    auto conflict = first;
    conflict.materialization = {9, 9};
    assert(cache.insert(conflict)
           == positive_action_cache_insert_result_v1::conflicting_result);
    const auto second = entry(11, std::byte{2});
    assert(cache.insert(second) == positive_action_cache_insert_result_v1::inserted);
    assert(cache.insert(entry(12, std::byte{3}))
           == positive_action_cache_insert_result_v1::capacity_exhausted);

    auto uncertified = entry(13, std::byte{4});
    uncertified.certified = 0;
    assert(cache.insert(uncertified)
           == positive_action_cache_insert_result_v1::invalid_entry);
}
