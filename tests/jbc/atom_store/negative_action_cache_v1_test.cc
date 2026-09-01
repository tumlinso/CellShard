#include <CellShard/artifact/atom_store/negative_action_cache_v1.hh>
#include <array>
#include <cassert>
using namespace cellshard::artifact::atom_store;
int main() {
    std::array<negative_action_cache_entry_v1, 1> storage{};
    negative_action_cache_v1 cache(storage.data(), storage.size());
    negative_action_cache_entry_v1 entry{};
    entry.action = {1, 2}; entry.source_content.bytes[0] = std::byte{1};
    entry.structure_epoch = 3; entry.evidence_generation = 4;
    entry.valid_through_generation = 6;
    entry.reason = negative_action_reason_v1::unsupported_capability;
    assert(cache.insert(entry));
    assert(cache.find(entry.action, entry.source_content, 3, 4) != nullptr);
    assert(cache.find(entry.action, entry.source_content, 3, 6) != nullptr);
    assert(cache.find(entry.action, entry.source_content, 3, 7) == nullptr);
    assert(!cache.insert(entry));
}
