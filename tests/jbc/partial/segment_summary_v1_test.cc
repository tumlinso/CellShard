#include <CellShard/compiler/partial/segment_summary_v1.hh>

#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using namespace cellshard::compiler::partial;

segment_summary_v1 one(std::uint64_t ordinal, std::uint64_t label) {
    return {{{2, label}, ordinal}, {{2, label}, ordinal}, 1, 1,
            {1, 1}, {1, 2}, 7, 1, 0};
}

void test_adversarial_boundaries() {
    auto left = one(0, 9);
    auto right = one(1, 9);
    auto merged = merge_segment_summaries_v1(left, right);
    assert(merged.valid() && merged.summary.segment_count == 1);
    right = one(1, 10);
    merged = merge_segment_summaries_v1(left, right);
    assert(merged.valid() && merged.summary.segment_count == 2);
    right = one(2, 10);
    assert(merge_segment_summaries_v1(left, right).code
           == segment_summary_code_v1::noncontiguous_ranges);
    left.segment_count = 2;
    assert(validate_segment_summary_v1(left)
           == segment_summary_code_v1::impossible_segment_count);
}

void test_randomized_differential_merge_tree() {
    std::mt19937_64 generator(0xbe5466cf34e90c6cULL);
    for (std::uint32_t trial = 0; trial < 512; ++trial) {
        std::vector<std::uint64_t> labels(128);
        for (auto &label : labels) label = 1 + generator() % 7;
        std::vector<segment_summary_v1> level;
        for (std::size_t index = 0; index < labels.size(); ++index) {
            level.push_back(one(index, labels[index]));
        }
        while (level.size() > 1) {
            std::vector<segment_summary_v1> next;
            for (std::size_t index = 0; index < level.size(); index += 2) {
                if (index + 1 == level.size()) {
                    next.push_back(level[index]);
                } else {
                    const auto merged = merge_segment_summaries_v1(
                        level[index], level[index + 1]);
                    assert(merged.valid());
                    next.push_back(merged.summary);
                }
            }
            level = std::move(next);
        }
        std::uint64_t expected = 1;
        for (std::size_t index = 1; index < labels.size(); ++index) {
            expected += labels[index] != labels[index - 1];
        }
        assert(level.front().segment_count == expected);
    }
}

} // namespace

int main() {
    test_adversarial_boundaries();
    test_randomized_differential_merge_tree();
    return 0;
}
