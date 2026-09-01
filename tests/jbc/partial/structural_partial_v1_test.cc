#include <CellShard/compiler/partial/structural_partial_v1.hh>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using namespace cellshard::compiler::partial;

structural_partial_view_v1 view(
    const structural_partial_record_v1 *records, std::uint64_t count) {
    return {records, count, {1, 10}, {1, 11}, 7,
            structural_partial_kind_v1::node_membership, 1};
}

structural_partial_record_v1 record(std::uint64_t identity) {
    return {{2, identity}, {}, identity, 0};
}

void test_adversarial_validation() {
    std::array<structural_partial_record_v1, 2> records{record(1), record(2)};
    assert(validate_structural_partial_v1(view(records.data(), records.size()))
               .valid());
    auto invalid = records;
    invalid[1] = invalid[0];
    assert(validate_structural_partial_v1(view(invalid.data(), invalid.size()))
               .code == structural_partial_validation_code_v1::
                            unordered_or_duplicate_record);
    invalid = records;
    invalid[0].object_identity = {9, 9};
    assert(validate_structural_partial_v1(view(invalid.data(), invalid.size()))
               .code == structural_partial_validation_code_v1::
                            unexpected_object_identity);
}

void test_duplicate_and_capacity_fail_closed() {
    std::array<structural_partial_record_v1, 2> left{record(1), record(3)};
    std::array<structural_partial_record_v1, 2> right{record(2), record(4)};
    std::array<structural_partial_record_v1, 4> output{};
    assert(merge_structural_partials_v1(
               view(left.data(), left.size()), view(right.data(), right.size()),
               output.data(), output.size())
               .merged());
    assert(output[0].subject_identity.local_identity == 1);
    assert(output[3].subject_identity.local_identity == 4);
    assert(merge_structural_partials_v1(
               view(left.data(), left.size()), view(right.data(), right.size()),
               output.data(), output.size() - 1)
               .code == structural_partial_merge_code_v1::capacity_overflow);
    right[0] = record(3);
    assert(merge_structural_partials_v1(
               view(left.data(), left.size()), view(right.data(), right.size()),
               output.data(), output.size())
               .code
           == structural_partial_merge_code_v1::duplicate_contribution);
}

void test_randomized_differential_merge() {
    std::mt19937_64 generator(0x9e3779b97f4a7c15ULL);
    for (std::uint32_t trial = 0; trial < 512; ++trial) {
        std::vector<structural_partial_record_v1> left;
        std::vector<structural_partial_record_v1> right;
        std::vector<structural_partial_record_v1> expected;
        for (std::uint64_t identity = 1; identity <= 64; ++identity) {
            const auto bucket = generator() % 3;
            if (bucket == 0) {
                left.push_back(record(identity));
                expected.push_back(record(identity));
            } else if (bucket == 1) {
                right.push_back(record(identity));
                expected.push_back(record(identity));
            }
        }
        if (left.empty() || right.empty()) {
            --trial;
            continue;
        }
        std::vector<structural_partial_record_v1> output(expected.size());
        const auto result = merge_structural_partials_v1(
            view(left.data(), left.size()), view(right.data(), right.size()),
            output.data(), output.size());
        assert(result.merged());
        assert(result.output_count == expected.size());
        for (std::size_t index = 0; index < expected.size(); ++index) {
            assert(output[index].subject_identity
                   == expected[index].subject_identity);
            assert(output[index].subject_canonical_ordinal
                   == expected[index].subject_canonical_ordinal);
        }
    }
}

} // namespace

int main() {
    test_adversarial_validation();
    test_duplicate_and_capacity_fail_closed();
    test_randomized_differential_merge();
    return 0;
}
