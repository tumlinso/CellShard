#include <CellShard/compiler/partial/gathered_panel_v1.hh>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <random>

namespace {
using namespace cellshard::compiler::partial;

void test_randomized_differential_gather() {
    std::array<double, 17 * 19> source{};
    for (std::size_t index = 0; index < source.size(); ++index) source[index] = double(index);
    std::mt19937_64 generator(0x3f84d5b5b5470917ULL);
    for (std::uint32_t trial = 0; trial < 512; ++trial) {
        std::array<gathered_axis_item_v1, 5> rows{};
        std::array<gathered_axis_item_v1, 7> columns{};
        for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = {{2, i + 1}, generator() % 17};
        for (std::size_t i = 0; i < columns.size(); ++i) columns[i] = {{3, i + 1}, generator() % 19};
        std::array<double, 35> output{};
        assert(gather_panel_values_v1(source.data(), 17, 19, 19,
                   rows.data(), rows.size(), columns.data(), columns.size(),
                   output.data(), output.size()).valid());
        for (std::size_t r = 0; r < rows.size(); ++r)
            for (std::size_t c = 0; c < columns.size(); ++c)
                assert(output[r * columns.size() + c]
                       == source[rows[r].canonical_ordinal * 19
                                 + columns[c].canonical_ordinal]);
    }
}

void test_validation_and_bounds() {
    std::array<gathered_axis_item_v1, 2> rows{{{{2, 1}, 0}, {{2, 2}, 1}}};
    std::array<gathered_axis_item_v1, 2> columns{{{{3, 1}, 0}, {{3, 2}, 1}}};
    std::array<double, 4> values{};
    gathered_panel_view_v1 panel{values.data(), values.size(), rows.data(), rows.size(),
        columns.data(), columns.size(), {1, 1}, {1, 2}, {1, 3}, 7, 8, 1, 0};
    assert(validate_gathered_panel_v1(panel).valid());
    panel.value_count = 3;
    assert(validate_gathered_panel_v1(panel).code
           == gathered_panel_code_v1::value_count_mismatch);
    rows[1].canonical_ordinal = 9;
    assert(gather_panel_values_v1(values.data(), 2, 2, 2, rows.data(), rows.size(),
               columns.data(), columns.size(), values.data(), values.size()).code
           == gathered_panel_code_v1::source_out_of_bounds);
}
}

int main() {
    test_randomized_differential_gather();
    test_validation_and_bounds();
    return 0;
}
