#include "null_transform.hpp"
#include <array>
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    std::uint32_t output[8]{};
    assert(matched_degree_null(corpus[0], 17, 64, output) != 0);
    std::array<unsigned, 6> before{}, after{};
    for (std::uint32_t i = 0; i < 8; ++i) { ++before[module_columns[i]]; ++after[output[i]]; }
    assert(before == after);
    for (std::uint32_t row = 0; row < 4; ++row)
        assert(module_offsets[row + 1] - module_offsets[row] == 2);
    return 0;
}
