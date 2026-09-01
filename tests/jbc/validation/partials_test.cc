#include "partials.hpp"
#include <cassert>
#include <cmath>
int main() {
    using namespace cellshard::jbc::validation;
    const auto partial = merge_partials(merge_partials(singleton_partial(1.0), singleton_partial(2.0)), singleton_partial(3.0));
    assert(partial.count == 3 && partial.sum == 6.0 && partial.sum_squares == 14.0);
    const double reference = std::log(std::exp(1.0) + std::exp(2.0) + std::exp(3.0));
    assert(std::abs(log_sum_exp(partial) - reference) < 1.0e-12);
    return 0;
}
