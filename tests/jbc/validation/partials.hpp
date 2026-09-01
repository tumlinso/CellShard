#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
namespace cellshard::jbc::validation {
struct numerical_partial {
    std::uint64_t count = 0;
    double sum = 0.0;
    double sum_squares = 0.0;
    double maximum = -std::numeric_limits<double>::infinity();
    double shifted_exp_sum = 0.0;
};
inline numerical_partial singleton_partial(double value) noexcept {
    return {1, value, value * value, value, 1.0};
}
inline numerical_partial merge_partials(const numerical_partial& left,
                                         const numerical_partial& right) noexcept {
    if (left.count == 0) return right;
    if (right.count == 0) return left;
    const double maximum = left.maximum > right.maximum ? left.maximum : right.maximum;
    const std::uint64_t count = right.count > UINT64_MAX - left.count ? UINT64_MAX : left.count + right.count;
    return {count, left.sum + right.sum,
            left.sum_squares + right.sum_squares, maximum,
            left.shifted_exp_sum * std::exp(left.maximum - maximum) +
            right.shifted_exp_sum * std::exp(right.maximum - maximum)};
}
inline double log_sum_exp(const numerical_partial& partial) noexcept {
    return partial.maximum + std::log(partial.shifted_exp_sum);
}
}  // namespace cellshard::jbc::validation
