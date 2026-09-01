#include <CellShard/compiler/discovery/support_signature/lsh_index_v1.hh>

#include <array>
#include <cassert>

namespace signature =
    cellshard::compiler::discovery::support_signature;

int main() {
    const std::uint64_t minima[] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        1, 2, 3, 4, 50, 60, 70, 80,
        9, 9, 9, 9, 10, 10, 10, 10};
    const signature::deterministic_minhash_view_v1 sketch{
        minima, 3, 8, 0, 99, {1, 1}, 2};
    std::array<signature::deterministic_lsh_entry_v1, 6> output{};
    auto result = signature::build_lsh_index_v1(
        sketch, 2, 4, 3, output.data(), output.size());
    assert(result.built() && result.view.entry_count == 6);
    assert(output[0].band <= output[1].band);
    bool shared_bucket = false;
    for (std::size_t index = 1; index < output.size(); ++index) {
        if (output[index - 1].band == output[index].band
            && output[index - 1].bucket_hash == output[index].bucket_hash) {
            shared_bucket = true;
        }
    }
    assert(shared_bucket);
    assert(!signature::authorizes_execution(result.view));
    result = signature::build_lsh_index_v1(
        sketch, 2, 4, 1, output.data(), output.size());
    assert(result.code
           == signature::deterministic_lsh_code_v1::invalid_bucket_bound);
    result = signature::build_lsh_index_v1(
        sketch, 4, 2, 2, output.data(), output.size());
    assert(result.code
           == signature::deterministic_lsh_code_v1::insufficient_output);
}
