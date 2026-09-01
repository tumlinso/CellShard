#include <CellShard/compiler/evidence/cross_dataset_template_v1.hh>

#include <array>
#include <cassert>

namespace evidence = cellshard::compiler::evidence;

int main() {
    std::array<evidence::dataset_template_binding_v1, 2> bindings{{
        {{1, 1}, {2, 1}, {3, 1}, {4, 1}, 1},
        {{1, 2}, {2, 2}, {3, 2}, {4, 2}, 1},
    }};
    evidence::cross_dataset_template_view_v1 view{
        bindings.data(), bindings.size(), bindings.size(), {5, 1}, {5, 2}};
    assert(evidence::validate_cross_dataset_template_v1(view).valid());
    assert(!evidence::establishes_biological_identity(view));

    auto malformed = view;
    bindings[1].order_identity = {};
    assert(evidence::validate_cross_dataset_template_v1(malformed).code
           == evidence::cross_dataset_template_validation_code_v1::invalid_binding_identity);
    bindings[1].order_identity = {3, 2};
    bindings[1].dataset_identity = bindings[0].dataset_identity;
    assert(evidence::validate_cross_dataset_template_v1(view).code
           == evidence::cross_dataset_template_validation_code_v1::unordered_or_duplicate_dataset);
}
