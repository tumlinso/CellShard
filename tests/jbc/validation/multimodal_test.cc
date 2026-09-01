#include "multimodal.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    modality_identity modalities[] = {{10, 100, 101, 200, 201}, {20, 100, 101, 300, 301}};
    assert(valid_multimodal_spine(modalities, 2)); modalities[1].entity_order_id = 102;
    assert(!valid_multimodal_spine(modalities, 2)); return 0;
}
