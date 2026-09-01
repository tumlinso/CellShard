#include "two_node.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    node_segment segments[2] = {{10, 30, 40, 0, 50, {1,2,3,4}, true},
                                {20, 30, 50, 50, 100, {1,2,3,4}, true}};
    assert(valid_two_node_slice(segments, 100)); segments[1].byte_begin = 51;
    assert(!valid_two_node_slice(segments, 100)); return 0;
}
