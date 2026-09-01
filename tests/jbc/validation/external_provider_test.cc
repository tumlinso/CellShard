#include "external_provider.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    const external_proposal proposal{10, 20, 1, {1,2,3,4}, true};
    assert(!accept_external(proposal, {10, 20, {1,2,3,4}, true}, 1));
    assert(accept_external(proposal, {11, 20, {1,2,3,4}, true}, 1));
    assert(!accept_external(proposal, {11, 20, {1,2,3,4}, false}, 1));
    return 0;
}
