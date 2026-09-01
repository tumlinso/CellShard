#include "fixtures.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    for (const auto& item : corpus) assert(valid_fixture(item));
    assert(corpus[0].biological && corpus[1].biological && !corpus[2].biological);
    assert(corpus[0].fixture_id > UINT32_MAX);
    return 0;
}
