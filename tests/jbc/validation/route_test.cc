#include "route.hpp"
#include <cassert>
int main() {
    using namespace cellshard::jbc::validation;
    const global_id hops[] = {100, 200};
    route_evidence route{1, 10, 20, 3, hops, 2, 4096, 4096, {1,2,3,4}, {1,2,3,4}, true};
    assert(valid_route(route)); route.resource_reserved = false; assert(!valid_route(route)); return 0;
}
