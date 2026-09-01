#include <CellShard/runtime/v2/route_calibration.hh>

#include <cassert>
#include <limits>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    content_digest identity{};
    identity.algorithm = digest_algorithm::legacy_fnv1a64;
    identity.used_bytes = 8;
    route_calibration record{route_table_id{4}, identity, 1, 2, 4096,
                             1ULL << 30, 800, 80, 1200, 32};
    assert(valid_route_calibration(record));
    assert(calibrated_route_nanoseconds(record, 4096) == 1128);
    assert(calibrated_route_nanoseconds(record, 4095)
           == std::numeric_limits<std::uint64_t>::max());
    record.destination_node = 1;
    assert(!valid_route_calibration(record));
    record.destination_node = 2;
    record.p99_fixed_nanoseconds = 799;
    assert(!valid_route_calibration(record));
}
