#include <CellShard/compiler/schedule/distributed_certificate.hh>
#include <array>
#include <cassert>
using namespace cellshard::compiler::schedule;
cellshard::content_digest d(std::byte b){cellshard::content_digest x{};x.algorithm=cellshard::digest_algorithm::legacy_fnv1a64;x.used_bytes=8;x.bytes[0]=b;return x;}
int main(){distributed_certificate c{portable_schedule_id{1},cellshard::partition_map_id{2},cellshard::route_table_id{3},d(std::byte{1}),d(std::byte{2}),2,5,9};std::array<participant_certificate,2>p{{{cellshard::partition_id{1},d(std::byte{3}),0,2,4},{cellshard::partition_id{2},d(std::byte{4}),2,3,5}}};assert(valid_distributed_certificate(c,p.data(),p.size()));p[1].partition=p[0].partition;assert(!valid_distributed_certificate(c,p.data(),p.size()));}
