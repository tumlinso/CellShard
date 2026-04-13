#include "CellShard/src/CellShard.hh"

int main() {
    ::cellshard::sharded::matrix matrix;
    ::cellshard::sharded::init(&matrix);
    return 0;
}
