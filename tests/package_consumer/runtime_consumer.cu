#include "src/CellShard.hh"

int main() {
    ::cellshard::sharded<::cellshard::sparse::compressed> matrix;
    ::cellshard::init(&matrix);
    return 0;
}
