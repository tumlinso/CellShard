#include <CellShard/ingest/dataset_ingest.cuh>

int main() {
    ::cellshard::ingest::dataset::manifest manifest;
    ::cellshard::ingest::dataset::init(&manifest);
    ::cellshard::ingest::dataset::clear(&manifest);
    return 0;
}
