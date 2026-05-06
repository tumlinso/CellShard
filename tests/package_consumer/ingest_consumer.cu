#include <CellShard/ingest/dataset_ingest.cuh>
#include <CellShard/ingest/mtx_series.cuh>

int main() {
    ::cellshard::ingest::dataset::manifest manifest;
    ::cellshard::ingest::dataset::init(&manifest);
    ::cellshard::ingest::dataset::clear(&manifest);
    ::cellshard::ingest::mtx_series::options opts;
    opts.slice_rows = 64u;
    return 0;
}
