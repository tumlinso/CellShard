#include <CellShard/access.hh>
#include <CellShard/core/cuda_compat.cuh>
#include <CellShard/artifact.hh>
#include <CellShard/io/pack/image_envelope.hh>
#include <CellShard/runtime/residency/host.hh>
#include <CellShard/runtime/source/local_file_source.hh>
#include <CellShard/runtime/layout/sharded.cuh>

int main() {
    ::cellshard::sharded<::cellshard::sparse::compressed> matrix;
    ::cellshard::init(&matrix);
    static_assert(::cellshard::access::is_archive_adapter<::cellshard::access::dense_fallback_binding>::value,
                  "installed access fallback headers should compile for package consumers");
    ::cellshard::artifact_catalog artifacts;
    ::cellshard::host_residency host;
    ::cellshard::local_file_source source;
    (void) artifacts;
    (void) host;
    (void) source;
    return 0;
}
