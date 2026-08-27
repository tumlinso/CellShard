

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-06: Extent, storage-object, and source contracts

Task revision: `263`; current project revision is in `todo-status.md`.

## Objective
Separate immutable storage objects/extents from mutable source locations and define the non-owning storage-independent exact-read boundary as CS-FOUND-I2C.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Implement I2C and fake in-memory exact-range source; defer local files unless mechanically inseparable.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `include/CellShard/artifact/extent.hh`
- `exclusive`: `include/CellShard/runtime/source.hh`
- `exclusive`: `include/CellShard/runtime/source/payload_source.hh`
- `exclusive`: `tests/payload_source_test.cc`
- `read`: `include/CellShard/identity`

## Dependencies
- `task`: `CS-FOUND-03`
<!-- todo-orchestrator:v2-managed:end -->
