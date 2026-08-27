

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-12: Legacy identity and sharded-runtime adapters

Task revision: `166`; current project revision is in `todo-status.md`.

## Objective
Provide explicit-context legacy CPEXEC01 and sharded adapters without defining new semantics through row shards.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Add explicit legacy context and row-partition adapter, mark CS-FOUND-LEGACY transition points, and preserve every listed public API.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `include/CellShard/io/pack/execution_payload.cuh`
- `exclusive`: `include/CellShard/runtime/layout/sharded.cuh`
- `exclusive`: `src/io/pack/execution_payload.cu`
- `exclusive`: `tests/execution_payload_test.cu`
- `exclusive`: `tests/foundation_legacy_adapter_test.cc`
- `forbidden`: `include/CellShard/runtime/residency/host.hh`
- `forbidden`: `include/CellShard/runtime/source/local_file_source.hh`
- `forbidden`: `src/runtime/residency/host.cc`
- `forbidden`: `src/runtime/source/local_file_source.cc`
- `read`: `include/CellShard/artifact`
- `read`: `include/CellShard/domain`
- `read`: `include/CellShard/runtime/host/sharded_host.cuh`

## Dependencies
- `task`: `CS-FOUND-04`
- `task`: `CS-FOUND-05`
- `task`: `CS-FOUND-06`
- `task`: `CS-FOUND-09`
- `checkpoint`: `CS-FOUND-G3`
<!-- todo-orchestrator:v2-managed:end -->
