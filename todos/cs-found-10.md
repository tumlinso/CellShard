

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-10: Local-file source and host residency

Task revision: `166`; current project revision is in `todo-status.md`.

## Objective
Reach G3, then resolve one image extent through LocalFileSource into one move-only host residency allocation as CS-FOUND-I5A.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Review and freeze I3A/I3B/I4 at G3, then implement pread-backed local exact reads and one-allocation host residency.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `include/CellShard/runtime/residency.hh`
- `exclusive`: `include/CellShard/runtime/residency/host.hh`
- `exclusive`: `include/CellShard/runtime/source/local_file_source.hh`
- `exclusive`: `src/runtime/residency/host.cc`
- `exclusive`: `src/runtime/source/local_file_source.cc`
- `exclusive`: `tests/host_residency_test.cc`
- `forbidden`: `include/CellShard/io/pack/execution_payload.cuh`
- `forbidden`: `src/io/pack/execution_payload.cu`
- `read`: `include/CellShard/artifact`
- `read`: `include/CellShard/io/pack/image_envelope.hh`
- `read`: `include/CellShard/runtime/source/payload_source.hh`

## Dependencies
- `task`: `CS-FOUND-07`
- `task`: `CS-FOUND-08`
- `task`: `CS-FOUND-09`
<!-- todo-orchestrator:v2-managed:end -->
