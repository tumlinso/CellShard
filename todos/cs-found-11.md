

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-11: Caller-owned CUDA allocator and device residency

Task revision: `302`; current project revision is in `todo-status.md`.

## Objective
Stage a validated host image through a caller allocator with one asynchronous H2D copy and move-only device residency as CS-FOUND-I5B.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Use the cuda skill, implement external allocator primary path plus transitional cudaMalloc wrapper, and run serialized correctness tests only.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `include/CellShard/runtime/residency/device.cuh`
- `exclusive`: `src/runtime/residency/device.cu`
- `exclusive`: `tests/device_residency_test.cu`
- `read`: `include/CellShard/artifact/image.hh`
- `read`: `include/CellShard/runtime/residency/host.hh`

## Dependencies
- `task`: `CS-FOUND-10`
- `checkpoint`: `CS-FOUND-G3`
<!-- todo-orchestrator:v2-managed:end -->
