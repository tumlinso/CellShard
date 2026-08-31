<!-- todo-orchestrator:v2-managed:start -->
# CS-JBC-RT20: Implement worker-local CUDA Graph lowering

Task revision: `312`; current project revision is in `todo-status.md`.

## Objective
Implement worker-local CUDA Graph lowering. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/jbc/runtime`
- `exclusive`: `include/CellShard/runtime/v2`
- `exclusive`: `src/runtime/v2`
- `exclusive`: `tests/jbc/runtime`
- `read`: `include/CellShard/runtime/residency/device.cuh`
- `read`: `include/CellShard/runtime/residency/host.hh`
- `read`: `include/CellShard/runtime/source/local_file_source.hh`
- `read`: `include/CellShard/runtime/source/payload_source.hh`
- `read`: `numaBraid`

## Dependencies
- `task`: `CS-JBC-RT19`
<!-- todo-orchestrator:v2-managed:end -->
