

<!-- todo-orchestrator:v2-managed:start -->
# CS-JBC-RT18: Implement reconstruction-aware admission and eviction

Task revision: `1041`; current project revision is in `todo-status.md`.

## Objective
Implement reconstruction-aware admission and eviction. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

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
- `task`: `CS-JBC-RT17`
<!-- todo-orchestrator:v2-managed:end -->
