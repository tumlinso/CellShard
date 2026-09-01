

<!-- todo-orchestrator:v2-managed:start -->
# CS-JBC-A05: Separate semantic, content, materialization, replica, and resident identities

Task revision: `1041`; current project revision is in `todo-status.md`.

## Objective
Separate semantic, content, materialization, replica, and resident identities. Deliver this as one isolated, reviewable step in the CellShard biological execution atom core workstream.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/CellShard/compiler/atom`
- `exclusive`: `src/compiler/atom`
- `exclusive`: `tests/jbc/atom`
- `read`: `include/CellShard/artifact/catalog.hh`
- `read`: `include/CellShard/artifact/image.hh`
- `read`: `include/CellShard/artifact/snapshot.hh`
- `read`: `include/CellShard/domain/descriptor.hh`
- `read`: `include/CellShard/domain/partition.hh`
- `read`: `include/CellShard/identity/strong_id.hh`

## Dependencies
- `task`: `CS-JBC-A04`
<!-- todo-orchestrator:v2-managed:end -->
