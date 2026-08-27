

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-07: Artifact catalog and snapshot validation

Task revision: `92`; current project revision is in `todo-status.md`.

## Objective
Freeze the descriptor ABI checkpoint, then implement separate in-memory artifact/source catalogs and snapshot validation as CS-FOUND-I3A.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Review I1/I2A/I2B/I2C, reach C1 and G2, then implement catalog/snapshot validation without persistent manifest format.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `include/CellShard/artifact/catalog.hh`
- `exclusive`: `include/CellShard/artifact/snapshot.hh`
- `exclusive`: `tests/foundation_snapshot_test.cc`
- `read`: `include/CellShard/artifact/extent.hh`
- `read`: `include/CellShard/artifact/image.hh`
- `read`: `include/CellShard/domain`
- `read`: `include/CellShard/identity`
- `read`: `include/CellShard/runtime/source/payload_source.hh`

## Dependencies
- `barrier`: `CS-FOUND-DESCRIPTOR-FANIN`
<!-- todo-orchestrator:v2-managed:end -->
