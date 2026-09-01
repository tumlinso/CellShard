

<!-- todo-orchestrator:v2-managed:start -->
# CS-JBC-ST19: Implement metadata-only reader and inspector

Task revision: `1041`; current project revision is in `todo-status.md`.

## Objective
Implement metadata-only reader and inspector. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `docs/SPEC_ATOM_STORE_V1.md`
- `exclusive`: `include/CellShard/artifact/atom_store`
- `exclusive`: `src/artifact/atom_store`
- `exclusive`: `tests/jbc/atom_store`
- `read`: `docs/SPEC_CSPACK_V1.md`
- `read`: `include/CellShard/artifact/catalog.hh`
- `read`: `include/CellShard/artifact/extent.hh`
- `read`: `include/CellShard/artifact/image.hh`
- `read`: `include/CellShard/artifact/snapshot.hh`
- `read`: `include/CellShard/io/pack/image_envelope.hh`

## Dependencies
- `task`: `CS-JBC-ST18`
<!-- todo-orchestrator:v2-managed:end -->
