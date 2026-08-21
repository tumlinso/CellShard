

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-09: CSPACK01 image publication and inspection

Task revision: `89`; current project revision is in `todo-status.md`.

## Objective
Integrate independently inspectable CPEXEC02 entries into unchanged CSPACK01 with atomic publication and expose CS-FOUND-I4.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `integration_exclusive`
- Result: `-`

## Next Action
Add new image store/inspect APIs, expose an extent, validate top-level offsets, and preserve the legacy writer.

## Ownership
- `exclusive`: `include/CellShard/io/pack/image_envelope.hh`
- `exclusive`: `src/io/pack/image_envelope.cc`
- `exclusive`: `tests/image_envelope_test.cc`
- `read`: `include/CellShard/io/pack/execution_payload.cuh`
- `read`: `include/CellShard/io/pack/packfile.cuh`
- `read`: `src/io/pack/execution_payload.cu`
- `read`: `src/io/pack/packfile.cu`

## Dependencies
- `task`: `CS-FOUND-07`
- `task`: `CS-FOUND-08`
<!-- todo-orchestrator:v2-managed:end -->
