

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-05: Projection and image descriptors

Task revision: `62`; current project revision is in `todo-status.md`.

## Objective
Represent producer-owned opaque execution images through producer ABI, target, projection, reuse, owning descriptor, and allocation-free view as CS-FOUND-I2B.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Implement only I2B without paths, runtime placement, or payload interpretation.

## Ownership
- `exclusive`: `include/CellShard/artifact/image.hh`
- `exclusive`: `tests/foundation_artifact_test.cc`
- `read`: `include/CellShard/domain`
- `read`: `include/CellShard/identity`

## Dependencies
- `task`: `CS-FOUND-03`
<!-- todo-orchestrator:v2-managed:end -->
