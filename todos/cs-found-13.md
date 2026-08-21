

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-13: End-to-end opaque image vertical slice

Task revision: `62`; current project revision is in `todo-status.md`.

## Objective
Freeze the image/residency checkpoint and prove the specified fake producer to host/device fake consumer pipeline with exact identity and byte preservation.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `integration_exclusive`
- Result: `-`

## Next Action
Review CPEXEC02/I4, source independence, allocation/copy ownership, and legacy isolation; reach C2/G4; then implement only the specified fake vertical-slice tests and reach G5.

## Ownership
- `exclusive`: `tests/opaque_image_pipeline_cuda_test.cu`
- `exclusive`: `tests/opaque_image_pipeline_test.cc`
- `read`: `include/CellShard/artifact`
- `read`: `include/CellShard/domain`
- `read`: `include/CellShard/identity`
- `read`: `include/CellShard/io/pack/image_envelope.hh`
- `read`: `include/CellShard/runtime/residency`
- `read`: `include/CellShard/runtime/source`

## Dependencies
- `task`: `CS-FOUND-07`
- `task`: `CS-FOUND-09`
- `task`: `CS-FOUND-10`
- `task`: `CS-FOUND-11`
- `task`: `CS-FOUND-12`
- `barrier`: `CS-FOUND-IMAGE-RESIDENCY-FANIN`
<!-- todo-orchestrator:v2-managed:end -->
