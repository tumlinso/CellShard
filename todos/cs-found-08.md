

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-08: Explicit CPEXEC02 codec

Task revision: `166`; current project revision is in `todo-status.md`.

## Objective
Implement deterministic explicit little-endian CPEXEC02 buffer encoding/decoding as CS-FOUND-I3B without CSPACK file integration.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Implement fixed-field/table codec, deterministic padding/checksum, size calculation, and malformed-buffer tests only.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `include/CellShard/io/pack/image_envelope.hh`
- `exclusive`: `src/io/pack/image_envelope.cc`
- `exclusive`: `tests/image_envelope_test.cc`
- `forbidden`: `include/CellShard/io/pack/execution_payload.cuh`
- `forbidden`: `src/io/pack/execution_payload.cu`
- `read`: `include/CellShard/artifact`
- `read`: `include/CellShard/domain`
- `read`: `include/CellShard/identity`

## Dependencies
- `checkpoint`: `CS-FOUND-C1`
- `checkpoint`: `CS-FOUND-G2`
<!-- todo-orchestrator:v2-managed:end -->
