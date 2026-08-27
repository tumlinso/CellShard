# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `cellshard-cpp-access-adapter-refactor` | status: closed | execution: closed | owner: codex | file: `todos/cellshard-cpp-access-adapter-refactor.md` | next: closed; reopen only for compatibility-header removal or broader external policy injection.

## Staleness Review
_No staleness review recorded yet._

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: yes, active workstreams: none.

<!-- todo-orchestrator:v2-managed:start -->
# Todo Status v2 Projection

Project revision: `186`

## Workstreams
- `CS-FOUND-00` | status: in_progress | execution: inactive | next: Complete CS-FOUND-01 and CS-FOUND-02 during bootstrap; later workers continue transactionally from CS-FOUND-03.
- `CS-FOUND-01` | status: done | execution: closed | next: Create the source-backed transition map, run the baseline gates, reach G0, and complete with evidence without architecture implementation.
- `CS-FOUND-02` | status: done | execution: closed | next: Add only the legacy codec/container baseline tests, run G1 gates, record results, reach G1, and stop without claiming CS-FOUND-03.
- `CS-FOUND-03` | status: done | execution: closed | next: Implement and validate CS-FOUND-I1 only; do not modify legacy identities.
- `CS-FOUND-04` | status: done | execution: closed | next: Implement only I2A in its exclusive files and complete positive/negative validation.
- `CS-FOUND-05` | status: done | execution: closed | next: Implement only I2B without paths, runtime placement, or payload interpretation.
- `CS-FOUND-06` | status: done | execution: closed | next: Implement I2C and fake in-memory exact-range source; defer local files unless mechanically inseparable.
- `CS-FOUND-07` | status: done | execution: closed | next: Review I1/I2A/I2B/I2C, reach C1 and G2, then implement catalog/snapshot validation without persistent manifest format.
- `CS-FOUND-08` | status: done | execution: closed | next: Implement fixed-field/table codec, deterministic padding/checksum, size calculation, and malformed-buffer tests only.
- `CS-FOUND-09` | status: done | execution: closed | next: Add new image store/inspect APIs, expose an extent, validate top-level offsets, and preserve the legacy writer.
- `CS-FOUND-10` | status: done | execution: closed | next: Review and freeze I3A/I3B/I4 at G3, then implement pread-backed local exact reads and one-allocation host residency.
- `CS-FOUND-12` | status: planned | execution: ready | next: Add explicit legacy context and row-partition adapter, mark CS-FOUND-LEGACY transition points, and preserve every listed public API.
- `CS-FOUND-11` | status: planned | execution: ready | next: Use the cuda skill, implement external allocator primary path plus transitional cudaMalloc wrapper, and run serialized correctness tests only.
- `CS-FOUND-13` | status: planned | execution: ready | next: Review CPEXEC02/I4, source independence, allocation/copy ownership, and legacy isolation; reach C2/G4; then implement only the specified fake vertical-slice tests and reach G5.
- `CS-FOUND-14` | status: planned | execution: ready | next: Update only implemented package/docs surfaces, add CS_FOUND_CONTRACT, run installed-package smoke, and reach G6 without rewriting the architecture review.
- `CS-FOUND-15` | status: planned | execution: ready | next: Run focused/full feasible CPU and CUDA tests, installed package smoke, static architecture scans, todo audit/reconcile/export, reach G7, and hand off with a broader physical-runtime investigation recommendation only.
- `cellshard-cpp-access-adapter-refactor` | status: closed | execution: idle | next: closed; reopen only for compatibility-header removal or broader external policy injection.
<!-- todo-orchestrator:v2-managed:end -->
