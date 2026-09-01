# CellShard source transition map for the joint biological compiler

This is the source-backed `CS-JBC-B02` classification of the live nested
CellShard checkout. The machine-readable companion is
[`planning/jbc/cellshard_source_transition_map_v1.csv`](../../planning/jbc/cellshard_source_transition_map_v1.csv).
The map changes no behavior, public header, format byte, central registry, or
historical workflow state.

## Provenance and authority

The exact input is the `CS-JBC-B02` object in the joint-compiler preledger
`proposed_todos.json`. Its newline-terminated compact sorted JSON SHA-256 is
`8662c4c0effd5819b79915cff2eda7ce9a0f02b64f5aca9710972f5c5f1e9953`;
the ledger-recorded package digest is
`6ef0c3ba6cc37a6b513209a0d830c541f5aab6f872b3eac7732cb8dc28945e2f`.
`sha256sum -c MANIFEST.sha256` passed for all 18 files in the source package.

Project Control observation `2026-09-01T05:53:20Z` established:

- CellShard commit `6ab8932704ac5988ac64853b3cf43e41e991ee98`;
- clean source-worktree fingerprint
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- Todo UUID `a52537a5-20db-4aeb-a126-dd0128c71fda`, revision `316`, and
  semantic/workflow fingerprint
  `d88ce4e9e66def7ca9c236ed479ef54ef999dbc7979fa0429cbbb5d07819b7c6`;
- zero skew among workflow, semantic-state, status, and export providers; and
- active task `CS-JBC-B02` in `CS-JBC-L-BOOTSTRAP`, with B03 still queued.

A separately timestamped pre-completion observation at
`2026-09-01T05:57:06Z` again reported revision `316`, the same commit, zero
provider skew, and semantic/workflow fingerprint
`17546892fb2e7b3df7de318c8d1cbc42fff767d744530d1cfa766224fcddd90b`.
Its bounded worktree cursor identified only this document and its CSV companion
as source changes, with fingerprint
`d373448f73a092887326d30e953ea987658ee62ca0f0d0ca242922aa684580e2`.
The fingerprint change at a stable revision reflects current workflow liveness
and the newly materialized scoped evidence, not an authority or source-base
change. A contemporaneous workflow sync kept B02 claimed at revision `316`.

The preledger classifies B02 as a settled required mechanism. Its exact
instruction is to inspect identity, domains, partitions, images, extents,
catalogs, snapshots, sources, residency, access adapters, sharded runtime,
distributed placement, CSH5, CSPACK, CPEXEC, CSHARD, tests, and CMake, then
classify each as preserve, generalize, compatibility, or retire.

## Classification meanings

- **Preserve** means the live surface already owns a required responsibility
  and remains an authoritative input or route.
- **Generalize** means retain the frozen surface and add an adjacent atom-aware
  contract; it never means silently widening the existing wire or ABI.
- **Compatibility** means keep behavior while callers migrate through an
  explicit adapter or versioned route. Compatibility is not new ownership.
- **Retire** identifies a role that must leave the JBC steady state. Nothing is
  deleted by B02; retirement requires a later owning task, migrated callers,
  differential fixtures, and an explicit compatibility decision.

## Source-backed map

| Area | Live evidence | Classification | Transition |
| --- | --- | --- | --- |
| Identity | `identity/strong_id.hh`, `identity/digest.hh` | Generalize | Keep the frozen 64-bit IDs and explicit digest vocabulary. Add namespace-qualified atom identities beside them; runtime pointers, device ordinals, paths, placement epochs, and service epochs never become biological identity. |
| Domains and partitions | `domain/descriptor.hh`, `domain/partition.hh` | Generalize | Reuse explicit domain, generation, map, order, contiguous/extent/opaque selection, and validation. Add exact atom coverage and distinct owner/proposal/halo/replica/contribution roles; equal shapes or row offsets never establish equivalence. |
| Images | `artifact/image.hh` | Generalize | Preserve projection keys, target capabilities, alignments, reuse, digests, dependencies, routes, and nonowning views. Adjacent atom contracts add species, ports, planes, evidence, affordances, coverage, and lineage. |
| Extents | `artifact/extent.hh` | Preserve | Storage objects, aligned checksummed extents, and operational source locations remain the storage thin waist. An extent is neither a biological partition nor an execution contribution. |
| Catalogs | `artifact/catalog.hh` | Generalize | Preserve separate immutable-artifact and operational-source catalogs and their cross-validation. Add indexed atom, grammar, basis, superatom, partial, physical-view, action, and lineage catalogs in the atom-store family. |
| Snapshots | `artifact/snapshot.hh` | Generalize | Preserve duplicate rejection, image dependency closure, cycle rejection, source reachability, and generation closure. Generalize roots to atom dictionaries, grammar/bases/partials/actions without adding source paths. |
| Sources | `runtime/source/payload_source.hh`, `runtime/source/local_file_source.hh` | Preserve | Keep the nonowning pointer-plus-ops exact-range boundary and local provider. Later providers extend capability explicitly; discovery, allocation, mmap, HDF5, or networking must not be hidden by the view. |
| Residency | `runtime/residency/host.hh`, `runtime/residency/device.cuh` | Generalize | Preserve move-only ownership, caller allocator, caller stream, digest, device, and reset semantics. Runtime v2 extends them to ordered multi-extent atom planes, readiness, leases, pins, topology, and reconstruction. |
| Access adapters | `access/adapter.cuh`, `access/fallback_adapters.cuh`, `tests/access_adapter_test.cc` | Compatibility | Preserve the archive/pack specialization seam and dense/compressed fallbacks. The closed `cellshard-cpp-access-adapter-refactor` is evidence only and is not reopened, duplicated, or repaired. |
| `sharded<T>` runtime | `runtime/layout/sharded.cuh`, `runtime/host/sharded_host.cuh`, `runtime/device/sharded_device.cuh` | Compatibility | Keep current row-centric stitched metadata, host materialization, and device upload. Never reinterpret `partition_offsets` or `shard_offsets` as biological identity; atom-native runtime v2 replaces its semantic role. |
| Distributed placement | `runtime/distributed/distributed.cuh`, `runtime/device_bindings.cuh` | Compatibility | Round-robin and greedy-by-bytes assignment remain local fallback machinery, not topology-aware or global-cost placement. Runtime v2 adds explicit topology, placement, routing, and transport contracts. |
| CSH5 | `io/csh5/api.cuh`, `src/io/csh5/`, `docs/FORMAT_ROLES.md` | Preserve | CSH5 stays the canonical metadata-rich durable archive and source for generated artifacts. Its direct conversion/fetch-to-GPU hot path is separately marked for retirement once artifact routes have complete fixtures. |
| CSPACK01 | `io/pack/packfile.cuh`, `src/io/pack/packfile.cu`, `docs/SPEC_CSPACK_V1.md` | Preserve | Keep the generated per-shard execution container, magic, and top-level table. It may transport compatible opaque entries but does not become a universal compiler database. |
| CPEXEC01 | `io/pack/execution_payload.cuh`, `src/io/pack/execution_payload.cu`, `tests/execution_payload_cpu_test.cc` | Compatibility | Preserve native legacy read/write and exact stale/corruption rejection. Do not give its conflated counters new identity ownership; retire direct hot use only after versioned consumers migrate. |
| CPEXEC02 | `io/pack/image_envelope.hh`, `src/io/pack/image_envelope.cc`, `tests/image_envelope_test.cc` | Preserve | Keep deterministic portable metadata, explicit image/projection/domain/extent identity, and metadata-only inspection inside unchanged CSPACK01. Add atom-store import/export rather than widening the envelope by implication. |
| CSHARD01 | `io/cshard/spec.hh`, `src/io/cshard/cshard.cc`, `tests/cshard_v1_test.cc` | Compatibility | Keep experimental inspect/validate/convert/read support. CSHARD is neither the current canonical archive nor the JBC atom store unless a later explicit format decision promotes it. |
| Sparse compute/layout policy | `include/CellShard/formats/`, `src/bucket/`, `src/convert/`, `src/repack/` | Retire | These legacy compute-policy surfaces migrate to Cellerator ownership. CellShard may store and move opaque payloads through narrow adapters; deletion waits for all callers and package consumers to migrate. |
| Direct CSH5/CPEXEC hot execution | `src/io/csh5/execution_runtime_part.hh`, `src/io/csh5/runtime/fetch.cc`, legacy execution payload staging | Retire | Steady state consumes published atom/artifact extents through explicit acquisition and residency. Retain canonical/reference fallbacks until differential validation proves replacement. |
| Tests | foundation, envelope, source, residency, adapter, pipeline, and package-consumer tests under `tests/` | Generalize | Preserve independent positive and malformed/stale/duplicate/cycle/capacity checks. Owning later tasks add deterministic, exact-oracle, randomized, null, complete-cost, and no-promotion fixtures under `tests/jbc/`. |
| CMake and umbrella exports | `CMakeLists.txt`, `CellShard/CellShard.hh`, `tests/package_consumer/` | Compatibility | Observe existing targets and exports but do not edit them in bootstrap. Only the designated integration task registers JBC targets, umbrella headers, installed exports, and package consumers. |

## Additive destination reservations

The map reserves names; it does not create registries or public exports. Later
owning tasks may populate:

```text
include/CellShard/compiler/atom/
include/CellShard/compiler/evidence/
include/CellShard/compiler/certification/
include/CellShard/compiler/discovery/
include/CellShard/compiler/composition/
include/CellShard/compiler/grammar/
include/CellShard/compiler/basis/
include/CellShard/compiler/partial/
include/CellShard/compiler/graph/
include/CellShard/compiler/schedule/
include/CellShard/artifact/atom_store/
include/CellShard/runtime/v2/
tests/jbc/
bench/jbc/
docs/JBC/evidence/
```

These remain subordinate to the shared Cellerator-owned mathematical and exact
coverage thin waist. CellShard owns recurrence evidence, exact global
certification, composition/basis/superatom/partial structure, global physical
realization, persistence, topology, placement, residency, and transport. It
does not absorb Cellerator numerical semantics, BioPrep policy, model
objectives, optimizer policy, pseudotime inference, or Baseplane science.

## Data flow and complexity

The required direction is additive and versioned:

```text
biological observations and typed relations
  -> bounded proposal evidence
  -> independent exact certification
  -> atom/composition/basis/superatom/partial catalogs
  -> immutable atom-store images and extents
  -> source acquisition
  -> host/device residency and transport
  -> Cellerator-owned local numerical execution
```

B02 itself is a cold, fixed-source classification pass: `O(F + R)` inspection
time for `F` scoped files and `R` recorded map rows, with `O(R)` document
storage and no runtime allocation or execution effect. The live catalog uses
linear lookup, and its duplicate/dependency validation can be quadratic in
catalog cardinality; that is acceptable for the bounded CS-FOUND baseline but
must be generalized to explicit indexed or sorted structures before large atom
catalogs enter hot acquisition. Explicit partition selection validates in
`O(K)` time and stores `O(K)` extents. Source reads, host allocation, H2D copy,
and legacy global sorts remain visible costs.

Future discovery tasks must bound candidates and peak memory using streaming,
top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or
caller-owned marks. Unrestricted all-pairs and subgraph enumeration are invalid
except as exact small-instance oracles. No steady-state runtime path may perform
discovery, catalog parsing, hidden allocation, global sorting, or topology
search.

## Failure and migration rules

- Malformed or stale identity/generation data is rejected before execution.
- Weak or unstable biological structure yields a valid no-candidate or
  no-promotion outcome; it is not promoted by naming alone.
- Capacity overflow, duplicate identity, incomplete exact coverage, dependency
  cycles, missing extents, and candidate explosion produce explicit diagnostics.
- Approximate discovery proposes only; independent exact certification owns
  executable coverage.
- Proposal overlap, physical-representation overlap, and execution-contribution
  overlap remain distinct, and every contribution has one exact owner unless a
  versioned partial algebra proves reconstruction.
- CSG1, CPE2, CPK1, CSH5, CSPACK01, CPEXEC01, CPEXEC02, and CSHARD01 bytes are
  not silently reinterpreted or mutated.

## Validation entry points

No build or runtime test is required for this documentation-and-manifest-only
classification. The existing independent references to retain are the
foundation identity/domain/artifact/snapshot, payload-source, host/device
residency, legacy-adapter, image-envelope, opaque-pipeline, CPEXEC01, CSHARD,
access-adapter, and package-consumer tests registered in `CMakeLists.txt`.

The exact later build entry points are:

```bash
# Standalone nested CellShard
cmake -S /home/tumlinson/Cellerator/components/CellShard \
  -B /home/tumlinson/Cellerator/components/CellShard/build \
  -DCELLSHARD_BUILD_TESTS=ON
cmake --build /home/tumlinson/Cellerator/components/CellShard/build \
  -j "$(nproc)"

# Embedded through Cellerator
cmake -S /home/tumlinson/Cellerator -B /home/tumlinson/Cellerator/build
cmake --build /home/tumlinson/Cellerator/build -j "$(nproc)"
```

The next bootstrap task may use this map to freeze the successor charter. B02
does not claim or implement that charter.
