# CellShard biology-native compiler successor charter v1

Status: **frozen successor charter**

Owner: `CS-JBC-B03`, lane `CS-JBC-L-BOOTSTRAP`

Scope: CellShard program ownership and convergence rules; no wire or runtime
behavior change

## Authority and precedence

This charter defines what new CellShard work must converge toward. It
supersedes the storage/staging-only and mandatory `CSH5 -> CSPACK -> GPU`
posture as a future architectural rule. Existing CS-FOUND documents, current
format specifications, tests, and implementations remain authoritative for
what exists and what must stay compatible today. They are not rewritten as if
the successor were already implemented.

The source baseline is the `CS-JBC-B02` transition map and its machine-readable
manifest. The exact preledger input is the `CS-JBC-B03` object in
`proposed_todos.json`; its newline-terminated compact sorted JSON SHA-256 is
`383db08c3bf66699a264dcc3c9fe52ba996e0816a364f3d75f3ad67072d997de`.
The ledger-recorded package digest is
`6ef0c3ba6cc37a6b513209a0d830c541f5aab6f872b3eac7732cb8dc28945e2f`.

Project Control observation `2026-09-01T06:00:29Z` established CellShard
commit `96a691e4a271fabd738ff5819eef6349ac3621a0`, Todo UUID
`a52537a5-20db-4aeb-a126-dd0128c71fda`, revision `318`, and semantic/workflow
fingerprint
`b810c2adff626604392a7abf6c198b88cf4d67a95c7f1c59e7b9fc76af7530c4`.
All providers had zero skew, the source worktree was clean, and B03 was the
only active task in the serial bootstrap lane.

A separately timestamped pre-completion observation at
`2026-09-01T06:02:53Z` again reported revision `318`, the same source commit,
zero provider skew, and semantic/workflow fingerprint
`60e48c2d5038fc669a3015f656c14098bbd12c8d408e0a326042bdaff486ad24`.
Its bounded source cursor identified only this charter, `AGENTS.md`, and
`docs/FORMAT_ROLES.md` as task-owned changes, with worktree fingerprint
`93fbaaad1b8218a6fbff48763fbfb8df0bebf6acc08e81b699b29c6a5d6a8e97`.
A contemporaneous opaque-handle sync kept B03 claimed at revision `318`. The
fingerprint change at stable authority revision records liveness and scoped
evidence materialization, not an authority-base change; no recovery was
performed.

## Mission

CellShard compiles recurrent biological organization into exact, reusable,
portable, and physically realizable artifacts. Its value is not the existence
of shards or containers. Its value is reducing total execution, movement,
residency, reconstruction, and communication cost by discovering and
certifying repeated structure across biological domains, operations, states,
modalities, trajectories, motifs, and partitions.

Compilation must preserve exact logical meaning. Approximate or statistical
evidence may propose reusable structure; it cannot authorize execution.
Independent exact certification owns coverage, contribution, and
reconstruction correctness.

## Ownership

CellShard owns:

- bounded recurrence discovery evidence and proposal records;
- exact global coverage and reconstruction certification;
- atom composition, grammars, bases, superatoms, partial results, and lineage;
- global physical-view realization and artifact publication;
- durable catalogs, snapshots, images, extents, sources, and format adapters;
- topology observation, placement, routing, acquisition, caching, residency,
  transport, leases, pins, readiness, and reconstruction;
- explicit complete-cost evidence for compiler and runtime promotion; and
- compatibility routes for frozen CellShard formats and runtime surfaces.

CellShard does not own:

- Cellerator mathematical operators, local kernel/layout selection, numerical
  accumulation policy, or execution semantics;
- BioPrep preprocessing, normalization, QC, marker, or workflow policy;
- model objectives, training, optimizer policy, Torch ownership, or
  framework-native execution;
- pseudotime, lineage inference, causal biological interpretation, or
  caller-specific science;
- Baseplane sequence science or sequence-specialized primitives; or
- a claim that recurrence exists merely because shapes, offsets, clusters, or
  labels agree.

Callers supply typed biological domains, exact orders, relations, observations,
values, constraints, capacities, and scientific strata. CellShard may compile
those facts but does not invent their scientific meaning.

## Successor contracts

The successor operates over a versioned atom-level thin waist that later
owning tasks will define. This charter freezes requirements, not field layouts,
species registries, or public headers.

Every executable unit must eventually expose:

- explicit, recoverable biological identity and exact logical coverage;
- typed input/output ports, domains, orders, and relation identity;
- separate immutable structure, mutable value, mutable state, materialization,
  replica, residency, and preference/cost generations;
- evidence provenance and an independently certified coverage result;
- physical affordances, resource requirements, dependencies, and lineage;
- one exact contribution owner, unless a versioned partial-result algebra
  proves reconstruction; and
- explicit failure and fallback semantics.

Proposal overlap, physical-representation overlap, and execution-contribution
overlap are different relations and must never be conflated. Equal extent,
shape, ordinal position, path, pointer, device ordinal, replica, service epoch,
or placement epoch never establishes biological identity.

## Compiler lifecycle

The cold compiler lifecycle is:

```text
typed biological observations and relations
  -> bounded proposal evidence
  -> independent exact certification
  -> atom/composition/grammar/basis/partial catalogs
  -> global physical-view and placement planning
  -> immutable images, extents, routes, and snapshots
  -> publication through versioned formats and providers
```

Cold builders may own declared temporary storage. Candidate generation must
bound time and peak memory with streaming, bounded top-L structures, sketches,
sparse maps, count/scan/fill, radix/sort, or caller-owned marks. Unrestricted
all-pairs or subgraph enumeration is allowed only as a declared exact
small-instance oracle.

The steady-state runtime lifecycle is:

```text
published snapshot
  -> exact source acquisition
  -> cache/residency/readiness binding
  -> topology-aware transport or local staging
  -> nonowning execution views
  -> Cellerator-owned numerical execution
```

Steady state performs no discovery, catalog parsing, hidden allocation, global
sorting, topology search, implicit device-wide synchronization, or silent
canonicalization. Public execution views remain explicit pointer-plus-count
records with visible capacity, ownership, stream, residency, and generation.

## Performance and promotion

Performance governs promotion. A compiler mechanism or physical realization
must reduce measured total cost through reuse, bytes moved, launches,
synchronization, residency pressure, reconstruction, communication, or
amortized preparation. Conceptual regularity alone is insufficient.

Every promoted mechanism records the applicable cold build time, peak memory,
candidate count, exact certification cost, artifact expansion, acquisition and
transfer cost, runtime benefit, expected reuse, break-even point, topology,
hardware/toolchain identity, correctness result, and strongest relevant
fallback. A measured no-candidate or no-promotion result is valid. Microkernel
speed alone cannot promote a compiler mechanism.

Malformed or stale identity/generation data, duplicate identity, incomplete
coverage, capacity overflow, dependency cycles, missing extents, and candidate
explosion produce deterministic diagnostics before execution. Weak or unstable
biological evidence produces an explicit no-promotion result.

## Compatibility and migration

- CSH5 remains the current canonical metadata-rich archive.
- CSPACK01 remains a generated execution container and may carry CPEXEC01 or
  CPEXEC02 through their existing routes; it is not the universal atom store.
- CPEXEC01 remains native legacy compatibility; CPEXEC02 remains the portable
  deterministic image envelope.
- CSHARD01 remains experimental standby and is not promoted by this charter.
- `sharded<T>`, local placement helpers, access adapters, and direct
  CSH5/CPEXEC execution remain compatibility/reference machinery while callers
  migrate.
- Existing CSG1, CPE2, CPK1, CSH5, CSPACK01, CPEXEC01, CPEXEC02, and CSHARD01
  bytes are never silently reinterpreted. A changed contract requires an
  adjacent version, explicit adapter, and differential fixtures.
- Cellerator-owned sparse compute/layout policy leaves CellShard only after
  callers and package consumers have migrated; this charter deletes nothing.
- The closed `cellshard-cpp-access-adapter-refactor` remains historical
  evidence and is not reopened or duplicated.

The preferred successor route is selected from published artifacts and sources
by complete measured cost. No archive or pack family is mandated globally.
Canonical CSH5 and legacy CSPACK routes remain independently testable
fallbacks until versioned replacements pass differential validation.

## Source and integration posture

The additive destinations reserved by B02 remain the convergence paths:

```text
include/CellShard/compiler/{atom,evidence,certification,discovery,composition}/
include/CellShard/compiler/{grammar,basis,partial,graph,schedule}/
include/CellShard/artifact/atom_store/
include/CellShard/runtime/v2/
tests/jbc/
bench/jbc/
docs/JBC/evidence/
```

Only designated integration tasks edit central registries, umbrella headers,
root CMake registration, installed exports, or package consumers. Provider and
consumer tasks publish source-linked fragments and fixtures without competing
central-file edits.

## Validation posture

Every implemented contract requires deterministic positive tests, malformed
and stale identity tests, duplicate/cycle/capacity tests, and randomized
properties appropriate to the mechanism. Approximate proposals are compared
with an exact rescan; certified output receives an independent exact validator.
At least one canonical/reference fallback remains differential-tested.

This B03 change is charter-only and alters no source, target, package surface,
or command. No build is required. Exact later build entry points are:

```bash
cmake -S /home/tumlinson/Cellerator/components/CellShard \
  -B /home/tumlinson/Cellerator/components/CellShard/build \
  -DCELLSHARD_BUILD_TESTS=ON
cmake --build /home/tumlinson/Cellerator/components/CellShard/build \
  -j "$(nproc)"

cmake -S /home/tumlinson/Cellerator -B /home/tumlinson/Cellerator/build
cmake --build /home/tumlinson/Cellerator/build -j "$(nproc)"
```

`CS-JBC-B04` owns the separate frozen CS-FOUND compatibility baseline. Nothing
in this charter claims or completes B04.
