# CS-JBC-B01 CellShard authority baseline

This record revalidates the nested CellShard repository and its transactional
Todo authority before successor-compiler implementation begins. It is evidence
for `CS-JBC-B01`, not a new ledger or architecture contract. No CellShard
behavior, wire format, Todo identity, historical task, or frozen interface is
changed here.

## Observation A — 2026-09-01T05:47:06Z

Project Control observed workspace `cellshard` with Todo revision `314` and
project UUID `a52537a5-20db-4aeb-a126-dd0128c71fda`. The semantic-state and
workflow providers agreed at revision `314`, with zero provider skew and read
authority fingerprint
`4e23168b2665bba906f1246367e3310552141e759583c8f109713fcfd324503f`.
`CS-JBC-B01` was the sole active JBC claim; `CS-JBC-B02` was blocked directly
on it and no JBC successor task was ready.

The same observation resolved the nested repository at detached commit
`5f6a502b4355732c4ed3cc873a25b8aec66d8338`, with observer working-tree
fingerprint
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
(no source changes) and Project Control Git common identity
`git-common-ea73477a9f286bc8`. The parent Cellerator index recorded exactly the
same gitlink at `components/CellShard` (mode `160000`). `origin/main` also
resolved to the same CellShard commit.

The read-only Todo workflow observation reported the active JBC run
`CS-JBC-RUN-V1`, lane `CS-JBC-L-BOOTSTRAP`, role `coordinator`, and task
`CS-JBC-B01`. The compatibility-era top-level `active_run_id` remained
`CS-POST-REMAP-RUN`; its sole lane was already closed, while the separately
listed `CS-JBC-RUN-V1` was active. Fifteen older sessions were classified as
stale:

```text
0473ff09-3630-496a-8931-9fe6f5370865
079c2f76-893f-4b96-ad9a-64e6655a7ce2
07e5dc96-b69f-4414-8928-8131b73c727e
08ce5a2a-7144-4834-9a4b-e138991d3f9b
1fabe89e-46b9-47ea-8b60-d3b4281fac03
35e0ae1e-c44f-4275-9989-380581759241
386001a1-e8db-4a22-a9c8-55e424faaad5
3b6d2b07-7987-47f7-87ce-7161b02840c4
40235a66-b8cd-480d-9438-1e2b59013086
5fd41251-0633-4cbc-834f-6fecfdc10e97
73ea1498-0315-4a7c-967f-432df7cc9681
a11b0c16-a8a8-49c3-8b08-da5801df663d
ba938929-8676-456c-a868-dec7ebdde9a6
f8a1e12f-bce7-4a56-9f27-4a0096d7d007
fae1bef6-e2f3-4b31-ac90-b635c946e098
```

They are historical recovery attention only: this task did not release,
recover, rewrite, or otherwise mutate them.

## Observation B — 2026-09-01T05:48:47Z

A separately timestamped Project Control and read-only Todo observation again
reported project UUID `a52537a5-20db-4aeb-a126-dd0128c71fda`, revision `314`,
detached CellShard commit
`5f6a502b4355732c4ed3cc873a25b8aec66d8338`, the same clean-source worktree
fingerprint, and zero provider skew. The authority fingerprint was
`407293d6398beacf15c9702e8c3bdacea5e137d3d43edbed5417b3914d085c5d`.
The fingerprint change without a Todo revision or repository change reflects
workflow liveness classification: the delegated dispatch crossed the observer
heartbeat window during evidence collection. The opaque Project Control handle
still validated the claim, and a contemporaneous `coordinate_task(sync)`
returned `status=claimed`, revision `314`, run `CS-JBC-RUN-V1`, lane
`CS-JBC-L-BOOTSTRAP`, and task `CS-JBC-B01`. No recovery action was taken.
The second read therefore added the current delegated session
`0aed495c-a0cb-4ec0-b667-136672a3b506` to the same fifteen stale-session IDs
listed above; it did not change their authority state.

Todo's independent SQLite checks reported schema version `10`, integrity
`ok`, no foreign-key errors, and canonical authority at:

```text
/home/tumlinson/Cellerator/.git/modules/components/CellShard/todo-orchestrator/a52537a5-20db-4aeb-a126-dd0128c71fda/state.sqlite3
```

The four modified generated files visible to raw Git during the claim are the
Todo snapshot and Markdown projections produced by the supported workflow:
`.todo-orchestrator/state.snapshot.json`, `todo-status.md`, `todos.md`, and
`todos/cs-jbc-b01.md`. They are not CellShard source changes. Project Control's
source-worktree view therefore correctly remained clean before this evidence
file was added.

## Frozen compatibility and interface baseline

Project Control reported these CS-FOUND interfaces frozen:

- `CS-FOUND-I1` v1;
- `CS-FOUND-I2A` v1, `CS-FOUND-I2B` v1, and `CS-FOUND-I2C` v1;
- `CS-FOUND-I3A` v1 and `CS-FOUND-I3B` v1;
- `CS-FOUND-I4` v2;
- `CS-FOUND-I5A` v1 and `CS-FOUND-I5B` v1.

The source-backed compatibility boundary remains unchanged:

- `.csh5` remains the canonical durable container and `.cspack` the generated
  execution artifact;
- the top-level `CSPACK01` container remains unchanged and continues to carry
  legacy `CPEXEC01` and portable `CPEXEC02` entries through their existing
  explicit compatibility routes;
- `sharded<MatrixT>`, its row-centric offsets, and local assignment helpers
  remain compatibility machinery rather than biological identity;
- structure, values, runtime state, location, and service epochs remain
  distinct from immutable biological/artifact identity; and
- the closed `cellshard-cpp-access-adapter-refactor` remains closed evidence.
  It was observed but not claimed, reopened, duplicated, repaired, or changed.

## Reproducible validation commands

The baseline is documentation-only, so no build was needed to establish it.
The exact build entry points recorded for later implementation validation are:

```bash
# Standalone CellShard
cmake -S /home/tumlinson/Cellerator/components/CellShard \
  -B /home/tumlinson/Cellerator/components/CellShard/build \
  -DCELLSHARD_BUILD_TESTS=ON
cmake --build /home/tumlinson/Cellerator/components/CellShard/build \
  -j "$(nproc)"

# Embedded through the parent Cellerator checkout
cmake -S /home/tumlinson/Cellerator -B /home/tumlinson/Cellerator/build
cmake --build /home/tumlinson/Cellerator/build -j "$(nproc)"
```

The authority observations themselves are reproducible with Project Control
`project_overview(project="cellshard")` and the following read-only Todo
commands:

```bash
TODO_ORCHESTRATOR_READ_ONLY=1 python \
  /home/tumlinson/.agents/skills/todo-orchestrator/scripts/todo.py \
  semantic workflow \
  --repo-root /home/tumlinson/Cellerator/components/CellShard --json
TODO_ORCHESTRATOR_READ_ONLY=1 python \
  /home/tumlinson/.agents/skills/todo-orchestrator/scripts/todo.py \
  doctor \
  --repo-root /home/tumlinson/Cellerator/components/CellShard --json
```

## Baseline conclusion

The nested commit, parent gitlink, origin reference, Todo UUID, revision, and
frozen compatibility interfaces agree. The only JBC implementation frontier
after acceptance of this baseline is `CS-JBC-B02`; it must be claimed in a
separate workflow step.
