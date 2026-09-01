# CSATOM v1 Atom-Store Specification

## Frozen name and scope

The collision-free format-family name is **CSATOM v1**, the file suffix is
`.csatom`, and the eight-byte file magic is `CSATOM01`. These identifiers do
not alias CSH5, CSPACK, CPE2, CPK1, or any execution-envelope family.

CSATOM is CellShard's immutable, atom-native persistence and lowering artifact.
It stores certified biological atom identity, exact coverage, dependencies,
portable payload sections, and source-linked lowering metadata. It is not the
canonical biological dataset, a mutable value store, a runtime-residency
record, an execution planner, or a replacement for CSPACK caches.

## Ownership and lifecycle

- CellShard owns the container, integrity, storage extents, and delivery.
- Proposal providers may contribute evidence but may not set certification.
- Exact certification is independently recorded before an atom is promotable.
- Cellerator owns numerical operation and physical projection semantics.
- Structure epoch, value generation, evidence generation, and cost freshness
  remain distinct identities.
- Published bytes are immutable; every semantic change creates a new atom-store
  identity and content digest.

Semantic identity names the biological meaning independent of bytes. Content
identity is the tagged SHA-256 digest of immutable bytes. Action identity names
the source-linked lowering operation. Materialization identity names one built
artifact under that action. Replica identity names one delivered copy without
changing the other four identities. These five types are not interchangeable.

Each immutable root-generation manifest records the store identity, monotonic
generation, structure epoch, root digest, prior root digest, and exact atom,
dependency, materialization, and replica counts. Generation one has no parent;
later generations must link a distinct strong parent digest.

## Required top-level model

Every v1 image has a fixed little-endian header followed by a bounded section
directory and aligned, non-overlapping sections. The header identifies:

1. schema, endian marker, header and total byte counts;
2. atom-store, catalog, structure, and certification identities;
3. section count and directory bounds;
4. a SHA-256 content digest, tagged by algorithm, covering the complete image
   with the digest field zeroed. Legacy FNV checksums are not valid CSATOM
   content identities.

Sections carry typed records for atoms, domains/orders, exact coverage,
dependencies, payload descriptors and bytes, lowering artifacts, and
provenance. Unknown required section types are rejected. Optional types may be
skipped only when their directory flags explicitly permit it.

The v1 large-arena header is 256 bytes and its directory uses 96-byte entries.
Both the directory and every section have at least 64-byte power-of-two
alignment. Directory entries carry a section kind, exactly one of required or
optional, 64-bit extent and record geometry, and an algorithm-tagged section
digest. Record geometry must account for the entire section when nonzero.

Atom payloads may also be fetched independently through a 128-byte atom-frame
header with magic `CSATMFR1`. A frame binds one semantic atom and one
materialization to an exact content digest, logical and encoded byte counts,
codec, alignment, and payload offset. Raw frames require identical logical and
encoded sizes; the frame bounds are validated before its payload is exposed.

An atom may span multiple ordered frames, and each frame may span multiple
storage extents. Frame-map records bind frame ordinal and logical range to an
extent-slice span. Extent slices bind atom, frame, storage object, extent, and
both extent-local and frame-local ranges. Serialized slices must exactly cover
the frame from byte zero without gaps or overlaps.

Encoded-replica descriptors name a delivered replica separately from its atom
and materialization. They bind both decoded and encoded SHA-256 identities,
byte counts, encoding, storage object, object range, and extent-slice span.
Identity encoding requires equal sizes and equal digest bytes; compressed or
provider-defined encodings retain independent encoded and decoded identities.

The positive action cache contains certified successes only. Its exact key is
the source-linked action identity, source content digest, and structure epoch;
a hit returns the immutable materialization, output digest, and evidence
generation. A conflicting successful result for an existing exact key is
rejected rather than silently replaced.

The separate negative action cache accepts only durable capability,
dependency-closure, or exact-certification failures. Each negative entry is
bound to the exact source digest and structure epoch and has an explicit
evidence-generation validity interval; transient delivery or resource failures
are not eligible.

All counts and byte offsets are unsigned 64-bit values. Stable records are
pointer-free and trivially copyable. Runtime pointers, paths, GPU ordinals,
streams, topology routes, and mutable source locations are forbidden in the
portable image.

## Validation contract

A reader must reject an image before exposure when any identity is zero, the
magic/schema/endian marker differs, arithmetic overflows, a section is
misaligned/out of bounds/overlapping, a required section is absent or
duplicated, a record references an unknown identity, exact coverage is
incomplete, dependency closure is invalid, or the image checksum differs.

No format-valid image is thereby executable. Runtime promotion additionally
requires independent exact certification, compatible lowering, generation
freshness, and the owning execution planner's capability checks.

## Compatibility

V1 bytes are never silently reinterpreted. Additive optional sections require
explicit optional flags. Any incompatible header, identity, directory, record,
checksum, or coverage change requires a new magic/schema and an explicit
conversion route. CSATOM may be transported inside a CSPACK extent, but its
inner identity and validation remain CSATOM-owned and distinct.
