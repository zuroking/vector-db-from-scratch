# Final Audit — vectordb from scratch

Phase 6 close-out, 2026-08-22. Companion to [ARCHITECTURE.md](ARCHITECTURE.md):
decisions are recorded there; this file records what was *accepted*,
what the system *cannot do*, and where it should go *next*.

## Verification summary

- Full suite: **144 passed**, coverage **100%** of 658 statements across
  all 13 modules (`pytest --cov=vectordb --cov-report=term-missing`).
- Strict mypy: clean over 22 source files.
- Slow quality gates (`pytest -m slow`): recall@10 = 1.0 at n=5000 for
  both L2 and cosine.
- Coverage closure followed the amended gap policy: reachable gaps got
  behavioural tests; the two defensive/unreachable branches carry
  `# pragma: no cover` with inline justification (HNSW 1-D vector guard
  behind Pydantic's `list[float]`; console-script delegation to Typer).
- The new `extend_candidates` test is mutation-verified: disabling the
  unseen-neighbour append in Algorithm 4's extension turns it red
  (recall drops 1.00 → 0.91 on the clustered fixture); the earlier
  line-dataset attempt did *not* discriminate and was replaced.

## Accepted architectural assumptions

All cross-reference [ARCHITECTURE.md](ARCHITECTURE.md); only their
consequences are restated here.

| Assumption | Consequence accepted |
|---|---|
| HNSW as sole index structure (decision #1) | No IVF/LSH fallback paths; single-graph memory profile |
| Not thread-safe (decision #2) | Synchronisation is entirely the caller's problem; no internal locks or contention cost |
| Custom binary format, schema_version = 1 (decision #3) | No third-party format ecosystem; strict version reject with **no migration path** |
| Soft-delete tombstones + explicit `rebuild()` only (decision #4) | Deleted nodes stay routable waypoints: search touches them, memory is reclaimed only by `rebuild()` |
| Dimensionality frozen at creation (decision #5) | No re-dimensioning without a fresh index + re-insert |
| One index = one metric (decision #6) | Metric dropped from `search()` entirely; mixed-metric workloads need separate indexes |
| Determinism via seeded level RNG | Bit-for-bit builds depend on numpy RNG behaviour per version; persistence avoids replaying RNG for restored levels (levels saved verbatim) |
| Core stays numpy-only | CLI/benchmark extras isolated behind `[cli]`; library progress is a plain callback, Rich attaches above it |

## Known limitations

1. **Cosine build cost.** Cosine distance recomputes norms per call
   during construction (~2× L2 build time at n=5000: 78 s vs 36 s).
   Correctness unaffected; flagged since Phase 3 as the first
   optimisation candidate.
2. **Post-load inserts diverge from an unsaved session.** RNG state is
   not persisted: after `load()`, the level RNG restarts from
   `IndexConfig.seed`, so *new* inserts may land on different layers
   than they would have without the round-trip. Searches over restored
   data are bit-identical (documented asymmetry, Phase 4).
3. **Soft deletes accumulate until `rebuild()`.** Tombstones remain in
   adjacency lists and vectors arrays. Long delete-heavy sessions grow
   memory and keep dead nodes on search paths; compaction is never
   implicit (locked decision #4).
4. **Benchmark ground truth is O(n · queries).** Brute-force scoring
   dominates wall-clock at large `--n`; the 20 000 soft limit warns but
   does not cap.
5. **Recall validation used synthetic uniform-random vectors only.**
   The Phase 2/3 gates (n=5000, recall@10 gate) draw i.i.d. Gaussians;
   real embeddings (e.g. sentence-transformers output) are highly
   clustered and low-intrinsic-dimension, which changes greedy-descent
   behaviour in both directions. Recall on such data is **not**
   validated by this test coverage — the clustered-data path exercised
   via `extend_candidates` tests is the only structured-data evidence.

## Future optimisation vectors

Each entry names the trigger that would justify starting it.

| Vector | Trigger |
|---|---|
| Norm caching / squared-distance shortcuts for cosine | Cosine builds become the dominant cost in real workflows (already ~2× at n=5000) |
| Quantization (scalar → product quantization) | Datasets exceed RAM budgets at full `<f8` precision; PQ gives ~32× compression at bounded recall cost |
| Disk-based ANN (e.g. mmap'd graph sections, DiskANN-style layout) | Working sets larger than RAM; persistence format already separates vectors from topology, easing the split |
| Parallel construction / batched inserts | Bulk-load latency becomes user-visible; requires care: determinism contract pins single-threaded insert order |
| **Thread safety via RWLock** (variant rejected at project start, decision #2) | Concurrent read-heavy serving workload with parallel `search()` while `insert`/`delete` are rare or quiesced — readers-share/writer-exclusive restores concurrency without touching single-thread semantics |

## Deliberately out of scope

Updates to stored vectors (delete + re-insert only), multi-index
transactions, replication/sharding, filtered/hybrid search. None are
regressions; all would be new architecture decisions requiring an
explicit architect review before implementation.
