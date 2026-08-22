# Architecture Decisions

Locked 2026-08-22. Source of truth for section 4 of `CLAUDE_PROMPT.md`;
do not re-open these questions without an explicit architect decision.

| # | Question | Decision |
|---|----------|----------|
| 1 | Index structure | **HNSW** — layered graph, greedy beam search, heuristic neighbour selection |
| 2 | Thread safety | **Not thread-safe**; documented single-thread contract, synchronisation is the caller's responsibility |
| 3 | Persistence format | **Custom binary format**: magic header + `schema_version`, serialised numpy arrays (vectors, adjacency lists, tower/layer assignments) |
| 4 | Deletion | **Soft-delete tombstones**; `rebuild()` provided as an explicit compaction maintenance operation, never implicit |
| 5 | Vector dimensionality | **Fixed at creation** via `IndexConfig.dim`; mismatches raise `DimensionMismatchError` |
| 6 | Distance metric | **One index = one metric**, `IndexConfig.metric` (enum) |

## Amendments

- `search(query: np.ndarray, k: int) -> list[SearchResult]` — the `metric`
  parameter was dropped entirely: metric selection lives in the index config
  only (architect amendment, 2026-08-22). Binding for Phase 3.
- Query dimensionality mismatch raises `DimensionMismatchError`.
- Soft-delete compaction is never tied to individual deletes; only
  `rebuild()` reconstructs internal structures.

## Phase 2 clarifications (locked 2026-08-22)

1. **Neighbour selection**: full SELECT-NEIGHBORS-HEURISTIC (Malkov &
   Yashunin, Algorithm 4) with diversity condition, implemented as a pure
   function; `keepPrunedConnections=True`, `extendCandidates` exposed as
   `IndexConfig.extend_candidates` (default `False`, following hnswlib).
2. **`rebuild()`**: mutates in place (`-> None`), replays surviving records
   in chronological first-insert order (not id order) into a fresh graph,
   then swaps state. Result ≡ fresh build of survivors; rebuild is
   idempotent and bit-for-bit reproducible via `IndexConfig.seed`.
3. **Entry point on delete**: never eagerly re-selected. Tombstones remain
   routable waypoints excluded from results; the entry pointer changes only
   when an insert reaches a strictly higher level or after `rebuild()`.
   Deleted ids are immediately reusable for new inserts.

## Determinism contract (Phase 2, amended Phase 3)

Identical `IndexConfig.seed` + identical insert sequence ⇒ identical
topology (levels, links, entry point) and identical search outputs. The
level RNG draws exactly once per insert; beam-search heap ties break by
internal node index. **Output ordering (Phase 3): results ascend by
distance; equal distances break by ascending id (lexicographic) — the
truncation cut uses the same comparator.**

## Phase 3 verification notes (2026-08-22)

- Sorting-contract tests were red-first (tie-break by id failed against
  the previous insertion-order behaviour); zero-vector end-to-end and
  payload round-trip tests are regression-grade (green on arrival).
- The green-batch Phase 2 groups (tombstones / entry point / neighbour
  selection) were validated by mutation testing: killing diversity
  selection, adding eager entry re-pick, and removing the tombstone
  filter each turned the corresponding tests red (2/3/1+ failures), with
  full-suite green restored after each revert.
- Large-scale recall gate (`pytest -m slow`, excluded from default runs):
  n=5000 → recall@10 = 1.0000 for both L2 (build 35.5s, 1.47 ms/query)
  and cosine (78.1s, 2.35 ms/query). Default parameters confirmed;
  cosine build cost (per-call norm recomputation) noted as the first
  optimisation candidate.

## Phase 4 clarifications (locked 2026-08-22)

1. **Levels on load()**: restored verbatim from the saved array — no RNG
   replay, no recomputation. Bit-for-bit round-trips never depend on numpy
   RNG reproducibility across versions. Consequence: after `load()` the
   level RNG restarts from `IndexConfig.seed`; this affects *only future*
   inserts.
2. **Entry point**: saved and restored as-is, including pointing at a
   tombstoned node at save time (`-1` encodes "none"). Never re-selected
   at load.
3. **`schema_version` mismatch**: strict reject via
   `SchemaVersionError(found, supported)`; there is deliberately **no
   migration path** — incompatible formats require an application-level
   re-export. This is a decision, not an open question.
4. **Corruption contract**: truncation raises `PersistenceError("truncated
   file: needed N bytes at <stage>, have M")` with the parse stage named;
   a foreign magic raises `PersistenceError("not a vectordb file: expected
   magic b'VDB1', found …")`. No bare `OSError`/`EOFError`/`ValueError`
   escapes the persistence layer.

### Binary format (`vectordb.persistence.binary`, schema_version = 1)

```
magic "VDB1" (4B) | schema_version (u32 LE) | json_len (u64 LE)
| metadata JSON (UTF-8, sorted keys): config, node_ids, payloads,
  insertion_order, entry_point (-1 = none), max_level, size
| vectors  n*dim × <f8   row-major, insertion order
| alive    n     × u8
| levels   n     × <i8
| adjacency_counts  L × <i8   per (node, layer) list, node-major
| adjacency_values  V × <i8   concatenated neighbour lists
```

Two distinct determinism guarantees, kept separate on purpose:

- **Save determinism** (`test_saved_bytes_are_deterministic`): saving the
  same built index twice produces byte-identical files on disk.
- **Round-trip fidelity** (`test_topology_and_search_are_bit_identical`):
  after `save` + `load`, topology and `search()` outputs are bit-identical
  to the original index's.

Known asymmetry (documented in `load_index`): RNG state is not persisted,
so *post-load inserts* diverge in level assignment from an equivalent
unsaved session even at equal seed. Searches over restored data are
unaffected.

Public API: `save_index(index, path)` / `load_index(path) -> HNSWIndex`.

## Phase 5 clarifications (locked 2026-08-22)

1. **Dependencies**: Typer + Rich live in the optional `[cli]` extra
   (`pip install ".[cli]"`); the core stays numpy-only. Console script:
   `vectordb` → `vectordb.cli.app:main`.
2. **JSONL input** for `build-index` / `insert`: one
   `{"id": str, "vector": [floats], "payload": {...}|null}` per line;
   dimensionality is pinned by the first record, violations are reported
   with the offending line number. All errors are friendly one-liners on
   stderr with exit code 1 — never tracebacks (`PersistenceError` /
   `SchemaVersionError` included).
3. **Benchmark soft limit**: `--n > 20_000` prints a stderr warning
   ("may take a while") and proceeds — a soft threshold, not a cap
   (`_BENCHMARK_N_SOFT_LIMIT`).
4. **Progress reporting**: the library stays Rich-free and exposes
   `on_progress(stage, fraction)` with stages `"build"` (single 1.0 emit),
   `"ground_truth"` and `"query"` (monotonic, ending at exactly 1.0). The
   CLI attaches a Rich progress bar only when stdout is a tty and `--json`
   is off.
5. **Benchmark determinism scope**: same seed ⇒ same dataset, queries and
   `recall_at_k`; wall-clock timings are excluded from the guarantee.
6. **Ground truth**: full linear scan via `vectordb.distances`. With
   record ids supplied (`exact_top_k_ids(..., ids=...)`), ties break by
   id lexicographically — exact parity with the search contract,
   truncation cut included, so tied distances can never be scored as
   recall misses. Without ids, ties fall back to stable row order
   (pure utility mode).

## Phase 6 close-out (2026-08-22)

Coverage closure per amended gap policy: reachable gaps got behavioural
tests; two defensive branches carry `# pragma: no cover` with inline
justification (HNSW's 1-D vector guard sits behind Pydantic's
`list[float]`; `main()` is pure Typer delegation). The new
`extend_candidates` test is mutation-verified — disabling the
unseen-neighbour append drops its clustered fixture from exact to 0.91
recall. Full numbers and the assumptions/limitations/optimisations
ledger live in [AUDIT.md](AUDIT.md).

## HNSW parameters (defaults)

| Parameter | Default | Notes |
|---|---|---|
| `m` | 16 | Max links per node per layer (layer 0: `2*m`) |
| `ef_construction` | 200 | Candidate list size during insert |
| `ef_search` | 50 | Candidate list size during search |
| level multiplier | `1 / ln(m)` | Exponential layer assignment |
| `extend_candidates` | `False` | Algorithm 4 candidate-pool extension |
| `seed` | `0` | Level-RNG seed; deterministic builds by default |

All tunable through `IndexConfig`; the chosen values do not block Phase 2.

## Process policies

- Every phase starts with a contract summary (signatures, invariants) for
  the architect before implementation.
- From Phase 4 onward, testing claims require real `pytest -v` output,
  not prose.
- Score semantics: `SearchResult.score` is a *distance* (lower = closer);
  cosine is reported as `1 - cosine_similarity`.
