# vectordb — a vector database built from scratch

A self-contained vector database implementing **HNSW** (Hierarchical
Navigable Small World graphs) on pure `numpy` — no FAISS, Chroma,
hnswlib, Annoy, ScaNN, or any other ANN library. Every part of the
retrieval path — distance metrics, graph construction, search, disk
persistence — is implemented and tested from first principles.

This is a portfolio project. The priority is algorithmic depth and
engineering rigor over production completeness.

## Results

Recall gate, measured on synthetic random vectors, `n=5000`, `k=10`,
against a brute-force ground truth (`pytest -m slow`):

| Metric | Recall@10 | Build time |
|---|---|---|
| L2 | 1.0000 | ~36 s |
| Cosine | 1.0000 | ~78 s |

Cosine's build cost is ~2x L2 at this scale — a known, documented tradeoff
from recomputing vector norms per distance call (see AUDIT.md).

- **Test suite:** 144 tests, 100% statement coverage (658/658 statements
  across 13 modules), strict mypy clean.
- **Slow gate:** dedicated recall benchmark on n=5000 for both metrics,
  isolated behind `pytest -m slow` (~2 min), run before every release.
- **Query latency:** sub-millisecond to low-millisecond per query at
  small-to-medium scale (e.g. ~1 ms mean at n=2000, dim=16 via the CLI
  `benchmark` command) — run `vectordb benchmark` on your own data/scale
  for numbers that reflect your workload; no single figure generalizes
  across dimensions and dataset sizes.

Full breakdown, methodology, and known limitations of these numbers —
including where they *don't* generalize — are in [AUDIT.md](AUDIT.md).

## Why this project is different

Most "vector DB from scratch" projects stop at "it returns nearest
neighbors." This one was built under a strict engineering process,
enforced across all 6 development phases:

- **Decisions locked before code.** Every architectural choice — index
  type, thread-safety model, delete strategy, persistence format,
  dimension handling, metric binding — was decided and written down
  *before* any implementation started. See [ARCHITECTURE.md](ARCHITECTURE.md).
- **Red-first TDD, verified not assumed.** Tests were written to fail
  first, against real stub behavior — not written to match code that
  already existed.
- **Mutation testing on the algorithmic core.** The diversity-heuristic
  neighbor selection, entry-point handling on delete, and tombstone
  filtering were each validated by deliberately reintroducing the bug
  they guard against and confirming the test suite catches it. The
  first version of the `extend_candidates` mutation test *didn't*
  discriminate — recall stayed 1.0 even with the mutation on a
  collinear-points fixture — and was rewritten against a clustered
  dataset where the flag's effect is real (recall drops 1.00 → 0.91
  when disabled). See AUDIT.md.
- **Bugs are reported, not hidden.** Real defects were found and fixed
  during development — a multi-layer graph traversal bug in HNSW insert,
  a cursor-aliasing bug in the persistence parser — each documented with
  root cause, not smoothed over as "tests passed."
- **Determinism as a first-class constraint.** Fixed seeds, verbatim
  (non-replayed) level restoration, and byte-identical save output are
  all independently tested, because a vector index that can't reproduce
  its own topology can't be debugged.

## Architecture

- **Index:** HNSW — layered graph, greedy beam search, diversity-heuristic
  neighbor selection (`extendCandidates` / `keepPrunedConnections`),
  soft-delete via tombstones, explicit `rebuild()` for compaction.
- **Distance metrics:** Cosine and L2, vectorized `numpy` implementations,
  bound to the index at creation (one index = one metric).
- **Persistence:** custom versioned binary format (`VDB1` magic header +
  schema version + little-endian array sections), not pickle — verbatim
  round-trip of graph topology, no recomputation on load.
- **Not thread-safe by design** — documented tradeoff, not an oversight.
  See AUDIT.md for the condition under which this would be revisited.

Full decision log, including rejected alternatives and their tradeoffs:
[ARCHITECTURE.md](ARCHITECTURE.md).

## Stack

- Python >= 3.12
- `numpy` — the only numeric dependency of the core
- Pydantic v2 for strict data validation
- Strict mypy, pytest
- CLI (optional extra): Typer + Rich — `pip install ".[cli]"`

## Development

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,cli]"
.venv\Scripts\python.exe -m pytest -v
.venv\Scripts\python.exe -m pytest -m slow -v          # recall gate, ~2 min
.venv\Scripts\python.exe -m mypy src --strict
.venv\Scripts\python.exe -m pytest --cov=vectordb --cov-report=term-missing
```

## CLI

After installing the `[cli]` extra:

```powershell
vectordb build-index --input records.jsonl --output index.vdb --metric l2
vectordb insert --index index.vdb --input more_records.jsonl
vectordb search --index index.vdb --query "0.1,0.2,..." -k 10
vectordb benchmark --n 5000 --dim 64 -k 10 --queries 100
```

Input format is JSONL, one record per line:
`{"id": "...", "vector": [...], "payload": {...} | null}`

## Project status

All 6 planned development phases are complete:

| Phase | Scope |
|---|---|
| 0 | Project skeleton, base models |
| 1 | Distance metrics (cosine, L2) |
| 2 | HNSW core (graph, insert, search, delete, rebuild) |
| 3 | Search & ranking contract, tie-breaking |
| 4 | Persistence (custom binary format) |
| 5 | CLI + benchmarks |
| 6 | 100% coverage, mutation testing, final audit |

See [AUDIT.md](AUDIT.md) for accepted assumptions, known limitations,
and future optimization vectors (quantization, disk-based ANN, parallel
construction, norm caching).
