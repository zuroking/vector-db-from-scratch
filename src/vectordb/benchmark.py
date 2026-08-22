"""Recall and latency benchmark against a brute-force baseline (Phase 5).

Library-level module: numpy-only, no CLI/Typer/Rich imports, so the core
stays dependency-pure. Progress is reported through an optional callback
(``on_progress(stage, fraction)``); the CLI attaches a Rich progress bar
on top of it.

Determinism: identical ``(config, n, k, n_queries, seed)`` produce the
identical dataset, queries and ``recall_at_k``. Wall-clock timings are,
by nature, not reproducible and excluded from that guarantee.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel, Field

from vectordb.core.config import DistanceMetric, IndexConfig
from vectordb.core.models import VectorRecord
from vectordb.distances import cosine_distance, l2_distance
from vectordb.index.hnsw import HNSWIndex

ProgressFn = Callable[[str, float], None]
"""Called as ``fn(stage, fraction)``, fraction in [0, 1].

Stages: ``"build"`` (emitted once at completion), ``"ground_truth"`` and
``"query"`` (both monotonic, ending at exactly 1.0).
"""


class BenchmarkReport(BaseModel):
    """Immutable result of one benchmark run."""

    n: int
    dim: int
    k: int
    n_queries: int
    metric: DistanceMetric
    build_seconds: float = Field(ge=0)
    recall_at_k: float = Field(ge=0, le=1)
    mean_query_ms: float = Field(ge=0)
    p99_query_ms: float = Field(ge=0)
    queries_per_second: float = Field(ge=0)


def exact_top_k_ids(
    dataset: np.ndarray,
    query: np.ndarray,
    k: int,
    metric: DistanceMetric,
    *,
    ids: Sequence[str] | None = None,
) -> list[int]:
    """Brute-force ground truth: row indices of the ``k`` closest vectors.

    Tie-break parity with :meth:`HNSWIndex.search` is load-bearing: with
    ``ids`` provided, ordering is distance ascending, then id ascending
    lexicographically -- the locked search contract, truncation cut
    included. Without ``ids``, ties fall back to stable row order (pure
    utility mode); :func:`run_benchmark` always supplies ids.

    Args:
        dataset: 2-D array of shape ``(n, dim)``.
        query: 1-D array of shape ``(dim,)``.
        k: Number of neighbours to return; may exceed ``n``.
        metric: Distance metric selecting the batched distance function.
        ids: Row-aligned unique record ids enabling contract-parity ties.

    Returns:
        Up to ``k`` dataset row indices, nearest first under the comparator.
    """
    dist_fn = cosine_distance if metric is DistanceMetric.COSINE else l2_distance
    data = np.asarray(dataset, dtype=np.float64)
    distances = dist_fn(np.asarray(query, dtype=np.float64), data)
    n = data.shape[0]

    if ids is None:
        order = np.argsort(distances, kind="stable")
        return [int(i) for i in order[:k]]

    # Vectorised (distance, id) sort: rank each row by its id position in
    # lexicographic order, then lexsort with distance as primary key.
    id_rank = np.empty(n, dtype=np.int64)
    id_rank[np.argsort(np.asarray(ids, dtype=str), kind="stable")] = np.arange(n)
    order = np.lexsort((id_rank, distances))
    return [int(i) for i in order[:k]]


def run_benchmark(
    config: IndexConfig,
    n: int,
    k: int = 10,
    n_queries: int = 100,
    seed: int = 0,
    on_progress: ProgressFn | None = None,
) -> BenchmarkReport:
    """Build an HNSW index over synthetic data and score it against brute force.

    Dataset and queries are drawn from one Gaussian stream seeded by
    ``seed``. Recall@k is the mean overlap between the HNSW top-k and the
    exact linear-scan top-k, divided by ``k``.

    Args:
        config: Frozen index configuration (dim/metric/params).
        n: Dataset size; callers wanting speed keep this modest -- ground
            truth is a full linear scan per query.
        k: Neighbours per query.
        n_queries: Number of synthetic queries.
        seed: Seed for both data generation and level assignment.
        on_progress: Optional callback receiving ``(stage, fraction)``,
            see :data:`ProgressFn`.

    Returns:
        Fully populated :class:`BenchmarkReport`.
    """

    def emit(stage: str, fraction: float) -> None:
        if on_progress is not None:
            on_progress(stage, fraction)

    rng = np.random.default_rng(seed)
    dataset = rng.normal(size=(n, config.dim))
    queries = rng.normal(size=(n_queries, config.dim))

    index = HNSWIndex(config)
    started = time.perf_counter()
    for i in range(n):
        index.insert(VectorRecord(id=f"vec{i}", vector=dataset[i].tolist()))
    build_seconds = time.perf_counter() - started
    emit("build", 1.0)

    record_ids = [f"vec{i}" for i in range(n)]
    truth: list[set[str]] = []
    for qi in range(n_queries):
        truth.append(
            {
                record_ids[row]
                for row in exact_top_k_ids(
                    dataset, queries[qi], k, config.metric, ids=record_ids
                )
            }
        )
        emit("ground_truth", (qi + 1) / n_queries)

    hits = 0
    latencies_ms: list[float] = []
    for qi in range(n_queries):
        started = time.perf_counter()
        results = index.search(queries[qi], k)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        found_ids = {result.id for result in results}
        hits += len(found_ids & truth[qi])
        emit("query", (qi + 1) / n_queries)

    total_query_seconds = sum(latencies_ms) / 1000.0
    return BenchmarkReport(
        n=n,
        dim=config.dim,
        k=k,
        n_queries=n_queries,
        metric=config.metric,
        build_seconds=build_seconds,
        recall_at_k=hits / (n_queries * k),
        mean_query_ms=float(np.mean(latencies_ms)),
        p99_query_ms=float(np.percentile(latencies_ms, 99)),
        queries_per_second=n_queries / total_query_seconds if total_query_seconds else 0.0,
    )
