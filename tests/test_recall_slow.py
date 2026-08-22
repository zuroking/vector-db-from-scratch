"""Large-scale recall gate (marked ``slow``; run explicitly via ``pytest -m slow``).

At n=200 the HNSW graph is nearly fully connected, so high recall there is
weak evidence. This gate validates the default parameter set (m=16,
ef_construction=200, ef_search=50) at a scale where layered navigation
actually matters.
"""

import time

import numpy as np
import pytest

from vectordb import DistanceMetric, IndexConfig, VectorRecord
from vectordb.distances import cosine_distance, l2_distance
from vectordb.index.hnsw import HNSWIndex

pytestmark = pytest.mark.slow

_N, _DIM, _K, _QUERIES = 5000, 8, 10, 30


@pytest.mark.parametrize("metric", [DistanceMetric.L2, DistanceMetric.COSINE])
def test_recall_at_10_on_5000_vectors(metric: DistanceMetric) -> None:
    rng = np.random.default_rng(101)
    vectors = rng.normal(size=(_N, _DIM))
    queries = rng.normal(size=(_QUERIES, _DIM))

    index = HNSWIndex(IndexConfig(dim=_DIM, metric=metric, seed=42))
    start = time.perf_counter()
    for i, vec in enumerate(vectors):
        index.insert(VectorRecord(id=f"v{i}", vector=vec.tolist()))
    build_seconds = time.perf_counter() - start

    dist_fn = l2_distance if metric is DistanceMetric.L2 else cosine_distance
    start = time.perf_counter()
    recalls: list[float] = []
    for query in queries:
        truth = set(
            np.argsort(dist_fn(query, vectors), kind="stable")[:_K].tolist()
        )
        got = {int(r.id[1:]) for r in index.search(query, k=_K)}
        recalls.append(len(truth & got) / _K)
    query_ms = (time.perf_counter() - start) * 1000 / _QUERIES

    mean_recall = sum(recalls) / len(recalls)
    print(
        f"\n[{metric.value}] n={_N} recall@{_K}={mean_recall:.4f} "
        f"min-per-query={min(recalls):.2f} build={build_seconds:.1f}s "
        f"avg-query={query_ms:.2f}ms"
    )

    assert mean_recall >= 0.9, f"recall@{_K}={mean_recall:.4f} below threshold"
