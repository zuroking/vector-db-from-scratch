"""Behavioural tests for the benchmark module (Phase 5).

The benchmark is library-level numpy-only code: brute-force ground truth,
HNSW recall@k and latency stats, deterministic under a fixed seed, with an
optional progress callback (the CLI attaches Rich on top of it).
"""

import numpy as np

from vectordb.benchmark import BenchmarkReport, exact_top_k_ids, run_benchmark
from vectordb.core.config import DistanceMetric, IndexConfig
from vectordb.core.models import VectorRecord
from vectordb.index.hnsw import HNSWIndex


def bench_config(**overrides: int) -> IndexConfig:
    fields: dict[str, int | DistanceMetric] = {
        "dim": 4,
        "metric": DistanceMetric.L2,
        "m": 8,
        "ef_construction": 32,
        "ef_search": 30,
        "seed": 3,
    }
    fields.update(overrides)
    return IndexConfig(**fields)  # type: ignore[arg-type]


class TestExactTopK:
    def test_query_equal_to_a_row_ranks_it_first(self) -> None:
        dataset = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [0.5, 0.5]])

        top = exact_top_k_ids(dataset, np.array([2.0, 2.0]), k=2, metric=DistanceMetric.L2)

        assert top[0] == 2

    def test_l2_orders_points_on_a_line(self) -> None:
        dataset = np.array([[0.0], [1.0], [2.0], [3.0]])

        top = exact_top_k_ids(dataset, np.array([0.6]), k=4, metric=DistanceMetric.L2)

        assert top == [1, 0, 2, 3]

    def test_cosine_orders_by_similarity(self) -> None:
        dataset = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        top = exact_top_k_ids(dataset, np.array([1.0, 1.0]), k=3, metric=DistanceMetric.COSINE)

        assert top == [2, 0, 1]

    def test_truncates_to_k(self) -> None:
        dataset = np.arange(20, dtype=np.float64).reshape(10, 2)

        top = exact_top_k_ids(dataset, np.zeros(2), k=3, metric=DistanceMetric.L2)

        assert len(top) == 3
        assert top == [0, 1, 2]


class TestTieBreakAgainstSearchContract:
    """Ground truth must rank ties exactly like HNSWIndex.search().

    Locked search contract: distance ascending, equal distances by id
    lexicographically; the truncation cut uses the same comparator. A
    divergent tie-break here would fabricate recall misses on tied
    distances -- not ANN error, just mismatched bookkeeping.
    """

    @staticmethod
    def tied_index(ids: list[str], far_value: float) -> tuple[HNSWIndex, np.ndarray]:
        config = IndexConfig(
            dim=1,
            metric=DistanceMetric.L2,
            m=4,
            ef_construction=16,
            ef_search=32,  # >= n: beam visits every live node deterministically
            seed=5,
        )
        index = HNSWIndex(config)
        for rid in ids:
            index.insert(VectorRecord(id=rid, vector=[far_value]))
        return index, np.array([[far_value]] * len(ids), dtype=np.float64)

    def test_ties_break_by_id_not_row_order(self) -> None:
        dataset = np.array([[0.0], [0.0]])

        top = exact_top_k_ids(
            dataset, np.array([0.0]), k=1,
            metric=DistanceMetric.L2, ids=["b", "a"],
        )

        assert top == [1]  # "a" wins although "b" is row 0

    def test_without_ids_falls_back_to_row_order(self) -> None:
        dataset = np.array([[0.0], [0.0]])

        top = exact_top_k_ids(dataset, np.array([0.0]), k=1, metric=DistanceMetric.L2)

        assert top == [0]

    def test_all_tied_dataset_matches_search_exactly(self) -> None:
        ids = ["h1", "a7", "k3", "b0", "z9", "c2", "m5", "f4"]
        index, dataset = self.tied_index(ids, far_value=5.0)
        query = np.array([5.0])
        k = 5

        got = [result.id for result in index.search(query, k)]
        truth_rows = exact_top_k_ids(
            dataset, query, k, metric=DistanceMetric.L2, ids=ids
        )

        assert [ids[row] for row in truth_rows] == got == sorted(ids)[:k]

    def test_partial_tie_cut_lands_inside_tie_group(self) -> None:
        config = IndexConfig(
            dim=1, metric=DistanceMetric.L2,
            m=4, ef_construction=16, ef_search=32, seed=5,
        )
        index = HNSWIndex(config)
        rows: list[tuple[str, float]] = [
            ("near_c", 0.0), ("tie_q1", 9.0), ("near_a", 1.0),
            ("tie_p9", 9.0), ("near_b", 2.0), ("tie_r5", 9.0),
        ]
        ids = [rid for rid, _ in rows]
        dataset = np.array([[value] for _, value in rows], dtype=np.float64)
        for rid, value in rows:
            index.insert(VectorRecord(id=rid, vector=[value]))
        query = np.array([0.2])  # distinct distances to all three near values
        k = 4  # cut falls inside the {p9, q1, r5} tie group

        got = [result.id for result in index.search(query, k)]
        truth_ids = [
            ids[row]
            for row in exact_top_k_ids(
                dataset, query, k, metric=DistanceMetric.L2, ids=ids
            )
        ]

        # Distinct near values rank unambiguously; the single cut slot goes
        # to the lexicographically smallest tied id -- on BOTH sides.
        assert truth_ids == ["near_c", "near_a", "near_b", "tie_p9"]
        assert got == truth_ids


class TestRunBenchmark:
    def test_report_is_fully_populated(self) -> None:
        report = run_benchmark(bench_config(), n=40, k=5, n_queries=7, seed=11)

        assert isinstance(report, BenchmarkReport)
        assert (report.n, report.dim, report.k, report.n_queries) == (40, 4, 5, 7)
        assert report.metric is DistanceMetric.L2
        assert 0.0 <= report.recall_at_k <= 1.0
        assert report.build_seconds > 0.0
        assert report.mean_query_ms > 0.0
        assert report.p99_query_ms > 0.0
        assert report.queries_per_second > 0.0

    def test_perfect_recall_on_small_dataset(self) -> None:
        report = run_benchmark(bench_config(), n=40, k=5, n_queries=7, seed=11)

        assert report.recall_at_k == 1.0

    def test_deterministic_given_same_seed(self) -> None:
        first = run_benchmark(bench_config(), n=40, k=5, n_queries=7, seed=11)
        second = run_benchmark(bench_config(), n=40, k=5, n_queries=7, seed=11)

        # Wall-clock timings (build_seconds, *_ms, qps) are excluded by
        # design; determinism covers the data-dependent outcome.
        assert first.recall_at_k == second.recall_at_k

    def test_progress_callback_receives_stages_and_fractions(self) -> None:
        calls: list[tuple[str, float]] = []

        run_benchmark(
            bench_config(), n=40, k=5, n_queries=7, seed=11,
            on_progress=lambda stage, fraction: calls.append((stage, fraction)),
        )

        stages = {stage for stage, _ in calls}
        assert {"ground_truth", "query"} <= stages
        for stage in ("ground_truth", "query"):
            fractions = [f for s, f in calls if s == stage]
            assert fractions == sorted(fractions), f"{stage} not monotonic"
            assert all(0.0 <= f <= 1.0 for f in fractions)
            assert fractions[-1] == 1.0
