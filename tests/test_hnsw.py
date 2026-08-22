"""Behavioural tests for the HNSW index core (Phase 2, wave 1)."""

import numpy as np
import pytest
from pydantic import JsonValue

from vectordb import (
    DistanceMetric,
    DimensionMismatchError,
    DuplicateIDError,
    EmptyIndexError,
    IDNotFoundError,
    IndexConfig,
    SearchResult,
    VectorRecord,
)
from vectordb.distances import cosine_distance, l2_distance
from vectordb.index.hnsw import HNSWIndex


def make_index(
    dim: int = 8,
    metric: DistanceMetric = DistanceMetric.L2,
    *,
    seed: int = 0,
    extend_candidates: bool = False,
) -> HNSWIndex:
    return HNSWIndex(
        IndexConfig(
            dim=dim,
            metric=metric,
            seed=seed,
            extend_candidates=extend_candidates,
        )
    )


def rec(rid: str, vec: list[float], payload: dict[str, JsonValue] | None = None) -> VectorRecord:
    return VectorRecord(id=rid, vector=vec, payload=payload)


def topology(index: HNSWIndex) -> tuple[object, ...]:
    """Order-sensitive structural fingerprint used for determinism checks."""
    return (
        index.entry_point_id,
        tuple(
            (
                rid,
                index.level_of(rid),
                tuple(
                    tuple(sorted(index.neighbor_ids(rid, layer)))
                    for layer in range(index.level_of(rid) + 1)
                ),
            )
            for rid in index.ids
        ),
    )


class TestEmptyIndex:
    def test_len_zero_and_contains_false(self) -> None:
        index = make_index()

        assert len(index) == 0
        assert index.contains("missing") is False
        assert index.entry_point_id is None
        assert index.ids == []

    def test_search_raises_empty_index_error(self) -> None:
        index = make_index()

        with pytest.raises(EmptyIndexError):
            index.search(np.ones(8), k=3)

    def test_delete_unknown_id_raises(self) -> None:
        index = make_index()

        with pytest.raises(IDNotFoundError):
            index.delete("ghost")


class TestSingleInsert:
    def test_insert_makes_record_findable_with_score_zero(self) -> None:
        index = make_index(dim=3)
        index.insert(rec("a", [1.0, 2.0, 3.0], payload={"kind": "point"}))

        results = index.search(np.array([1.0, 2.0, 3.0]), k=1)

        assert len(index) == 1
        assert index.contains("a") is True
        assert [r.id for r in results] == ["a"]
        assert results[0].score == pytest.approx(0.0)
        assert results[0].payload == {"kind": "point"}

    def test_first_insert_becomes_entry_point(self) -> None:
        index = make_index(dim=3)
        index.insert(rec("a", [1.0, 0.0, 0.0]))

        assert index.entry_point_id == "a"
        assert index.level_of("a") >= 0


class TestInsertValidation:
    def test_duplicate_live_id_rejected(self) -> None:
        index = make_index(dim=2)
        index.insert(rec("a", [1.0, 0.0]))

        with pytest.raises(DuplicateIDError) as exc_info:
            index.insert(rec("a", [0.0, 1.0]))

        assert exc_info.value.record_id == "a"

    def test_dimension_mismatch_rejected(self) -> None:
        index = make_index(dim=4)

        with pytest.raises(DimensionMismatchError) as exc_info:
            index.insert(rec("bad", [1.0, 2.0]))

        assert exc_info.value.expected == 4
        assert exc_info.value.actual == 2

    def test_failed_insert_leaves_index_unchanged(self) -> None:
        index = make_index(dim=4)
        index.insert(rec("a", [1.0, 0.0, 0.0, 0.0]))

        with pytest.raises(DimensionMismatchError):
            index.insert(rec("bad", [1.0]))

        assert len(index) == 1
        assert index.contains("bad") is False


class TestSearchValidation:
    def test_query_dimension_mismatch_raises(self) -> None:
        index = make_index(dim=3)
        index.insert(rec("a", [1.0, 0.0, 0.0]))

        with pytest.raises(DimensionMismatchError) as exc_info:
            index.search(np.ones(5), k=1)

        assert exc_info.value.expected == 3
        assert exc_info.value.actual == 5

    @pytest.mark.parametrize("bad_k", [0, -2])
    def test_non_positive_k_rejected(self, bad_k: int) -> None:
        index = make_index(dim=2)
        index.insert(rec("a", [1.0, 0.0]))

        with pytest.raises(ValueError, match="k"):
            index.search(np.array([1.0, 0.0]), k=bad_k)

    def test_non_1d_query_rejected(self) -> None:
        index = make_index(dim=2)
        index.insert(rec("a", [1.0, 0.0]))

        with pytest.raises(ValueError, match="1-D"):
            index.search(np.zeros((2, 2)), k=1)

    def test_k_above_live_count_returns_all_live_sorted(self) -> None:
        index = make_index(dim=2, seed=3)
        index.insert(rec("near", [1.0, 0.0]))
        index.insert(rec("mid", [2.0, 0.0]))
        index.insert(rec("far", [10.0, 0.0]))

        results = index.search(np.array([1.0, 0.0]), k=50)

        assert [r.id for r in results] == ["near", "mid", "far"]
        assert results[0].score < results[1].score < results[2].score


class TestExactnessOnTinyDatasets:
    @pytest.mark.parametrize("metric", [DistanceMetric.L2, DistanceMetric.COSINE])
    def test_small_dataset_search_is_exact(self, metric: DistanceMetric) -> None:
        rng = np.random.default_rng(11)
        vectors = rng.normal(size=(12, 4))
        index = make_index(dim=4, metric=metric, seed=5)
        for i, vec in enumerate(vectors):
            index.insert(rec(f"v{i}", vec.tolist()))

        dist_fn = l2_distance if metric is DistanceMetric.L2 else cosine_distance
        for q_idx in range(12):
            query = vectors[q_idx]
            truth = set(
                np.argsort(dist_fn(query, vectors), kind="stable")[:3].tolist()
            )
            got = {r.id for r in index.search(query, k=3)}

            assert got == {f"v{i}" for i in truth}


class TestDeterminism:
    def test_same_seed_same_insert_sequence_same_topology(self) -> None:
        rng = np.random.default_rng(99)
        vectors = rng.normal(size=(40, 6))

        first = make_index(dim=6, seed=123)
        second = make_index(dim=6, seed=123)
        for i, vec in enumerate(vectors):
            record = rec(f"v{i}", vec.tolist())
            first.insert(record)
            second.insert(record)

        assert topology(first) == topology(second)

    def test_entry_point_sits_on_the_top_level(self) -> None:
        rng = np.random.default_rng(31)
        index = make_index(dim=4, seed=17)
        for i, vec in enumerate(rng.normal(size=(80, 4))):
            index.insert(rec(f"v{i}", vec.tolist()))

        assert index.entry_point_id is not None
        top = max(index.level_of(rid) for rid in index.ids)

        assert index.level_of(index.entry_point_id) == top

    def test_layer_frequencies_decrease_with_height(self) -> None:
        """Pyramid property of the exponential level assignment (Malkov-Yashunin)."""
        rng = np.random.default_rng(77)
        index = make_index(dim=4, seed=21)
        for i, vec in enumerate(rng.normal(size=(150, 4))):
            index.insert(rec(f"v{i}", vec.tolist()))

        counts: dict[int, int] = {}
        for rid in index.ids:
            counts[index.level_of(rid)] = counts.get(index.level_of(rid), 0) + 1

        heights_top_down = sorted(counts, reverse=True)
        # Pyramid property: higher layers hold strictly fewer nodes,
        # so reading top-down yields non-decreasing frequencies.
        top_down_counts = [counts[h] for h in heights_top_down]

        assert top_down_counts == sorted(top_down_counts)
        assert len(heights_top_down) >= 2, "150 nodes should populate at least two layers"


class TestRecallAgainstBruteForce:
    @pytest.mark.parametrize("metric", [DistanceMetric.L2, DistanceMetric.COSINE])
    def test_recall_at_10_beats_threshold(self, metric: DistanceMetric) -> None:
        rng = np.random.default_rng(7)
        n, dim, k = 200, 8, 10
        vectors = rng.normal(size=(n, dim))
        queries = rng.normal(size=(25, dim))

        index = make_index(dim=dim, metric=metric, seed=42)
        for i, vec in enumerate(vectors):
            index.insert(rec(f"v{i}", vec.tolist()))

        dist_fn = l2_distance if metric is DistanceMetric.L2 else cosine_distance
        recalls: list[float] = []
        for query in queries:
            truth = set(np.argsort(dist_fn(query, vectors), kind="stable")[:k].tolist())
            approx = {int(r.id[1:]) for r in index.search(query, k=k)}
            recalls.append(len(truth & approx) / k)

        mean_recall = sum(recalls) / len(recalls)
        assert mean_recall >= 0.9, f"recall@{k}={mean_recall:.3f} below threshold"

    @pytest.mark.parametrize("metric", [DistanceMetric.L2, DistanceMetric.COSINE])
    def test_scores_are_monotonically_nondecreasing(self, metric: DistanceMetric) -> None:
        rng = np.random.default_rng(13)
        vectors = rng.normal(size=(60, 5))
        index = make_index(dim=5, metric=metric, seed=9)
        for i, vec in enumerate(vectors):
            index.insert(rec(f"v{i}", vec.tolist()))

        results = index.search(rng.normal(size=5), k=15)

        scores = [r.score for r in results]
        assert scores == sorted(scores)


class TestTombstones:
    def test_delete_hides_record_and_shrinks_len(self) -> None:
        index = make_index(dim=2, seed=5)
        index.insert(rec("a", [0.0, 0.0]))
        index.insert(rec("b", [5.0, 0.0]))
        index.delete("a")

        assert len(index) == 1
        assert index.contains("a") is False
        assert "a" not in [r.id for r in index.search(np.array([0.0, 0.0]), k=5)]

    def test_double_delete_raises(self) -> None:
        index = make_index(dim=2)
        index.insert(rec("a", [1.0, 0.0]))
        index.delete("a")

        with pytest.raises(IDNotFoundError):
            index.delete("a")

    def test_deleted_id_is_reusable(self) -> None:
        index = make_index(dim=2)
        index.insert(rec("a", [1.0, 0.0]))
        index.delete("a")

        index.insert(rec("a", [0.0, 2.0]))

        assert index.contains("a") is True
        results = index.search(np.array([0.0, 2.0]), k=1)
        assert [r.id for r in results] == ["a"]
        assert results[0].score == pytest.approx(0.0)

    def test_k_above_live_count_after_deletes_returns_all_live(self) -> None:
        index = make_index(dim=2, seed=4)
        for rid, vec in [("a", [0.0, 0.0]), ("b", [1.0, 0.0]), ("c", [2.0, 0.0])]:
            index.insert(rec(rid, vec))
        index.delete("b")

        results = index.search(np.array([0.0, 0.0]), k=10)

        assert [r.id for r in results] == ["a", "c"]

    def test_recall_survives_deleting_a_third_of_nodes(self) -> None:
        rng = np.random.default_rng(23)
        vectors = rng.normal(size=(60, 6))
        index = make_index(dim=6, seed=8)
        for i, vec in enumerate(vectors):
            index.insert(rec(f"v{i}", vec.tolist()))

        doomed = {f"v{i}" for i in range(0, 60, 3)}
        for rid in doomed:
            index.delete(rid)
        live = np.array([v for i, v in enumerate(vectors) if f"v{i}" not in doomed])

        queries = rng.normal(size=(15, 6))
        hits = 0
        total = 0
        for query in queries:
            truth = set(np.argsort(l2_distance(query, live), kind="stable")[:3].tolist())
            got_ids = [r.id for r in index.search(query, k=3)]
            live_ids = [f"v{i}" for i in range(60) if f"v{i}" not in doomed]
            got = {live_ids.index(rid) for rid in got_ids}
            hits += len(truth & got)
            total += 3
            assert all(rid not in doomed for rid in got_ids)

        assert hits / total >= 0.9


class TestEntryPointOnDelete:
    """Locked decision #3: no eager reselection; tombstones stay routable."""

    def test_entry_point_pointer_is_not_reselected(self) -> None:
        index = make_index(dim=3, seed=6)
        for i, vec in enumerate(np.random.default_rng(6).normal(size=(10, 3))):
            index.insert(rec(f"v{i}", vec.tolist()))
        entry_before = index.entry_point_id
        assert entry_before is not None

        index.delete(entry_before)

        assert index.entry_point_id == entry_before

    def test_search_after_deleting_entry_point_stays_exact(self) -> None:
        vectors = np.random.default_rng(12).normal(size=(15, 3))
        index = make_index(dim=3, seed=2)
        for i, vec in enumerate(vectors):
            index.insert(rec(f"v{i}", vec.tolist()))

        entry = index.entry_point_id
        assert entry is not None
        index.delete(entry)
        live_ids = index.ids
        live_matrix = np.array(
            [vectors[int(rid[1:])] for rid in live_ids]
        )

        query = np.array([0.5, -0.5, 0.25])
        truth_ids = {
            live_ids[i]
            for i in np.argsort(l2_distance(query, live_matrix), kind="stable")[:4]
        }
        got_ids = {r.id for r in index.search(query, k=4)}

        assert got_ids == truth_ids
        assert entry not in got_ids

    def test_insert_after_entry_point_delete_keeps_graph_working(self) -> None:
        vectors = np.random.default_rng(15).normal(size=(10, 3))
        index = make_index(dim=3, seed=3)
        for i, vec in enumerate(vectors):
            index.insert(rec(f"v{i}", vec.tolist()))

        index.delete(index.entry_point_id or "")
        index.insert(rec("fresh", [10.0, 10.0, 10.0]))

        results = index.search(np.array([10.0, 10.0, 10.0]), k=1)

        assert results[0].id == "fresh"


class TestRebuild:
    def test_rebuild_drops_tombstones_and_keeps_recall(self) -> None:
        rng = np.random.default_rng(19)
        vectors = rng.normal(size=(50, 5))
        index = make_index(dim=5, seed=10)
        for i, vec in enumerate(vectors):
            index.insert(rec(f"v{i}", vec.tolist()))
        for i in range(0, 50, 2):
            index.delete(f"v{i}")

        index.rebuild()

        assert len(index) == 25
        assert index.contains("v1") is True
        assert index.contains("v0") is False
        query = rng.normal(size=5)
        odd_ids = [f"v{i}" for i in range(1, 50, 2)]
        truth_positions = np.argsort(
            l2_distance(query, vectors[1::2]), kind="stable"
        )[:3]
        truth = {odd_ids[p] for p in truth_positions}
        got = {r.id for r in index.search(query, k=3)}
        assert got == truth

    def test_rebuild_equals_fresh_build_of_survivors_in_order(self) -> None:
        rng = np.random.default_rng(29)
        vectors = rng.normal(size=(20, 4))
        rebuilt = make_index(dim=4, seed=11)
        for i, vec in enumerate(vectors):
            rebuilt.insert(rec(f"v{i}", vec.tolist()))
        for i in (3, 7, 11):
            rebuilt.delete(f"v{i}")
        rebuilt.rebuild()

        fresh = make_index(dim=4, seed=11)
        for i in range(20):
            if i not in (3, 7, 11):
                fresh.insert(rec(f"v{i}", vectors[i].tolist()))

        assert topology(rebuilt) == topology(fresh)

    def test_rebuild_is_idempotent(self) -> None:
        rng = np.random.default_rng(31)
        index = make_index(dim=4, seed=13)
        for i, vec in enumerate(rng.normal(size=(25, 4))):
            index.insert(rec(f"v{i}", vec.tolist()))
        index.delete("v2")
        index.delete("v9")
        index.rebuild()
        snapshot = topology(index)

        index.rebuild()

        assert topology(index) == snapshot

    def test_full_delete_then_rebuild_resets_entry_point(self) -> None:
        index = make_index(dim=2, seed=1)
        index.insert(rec("a", [1.0, 0.0]))
        index.insert(rec("b", [0.0, 1.0]))
        index.delete("a")
        index.delete("b")

        # Waypoint pointer persists by design even with zero live records.
        assert len(index) == 0
        assert index.entry_point_id == "b"
        with pytest.raises(EmptyIndexError):
            index.search(np.array([1.0, 1.0]), k=1)

        index.rebuild()

        assert index.entry_point_id is None
        assert index.ids == []


class TestExtendCandidatesFlag:
    def test_flag_on_builds_working_index_on_clustered_data(self) -> None:
        rng = np.random.default_rng(41)
        cluster_a = rng.normal(loc=-6.0, size=(20, 4))
        cluster_b = rng.normal(loc=6.0, size=(20, 4))
        vectors = np.vstack([cluster_a, cluster_b])

        index = make_index(dim=4, seed=14, extend_candidates=True)
        for i, vec in enumerate(vectors):
            index.insert(rec(f"v{i}", vec.tolist()))

        recalls: list[float] = []
        for query in rng.normal(size=(10, 4)):
            truth = set(np.argsort(l2_distance(query, vectors), kind="stable")[:5].tolist())
            got = {int(r.id[1:]) for r in index.search(query, k=5)}
            recalls.append(len(truth & got) / 5)

        assert sum(recalls) / len(recalls) >= 0.9

    def test_flag_on_stays_deterministic(self) -> None:
        rng = np.random.default_rng(43)
        vectors = rng.normal(size=(30, 4))

        def build() -> HNSWIndex:
            index = make_index(dim=4, seed=15, extend_candidates=True)
            for i, vec in enumerate(vectors):
                index.insert(rec(f"v{i}", vec.tolist()))
            return index

        assert topology(build()) == topology(build())

    def test_extension_reaches_beyond_visited_beam(self) -> None:
        """With ef_construction < n the beam misses nodes; the extension
        loop is what pulls in their neighbours (Algorithm 4's clustered-data
        case). Two tight far-apart clusters with m=3/ef_c=4 leave intra-
        cluster links sparse: without the extension this config measures
        recall@5 = 0.92, with it the search is exact (verified by mutation:
        disabling the unseen-neighbour append turns this test red)."""
        rng = np.random.default_rng(99)
        vectors = np.vstack([
            rng.normal(loc=-10.0, scale=0.3, size=(40, 2)),
            rng.normal(loc=10.0, scale=0.3, size=(40, 2)),
        ])
        queries = np.vstack([
            rng.normal(loc=-10.0, scale=0.5, size=(10, 2)),
            rng.normal(loc=10.0, scale=0.5, size=(10, 2)),
        ])

        index = HNSWIndex(
            IndexConfig(
                dim=2,
                metric=DistanceMetric.L2,
                m=3,
                ef_construction=4,
                ef_search=16,
                seed=5,
                extend_candidates=True,
            )
        )
        for i, vec in enumerate(vectors):
            index.insert(rec(f"v{i}", vec.tolist()))

        hits = 0
        for query in queries:
            truth = set(
                np.argsort(np.sum((vectors - query) ** 2, axis=1))[:5].tolist()
            )
            got = {int(r.id[1:]) for r in index.search(query, k=5)}
            hits += len(truth & got)

        assert hits / (len(queries) * 5) == 1.0

    def test_neighbor_ids_layer_out_of_range_raises(self) -> None:
        index = make_index(dim=2, seed=3)
        index.insert(rec("a", [1.0, 0.0]))
        level = index.level_of("a")

        with pytest.raises(ValueError, match="layer"):
            index.neighbor_ids("a", level + 1)


class TestSortingContract:
    """Phase 3: output ordering is part of the public contract."""

    def test_identical_vectors_tie_break_by_id(self) -> None:
        index = make_index(dim=3, seed=7)
        base = [1.0, 2.0, 3.0]
        for record_id in ("zeta", "alpha", "mid"):
            index.insert(rec(record_id, base))

        results = index.search(np.array(base), k=3)

        assert [r.id for r in results] == ["alpha", "mid", "zeta"]
        assert len({r.score for r in results}) == 1

    def test_ties_prefer_smaller_id_at_truncation_boundary(self) -> None:
        index = make_index(dim=2, seed=8)
        base = [0.5, -0.5]
        for record_id in ("delta", "bravo", "charlie", "alpha"):
            index.insert(rec(record_id, base))

        results = index.search(np.array(base), k=2)

        assert [r.id for r in results] == ["alpha", "bravo"]

    def test_zero_vector_end_to_end_cosine_convention(self) -> None:
        index = make_index(dim=2, metric=DistanceMetric.COSINE, seed=9)
        index.insert(rec("zero", [0.0, 0.0]))
        index.insert(rec("near", [1.0, 0.0]))

        results = index.search(np.array([1.0, 0.0]), k=2)

        assert [r.id for r in results] == ["near", "zero"]
        assert results[1].score == pytest.approx(1.0)

    def test_payload_round_trip_for_multiple_records(self) -> None:
        index = make_index(dim=2, seed=10)
        expected: dict[str, dict[str, JsonValue] | None] = {
            "a": {"n": 1},
            "b": {"n": 2},
            "c": None,
        }
        for record_id, payload in expected.items():
            index.insert(rec(record_id, [float(len(record_id)), 0.0], payload))

        by_id = {r.id: r.payload for r in index.search(np.zeros(2), k=3)}

        assert by_id == expected
