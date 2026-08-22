"""Unit tests for the pure neighbour-selection heuristic (Algorithm 4)."""

import numpy as np
import pytest

from vectordb.distances import l2_distance
from vectordb.index.hnsw import select_neighbors_heuristic


class TestDiversityCondition:
    def test_diverse_direction_beats_closer_collinear_candidate(self) -> None:
        # B is closest; A lies on B's direction from q; C spans a new one.
        # Simple nearest-M would pick {B, A}; the heuristic must pick {B, C}.
        query = np.array([0.0, 0.0])
        candidates = np.array(
            [[1.0, 0.0], [0.5, 0.0], [0.0, 1.0]]  # A, B, C
        )

        picked = select_neighbors_heuristic(
            query, candidates, m=2, dist_fn=l2_distance
        )

        assert picked == [1, 2]

    def test_keep_pruned_backfills_free_slots(self) -> None:
        # All candidates share one ray: only X0 survives diversity, the
        # rest are pruned -- and must come back as backfill up to m.
        query = np.array([0.0, 0.0])
        candidates = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])

        picked = select_neighbors_heuristic(
            query, candidates, m=3, dist_fn=l2_distance,
            keep_pruned_connections=True,
        )

        assert picked == [0, 1, 2]

    def test_keep_pruned_disabled_returns_only_diverse(self) -> None:
        query = np.array([0.0, 0.0])
        candidates = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])

        picked = select_neighbors_heuristic(
            query, candidates, m=3, dist_fn=l2_distance,
            keep_pruned_connections=False,
        )

        assert picked == [0]

    def test_cap_m_limits_output(self) -> None:
        rng = np.random.default_rng(3)
        query = np.zeros(4)
        candidates = rng.normal(size=(10, 4))

        picked = select_neighbors_heuristic(
            query, candidates, m=4, dist_fn=l2_distance
        )

        assert len(picked) == 4
        assert len(set(picked)) == 4

    def test_equidistant_candidates_resolve_by_position(self) -> None:
        query = np.array([0.0])
        candidates = np.array([[1.0], [-1.0]])  # identical distances

        picked = select_neighbors_heuristic(
            query, candidates, m=2, dist_fn=l2_distance,
            keep_pruned_connections=False,
        )

        assert picked == [0, 1]
