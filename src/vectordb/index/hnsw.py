"""HNSW index implementation (Phase 2).

Hierarchical Navigable Small World graphs after Malkov & Yashunin (2018).
Locked architectural constraints:

- Not thread-safe (decision #2): callers must synchronise externally.
- Soft-delete tombstones (decision #4): deleted nodes stay routable
  waypoints but are excluded from results; compaction happens only in
  ``rebuild()``.
- The entry point is never eagerly re-selected on delete; it changes only
  when a newly inserted node reaches a strictly higher level.
- All numeric work runs through :mod:`vectordb.distances` in float64;
  the level RNG is seeded from ``IndexConfig.seed`` so identical insert
  sequences produce identical topologies (bit-for-bit reproducibility).
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from typing import NamedTuple, TypeAlias

import numpy as np
from pydantic import JsonValue

from vectordb.core.config import DistanceMetric, IndexConfig
from vectordb.core.exceptions import (
    DimensionMismatchError,
    DuplicateIDError,
    EmptyIndexError,
    IDNotFoundError,
)
from vectordb.core.models import SearchResult, VectorRecord
from vectordb.distances import cosine_distance, l2_distance

DistanceFn: TypeAlias = Callable[[np.ndarray, np.ndarray], np.ndarray]

_INITIAL_CAPACITY = 64
# Paper: M_max0 = 2*M -- layer 0 carries twice the link budget of upper layers.
_LAYER0_M_MAX_FACTOR = 2


def select_neighbors_heuristic(
    query: np.ndarray,
    candidates: np.ndarray,
    m: int,
    dist_fn: DistanceFn,
    keep_pruned_connections: bool = True,
) -> list[int]:
    """Diversity-aware neighbour selection (Malkov & Yashunin, Algorithm 4).

    A candidate is accepted only while it is closer to ``query`` than to
    every already-selected neighbour; candidates lying in an already-covered
    direction are pruned. With ``keep_pruned_connections`` the pruned pool
    backfills free slots up to ``m`` (the paper's keepPrunedConnections).

    Args:
        query: 1-D vector of shape ``(dim,)``.
        candidates: 2-D array of shape ``(c, dim)``.
        m: Maximum number of neighbours to return.
        dist_fn: Batched distance function from :mod:`vectordb.distances`.
        keep_pruned_connections: Whether pruned candidates may backfill slots.

    Returns:
        Selected indices into ``candidates``, in acceptance order. Fully
        deterministic: ties are broken by stable distance sort, then by
        candidate position.
    """
    d_to_query = dist_fn(query, candidates)
    order = np.argsort(d_to_query, kind="stable")

    selected: list[int] = []
    pruned: list[int] = []
    for idx_value in order:
        idx = int(idx_value)
        if len(selected) >= m:
            break
        if selected:
            # Accept e iff dist(e, q) < dist(e, r) for every r already in R.
            d_to_selected = dist_fn(candidates[idx], candidates[selected])
            if bool(np.any(d_to_selected < d_to_query[idx])):
                pruned.append(idx)
                continue
        selected.append(idx)

    if keep_pruned_connections:
        for idx in pruned:
            if len(selected) >= m:
                break
            selected.append(idx)
    return selected


class IndexSnapshot(NamedTuple):
    """Complete internal state, sufficient for byte-exact persistence.

    ``entry_point`` uses ``-1`` to encode "no entry point" because the
    binary format carries no nullable integers.
    """

    vectors: np.ndarray
    alive: np.ndarray
    levels: np.ndarray
    links: list[list[list[int]]]
    node_ids: list[str]
    payloads: list[dict[str, JsonValue] | None]
    insertion_order: list[str]
    entry_point: int
    max_level: int
    size: int


class HNSWIndex:
    """Approximate nearest-neighbour index over an HNSW layered graph.

    Usage contract: one instance binds one dimensionality and one distance
    metric via ``IndexConfig``; both are immutable for the index lifetime.

    Complexity (n = live vector count):
        insert: O(ef_construction * M * log n) distance evaluations,
        search: O(max(ef_search, k) * M * log n) distance evaluations.
    """

    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        self._dist_fn: DistanceFn = (
            cosine_distance if config.metric is DistanceMetric.COSINE else l2_distance
        )
        self._rng = np.random.default_rng(config.seed)

        self._vectors = np.empty((_INITIAL_CAPACITY, config.dim), dtype=np.float64)
        self._alive = np.zeros(_INITIAL_CAPACITY, dtype=bool)
        self._levels = np.zeros(_INITIAL_CAPACITY, dtype=np.int64)

        self._node_ids: list[str] = []
        self._payloads: list[dict[str, JsonValue] | None] = []
        # _links[node][layer] -> list of neighbour node indices.
        self._links: list[list[list[int]]] = []
        self._id_to_node: dict[str, int] = {}
        self._insertion_order: list[str] = []

        self._entry_point: int | None = None
        self._max_level = -1
        self._size = 0

    # -- mutation ---------------------------------------------------------

    def insert(self, record: VectorRecord) -> None:
        """Insert one record, linking it into every layer up to its level.

        A previously deleted id becomes reusable: the tombstone freed the
        name, and the new insert creates a fresh node.

        Raises:
            DimensionMismatchError: If record dimensionality differs from config.
            DuplicateIDError: If the id already belongs to a live record.
            ValueError: If the stored vector is not 1-D.
        """
        vector = np.asarray(record.vector, dtype=np.float64)
        if vector.ndim != 1:  # pragma: no cover - defensive: VectorRecord
            # declares `vector: list[float]`, so Pydantic already rejects
            # any nested (non-1-D) shape before insert() sees the record.
            raise ValueError("record.vector must be 1-D")
        if vector.shape[0] != self._config.dim:
            raise DimensionMismatchError(
                expected=self._config.dim, actual=int(vector.shape[0])
            )
        if record.id in self._id_to_node:
            raise DuplicateIDError(record.id)

        level = self._draw_level()
        node = self._allocate(record.id, vector, record.payload, level)

        if self._entry_point is None:
            self._entry_point = node
            self._max_level = level
        else:
            self._connect(node, level)

        self._id_to_node[record.id] = node
        self._insertion_order.append(record.id)
        self._size += 1

    def delete(self, record_id: str) -> None:
        """Soft-delete a live record (tombstone); graph links are kept.

        The tombstoned node remains a routable waypoint so connectivity and
        recall survive without repair. It is excluded from results, its id
        becomes reusable, and ``__len__`` decreases.

        Raises:
            IDNotFoundError: If the id is absent or already deleted.
        """
        node = self._require_live(record_id)
        self._alive[node] = False
        self._size -= 1
        del self._id_to_node[record_id]
        # Locked decision #3: a tombstoned entry point intentionally remains
        # the navigational root; replacement happens only via a strictly
        # higher insert or via rebuild(). No eager reselection.

    def rebuild(self) -> None:
        """Compact tombstones in place, replaying survivors in first-insert order.

        HNSW topology depends on the insertion sequence, so compaction
        replays surviving records chronologically into a fresh graph and
        then swaps it in. Locked guarantees (architect decision #2):

        - the result is identical to building a brand-new index from just
          the surviving records with the same ``IndexConfig``;
        - repeated ``rebuild()`` calls are idempotent;
        - afterwards the entry pointer refers to a live node (or is
          ``None`` when no record survived);
        - mutation is in-place; ``None`` is returned on purpose.
        """
        survivor_ids = [
            rid for rid in self._insertion_order if rid in self._id_to_node
        ]
        fresh = HNSWIndex(self._config)
        for record_id in survivor_ids:
            node = self._id_to_node[record_id]
            fresh.insert(
                VectorRecord(
                    id=record_id,
                    vector=self._vectors[node].tolist(),
                    payload=self._payloads[node],
                )
            )
        self._adopt(fresh)

    def _adopt(self, other: HNSWIndex) -> None:
        """Swap in all mutable state from ``other`` (compaction step)."""
        self._rng = other._rng
        self._vectors = other._vectors
        self._alive = other._alive
        self._levels = other._levels
        self._node_ids = other._node_ids
        self._payloads = other._payloads
        self._links = other._links
        self._id_to_node = other._id_to_node
        self._insertion_order = other._insertion_order
        self._entry_point = other._entry_point
        self._max_level = other._max_level
        self._size = other._size

    # -- query ------------------------------------------------------------

    def search(self, query: np.ndarray, k: int) -> list[SearchResult]:
        """Return up to ``k`` live records nearest to ``query``, best first.

        Sorting contract (locked Phase 3): results ascend by distance;
        equal distances are broken by ascending id (lexicographic), so
        identical vectors under different ids surface in a stable,
        insertion-order-independent sequence. ``k`` above the live count
        simply returns every live record.

        Raises:
            DimensionMismatchError: If query dimensionality differs from config.
            ValueError: If ``k`` is not positive or the query is not 1-D.
            EmptyIndexError: If the index holds no live records.
        """
        if k < 1:
            raise ValueError(f"k must be a positive integer, got {k}")
        q = np.asarray(query, dtype=np.float64)
        if q.ndim != 1:
            raise ValueError(f"query must be 1-D, got {q.ndim}-D")
        if q.shape[0] != self._config.dim:
            raise DimensionMismatchError(
                expected=self._config.dim, actual=int(q.shape[0])
            )
        if self._size == 0 or self._entry_point is None:
            raise EmptyIndexError("search on an index with zero live records")

        ep = self._entry_point
        for layer in range(self._max_level, 0, -1):
            ep = self._search_layer(q, [ep], 1, layer)[0][1]

        ef = max(self._config.ef_search, k)
        found = self._search_layer(q, [ep], ef, 0)

        live: list[tuple[float, str, int]] = [
            (dist, self._node_ids[node], node)
            for dist, node in found
            if self._alive[node]
        ]
        # Truncation uses the same comparator as the contract: among equal
        # distances the lexicographically smaller id survives the cut.
        live.sort(key=lambda entry: (entry[0], entry[1]))
        return [
            SearchResult(id=record_id, score=dist, payload=self._payloads[node])
            for dist, record_id, node in live[:k]
        ]

    # -- introspection ------------------------------------------------------

    def contains(self, record_id: str) -> bool:
        """True if ``record_id`` refers to a live (non-deleted) record."""
        return record_id in self._id_to_node

    def __len__(self) -> int:
        """Number of live (non-tombstoned) records."""
        return self._size

    @property
    def entry_point_id(self) -> str | None:
        """Id of the current graph entry point; ``None`` while empty."""
        if self._entry_point is None:
            return None
        return self._node_ids[self._entry_point]

    @property
    def ids(self) -> list[str]:
        """Live ids in chronological first-insert order."""
        return [rid for rid in self._insertion_order if rid in self._id_to_node]

    def level_of(self, record_id: str) -> int:
        """Top layer index of the given live record.

        Raises:
            IDNotFoundError: If the id is absent or deleted.
        """
        return int(self._levels[self._require_live(record_id)])

    def neighbor_ids(self, record_id: str, layer: int) -> list[str]:
        """Neighbour ids of a live record at one layer, in link order.

        Raises:
            IDNotFoundError: If the id is absent or deleted.
            ValueError: If ``layer`` is outside ``0..level_of(record_id)``.
        """
        node = self._require_live(record_id)
        top = int(self._levels[node])
        if layer < 0 or layer > top:
            raise ValueError(f"layer {layer} outside 0..{top} for {record_id!r}")
        return [self._node_ids[nb] for nb in self._links[node][layer]]

    # -- internals ----------------------------------------------------------

    @property
    def config(self) -> IndexConfig:
        """The frozen configuration this index was created with."""
        return self._config

    def _snapshot_for_persistence(self) -> IndexSnapshot:
        """Export full internal state for the persistence layer (Phase 4)."""
        node_count = len(self._node_ids)
        return IndexSnapshot(
            vectors=self._vectors[:node_count].copy(),
            alive=self._alive[:node_count].copy(),
            levels=self._levels[:node_count].copy(),
            links=[[[nb for nb in layer] for layer in per_node] for per_node in self._links],
            node_ids=list(self._node_ids),
            payloads=list(self._payloads),
            insertion_order=list(self._insertion_order),
            entry_point=-1 if self._entry_point is None else self._entry_point,
            max_level=self._max_level,
            size=self._size,
        )

    def _restore_from_persistence(self, snapshot: IndexSnapshot) -> None:
        """Adopt a persisted snapshot verbatim (Phase 4 decisions #1 and #2).

        Levels, links and the entry pointer are restored exactly as saved --
        nothing structural is recomputed, so round-trips never depend on RNG
        reproducibility across numpy versions. A tombstoned entry point at
        save time stays the entry point after load. The level RNG restarts
        from ``IndexConfig.seed`` and influences only *future* inserts.
        """
        node_count = len(snapshot.node_ids)
        self._vectors = np.array(snapshot.vectors, dtype=np.float64, copy=True)
        self._alive = np.array(snapshot.alive, dtype=bool, copy=True)
        self._levels = np.array(snapshot.levels, dtype=np.int64, copy=True)
        # Fresh mutable containers: future inserts append to these lists.
        self._links = [
            [list(layer) for layer in per_node] for per_node in snapshot.links
        ]
        self._node_ids = list(snapshot.node_ids)
        self._payloads = list(snapshot.payloads)
        self._insertion_order = list(snapshot.insertion_order)
        self._id_to_node = {
            record_id: node
            for node, record_id in enumerate(snapshot.node_ids)
            if bool(snapshot.alive[node])
        }
        self._entry_point = (
            None if snapshot.entry_point < 0 else snapshot.entry_point
        )
        self._max_level = snapshot.max_level
        self._size = snapshot.size
        self._rng = np.random.default_rng(self._config.seed)

    def _require_live(self, record_id: str) -> int:
        node = self._id_to_node.get(record_id)
        if node is None:
            raise IDNotFoundError(record_id)
        return node

    def _draw_level(self) -> int:
        uniform = 1.0 - float(self._rng.random())  # in (0, 1]: avoids log(0)
        return int(-math.log(uniform) * self._config.level_multiplier)

    def _ensure_capacity(self, needed: int) -> None:
        capacity = self._vectors.shape[0]
        if needed <= capacity:
            return
        new_capacity = max(needed, 2 * capacity)
        vectors = np.empty((new_capacity, self._config.dim), dtype=np.float64)
        vectors[:capacity] = self._vectors[:capacity]
        alive = np.zeros(new_capacity, dtype=bool)
        alive[:capacity] = self._alive[:capacity]
        levels = np.zeros(new_capacity, dtype=np.int64)
        levels[:capacity] = self._levels[:capacity]
        self._vectors, self._alive, self._levels = vectors, alive, levels

    def _allocate(
        self,
        record_id: str,
        vector: np.ndarray,
        payload: dict[str, JsonValue] | None,
        level: int,
    ) -> int:
        node = len(self._node_ids)
        self._ensure_capacity(node + 1)
        self._vectors[node] = vector
        self._alive[node] = True
        self._levels[node] = level
        self._node_ids.append(record_id)
        self._payloads.append(payload)
        self._links.append([[] for _ in range(level + 1)])
        return node

    def _connect(self, node: int, level: int) -> None:
        """Wire a freshly allocated node into the graph (paper Algorithm 1)."""
        query = self._vectors[node]
        assert self._entry_point is not None  # guaranteed by insert()

        ep = self._entry_point
        for layer in range(self._max_level, level, -1):
            ep = self._search_layer(query, [ep], 1, layer)[0][1]

        eps = [ep]
        for layer in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(
                query, eps, self._config.ef_construction, layer
            )
            candidates = self._maybe_extend(query, candidates, layer)
            selected = self._select_from_pairs(query, candidates, self._config.m)

            self._links[node][layer] = list(selected)
            m_max = (
                self._config.m * _LAYER0_M_MAX_FACTOR
                if layer == 0
                else self._config.m
            )
            for other in selected:
                other_links = self._links[other][layer]
                other_links.append(node)
                if len(other_links) > m_max:
                    self._shrink(other, layer, m_max)
            # Paper: W feeds the next lower layer -- as bare node indices.
            eps = [node for _, node in candidates]

        if level > self._max_level:
            self._entry_point = node
            self._max_level = level

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Best-first beam search within one layer (paper Algorithm 2).

        Returns at most ``ef`` ``(distance, node)`` pairs sorted ascending;
        ties break by node index, keeping runs deterministic.
        """
        visited = set(entry_points)
        entry_dists = self._dist_fn(query, self._vectors[entry_points])

        candidates: list[tuple[float, int]] = []  # min-heap by distance
        results: list[tuple[float, int]] = []  # max-heap via negated distance
        for entry_dist, node in zip(
            (float(d) for d in entry_dists), entry_points
        ):
            heapq.heappush(candidates, (entry_dist, node))
            heapq.heappush(results, (-entry_dist, node))

        while candidates:
            dist_c, node_c = heapq.heappop(candidates)
            if dist_c > -results[0][0] and len(results) >= ef:
                break  # closest remaining candidate is worse than the beam
            fresh = [
                nb for nb in self._links[node_c][layer] if nb not in visited
            ]
            if not fresh:
                continue
            visited.update(fresh)
            fresh_dists = self._dist_fn(query, self._vectors[fresh])
            for dist_n, node_n in zip((float(d) for d in fresh_dists), fresh):
                if len(results) < ef:
                    heapq.heappush(candidates, (dist_n, node_n))
                    heapq.heappush(results, (-dist_n, node_n))
                elif dist_n < -results[0][0]:
                    heapq.heappush(candidates, (dist_n, node_n))
                    heapq.heapreplace(results, (-dist_n, node_n))

        return sorted((-neg_dist, node) for neg_dist, node in results)

    def _maybe_extend(
        self,
        query: np.ndarray,
        pairs: list[tuple[float, int]],
        layer: int,
    ) -> list[tuple[float, int]]:
        """Apply the paper's extendCandidates flag (off by default)."""
        if not self._config.extend_candidates:
            return pairs
        seen = {node for _, node in pairs}
        frontier = [node for _, node in pairs]
        for node in frontier:
            for nb in self._links[node][layer]:
                if nb not in seen:
                    seen.add(nb)
                    frontier.append(nb)
        dists = self._dist_fn(query, self._vectors[frontier])
        extended = [
            (float(dist), node) for dist, node in zip(dists, frontier)
        ]
        return sorted(extended)

    def _select_from_pairs(
        self,
        query: np.ndarray,
        pairs: list[tuple[float, int]],
        m: int,
    ) -> list[int]:
        """Run the diversity heuristic over ``(distance, node)`` pairs."""
        nodes = [node for _, node in pairs]
        indices = select_neighbors_heuristic(
            query,
            self._vectors[nodes],
            m,
            self._dist_fn,
            keep_pruned_connections=True,
        )
        return [nodes[i] for i in indices]

    def _shrink(self, node: int, layer: int, m_max: int) -> None:
        """Re-select a node's links after overflow (paper's shrink step)."""
        links = self._links[node][layer]
        kept = select_neighbors_heuristic(
            self._vectors[node],
            self._vectors[links],
            m_max,
            self._dist_fn,
            keep_pruned_connections=True,
        )
        self._links[node][layer] = [links[i] for i in kept]
