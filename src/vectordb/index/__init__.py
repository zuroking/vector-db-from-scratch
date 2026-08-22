"""Index structures (Phase 2).

Will host the HNSW implementation selected by architectural decision #1:
multi-layer navigation, greedy beam search with ``ef`` candidate lists,
heuristic neighbour selection, and soft-delete tombstones (#4).
"""
