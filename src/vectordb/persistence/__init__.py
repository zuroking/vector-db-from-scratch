"""Persistence layer (Phase 4).

Custom binary format per architectural decision #3: magic header +
``schema_version``, followed by serialised numpy arrays (vectors, adjacency
lists, HNSW tower/layer assignments).
"""

from vectordb.persistence.binary import MAGIC, SCHEMA_VERSION, load_index, save_index

__all__ = ["MAGIC", "SCHEMA_VERSION", "load_index", "save_index"]
