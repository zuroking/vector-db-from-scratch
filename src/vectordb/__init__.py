"""vectordb -- a vector database built from scratch (HNSW, numpy-only)."""

from vectordb.core.config import DistanceMetric, IndexConfig
from vectordb.core.exceptions import (
    DimensionMismatchError,
    DuplicateIDError,
    EmptyIndexError,
    IDNotFoundError,
    PersistenceError,
    SchemaVersionError,
    VectorDBError,
)
from vectordb.core.models import SearchResult, VectorRecord

__version__ = "0.1.0"

__all__ = [
    "DimensionMismatchError",
    "DistanceMetric",
    "DuplicateIDError",
    "EmptyIndexError",
    "IDNotFoundError",
    "IndexConfig",
    "PersistenceError",
    "SchemaVersionError",
    "SearchResult",
    "VectorDBError",
    "VectorRecord",
    "__version__",
]
