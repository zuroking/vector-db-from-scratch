"""Typed exception hierarchy for the vector database."""


class VectorDBError(Exception):
    """Base class for every error raised by this library."""


class DimensionMismatchError(VectorDBError):
    """A vector or query dimensionality does not match ``IndexConfig.dim``."""

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"dimension mismatch: index expects {expected}, got {actual}")


class DuplicateIDError(VectorDBError):
    """An insert references an id that already exists in the index."""

    def __init__(self, record_id: str) -> None:
        self.record_id = record_id
        super().__init__(f"duplicate vector id: {record_id!r}")


class IDNotFoundError(VectorDBError):
    """A delete or lookup references an id absent from the index."""

    def __init__(self, record_id: str) -> None:
        self.record_id = record_id
        super().__init__(f"id not found: {record_id!r}")


class EmptyIndexError(VectorDBError):
    """A search was requested on an index holding zero live vectors."""


class PersistenceError(VectorDBError):
    """Saving or loading an index failed."""


class SchemaVersionError(PersistenceError):
    """A saved file carries an unsupported ``schema_version``.

    Strict reject by design (locked decision): there is deliberately no
    migration path between format versions -- incompatible files require
    an explicit application-level re-export.
    """

    def __init__(self, found: int, supported: int) -> None:
        self.found = found
        self.supported = supported
        super().__init__(
            f"unsupported schema version {found}; this build supports {supported}"
        )
