"""Public data models shared across layers."""

import math

from pydantic import BaseModel, Field, JsonValue, field_validator


class VectorRecord(BaseModel):
    """A single stored vector with its identity and optional metadata.

    Dimensionality is not enforced here: the index layer validates it
    against ``IndexConfig.dim`` and raises ``DimensionMismatchError``.
    """

    id: str = Field(min_length=1, description="Unique caller-provided identifier.")
    vector: list[float] = Field(description="Embedding components; must be finite.")
    payload: dict[str, JsonValue] | None = Field(
        default=None,
        description="Arbitrary JSON-serializable metadata returned with search results.",
    )

    @field_validator("vector")
    @classmethod
    def _validate_finite_components(cls, value: list[float]) -> list[float]:
        """Reject empty vectors and non-finite components (NaN / inf)."""
        if not value:
            raise ValueError("vector must contain at least one component")
        for component in value:
            if not math.isfinite(component):
                raise ValueError(f"vector components must be finite, got {component!r}")
        return value


class SearchResult(BaseModel):
    """One nearest-neighbour hit.

    ``score`` always carries a *distance*: lower means closer, for both
    metrics (cosine is reported as cosine distance ``1 - similarity``).
    """

    id: str
    score: float
    payload: dict[str, JsonValue] | None = None
