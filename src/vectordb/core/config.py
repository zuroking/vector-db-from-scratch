"""Index configuration models."""

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class DistanceMetric(StrEnum):
    """Distance metrics supported by an index."""

    COSINE = "cosine"
    L2 = "l2"


class IndexConfig(BaseModel):
    """Immutable configuration bound to a single index instance.

    One index fixes exactly one dimensionality and one distance metric at
    creation time (architectural decisions #5 and #6); neither changes over
    the index lifetime. HNSW defaults follow common production settings;
    the level multiplier is pinned to ``1 / ln(M)`` per the locked spec.
    """

    # Решение раздела 4: конфиг неизменяем после создания --
    # заморозка исключает мутации в обход валидации.
    model_config = ConfigDict(frozen=True)

    dim: PositiveInt
    metric: DistanceMetric

    m: int = Field(
        default=16,
        ge=2,
        description="Max bidirectional links per node per layer (HNSW ``M``).",
    )
    ef_construction: int = Field(
        default=200,
        ge=1,
        description="Dynamic candidate-list size during insertion.",
    )
    ef_search: int = Field(
        default=50,
        ge=1,
        description="Default dynamic candidate-list size during search.",
    )
    extend_candidates: bool = Field(
        default=False,
        description=(
            "SELECT-NEIGHBORS-HEURISTIC extendCandidates flag: enlarge the "
            "candidate pool with neighbours-of-candidates before pruning. "
            "Targets highly clustered data; off by default following "
            "common practice (hnswlib omits it as well)."
        ),
    )
    seed: int = Field(
        default=0,
        ge=0,
        description=(
            "Seed of the level-assignment RNG; deterministic by default "
            "so builds are reproducible (needed for bit-for-bit round-trip tests)."
        ),
    )

    @property
    def level_multiplier(self) -> float:
        """Multiplier of the exponential level distribution, fixed to ``1 / ln(M)``."""
        return 1.0 / math.log(self.m)
