"""Tests for IndexConfig and DistanceMetric validation."""

import math

import pytest
from pydantic import ValidationError

from vectordb import DistanceMetric, IndexConfig


class TestDistanceMetric:
    def test_metric_values(self) -> None:
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.L2.value == "l2"


class TestIndexConfigDefaults:
    def test_hnsw_defaults_match_locked_specification(self) -> None:
        config = IndexConfig(dim=128, metric=DistanceMetric.L2)

        assert config.m == 16
        assert config.ef_construction == 200
        assert config.ef_search == 50
        assert config.seed == 0
        assert config.extend_candidates is False

    def test_level_multiplier_is_one_over_ln_m(self) -> None:
        config = IndexConfig(dim=8, metric=DistanceMetric.COSINE)

        assert config.level_multiplier == pytest.approx(1.0 / math.log(16))


class TestIndexConfigValidation:
    def test_config_is_frozen(self) -> None:
        config = IndexConfig(dim=8, metric=DistanceMetric.COSINE)

        with pytest.raises(ValidationError):
            config.dim = 16  # type: ignore[misc]

    def test_zero_dim_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IndexConfig(dim=0, metric=DistanceMetric.L2)

    def test_negative_dim_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IndexConfig(dim=-3, metric=DistanceMetric.L2)

    @pytest.mark.parametrize("bad_m", [0, 1, -16])
    def test_m_below_two_rejected(self, bad_m: int) -> None:
        with pytest.raises(ValidationError):
            IndexConfig(dim=8, metric=DistanceMetric.L2, m=bad_m)

    @pytest.mark.parametrize("bad_value", [0, -1])
    def test_ef_construction_must_be_positive(self, bad_value: int) -> None:
        with pytest.raises(ValidationError):
            IndexConfig(dim=8, metric=DistanceMetric.L2, ef_construction=bad_value)

    def test_ef_search_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            IndexConfig(dim=8, metric=DistanceMetric.L2, ef_search=0)

    def test_unknown_metric_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IndexConfig(dim=8, metric="manhattan")  # type: ignore[arg-type]
