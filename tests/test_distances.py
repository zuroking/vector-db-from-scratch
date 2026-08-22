"""Analytic and behavioural tests for vectordb.distances."""

import math
import warnings
from collections.abc import Callable

import numpy as np
import pytest

from vectordb.core.exceptions import DimensionMismatchError
from vectordb.distances import cosine_distance, l2_distance

DistanceFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class TestL2Distance:
    def test_known_analytic_value(self) -> None:
        query = np.array([0.0, 0.0])
        matrix = np.array([[3.0, 4.0], [-6.0, 8.0]])

        result = l2_distance(query, matrix)

        assert result == pytest.approx([5.0, 10.0])

    def test_identical_vectors_have_zero_distance(self) -> None:
        query = np.array([1.0, 2.0, 3.0])
        matrix = np.array([[1.0, 2.0, 3.0], [-1.0, 5.0, 0.0]])

        result = l2_distance(query, matrix)

        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(math.sqrt(4.0 + 9.0 + 9.0))

    def test_matches_manual_computation_on_random_batch(self) -> None:
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(50, 8))
        query = rng.normal(size=8)

        result = l2_distance(query, matrix)

        expected = np.sqrt(((matrix - query) ** 2).sum(axis=1))
        assert result == pytest.approx(expected)

    def test_dimension_mismatch_raises(self) -> None:
        query = np.zeros(3)
        matrix = np.zeros((4, 5))

        with pytest.raises(DimensionMismatchError) as exc_info:
            l2_distance(query, matrix)

        assert exc_info.value.expected == 5
        assert exc_info.value.actual == 3

    def test_non_1d_query_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            l2_distance(np.zeros((2, 3)), np.zeros((4, 3)))

    def test_non_2d_matrix_rejected(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            l2_distance(np.zeros(3), np.zeros(3))

    def test_empty_matrix_returns_empty_result(self) -> None:
        result = l2_distance(np.ones(3), np.empty((0, 3)))

        assert result.shape == (0,)
        assert result.dtype == np.float64


class TestCosineDistance:
    def test_identical_vectors_have_near_zero_distance(self) -> None:
        query = np.array([1.0, 2.0, 3.0])
        matrix = np.array([[1.0, 2.0, 3.0], [-1.0, 5.0, 0.0]])

        result = cosine_distance(query, matrix)

        assert result[0] == pytest.approx(0.0, abs=1e-12)
        assert result[1] == pytest.approx(
            1.0 - (-1.0 + 10.0 + 0.0) / (math.sqrt(14.0) * math.sqrt(26.0))
        )

    def test_orthogonal_vectors_have_unit_distance(self) -> None:
        result = cosine_distance(np.array([1.0, 0.0]), np.array([[0.0, 1.0]]))

        assert result == pytest.approx([1.0])

    def test_opposite_vectors_have_maximal_distance(self) -> None:
        result = cosine_distance(np.array([1.0, 0.0]), np.array([[-1.0, 0.0]]))

        assert result == pytest.approx([2.0])

    def test_zero_query_yields_unit_distance_without_warnings(self) -> None:
        query = np.zeros(3)
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = cosine_distance(query, matrix)

        assert result == pytest.approx([1.0, 1.0])

    def test_zero_matrix_row_isolated_from_live_neighbours(self) -> None:
        query = np.array([1.0, 0.0])
        matrix = np.array([[2.0, 0.0], [0.0, 0.0]])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = cosine_distance(query, matrix)

        assert result == pytest.approx([0.0, 1.0])

    def test_dimension_mismatch_raises(self) -> None:
        query = np.zeros(7)
        matrix = np.zeros((2, 3))

        with pytest.raises(DimensionMismatchError) as exc_info:
            cosine_distance(query, matrix)

        assert exc_info.value.expected == 3
        assert exc_info.value.actual == 7

    def test_empty_matrix_returns_empty_result(self) -> None:
        result = cosine_distance(np.ones(3), np.empty((0, 3)))

        assert result.shape == (0,)
        assert result.dtype == np.float64


class TestPrecisionAndDtypes:
    @pytest.mark.parametrize("distance_fn", [l2_distance, cosine_distance])
    def test_float32_input_upcast_to_float64_result(
        self, distance_fn: DistanceFn
    ) -> None:
        query32 = np.array([1.0, 2.0], dtype=np.float32)
        matrix32 = np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.float32)

        result = distance_fn(query32, matrix32)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result == pytest.approx(distance_fn(query32.astype(np.float64),
                                                   matrix32.astype(np.float64)))
