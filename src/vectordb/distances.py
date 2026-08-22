"""Vectorised distance implementations (Phase 1).

Both functions share one canonical batched form: a single ``query`` vector
against a ``(n, dim)`` matrix of stored vectors, computed in float64 so that
results are independent of input dtype and reproducible bit-for-bit.
"""

from __future__ import annotations

import numpy as np

from vectordb.core.exceptions import DimensionMismatchError


def _prepare_query(vector: np.ndarray) -> np.ndarray:
    """Normalise a query to a 1-D float64 array."""
    prepared = np.asarray(vector, dtype=np.float64)
    if prepared.ndim != 1:
        raise ValueError(f"query must be 1-D, got {prepared.ndim}-D")
    return prepared


def _prepare_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalise a stored-vector batch to a 2-D float64 array."""
    prepared = np.asarray(matrix, dtype=np.float64)
    if prepared.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got {prepared.ndim}-D")
    return prepared


def _check_dimensions(query: np.ndarray, matrix: np.ndarray) -> None:
    """Raise unless query length equals matrix width."""
    if query.shape[0] != matrix.shape[1]:
        raise DimensionMismatchError(expected=matrix.shape[1], actual=query.shape[0])


def l2_distance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Euclidean (L2) distance from ``query`` to every row of ``matrix``.

    Complexity: O(n * dim) time, O(n * dim) temporary memory.

    Args:
        query: 1-D array of shape ``(dim,)``.
        matrix: 2-D array of shape ``(n, dim)``.

    Returns:
        float64 array of shape ``(n,)`` with distances.

    Raises:
        DimensionMismatchError: If query length differs from matrix width.
        ValueError: If inputs have the wrong rank.
    """
    q = _prepare_query(query)
    m = _prepare_matrix(matrix)
    _check_dimensions(q, m)
    diff = m - q
    distances: np.ndarray = np.sqrt((diff * diff).sum(axis=1))
    return distances


def cosine_distance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine distance ``1 - cos(query, row)`` for every row of ``matrix``.

    Zero-vector convention: a zero query or zero row has no direction, so
    its similarity is defined as 0.0 (distance 1.0). This keeps search
    robust against degenerate vectors -- no NaN, no RuntimeWarning.

    Complexity: O(n * dim) time, dominated by the matrix-vector product.

    Args:
        query: 1-D array of shape ``(dim,)``.
        matrix: 2-D array of shape ``(n, dim)``.

    Returns:
        float64 array of shape ``(n,)`` with distances.

    Raises:
        DimensionMismatchError: If query length differs from matrix width.
        ValueError: If inputs have the wrong rank.
    """
    q = _prepare_query(query)
    m = _prepare_matrix(matrix)
    _check_dimensions(q, m)

    q_norm = float(np.linalg.norm(q))
    row_norms = np.linalg.norm(m, axis=1)

    # Replace zero norms by 1 only as a division guard; the affected
    # similarities are overwritten below via the degenerate mask.
    safe_q_norm = q_norm if q_norm > 0.0 else 1.0
    safe_row_norms = np.where(row_norms > 0.0, row_norms, 1.0)

    sims = (m @ q) / (safe_row_norms * safe_q_norm)
    degenerate = (row_norms == 0.0) | (q_norm == 0.0)
    sims = np.where(degenerate, 0.0, sims)
    # Floating point can push |sim| epsilon-above 1; clip keeps distance in [0, 2].
    return 1.0 - np.clip(sims, -1.0, 1.0)
