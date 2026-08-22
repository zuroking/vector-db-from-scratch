"""Tests for VectorRecord and SearchResult validation."""

import math

import pytest
from pydantic import ValidationError

from vectordb import SearchResult, VectorRecord


class TestVectorRecord:
    def test_valid_record_fields(self) -> None:
        record = VectorRecord(id="a1", vector=[0.1, -2.0, 3.5], payload={"tag": "x"})

        assert record.id == "a1"
        assert record.vector == [0.1, -2.0, 3.5]
        assert record.payload == {"tag": "x"}

    def test_payload_defaults_to_none(self) -> None:
        record = VectorRecord(id="a1", vector=[1.0])

        assert record.payload is None

    def test_int_components_coerced_to_float(self) -> None:
        record = VectorRecord(id="a1", vector=[1, 2])

        assert record.vector == [1.0, 2.0]

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VectorRecord(id="", vector=[1.0])

    def test_empty_vector_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VectorRecord(id="a1", vector=[])

    def test_nan_component_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VectorRecord(id="a1", vector=[1.0, math.nan])

    def test_infinite_component_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VectorRecord(id="a1", vector=[math.inf, 1.0])

    def test_non_json_payload_rejected(self) -> None:
        raw = {"id": "a1", "vector": [1.0], "payload": {"bad": {1, 2}}}

        with pytest.raises(ValidationError):
            VectorRecord.model_validate(raw)


class TestSearchResult:
    def test_valid_result(self) -> None:
        result = SearchResult(id="a1", score=0.25, payload={"k": 1})

        assert result.id == "a1"
        assert result.score == 0.25
        assert result.payload == {"k": 1}

    def test_payload_defaults_to_none(self) -> None:
        result = SearchResult(id="a1", score=1.5)

        assert result.payload is None
