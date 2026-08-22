"""Behavioural tests for binary persistence (Phase 4).

Locked decisions under test (architect clarifications, 2026-08-22):

1. ``levels``/``links``/entry point are restored verbatim from the file --
   no RNG replay, no structural recomputation on load.
2. A tombstoned entry point at save time is still the entry point after
   load.
3. ``schema_version`` mismatch is a strict reject via ``SchemaVersionError``
   (no migration path).
4. Corruption surfaces as ``PersistenceError``: truncation errors name the
   parse stage; a foreign magic names both expected and found bytes.
"""

from pathlib import Path

import json
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import JsonValue

from vectordb import (
    DistanceMetric,
    EmptyIndexError,
    IndexConfig,
    PersistenceError,
    SchemaVersionError,
    VectorRecord,
)
from vectordb.index.hnsw import HNSWIndex
from vectordb.persistence import MAGIC, SCHEMA_VERSION, load_index, save_index

_HEADER_LEN = 4 + 4 + 8  # magic + schema_version + json_len


def build_index(n: int = 24, dim: int = 4) -> HNSWIndex:
    """Deterministic small index: fixed config, seeded vectors, mixed payloads."""
    config = IndexConfig(
        dim=dim,
        metric=DistanceMetric.L2,
        m=4,
        ef_construction=16,
        ef_search=8,
        seed=7,
    )
    index = HNSWIndex(config)
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(n, dim))
    for i, vector in enumerate(vectors):
        payload: dict[str, JsonValue] | None = None
        if i % 3 == 0:
            payload = {"i": i, "tags": [f"t{i}", "x"], "meta": {"ok": True}}
        index.insert(
            VectorRecord(id=f"id{i:02d}", vector=vector.tolist(), payload=payload)
        )
    return index


def topology(index: HNSWIndex) -> tuple[object, ...]:
    """Order-sensitive structural fingerprint (same idiom as test_hnsw)."""
    return (
        index.entry_point_id,
        len(index),
        tuple(index.ids),
        tuple(
            (
                rid,
                index.level_of(rid),
                tuple(
                    tuple(index.neighbor_ids(rid, layer))
                    for layer in range(index.level_of(rid) + 1)
                ),
            )
            for rid in index.ids
        ),
    )


class TestRoundTrip:
    def test_config_survives(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"

        save_index(index, path)
        loaded = load_index(path)

        assert isinstance(loaded, HNSWIndex)
        assert loaded.config == index.config

    def test_topology_and_search_are_bit_identical(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"
        query = np.array([0.5, -0.25, 1.0, 0.0])

        save_index(index, path)
        loaded = load_index(path)

        assert topology(loaded) == topology(index)
        assert loaded.search(query, k=5) == index.search(query, k=5)

    def test_nested_payloads_survive(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"

        save_index(index, path)
        loaded = load_index(path)

        for result in loaded.search(np.zeros(4), k=len(index)):
            original = index.search(np.zeros(4), k=len(index))
            match = [r for r in original if r.id == result.id]
            assert match, f"loaded id {result.id!r} missing from original"
            assert result.payload == match[0].payload

    def test_tombstones_survive(self, tmp_path: Path) -> None:
        index = build_index()
        index.delete("id03")
        index.delete("id11")
        path = tmp_path / "index.vdb"
        query = np.zeros(4)

        save_index(index, path)
        loaded = load_index(path)

        assert len(loaded) == len(index) == 22
        assert loaded.contains("id03") is False
        assert loaded.contains("id11") is False
        assert topology(loaded) == topology(index)
        surviving = {r.id for r in loaded.search(query, k=100)}
        assert "id03" not in surviving and "id11" not in surviving

    def test_tombstoned_entry_point_single_record(self, tmp_path: Path) -> None:
        """Decision #2 in its purest form: the only node, deleted, still ep."""
        index = build_index(n=1)
        index.delete("id00")
        path = tmp_path / "index.vdb"

        assert index.entry_point_id == "id00"  # tombstoned, not re-selected

        save_index(index, path)
        loaded = load_index(path)

        assert loaded.entry_point_id == "id00"
        assert len(loaded) == 0
        with pytest.raises(EmptyIndexError):
            loaded.search(np.zeros(4), k=1)

    def test_tombstoned_entry_point_verbatim_multi(self, tmp_path: Path) -> None:
        index = build_index()
        index.delete("id05")
        path = tmp_path / "index.vdb"
        entry_before = index.entry_point_id

        save_index(index, path)
        loaded = load_index(path)

        assert loaded.entry_point_id == entry_before

    def test_empty_index_round_trips(self, tmp_path: Path) -> None:
        index = build_index(n=0)
        path = tmp_path / "empty.vdb"

        save_index(index, path)
        loaded = load_index(path)

        assert len(loaded) == 0
        assert loaded.entry_point_id is None
        assert loaded.ids == []
        with pytest.raises(EmptyIndexError):
            loaded.search(np.zeros(4), k=1)

    def test_insert_after_load_appends(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"
        fresh = [1.0, 1.0, 1.0, 1.0]

        save_index(index, path)
        loaded = load_index(path)
        loaded.insert(VectorRecord(id="post-load", vector=fresh))

        assert loaded.contains("post-load")
        assert len(loaded) == len(index) + 1
        assert loaded.search(np.array(fresh), k=1)[0].id == "post-load"

    def test_save_does_not_mutate_source_index(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"
        query = np.array([0.5, -0.25, 1.0, 0.0])
        before = (len(index), index.search(query, k=5), topology(index))

        save_index(index, path)

        assert (len(index), index.search(query, k=5), topology(index)) == before

    def test_saved_bytes_are_deterministic(self, tmp_path: Path) -> None:
        index = build_index()

        first = tmp_path / "a.vdb"
        second = tmp_path / "b.vdb"
        save_index(index, first)
        save_index(index, second)

        assert first.read_bytes() == second.read_bytes()
        assert first.read_bytes().startswith(MAGIC)


class TestCorruption:
    def test_wrong_magic_rejected_with_found_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "zip.vdb"
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 64)

        with pytest.raises(PersistenceError) as excinfo:
            load_index(path)

        message = str(excinfo.value)
        assert "not a vectordb file" in message
        assert "PK" in message

    def test_short_file_rejected_at_magic_stage(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.vdb"
        path.write_bytes(b"VDB")

        with pytest.raises(PersistenceError) as excinfo:
            load_index(path)

        message = str(excinfo.value)
        assert "truncated" in message
        assert "magic" in message

    def test_schema_version_mismatch_is_strict_reject(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"
        save_index(index, path)

        raw = bytearray(path.read_bytes())
        raw[4:8] = (SCHEMA_VERSION + 1).to_bytes(4, "little")
        path.write_bytes(bytes(raw))

        with pytest.raises(SchemaVersionError) as excinfo:
            load_index(path)

        assert excinfo.value.found == SCHEMA_VERSION + 1
        assert excinfo.value.supported == SCHEMA_VERSION

    def test_truncated_header_names_stage(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"
        save_index(index, path)
        path.write_bytes(path.read_bytes()[:10])  # cut inside json_len

        with pytest.raises(PersistenceError) as excinfo:
            load_index(path)

        message = str(excinfo.value)
        assert "truncated" in message
        assert "json_len" in message

    def test_truncated_json_section_names_stage(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"
        save_index(index, path)

        raw = path.read_bytes()
        json_len = int.from_bytes(raw[8:16], "little")
        path.write_bytes(raw[:_HEADER_LEN + json_len // 2])

        with pytest.raises(PersistenceError) as excinfo:
            load_index(path)

        message = str(excinfo.value)
        assert "truncated" in message
        assert "metadata" in message

    def test_truncated_array_section_names_stage(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"
        save_index(index, path)

        raw = path.read_bytes()
        json_len = int.from_bytes(raw[8:16], "little")
        path.write_bytes(raw[:_HEADER_LEN + json_len + 10])  # inside vectors

        with pytest.raises(PersistenceError) as excinfo:
            load_index(path)

        message = str(excinfo.value)
        assert "truncated" in message
        assert "vectors" in message

    def test_missing_file_raises_persistence_error(self, tmp_path: Path) -> None:
        with pytest.raises(PersistenceError):
            load_index(tmp_path / "does-not-exist.vdb")


def _rewrite_metadata(
    path: Path, mutate: Callable[[dict[str, JsonValue]], None]
) -> None:
    """Decode a saved file's metadata JSON, apply ``mutate``, re-encode in
    place (json_len adjusted; array sections untouched)."""
    raw = path.read_bytes()
    json_len = int.from_bytes(raw[8:16], "little")
    metadata = json.loads(raw[_HEADER_LEN:_HEADER_LEN + json_len].decode("utf-8"))
    mutate(metadata)
    new_json = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    path.write_bytes(
        raw[:8] + len(new_json).to_bytes(8, "little") + new_json + raw[_HEADER_LEN + json_len:]
    )


def _assemble_file(path: Path, *, counts: list[int], values: list[int]) -> None:
    """Hand-assemble a minimal valid single-node file around the given
    adjacency arrays, to exercise guards truncation of real files can't reach."""
    meta = {
        "config": IndexConfig(dim=1, metric=DistanceMetric.L2).model_dump(mode="json"),
        "node_ids": ["n0"],
        "payloads": [None],
        "insertion_order": ["n0"],
        "entry_point": 0,
        "max_level": 0,
        "size": 1,
    }
    meta_bytes = json.dumps(
        meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    sections = (
        np.array([[1.0]], dtype="<f8").tobytes(),   # vectors: n*dim
        np.array([1], dtype=np.uint8).tobytes(),    # alive
        np.array([0], dtype="<i8").tobytes(),       # levels: one node at layer 0
        np.array(counts, dtype="<i8").tobytes(),
        np.array(values, dtype="<i8").tobytes(),
    )
    path.write_bytes(
        MAGIC
        + SCHEMA_VERSION.to_bytes(4, "little")
        + len(meta_bytes).to_bytes(8, "little")
        + meta_bytes
        + b"".join(sections)
    )


class TestSaveFailure:
    def test_unwritable_target_raises_persistence_error(
        self, tmp_path: Path
    ) -> None:
        index = build_index()

        with pytest.raises(PersistenceError) as excinfo:
            save_index(index, tmp_path / "no-such-dir" / "index.vdb")

        assert "failed to write" in str(excinfo.value)


class TestSemanticCorruption:
    """Internally inconsistent bytes that are *not* mere truncation."""

    def test_undecodable_metadata_json_rejected(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"
        save_index(index, path)

        raw = bytearray(path.read_bytes())
        json_len = int.from_bytes(raw[8:16], "little")
        raw[_HEADER_LEN:_HEADER_LEN + json_len] = b"\xff" * json_len
        path.write_bytes(bytes(raw))

        with pytest.raises(PersistenceError) as excinfo:
            load_index(path)

        assert "corrupt metadata JSON" in str(excinfo.value)

    def test_invalid_config_section_rejected(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"
        save_index(index, path)

        def corrupt_config(meta: dict[str, JsonValue]) -> None:
            config = meta["config"]
            assert isinstance(config, dict)
            config["dim"] = -1

        _rewrite_metadata(path, corrupt_config)

        with pytest.raises(PersistenceError) as excinfo:
            load_index(path)

        assert "corrupt config section" in str(excinfo.value)

    def test_metadata_length_mismatch_rejected(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"
        save_index(index, path)

        def drop_payload(meta: dict[str, JsonValue]) -> None:
            payloads = meta["payloads"]
            assert isinstance(payloads, list)
            payloads.pop()

        _rewrite_metadata(path, drop_payload)

        with pytest.raises(PersistenceError) as excinfo:
            load_index(path)

        message = str(excinfo.value)
        assert "inconsistent metadata lengths" in message

    def test_entry_point_out_of_range_rejected(self, tmp_path: Path) -> None:
        index = build_index()
        path = tmp_path / "index.vdb"
        save_index(index, path)
        _rewrite_metadata(path, lambda meta: meta.update(entry_point=9999))

        with pytest.raises(PersistenceError) as excinfo:
            load_index(path)

        message = str(excinfo.value)
        assert "inconsistent header fields" in message

    def test_negative_adjacency_count_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "neg-count.vdb"
        _assemble_file(path, counts=[-1], values=[])

        with pytest.raises(PersistenceError) as excinfo:
            load_index(path)

        message = str(excinfo.value)
        assert "negative adjacency count" in message

    def test_adjacency_value_out_of_range_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad-value.vdb"
        _assemble_file(path, counts=[2], values=[0, 5])  # only node 0 exists

        with pytest.raises(PersistenceError) as excinfo:
            load_index(path)

        message = str(excinfo.value)
        assert "adjacency value out of range" in message
