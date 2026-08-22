"""Custom binary serialisation for :class:`HNSWIndex` (Phase 4).

File layout (all integers little-endian, no padding, sections in fixed
order; locked in ARCHITECTURE.md):

    magic "VDB1" (4B)
    schema_version (u32)
    json_len (u64)
    metadata JSON (json_len bytes, UTF-8): config, node ids, payloads,
        insertion order, entry point (-1 encodes none), max_level, size
    vectors  n*dim * <f8   row-major, insertion order
    alive    n     * u8
    levels   n     * <i8
    adjacency_counts  L * <i8   per (node, layer) list, node-major
    adjacency_values  V * <i8   concatenated neighbour lists

Locked corruption contract: premature EOF raises ``PersistenceError`` with
the parse stage in the message; a foreign magic raises ``PersistenceError``
naming both expected and found bytes; an unsupported ``schema_version``
raises ``SchemaVersionError`` (strict reject, no migration path).
"""

from __future__ import annotations

import json
import os

import numpy as np
from pydantic import ValidationError

from vectordb.core.config import IndexConfig
from vectordb.core.exceptions import PersistenceError, SchemaVersionError
from vectordb.index.hnsw import HNSWIndex, IndexSnapshot

MAGIC = b"VDB1"
SCHEMA_VERSION = 1


def save_index(index: HNSWIndex, path: str | os.PathLike[str]) -> None:
    """Serialise ``index`` to ``path`` in the custom binary format.

    Reads a full snapshot through the index's persistence seam; the source
    index is never mutated. Output bytes are deterministic for a given
    build (sorted JSON keys, fixed-width little-endian arrays).
    """
    snapshot = index._snapshot_for_persistence()
    metadata = {
        "config": index.config.model_dump(mode="json"),
        "node_ids": snapshot.node_ids,
        "payloads": snapshot.payloads,
        "insertion_order": snapshot.insertion_order,
        "entry_point": snapshot.entry_point,
        "max_level": snapshot.max_level,
        "size": snapshot.size,
    }
    json_bytes = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    counts: list[int] = []
    values: list[int] = []
    for per_node in snapshot.links:
        for layer in per_node:
            counts.append(len(layer))
            values.extend(layer)

    sections = (
        np.ascontiguousarray(snapshot.vectors, dtype="<f8").tobytes(),
        np.asarray(snapshot.alive, dtype=np.uint8).tobytes(),
        np.asarray(snapshot.levels, dtype="<i8").tobytes(),
        np.asarray(counts, dtype="<i8").tobytes(),
        np.asarray(values, dtype="<i8").tobytes(),
    )

    try:
        with open(path, "wb") as stream:
            stream.write(MAGIC)
            stream.write(np.uint32(SCHEMA_VERSION).astype("<u4").tobytes())
            stream.write(np.uint64(len(json_bytes)).astype("<u8").tobytes())
            stream.write(json_bytes)
            for section in sections:
                stream.write(section)
    except OSError as exc:
        raise PersistenceError(f"failed to write {path!s}: {exc}") from exc


def load_index(path: str | os.PathLike[str]) -> HNSWIndex:
    """Deserialise an index previously written by :func:`save_index`.

    The graph is adopted verbatim from the file: levels, links and the
    entry pointer are exactly as saved (locked decisions #1 and #2).

    .. warning::
        RNG state is NOT restored -- post-load inserts will diverge in
        level assignment from an equivalent unsaved session, even with
        identical seed. After ``load()`` the level RNG restarts from
        ``IndexConfig.seed``, while a never-saved session keeps drawing
        from wherever its insert history left it. Searches on the loaded
        portion are unaffected; only *new* nodes may land on different
        layers than they would have without the save/load round-trip.

    Raises:
        PersistenceError: Unreadable file, foreign magic or any truncated/
            internally inconsistent byte stream (parse stage in message).
        SchemaVersionError: Unsupported ``schema_version`` (strict reject).
    """
    try:
        with open(path, "rb") as stream:
            raw = stream.read()
    except OSError as exc:
        raise PersistenceError(f"cannot read {path!s}: {exc}") from exc
    return _parse(raw, path)


class _Cursor:
    """Bounds-checked byte reader; every shortfall names its parse stage."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw
        self._pos = 0

    def take(self, count: int, stage: str) -> bytes:
        end = self._pos + count
        if end > len(self._raw):
            raise PersistenceError(
                f"truncated file: needed {count} bytes at {stage}, "
                f"have {len(self._raw) - self._pos}"
            )
        chunk = self._raw[self._pos : end]
        self._pos = end
        return chunk


def _parse(raw: bytes, path: str | os.PathLike[str]) -> HNSWIndex:
    cursor = _Cursor(raw)

    magic = cursor.take(4, "magic")
    if magic != MAGIC:
        raise PersistenceError(
            f"not a vectordb file: expected magic {MAGIC!r}, found {magic!r}"
        )

    version = int.from_bytes(cursor.take(4, "schema_version"), "little")
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(found=version, supported=SCHEMA_VERSION)

    json_len = int.from_bytes(cursor.take(8, "json_len"), "little")
    meta_bytes = cursor.take(json_len, "metadata_json")
    try:
        metadata = json.loads(meta_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersistenceError(f"corrupt metadata JSON in {path!s}: {exc}") from exc

    try:
        config = IndexConfig(**metadata["config"])
    except (KeyError, TypeError, ValidationError) as exc:
        raise PersistenceError(f"corrupt config section in {path!s}: {exc}") from exc

    node_ids = [str(rid) for rid in metadata["node_ids"]]
    payloads = metadata["payloads"]
    insertion_order = [str(rid) for rid in metadata["insertion_order"]]
    entry_point = int(metadata["entry_point"])
    max_level = int(metadata["max_level"])
    size = int(metadata["size"])
    n = len(node_ids)
    dim = config.dim

    if len(payloads) != n or len(insertion_order) > n:
        raise PersistenceError(
            f"inconsistent metadata lengths in {path!s}: "
            f"{n} ids, {len(payloads)} payloads, {len(insertion_order)} ordered"
        )
    if not -1 <= entry_point < n or not -1 <= max_level or not 0 <= size <= n:
        raise PersistenceError(f"inconsistent header fields in {path!s}")

    vectors = (
        np.frombuffer(cursor.take(n * dim * 8, "vectors"), dtype="<f8")
        .astype(np.float64, copy=True)
        .reshape(n, dim)
    )
    alive = np.frombuffer(cursor.take(n, "alive"), dtype=np.uint8) != 0
    levels = np.frombuffer(cursor.take(n * 8, "levels"), dtype="<i8").astype(
        np.int64, copy=True
    )

    total_lists = sum(int(level) + 1 for level in levels.tolist())
    counts = np.frombuffer(
        cursor.take(total_lists * 8, "adjacency_counts"), dtype="<i8"
    )
    if bool(np.any(counts < 0)):
        raise PersistenceError(f"negative adjacency count in {path!s}")
    total_values = int(counts.sum())
    values = np.frombuffer(
        cursor.take(total_values * 8, "adjacency_values"), dtype="<i8"
    )
    if bool(np.any(values < 0)) or bool(np.any(values >= max(n, 1))):
        raise PersistenceError(f"adjacency value out of range in {path!s}")

    links: list[list[list[int]]] = []
    flat_values = values.tolist()
    flat_counts = counts.tolist()
    count_pos = 0
    value_pos = 0
    for node_level in levels.tolist():
        node_layers: list[list[int]] = []
        for _ in range(int(node_level) + 1):
            width = int(flat_counts[count_pos])
            count_pos += 1
            node_layers.append(
                [int(nb) for nb in flat_values[value_pos : value_pos + width]]
            )
            value_pos += width
        links.append(node_layers)

    index = HNSWIndex(config)
    index._restore_from_persistence(
        IndexSnapshot(
            vectors=vectors,
            alive=alive,
            levels=levels,
            links=links,
            node_ids=node_ids,
            payloads=list(payloads),
            insertion_order=insertion_order,
            entry_point=entry_point,
            max_level=max_level,
            size=size,
        )
    )
    return index
