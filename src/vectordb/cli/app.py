"""Typer CLI for vectordb (Phase 5).

Thin wrapper only: every behavioural path goes through the public library
API (index, persistence, benchmark). Errors surface as friendly one-line
messages on stderr with exit code 1 -- never tracebacks.
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from vectordb.benchmark import ProgressFn, run_benchmark
from vectordb.core.config import DistanceMetric, IndexConfig
from vectordb.core.exceptions import (
    DimensionMismatchError,
    DuplicateIDError,
    EmptyIndexError,
    PersistenceError,
)
from vectordb.core.models import VectorRecord
from vectordb.index.hnsw import HNSWIndex
from vectordb.persistence import load_index, save_index

app = typer.Typer(
    help="vectordb -- build, search and benchmark a numpy-only HNSW index.",
    no_args_is_help=True,
)

# Soft (non-blocking) threshold: above it the CLI warns that brute-force
# ground truth may take a while but always proceeds.
_BENCHMARK_N_SOFT_LIMIT = 20_000


class MetricChoice(str, Enum):
    """Distance metric selector mirroring ``DistanceMetric`` values."""

    L2 = "l2"
    COSINE = "cosine"


def _fail(message: str) -> NoReturn:
    """Print a friendly one-line error and exit 1."""
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _to_distance_metric(choice: MetricChoice) -> DistanceMetric:
    return DistanceMetric(choice.value)


def _read_records(path: Path) -> list[VectorRecord]:
    """Parse a JSONL file into validated records.

    One record per line: ``{"id": str, "vector": [floats], "payload": {...}|null}``.
    Dimensionality is pinned by the first record; any later deviation is a
    hard, line-attributed error. Blank lines are skipped.

    Raises:
        typer.Exit: On malformed JSON, missing keys, validation failures or
            a dimension conflict (all via :func:`_fail`).
    """
    records: list[VectorRecord] = []
    dim: int | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("expected a JSON object per line")
                record = VectorRecord(
                    id=str(row["id"]),
                    vector=row["vector"],
                    payload=row.get("payload"),
                )
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValidationError,
            ) as exc:
                _fail(f"{path}: invalid JSONL at line {line_no}: {exc}")
            vector_len = len(record.vector)
            if dim is None:
                dim = vector_len
            elif vector_len != dim:
                _fail(
                    f"{path}: dimension mismatch at line {line_no}: "
                    f"expected {dim}, got {vector_len}"
                )
            records.append(record)
    return records


def _load_or_fail(path: Path) -> HNSWIndex:
    """Load a saved index, mapping persistence errors to friendly exits."""
    try:
        return load_index(path)
    except PersistenceError as exc:  # SchemaVersionError subclasses this
        _fail(str(exc))


def _parse_query(query: str) -> np.ndarray:
    """Parse ``"0.1,0.2"`` into a float64 vector."""
    try:
        return np.array(
            [float(part) for part in query.split(",")], dtype=np.float64
        )
    except ValueError:
        _fail(f"invalid --query {query!r}: expected comma-separated floats")


@app.command("build-index")
def build_index_command(
    input_path: Annotated[
        Path, typer.Option("--input", help="JSONL file: one {id, vector, payload} per line.")
    ],
    output: Annotated[Path, typer.Option("--output", help="Target .vdb file.")],
    metric: Annotated[MetricChoice, typer.Option("--metric")] = MetricChoice.L2,
    m: Annotated[int, typer.Option(min=2)] = 16,
    ef_construction: Annotated[int, typer.Option(min=1)] = 200,
    ef_search: Annotated[int, typer.Option(min=1)] = 50,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Build an HNSW index from a JSONL file and save it.

    Dimensionality is taken from the first record and validated across the
    whole file; metric and HNSW parameters are frozen into the saved config.
    """
    records = _read_records(input_path)
    if not records:
        _fail(f"{input_path}: no records found")
    config = IndexConfig(
        dim=len(records[0].vector),
        metric=_to_distance_metric(metric),
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        seed=seed,
    )
    index = HNSWIndex(config)
    for record in records:
        try:
            index.insert(record)
        except DuplicateIDError:
            _fail(f"{input_path}: duplicate id {record.id!r} in input")
    save_index(index, output)
    typer.echo(f"built index with {len(index)} vectors -> {output}")


@app.command()
def insert(
    index: Annotated[
        Path, typer.Option("--index", help="Existing .vdb file (updated in place).")
    ],
    input_path: Annotated[Path, typer.Option("--input", help="JSONL file with new records.")],
) -> None:
    """Insert JSONL records into an existing saved index."""
    loaded = _load_or_fail(index)
    for record in _read_records(input_path):
        if len(record.vector) != loaded.config.dim:
            _fail(
                f"{input_path}: dimension mismatch at id {record.id!r}: "
                f"index expects {loaded.config.dim}, got {len(record.vector)}"
            )
        try:
            loaded.insert(record)
        except DuplicateIDError:
            _fail(f"{input_path}: id {record.id!r} already exists in index")
    save_index(loaded, index)
    typer.echo(f"inserted into {index}; index now holds {len(loaded)} live vectors")


@app.command()
def search(
    index: Annotated[Path, typer.Option("--index", help="Saved .vdb file.")],
    query: Annotated[
        str, typer.Option("--query", help='Comma-separated floats, e.g. "0.1,0.2".')
    ],
    k: Annotated[int, typer.Option("-k", "--top-k", min=1)] = 10,
) -> None:
    """Search a saved index and print results as a table."""
    loaded = _load_or_fail(index)
    q = _parse_query(query)
    try:
        results = loaded.search(q, k)
    except (DimensionMismatchError, EmptyIndexError) as exc:
        _fail(str(exc))

    table = Table(title=f"top-{k} results", show_header=True)
    table.add_column("#", justify="right")
    table.add_column("id")
    table.add_column("score", justify="right")
    table.add_column("payload")
    for rank, result in enumerate(results, start=1):
        payload = (
            json.dumps(result.payload, ensure_ascii=False)
            if result.payload is not None
            else ""
        )
        table.add_row(str(rank), result.id, f"{result.score:.6g}", payload)
    Console().print(table)


@app.command()
def benchmark(
    n: Annotated[int, typer.Option("--n", min=1, help="Dataset size.")],
    dim: Annotated[int, typer.Option("--dim", min=1)] = 64,
    k: Annotated[int, typer.Option("-k", "--top-k", min=1)] = 10,
    queries: Annotated[int, typer.Option("--queries", min=1)] = 100,
    metric: Annotated[MetricChoice, typer.Option("--metric")] = MetricChoice.L2,
    seed: Annotated[int, typer.Option()] = 0,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Benchmark recall@k and latency against a brute-force baseline."""
    if n > _BENCHMARK_N_SOFT_LIMIT:
        typer.secho(
            f"warning: --n {n} exceeds {_BENCHMARK_N_SOFT_LIMIT}; "
            "brute-force ground truth may take a while",
            fg=typer.colors.YELLOW,
            err=True,
        )

    config = IndexConfig(dim=dim, metric=_to_distance_metric(metric), seed=seed)
    on_progress = _make_rich_progress() if _use_progress_bar(json_output) else None
    report = run_benchmark(
        config, n=n, k=k, n_queries=queries, seed=seed, on_progress=on_progress
    )

    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    table = Table(title="HNSW vs brute-force benchmark", show_header=True)
    table.add_column("metric")
    table.add_column("value", justify="right")
    rows: list[tuple[str, str]] = [
        ("n / dim / k / queries", f"{report.n} / {report.dim} / {report.k} / {report.n_queries}"),
        ("distance metric", report.metric.value),
        ("build time, s", f"{report.build_seconds:.3f}"),
        ("recall@k", f"{report.recall_at_k:.4f}"),
        ("mean query, ms", f"{report.mean_query_ms:.3f}"),
        ("p99 query, ms", f"{report.p99_query_ms:.3f}"),
        ("queries/s", f"{report.queries_per_second:.1f}"),
    ]
    for name, value in rows:
        table.add_row(name, value)
    Console().print(table)


def _use_progress_bar(json_output: bool) -> bool:
    """Rich progress only for interactive terminals; never for --json pipes."""
    return sys.stdout.isatty() and not json_output


def _make_rich_progress() -> ProgressFn:
    """Attach stage-aware Rich progress; returns an ``on_progress`` callback."""
    from rich.progress import Progress, TaskID

    progress = Progress()
    tasks: dict[str, TaskID] = {}

    def on_progress(stage: str, fraction: float) -> None:
        if stage not in tasks:
            tasks[stage] = progress.add_task(stage, total=1.0)
        progress.update(tasks[stage], completed=fraction)

    progress.start()
    return on_progress


def main() -> None:  # pragma: no cover - pure delegation to Typer's own
    # console-script plumbing; a test here would exercise typer, not us.
    """Console-script entry point."""
    app()
