"""Behavioural tests for the Typer CLI (Phase 5).

The CLI is a thin wrapper: every behavioural path goes through the public
library API (index, persistence, benchmark). Errors surface as friendly
one-line messages with exit code 1 -- never tracebacks.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from vectordb import VectorRecord
from vectordb.cli.app import _BENCHMARK_N_SOFT_LIMIT, app
from vectordb.core.config import DistanceMetric, IndexConfig
from vectordb.index.hnsw import HNSWIndex
from vectordb.persistence import SCHEMA_VERSION, load_index, save_index

runner = CliRunner()


def all_output(result: Result) -> str:
    """stdout + stderr regardless of the click version's mixing behaviour."""
    parts = [result.output]
    try:
        parts.append(result.stderr or "")
    except (ValueError, AttributeError):
        pass  # click <8.2 mixes stderr into output; nothing extra to read
    return "\n".join(parts)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    lines = [json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def base_rows(dim: int = 2) -> list[dict[str, object]]:
    return [
        {"id": "id00", "vector": [0.0] * dim, "payload": {"color": "red"}},
        {"id": "id01", "vector": [1.0] * dim},
        {"id": "id02", "vector": [10.0] * dim},
    ]


def invoke(args: list[str]) -> Result:
    return runner.invoke(app, args)


class TestBuildAndSearch:
    def test_build_then_search_round_trip(self, tmp_path: Path) -> None:
        data = write_jsonl(tmp_path / "records.jsonl", base_rows())
        out = tmp_path / "index.vdb"

        built = invoke(["build-index", "--input", str(data), "--output", str(out)])
        assert built.exit_code == 0, all_output(built)

        found = invoke(
            ["search", "--index", str(out), "--query", "1.0,1.0", "-k", "2"]
        )
        assert found.exit_code == 0, all_output(found)
        text = all_output(found)
        assert "id01" in text and "id00" in text
        assert text.index("id01") < text.index("id00")

    def test_payload_reaches_search_output(self, tmp_path: Path) -> None:
        data = write_jsonl(tmp_path / "records.jsonl", base_rows())
        out = tmp_path / "index.vdb"
        invoke(["build-index", "--input", str(data), "--output", str(out)])

        found = invoke(["search", "--index", str(out), "--query", "0,0", "-k", "1"])

        assert found.exit_code == 0
        assert "red" in all_output(found)

    def test_insert_appends_and_persists(self, tmp_path: Path) -> None:
        data = write_jsonl(tmp_path / "records.jsonl", base_rows())
        out = tmp_path / "index.vdb"
        invoke(["build-index", "--input", str(data), "--output", str(out)])
        extra = write_jsonl(
            tmp_path / "extra.jsonl",
            [{"id": "id03", "vector": [5.0, 5.0]}],
        )

        inserted = invoke(["insert", "--index", str(out), "--input", str(extra)])
        assert inserted.exit_code == 0, all_output(inserted)

        assert len(load_index(out)) == 4
        found = invoke(["search", "--index", str(out), "--query", "5.0,5.0", "-k", "1"])
        assert "id03" in all_output(found)


class TestFriendlyErrors:
    def test_missing_index_file(self, tmp_path: Path) -> None:
        result = invoke(
            ["search", "--index", str(tmp_path / "nope.vdb"), "--query", "0,0", "-k", "1"]
        )

        assert result.exit_code == 1
        text = all_output(result)
        assert "Traceback" not in text
        assert "nope.vdb" in text

    def test_wrong_magic_file(self, tmp_path: Path) -> None:
        bogus = tmp_path / "zip.vdb"
        bogus.write_bytes(b"PK\x03\x04" + b"\x00" * 32)

        result = invoke(["search", "--index", str(bogus), "--query", "0,0", "-k", "1"])

        assert result.exit_code == 1
        text = all_output(result)
        assert "Traceback" not in text
        assert "not a vectordb file" in text

    def test_schema_version_mismatch(self, tmp_path: Path) -> None:
        index = HNSWIndex(IndexConfig(dim=2, metric=DistanceMetric.L2))
        index.insert(VectorRecord(id="a", vector=[0.0, 0.0]))
        target = tmp_path / "old.vdb"
        save_index(index, target)
        raw = bytearray(target.read_bytes())
        raw[4:8] = (SCHEMA_VERSION + 1).to_bytes(4, "little")
        target.write_bytes(bytes(raw))

        result = invoke(["search", "--index", str(target), "--query", "0,0", "-k", "1"])

        assert result.exit_code == 1
        text = all_output(result)
        assert "Traceback" not in text
        assert "schema version" in text

    def test_malformed_jsonl_names_the_line(self, tmp_path: Path) -> None:
        data = tmp_path / "broken.jsonl"
        data.write_text('{"id": "a", "vector": [0.0]}\n{not json}\n', encoding="utf-8")
        out = tmp_path / "index.vdb"

        result = invoke(["build-index", "--input", str(data), "--output", str(out)])

        assert result.exit_code == 1
        text = all_output(result)
        assert "Traceback" not in text
        assert "line 2" in text

    def test_dimension_conflict_between_lines(self, tmp_path: Path) -> None:
        data = tmp_path / "ragged.jsonl"
        data.write_text(
            '{"id": "a", "vector": [0.0, 0.0]}\n'
            '{"id": "b", "vector": [1.0, 2.0, 3.0]}\n',
            encoding="utf-8",
        )
        out = tmp_path / "index.vdb"

        result = invoke(["build-index", "--input", str(data), "--output", str(out)])

        assert result.exit_code == 1
        text = all_output(result)
        assert "Traceback" not in text
        assert "dimension" in text.lower()


class TestBenchmarkCommand:
    def test_small_benchmark_runs_and_reports_recall(self) -> None:
        result = invoke(
            [
                "benchmark", "--n", "60", "--dim", "4",
                "-k", "5", "--queries", "5", "--seed", "1",
            ]
        )

        assert result.exit_code == 0, all_output(result)
        text = all_output(result)
        assert "recall" in text.lower()
        assert "l2" in text.lower()

    def test_json_flag_emits_parseable_report(self) -> None:
        result = invoke(
            [
                "benchmark", "--n", "60", "--dim", "4",
                "-k", "5", "--queries", "5", "--seed", "1", "--json",
            ]
        )

        assert result.exit_code == 0, all_output(result)
        payload = json.loads(all_output(result))
        assert payload["n"] == 60
        assert 0.0 <= payload["recall_at_k"] <= 1.0

    def test_soft_warning_above_limit_but_still_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import vectordb.cli.app as app_module

        monkeypatch.setattr(app_module, "_BENCHMARK_N_SOFT_LIMIT", 10)
        result = invoke(
            [
                "benchmark", "--n", "12", "--dim", "4",
                "-k", "3", "--queries", "2", "--seed", "1",
            ]
        )

        assert result.exit_code == 0, all_output(result)
        assert "may take" in all_output(result)

    def test_default_soft_limit_is_20000(self) -> None:
        assert _BENCHMARK_N_SOFT_LIMIT == 20_000


class TestInputParsingEdges:
    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        data = tmp_path / "gappy.jsonl"
        data.write_text(
            '\n{"id": "a", "vector": [0.0, 0.0]}\n\n'
            '{"id": "b", "vector": [1.0, 1.0]}\n',
            encoding="utf-8",
        )
        out = tmp_path / "index.vdb"

        result = invoke(
            ["build-index", "--input", str(data), "--output", str(out)]
        )

        assert result.exit_code == 0, all_output(result)
        assert len(load_index(out)) == 2

    def test_non_object_json_line_rejected(self, tmp_path: Path) -> None:
        data = tmp_path / "array.jsonl"
        data.write_text('{"id": "a", "vector": [0.0]}\n[1, 2]\n', encoding="utf-8")

        result = invoke(["build-index", "--input", str(data), "--output", str(tmp_path / "i.vdb")])

        assert result.exit_code == 1
        text = all_output(result)
        assert "Traceback" not in text
        assert "line 2" in text

    def test_empty_input_file_rejected(self, tmp_path: Path) -> None:
        data = tmp_path / "empty.jsonl"
        data.write_text("", encoding="utf-8")

        result = invoke(["build-index", "--input", str(data), "--output", str(tmp_path / "i.vdb")])

        assert result.exit_code == 1
        text = all_output(result)
        assert "no records" in text

    def test_duplicate_id_within_build_input_rejected(self, tmp_path: Path) -> None:
        data = write_jsonl(
            tmp_path / "dups.jsonl",
            [
                {"id": "same", "vector": [0.0, 0.0]},
                {"id": "same", "vector": [1.0, 1.0]},
            ],
        )

        result = invoke(["build-index", "--input", str(data), "--output", str(tmp_path / "i.vdb")])

        assert result.exit_code == 1
        text = all_output(result)
        assert "duplicate id" in text
        assert "'same'" in text


class TestInsertValidation:
    def test_dimension_mismatch_against_index_rejected(self, tmp_path: Path) -> None:
        data = write_jsonl(tmp_path / "records.jsonl", base_rows())
        out = tmp_path / "index.vdb"
        invoke(["build-index", "--input", str(data), "--output", str(out)])
        extra = write_jsonl(
            tmp_path / "wide.jsonl",
            [{"id": "w1", "vector": [1.0, 2.0, 3.0]}],
        )

        result = invoke(["insert", "--index", str(out), "--input", str(extra)])

        assert result.exit_code == 1
        text = all_output(result)
        assert "dimension mismatch" in text
        assert "index expects 2" in text

    def test_id_already_in_index_rejected(self, tmp_path: Path) -> None:
        data = write_jsonl(tmp_path / "records.jsonl", base_rows())
        out = tmp_path / "index.vdb"
        invoke(["build-index", "--input", str(data), "--output", str(out)])
        extra = write_jsonl(
            tmp_path / "clash.jsonl",
            [{"id": "id00", "vector": [5.0, 5.0]}],
        )

        result = invoke(["insert", "--index", str(out), "--input", str(extra)])

        assert result.exit_code == 1
        assert "already exists" in all_output(result)


class TestSearchValidation:
    def test_query_with_wrong_dimensionality(self, tmp_path: Path) -> None:
        data = write_jsonl(tmp_path / "records.jsonl", base_rows())
        out = tmp_path / "index.vdb"
        invoke(["build-index", "--input", str(data), "--output", str(out)])

        result = invoke(
            ["search", "--index", str(out), "--query", "0,0,0", "-k", "1"]
        )

        assert result.exit_code == 1
        text = all_output(result)
        assert "Traceback" not in text
        assert "dimension" in text.lower()

    def test_unparseable_query_fails_friendly(self, tmp_path: Path) -> None:
        data = write_jsonl(tmp_path / "records.jsonl", base_rows())
        out = tmp_path / "index.vdb"
        invoke(["build-index", "--input", str(data), "--output", str(out)])

        result = invoke(
            ["search", "--index", str(out), "--query", "abc,0", "-k", "1"]
        )

        assert result.exit_code == 1
        text = all_output(result)
        assert "Traceback" not in text
        assert "invalid --query" in text

    def test_search_on_empty_index(self, tmp_path: Path) -> None:
        index = HNSWIndex(IndexConfig(dim=2, metric=DistanceMetric.L2))
        out = tmp_path / "empty.vdb"
        save_index(index, out)

        result = invoke(["search", "--index", str(out), "--query", "0,0", "-k", "1"])

        assert result.exit_code == 1
        text = all_output(result)
        assert "Traceback" not in text
        assert "zero live records" in text


class TestProgressBarPath:
    def test_benchmark_runs_with_rich_progress_attached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import vectordb.cli.app as app_module

        # CliRunner stdout is never a tty; force the interactive branch so
        # _make_rich_progress() and its stage callbacks execute end-to-end.
        monkeypatch.setattr(app_module, "_use_progress_bar", lambda json_flag: True)
        result = invoke(
            [
                "benchmark", "--n", "30", "--dim", "4",
                "-k", "3", "--queries", "3", "--seed", "1",
            ]
        )

        assert result.exit_code == 0, all_output(result)
        assert "recall" in all_output(result).lower()
