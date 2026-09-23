"""Carga y validación genérica del benchmark de retrieval v2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

from common.config import get_dataset_paths


BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = BENCHMARK_DIR / "retrieval_benchmark_v2.jsonl"

TICKERS = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA")
FISCAL_YEARS = (2024, 2025)
ITEMS = ("1A", "7", "7A", "8")
EXPECTED_KEYS = frozenset(product(TICKERS, FISCAL_YEARS, ITEMS))
PILOT_IDS = frozenset(f"g3-{number:03d}" for number in range(8, 14))
EVIDENCE_TYPES = frozenset({"prose", "numeric", "table_adjacent"})
RECORD_FIELDS = frozenset({
    "question_id",
    "ticker",
    "fiscal_year",
    "item",
    "question",
    "evidence_text",
    "char_start",
    "char_end",
    "matching_method",
    "evidence_type",
})


class BenchmarkValidationError(ValueError):
    """El benchmark no satisface una o más invariantes."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_benchmark(path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    """Carga los registros del JSONL canónico sin transformarlos."""
    return _load_jsonl(path)


def load_sections(path: Path | None = None) -> dict[tuple[str, int, str], str]:
    """Carga las 48 secciones SEC y rechaza claves duplicadas."""
    source_path = path or get_dataset_paths().sections
    sections: dict[tuple[str, int, str], str] = {}
    for row in _load_jsonl(source_path):
        key = str(row["ticker"]), int(row["fiscal_year"]), str(row["item"])
        if key in sections:
            raise BenchmarkValidationError(f"Sección duplicada: {key}")
        sections[key] = str(row["texto"])
    return sections


def audit_sections(path: Path | None = None) -> dict[str, Any]:
    """Comprueba el inventario canónico 6 × 2 × 4."""
    sections = load_sections(path)
    keys = set(sections)
    return {
        "n_sections": len(sections),
        "n_unique_combinations": len(keys),
        "expected_combinations_present": keys == EXPECTED_KEYS,
        "missing": [list(key) for key in sorted(EXPECTED_KEYS - keys)],
        "unexpected": [list(key) for key in sorted(keys - EXPECTED_KEYS)],
        "by_ticker": dict(sorted(Counter(key[0] for key in keys).items())),
        "by_year": dict(sorted(Counter(key[1] for key in keys).items())),
        "by_item": dict(sorted(Counter(key[2] for key in keys).items())),
    }


def _record_key(record: dict[str, Any]) -> tuple[str, int, str] | None:
    ticker = record.get("ticker")
    year = record.get("fiscal_year")
    item = record.get("item")
    if (
        not isinstance(ticker, str)
        or not isinstance(year, int)
        or isinstance(year, bool)
        or not isinstance(item, str)
    ):
        return None
    return ticker, year, item


def validate_benchmark(
    benchmark_path: Path = BENCHMARK_PATH,
    source_path: Path | None = None,
    *,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Valida el JSONL contra las secciones sin depender del chunking."""
    records = load_benchmark(benchmark_path)
    sections = load_sections(source_path)

    ids = [record.get("question_id") for record in records]
    keys = [_record_key(record) for record in records]
    valid_keys = [key for key in keys if key is not None]
    key_counts = Counter(valid_keys)
    ticker_counts = Counter(key[0] for key in valid_keys)
    year_counts = Counter(key[1] for key in valid_keys)
    item_counts = Counter(key[2] for key in valid_keys)

    schema_results: list[bool] = []
    offset_results: list[bool] = []
    exact_results: list[bool] = []
    uniqueness_counts: dict[str, int] = {}
    for position, (record, key) in enumerate(zip(records, keys), 1):
        schema_results.append(set(record) == RECORD_FIELDS)
        source = sections.get(key, "") if key is not None else ""
        start = record.get("char_start")
        end = record.get("char_end")
        evidence = record.get("evidence_text")
        valid_offsets = (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end <= len(source)
        )
        offset_results.append(valid_offsets)
        exact_results.append(
            valid_offsets
            and isinstance(evidence, str)
            and bool(evidence)
            and source[start:end] == evidence
        )
        label = str(record.get("question_id", f"row-{position}"))
        uniqueness_counts[label] = (
            source.count(evidence) if isinstance(evidence, str) else 0
        )

    checks = {
        "exactly_48_questions": len(records) == 48,
        "valid_schema": all(schema_results),
        "unique_question_ids": (
            all(isinstance(question_id, str) and question_id for question_id in ids)
            and len(ids) == len(set(ids))
        ),
        "one_per_ticker_year_item": (
            set(key_counts) == EXPECTED_KEYS
            and all(count == 1 for count in key_counts.values())
        ),
        "eight_per_ticker": (
            set(ticker_counts) == set(TICKERS)
            and all(ticker_counts[ticker] == 8 for ticker in TICKERS)
        ),
        "twenty_four_per_year": (
            set(year_counts) == set(FISCAL_YEARS)
            and all(year_counts[year] == 24 for year in FISCAL_YEARS)
        ),
        "twelve_per_item": (
            set(item_counts) == set(ITEMS)
            and all(item_counts[item] == 12 for item in ITEMS)
        ),
        "known_evidence_types": all(
            record.get("evidence_type") in EVIDENCE_TYPES
            for record in records
        ),
        "exact_literal_matching_method": all(
            record.get("matching_method") == "exact_literal"
            for record in records
        ),
        "valid_half_open_offsets": all(offset_results),
        "evidence_exact_in_source": all(exact_results),
        "unique_evidence_in_section": all(
            count == 1 for count in uniqueness_counts.values()
        ),
        "no_chunk_id_dependency": all(
            "chunk_id" not in record for record in records
        ),
        "no_pilot_id_overlap": not (set(ids) & PILOT_IDS),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "questions": len(records),
            "by_ticker": dict(sorted(ticker_counts.items())),
            "by_year": dict(sorted(year_counts.items())),
            "by_item": dict(sorted(item_counts.items())),
            "by_evidence_type": dict(sorted(Counter(
                str(record.get("evidence_type")) for record in records
            ).items())),
        },
        "uniqueness_counts": uniqueness_counts,
    }
    if raise_on_error and not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise BenchmarkValidationError(f"Validaciones fallidas: {failed}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida el benchmark de retrieval v2"
    )
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    report = {
        "benchmark": str(args.benchmark.resolve()),
        "section_audit": audit_sections(args.source),
        "benchmark_validation": validate_benchmark(
            args.benchmark, args.source
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
