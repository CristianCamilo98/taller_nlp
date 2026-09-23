"""Pruebas deterministas del benchmark candidato del Bloque 6."""

from __future__ import annotations

import json
import hashlib
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from common.config import get_dataset_paths
from dani.experiments.benchmark.benchmark_v2 import (
    BENCHMARK_DIR,
    BENCHMARK_PATH,
    EVIDENCE_TYPES,
    EXPECTED_KEYS,
    FISCAL_YEARS,
    ITEMS,
    PILOT_IDS,
    RECORD_FIELDS,
    TICKERS,
    audit_sections,
    load_benchmark,
    load_sections,
    validate_benchmark,
)


FROZEN_SHA256 = (
    "6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class SectionInventoryTests(unittest.TestCase):
    def test_source_has_exactly_the_48_required_sections(self) -> None:
        audit = audit_sections()
        self.assertEqual(audit["n_sections"], 48)
        self.assertEqual(audit["n_unique_combinations"], 48)
        self.assertTrue(audit["expected_combinations_present"])
        self.assertEqual(audit["missing"], [])
        self.assertEqual(audit["unexpected"], [])
        self.assertEqual(audit["by_ticker"], {ticker: 8 for ticker in TICKERS})
        self.assertEqual(audit["by_year"], {year: 24 for year in FISCAL_YEARS})
        self.assertEqual(audit["by_item"], {item: 12 for item in ITEMS})


class CandidateArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_benchmark(BENCHMARK_PATH)
        cls.sections = load_sections()

    def test_schema_count_and_unique_non_pilot_ids(self) -> None:
        ids = [record["question_id"] for record in self.records]
        self.assertEqual(len(self.records), 48)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(set(ids) & PILOT_IDS)
        for record in self.records:
            self.assertEqual(set(record), RECORD_FIELDS)
            self.assertNotIn("chunk_id", record)

    def test_balanced_six_by_two_by_four_coverage(self) -> None:
        keys = [
            (record["ticker"], record["fiscal_year"], record["item"])
            for record in self.records
        ]
        self.assertEqual(set(keys), EXPECTED_KEYS)
        self.assertTrue(all(count == 1 for count in Counter(keys).values()))
        self.assertEqual(
            Counter(record["ticker"] for record in self.records),
            Counter({ticker: 8 for ticker in TICKERS}),
        )
        self.assertEqual(
            Counter(record["fiscal_year"] for record in self.records),
            Counter({year: 24 for year in FISCAL_YEARS}),
        )
        self.assertEqual(
            Counter(record["item"] for record in self.records),
            Counter({item: 12 for item in ITEMS}),
        )

    def test_exact_half_open_spans_are_unique(self) -> None:
        for record in self.records:
            key = (
                record["ticker"],
                record["fiscal_year"],
                record["item"],
            )
            source = self.sections[key]
            start = record["char_start"]
            end = record["char_end"]
            evidence = record["evidence_text"]
            self.assertEqual(source[start:end], evidence)
            self.assertEqual(end - start, len(evidence))
            self.assertEqual(source.count(evidence), 1)
            self.assertEqual(record["matching_method"], "exact_literal")

    def test_evidence_types_include_exactly_four_table_cases(self) -> None:
        counts = Counter(record["evidence_type"] for record in self.records)
        self.assertEqual(set(counts), EVIDENCE_TYPES)
        self.assertEqual(counts, Counter({
            "prose": 34,
            "numeric": 10,
            "table_adjacent": 4,
        }))

    def test_frozen_file_has_expected_sha256(self) -> None:
        digest = hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest().upper()
        self.assertEqual(digest, FROZEN_SHA256)

    def test_all_48_spans_have_full_chunk_containment(self) -> None:
        chunks_by_section: dict[tuple[str, int, str], list[dict]] = defaultdict(
            list
        )
        for chunk in read_jsonl(get_dataset_paths().chunks):
            key = (
                str(chunk["ticker"]),
                int(chunk["fiscal_year"]),
                str(chunk["item"]),
            )
            chunks_by_section[key].append(chunk)

        for record in self.records:
            key = (
                record["ticker"],
                record["fiscal_year"],
                record["item"],
            )
            containing = [
                chunk for chunk in chunks_by_section[key]
                if int(chunk["inicio_car"]) <= record["char_start"]
                and int(chunk["fin_car"]) >= record["char_end"]
            ]
            self.assertTrue(containing, record["question_id"])

    def test_questions_keep_spanish_query_form(self) -> None:
        for record in self.records:
            self.assertTrue(
                record["question"].startswith("¿"), record["question_id"]
            )
            self.assertTrue(
                record["question"].endswith("?"), record["question_id"]
            )

    def test_full_validator_passes_every_check(self) -> None:
        report = validate_benchmark()
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(all(
            count == 1 for count in report["uniqueness_counts"].values()
        ))

    def test_jsonl_is_the_only_data_source_and_has_no_model_dependency(
        self,
    ) -> None:
        module_source = (BENCHMARK_DIR / "benchmark_v2.py").read_text(
            encoding="utf-8"
        )
        benchmark_source = BENCHMARK_PATH.read_text(encoding="utf-8")
        self.assertEqual(BENCHMARK_PATH.name, "retrieval_benchmark_v2.jsonl")
        self.assertFalse(
            (BENCHMARK_DIR / "retrieval_benchmark_v2_candidate.jsonl").exists()
        )
        for removed_name in (
            "CandidateSpec",
            "CANDIDATE_SPECS",
            "build_candidate",
            "write_candidate",
            '"--write"',
            "dani-b6-",
        ):
            self.assertNotIn(removed_name, module_source)
        for dependency in (
            "dani.experiments.runner",
            "dani.experiments.embeddings",
            "dani.experiments.retriever",
        ):
            self.assertNotIn(dependency, module_source)
        for experiment_number in range(4):
            marker = f"e{experiment_number}_"
            self.assertNotIn(marker, module_source)
            self.assertNotIn(marker, benchmark_source)


if __name__ == "__main__":
    unittest.main()
