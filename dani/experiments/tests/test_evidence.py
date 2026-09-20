"""Tests deterministas del ground truth basado en spans de sección."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd

from common.config import get_dataset_paths
from dani.experiments.config import ExperimentConfig
from dani.experiments.evaluation import RetrievalEvaluator
from dani.experiments.evidence import (
    EvidenceMapper,
    EvidenceResolutionError,
    EvidenceSpan,
    build_ground_truth_artifact,
)


def section(text: str, ticker: str = "AAA") -> dict:
    return {"ticker": ticker, "fiscal_year": 2025, "item": "7", "texto": text}


def question(anchor: str, question_id: str = "q1") -> dict:
    return {
        "id": question_id,
        "ticker": "AAA",
        "fiscal_year": 2025,
        "item": "7",
        "ancla_texto": anchor,
    }


def chunk(
    chunk_id: str, source: str, start: int, end: int, ticker: str = "AAA"
) -> dict:
    return {
        "chunk_id": chunk_id,
        "ticker": ticker,
        "fiscal_year": 2025,
        "item": "7",
        "inicio_car": start,
        "fin_car": end,
        "texto": source[start:end],
    }


class EvidenceSpanTests(unittest.TestCase):
    def test_frozen_and_valid_offsets(self) -> None:
        span = EvidenceSpan(
            "q", "AAA", 2025, "7", "evidence", 2, 10, "exact_literal"
        )
        span.validate_against("xxevidenceyy")
        with self.assertRaises(FrozenInstanceError):
            span.char_start = 3  # type: ignore[misc]
        with self.assertRaises(ValueError):
            EvidenceSpan("q", "AAA", 2025, "7", "x", 1, 1, "exact_literal")
        with self.assertRaises(ValueError):
            EvidenceSpan("q", "AAA", 2025, "7", "x", -1, 0, "exact_literal")


class MatchingTests(unittest.TestCase):
    def test_unique_exact_match_and_offset_invariant(self) -> None:
        mapper = EvidenceMapper([section("before exact evidence after")], [])
        span = mapper.resolve(question("exact evidence"))
        self.assertEqual(span.matching_method, "exact_literal")
        self.assertEqual((span.char_start, span.char_end), (7, 21))
        self.assertEqual(
            mapper.section_for(span)["texto"][span.char_start:span.char_end],
            span.evidence_text,
        )

    def test_not_found_and_multiple_are_not_silently_resolved(self) -> None:
        mapper = EvidenceMapper([section("one repeated and repeated")], [])
        with self.assertRaises(EvidenceResolutionError) as multiple:
            mapper.resolve(question("repeated"))
        self.assertEqual(multiple.exception.status, "MULTIPLE")
        with self.assertRaises(EvidenceResolutionError) as missing:
            mapper.resolve(question("absent"))
        self.assertEqual(missing.exception.status, "NOT_FOUND")

    def test_whitespace_fallback_maps_back_to_original_source(self) -> None:
        source = "before evidence\n\twith   spacing after"
        mapper = EvidenceMapper([section(source)], [])
        span = mapper.resolve(question("evidence with spacing"))
        self.assertEqual(span.matching_method, "whitespace_normalized")
        self.assertEqual(span.source_text_span, "evidence\n\twith   spacing")
        span.validate_against(source)


class ProjectionTests(unittest.TestCase):
    def test_full_partial_outside_and_metadata_rules(self) -> None:
        source = "0123456789ABCDEFGHIJ"
        chunks = [
            chunk("full-1", source, 0, 10),
            chunk("full-2", source, 4, 12),
            chunk("partial-left", source, 3, 7),
            chunk("partial-right", source, 8, 14),
            chunk("outside", source, 10, 15),
            chunk("wrong-metadata", source, 0, 20, ticker="BBB"),
        ]
        mapper = EvidenceMapper([section(source), section(source, "BBB")], chunks)
        span = EvidenceSpan(
            "q", "AAA", 2025, "7", "56789", 5, 10, "exact_literal"
        )
        projection = mapper.project(span)
        self.assertEqual(
            projection["full_containment_chunk_ids"], ["full-1", "full-2"]
        )
        self.assertEqual(
            projection["partial_overlap_chunk_ids"],
            ["partial-left", "partial-right"],
        )
        self.assertEqual(projection["status"], "full_containment")


class RealDatasetTests(unittest.TestCase):
    def test_six_real_spans_have_full_containment_and_e0_parity(self) -> None:
        paths = get_dataset_paths()
        mapper = EvidenceMapper.from_dataset()
        self.assertEqual(mapper.validate_chunk_offsets(), 1749)
        metadata = pd.read_parquet(paths.chunks_meta)
        evaluator = RetrievalEvaluator(
            ExperimentConfig().golden_path, metadata, max_k=10
        )
        for current_question in evaluator.questions:
            span = mapper.resolve(current_question)
            projection = mapper.project(span)
            old = evaluator.relevant_chunk_ids(current_question)
            self.assertEqual(span.matching_method, "exact_literal")
            self.assertTrue(projection["full_containment_chunk_ids"])
            self.assertEqual(
                set(old), set(projection["full_containment_chunk_ids"])
            )
        self.assertEqual(len(evaluator.questions), 6)

    def test_artifact_summary_and_schema(self) -> None:
        artifact = build_ground_truth_artifact()
        summary = artifact["summary"]
        self.assertEqual(artifact["schema_version"], "1")
        self.assertEqual(summary["n_questions"], 6)
        self.assertEqual(summary["exact_unique_matches"], 6)
        self.assertEqual(summary["normalized_matches"], 0)
        self.assertEqual(summary["ambiguous_matches"], 0)
        self.assertEqual(summary["not_found"], 0)
        self.assertEqual(summary["evidence_with_full_containment"], 6)
        self.assertEqual(summary["split_evidence"], 0)
        self.assertEqual(summary["e0_relevant_set_parity"], 6)
        self.assertEqual(len(artifact["evidence"]), 6)


class ImportSafetyTests(unittest.TestCase):
    def test_import_does_not_load_embedding_model_or_faiss(self) -> None:
        environment = dict(os.environ)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        command = (
            "import sys; import dani.experiments.evidence; "
            "import dani.experiments.evaluation; "
            "assert 'sentence_transformers' not in sys.modules; "
            "assert 'faiss' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", command],
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()

