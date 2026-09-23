"""Infraestructura de evaluación del benchmark v2, sin cargar modelos."""

from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from common.config import get_dataset_paths
from common.eval.retrieval_metrics import evaluate_rankings
from dani.experiments.benchmark.benchmark_v2 import (
    BENCHMARK_NAME,
    BENCHMARK_PATH,
    FROZEN_SHA256,
    BenchmarkValidationError,
    verify_frozen_benchmark,
)
from dani.experiments.config import (
    ExperimentConfig,
    e1_bge_large_config,
    e2_e5_large_v2_config,
    e3_qwen3_embedding_06b_config,
)
from dani.experiments.evaluation import (
    EVALUATION_VIEWS,
    BenchmarkV2Evaluator,
    EvidenceContainmentError,
    evaluate_benchmark_views,
)
from dani.experiments.runner import (
    BENCHMARK_V2,
    PILOT,
    ExperimentRunner,
    IndexProvenanceError,
    default_output_path,
    sha256_file,
    validate_index_options,
    validate_reused_index,
)
from dani.experiments.retriever import SearchOutcome


class EmptyRetriever:
    """Retriever sintético: permite probar el contrato sin embeddings."""

    def search(self, *args, **kwargs) -> SearchOutcome:
        return SearchOutcome([], 0.01, 0.02, 0.03, 0.06)


class FrozenBenchmarkTests(unittest.TestCase):
    def test_runtime_verifier_accepts_only_the_frozen_bytes(self) -> None:
        self.assertEqual(verify_frozen_benchmark(), FROZEN_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "changed.jsonl"
            changed.write_bytes(BENCHMARK_PATH.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                BenchmarkValidationError, "SHA-256 de benchmark inválido"
            ):
                verify_frozen_benchmark(changed)

    def test_frozen_benchmark_has_48_questions(self) -> None:
        records = [
            json.loads(line)
            for line in BENCHMARK_PATH.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual(len(records), 48)
        self.assertEqual(len({row["question_id"] for row in records}), 48)


class EvidenceProjectionTests(unittest.TestCase):
    @staticmethod
    def metadata() -> pd.DataFrame:
        return pd.DataFrame([
            {
                "chunk_id": "contains-wide", "ticker": "AAA",
                "fiscal_year": 2025, "item": "7", "inicio_car": 0,
                "fin_car": 100,
            },
            {
                "chunk_id": "contains-tight", "ticker": "AAA",
                "fiscal_year": 2025, "item": "7", "inicio_car": 20,
                "fin_car": 50,
            },
            {
                "chunk_id": "partial", "ticker": "AAA",
                "fiscal_year": 2025, "item": "7", "inicio_car": 35,
                "fin_car": 70,
            },
            {
                "chunk_id": "other-item", "ticker": "AAA",
                "fiscal_year": 2025, "item": "8", "inicio_car": 0,
                "fin_car": 100,
            },
        ])

    @staticmethod
    def question() -> dict:
        return {
            "question_id": "q-multiple", "ticker": "AAA",
            "fiscal_year": 2025, "item": "7", "char_start": 25,
            "char_end": 45,
        }

    def test_mapping_keeps_every_full_containment_chunk(self) -> None:
        relevant, pool_size = BenchmarkV2Evaluator.relevant_chunks(
            self.question(), self.metadata()
        )
        self.assertEqual(relevant, ["contains-wide", "contains-tight"])
        self.assertEqual(pool_size, 3)

    def test_zero_full_containment_fails_closed(self) -> None:
        question = {**self.question(), "question_id": "q-zero",
                    "char_start": 101, "char_end": 110}
        with self.assertRaisesRegex(
            EvidenceContainmentError, "cero chunks con full containment"
        ):
            BenchmarkV2Evaluator.relevant_chunks(question, self.metadata())

    def test_canonical_48_all_have_containment_and_one_has_multiple(self) -> None:
        metadata = pd.read_parquet(get_dataset_paths().chunks_meta)
        evaluator = BenchmarkV2Evaluator(metadata)
        mappings = [
            evaluator.relevant_chunks(question, metadata)[0]
            for question in evaluator.questions
        ]
        self.assertEqual(len(mappings), 48)
        self.assertTrue(all(mappings))
        self.assertTrue(any(len(chunk_ids) > 1 for chunk_ids in mappings))

    def test_per_question_output_contains_derived_ground_truth(self) -> None:
        metadata = pd.read_parquet(get_dataset_paths().chunks_meta)
        evaluator = BenchmarkV2Evaluator(metadata)
        views, rows = evaluator.evaluate(EmptyRetriever())  # type: ignore[arg-type]
        self.assertEqual(len(rows), 48)
        self.assertEqual(views["ALL-48"]["n_questions"], 48)
        required = {
            "question_id", "query", "ticker", "fiscal_year", "item",
            "evidence_type", "evidence_span", "candidate_pool_size",
            "relevant_chunk_ids", "n_relevant_chunks", "ranking",
            "first_relevant_rank", "latency_s",
        }
        for row in rows:
            self.assertEqual(set(row), required)
            self.assertGreater(row["candidate_pool_size"], 0)
            self.assertEqual(
                row["n_relevant_chunks"], len(row["relevant_chunk_ids"])
            )
            self.assertGreater(row["n_relevant_chunks"], 0)


class MetricsAndViewsTests(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict]:
        rows = []
        for item in ("1A", "7", "7A", "8"):
            for number in range(12):
                relevant = [f"{item}-{number}-a", f"{item}-{number}-b"]
                rows.append({
                    "item": item,
                    "relevant_chunk_ids": relevant,
                    "ranking": [
                        {"chunk_id": "noise"},
                        {"chunk_id": relevant[1]},
                        {"chunk_id": relevant[0]},
                    ],
                })
        return rows

    def test_recall_and_mrr_use_first_of_multiple_relevant_chunks(self) -> None:
        metrics = evaluate_rankings(self.rows()[:1])
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["recall@10"], 1.0)
        self.assertEqual(metrics["mrr@10"], 0.5)

    def test_predefined_views_have_48_12_12_12_12_36_questions(self) -> None:
        views = evaluate_benchmark_views(self.rows())
        self.assertEqual(tuple(views), EVALUATION_VIEWS)
        self.assertEqual(views["ALL-48"]["n_questions"], 48)
        for item in ("1A", "7", "7A", "8"):
            self.assertEqual(views[f"ITEM-{item}-12"]["n_questions"], 12)
        self.assertEqual(views["NON-7A-36"]["n_questions"], 36)
        self.assertEqual(views["ITEM-7A-12"]["mrr@10"], 0.5)
        self.assertEqual(views["NON-7A-36"]["recall@3"], 1.0)


class SelectionAndNamingTests(unittest.TestCase):
    def test_runner_default_still_selects_the_six_question_pilot(self) -> None:
        metadata = pd.read_parquet(get_dataset_paths().chunks_meta)
        evaluator = ExperimentRunner(ExperimentConfig())._build_evaluator(metadata)
        self.assertEqual(len(evaluator.questions), 6)

    def test_pilot_and_v2_default_filenames_do_not_collide(self) -> None:
        configs = (
            ExperimentConfig(),
            e1_bge_large_config(),
            e2_e5_large_v2_config(),
            e3_qwen3_embedding_06b_config(),
        )
        for config in configs:
            pilot = default_output_path(config, PILOT)
            benchmark = default_output_path(config, BENCHMARK_V2)
            self.assertEqual(pilot.name, f"{config.experiment_id}.json")
            self.assertEqual(
                benchmark.name,
                f"{config.experiment_id}_benchmark_v2.json",
            )
            self.assertNotEqual(pilot, benchmark)

    def test_manifest_constants_are_frozen_and_predefined(self) -> None:
        self.assertEqual(BENCHMARK_NAME, "retrieval_benchmark_v2")
        self.assertEqual(FROZEN_SHA256, verify_frozen_benchmark())
        self.assertEqual(len(EVALUATION_VIEWS), 6)


class ReusedIndexProvenanceTests(unittest.TestCase):
    """Prueba la provenance FAISS con artefactos sintéticos ligeros."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.index_path = self.root / "corpus.faiss"
        index = faiss.IndexFlatIP(1024)
        vectors = np.zeros((2, 1024), dtype=np.float32)
        vectors[:, 0] = 1.0
        index.add(vectors)
        faiss.write_index(index, str(self.index_path))
        self.chunks_path = self.root / "chunks.jsonl"
        self.meta_path = self.root / "chunks_meta.parquet"
        self.chunks_path.write_bytes(b"chunk-0\nchunk-1\n")
        self.meta_path.write_bytes(b"ordered-metadata")
        self.manifest_path = self.root / "result.json"
        self.config = e3_qwen3_embedding_06b_config()
        self.base_manifest = {
            "manifest": {
                "identity": {"experiment_id": self.config.experiment_id},
                "config": self.config.to_dict(),
                "embedding": {
                    "model_name": self.config.model_name,
                    "model_revision": self.config.model_revision,
                    "dimension": self.config.expected_dimension,
                    "normalize": self.config.normalize_embeddings,
                    "document_formatting": self.config.document_format,
                },
                "index": {
                    "sha256": sha256_file(self.index_path),
                    "type": "IndexFlatIP",
                    "d": 1024,
                    "ntotal": 2,
                },
                "inputs": {
                    "n_chunks": 2,
                    "sha256": {
                        "chunks_jsonl": sha256_file(self.chunks_path),
                        "chunks_meta_parquet": sha256_file(self.meta_path),
                    },
                },
            },
            "metrics": {"must_not_be_used_for_provenance": True},
            "per_question": [{"must_not_be_used_for_provenance": True}],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(self, mutation=None) -> Path:
        payload = copy.deepcopy(self.base_manifest)
        if mutation is not None:
            mutation(payload["manifest"])
        self.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return self.manifest_path

    def validate(self, config=None) -> dict:
        return validate_reused_index(
            index_path=self.index_path,
            index_manifest_path=self.manifest_path,
            config=config or self.config,
            chunks_path=self.chunks_path,
            chunks_meta_path=self.meta_path,
            n_chunks=2,
        )

    def test_index_without_manifest_is_rejected(self) -> None:
        with self.assertRaises(IndexProvenanceError):
            validate_index_options(self.index_path, None)

    def test_manifest_without_index_is_rejected(self) -> None:
        with self.assertRaises(IndexProvenanceError):
            validate_index_options(None, self.manifest_path)

    def test_wrong_faiss_sha_is_rejected(self) -> None:
        self.write_manifest(
            lambda manifest: manifest["index"].update({"sha256": "0" * 64})
        )
        with self.assertRaisesRegex(IndexProvenanceError, "index.sha256"):
            self.validate()

    def test_same_1024_dimension_but_wrong_model_is_rejected(self) -> None:
        self.write_manifest()
        with self.assertRaisesRegex(IndexProvenanceError, "config.model_name"):
            self.validate(e1_bge_large_config())

    def test_wrong_revision_is_rejected(self) -> None:
        def mutate(manifest) -> None:
            manifest["config"]["model_revision"] = "wrong-revision"
            manifest["embedding"]["model_revision"] = "wrong-revision"

        self.write_manifest(mutate)
        with self.assertRaisesRegex(IndexProvenanceError, "model_revision"):
            self.validate()

    def test_wrong_chunks_jsonl_hash_is_rejected(self) -> None:
        self.write_manifest(lambda manifest: manifest["inputs"]["sha256"].update(
            {"chunks_jsonl": "0" * 64}
        ))
        with self.assertRaisesRegex(IndexProvenanceError, "chunks_jsonl"):
            self.validate()

    def test_wrong_chunks_metadata_hash_is_rejected(self) -> None:
        self.write_manifest(lambda manifest: manifest["inputs"]["sha256"].update(
            {"chunks_meta_parquet": "0" * 64}
        ))
        with self.assertRaisesRegex(IndexProvenanceError, "chunks_meta_parquet"):
            self.validate()

    def test_wrong_dimension_is_rejected(self) -> None:
        self.write_manifest(
            lambda manifest: manifest["index"].update({"d": 768})
        )
        with self.assertRaisesRegex(IndexProvenanceError, "index.d"):
            self.validate()

    def test_wrong_ntotal_is_rejected(self) -> None:
        self.write_manifest(
            lambda manifest: manifest["index"].update({"ntotal": 3})
        )
        with self.assertRaisesRegex(IndexProvenanceError, "index.ntotal"):
            self.validate()

    def test_valid_index_and_manifest_are_accepted(self) -> None:
        self.write_manifest()
        provenance = self.validate()
        self.assertEqual(provenance["index_sha256"], sha256_file(self.index_path))
        self.assertEqual(
            provenance["source_experiment_id"], self.config.experiment_id
        )
        self.assertEqual(provenance["source_model_name"], self.config.model_name)
        self.assertNotIn("metrics", provenance)
        self.assertNotIn("per_question", provenance)

    def test_no_index_keeps_the_normal_path(self) -> None:
        self.assertIsNone(validate_index_options(None, None))


if __name__ == "__main__":
    unittest.main()
