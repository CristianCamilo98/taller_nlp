"""Tests unitarios y de integridad del experimento E0, sin cargar BGE."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import pandas as pd

from common.benchmark_config import BENCHMARK
from common.config import get_dataset_paths
from dani.experiments.config import EXPECTED_E0_METRICS, ExperimentConfig
from dani.experiments.embeddings import BgeV15Adapter
from dani.experiments.evaluation import RetrievalEvaluator
from dani.experiments.retriever import DenseFaissRetriever, SearchOutcome
from dani.experiments.runner import ExperimentRunner, create_manifest


class FakeEncoder:
    def __init__(self, dimension: int = 3):
        self.dimension = dimension
        self.calls: list[dict[str, Any]] = []
        self.max_seq_length = 512
        self.device = "cpu"

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        self.calls.append({"texts": list(texts), **kwargs})
        base = np.zeros((len(texts), self.dimension), dtype=np.float32)
        base[:, 0] = 1.0
        return base


class StaticAdapter:
    def __init__(self, query_vector: Sequence[float]):
        self.query_vector = np.asarray([query_vector], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.query_vector

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise AssertionError("Este test no debe codificar documentos")

    def metadata(self) -> dict[str, Any]:
        return {}


class ConfigTests(unittest.TestCase):
    def test_e0_is_frozen_serializable_and_matches_common(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(config.model_name, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.model_name, BENCHMARK.embedding_model)
        self.assertEqual(config.query_prefix, BENCHMARK.embedding_query_prefix)
        self.assertTrue(config.normalize_embeddings)
        self.assertEqual(config.expected_dimension, 384)
        self.assertEqual(config.index_type, "IndexFlatIP")
        self.assertEqual(config.k_values, (1, 3, 5, 10))
        json.dumps(config.to_dict())
        with self.assertRaises(FrozenInstanceError):
            config.model_name = "otro"  # type: ignore[misc]


class EmbeddingAdapterTests(unittest.TestCase):
    def test_query_and_document_formatting_shape_and_normalization(self) -> None:
        config = ExperimentConfig(expected_dimension=3, embedding_batch_size=2)
        encoder = FakeEncoder(dimension=3)
        adapter = BgeV15Adapter(config, encoder=encoder)

        query = adapter.encode_query("hello")
        documents = adapter.encode_documents(["doc one", "doc two"])

        self.assertEqual(query.shape, (1, 3))
        self.assertEqual(documents.shape, (2, 3))
        self.assertEqual(
            encoder.calls[0]["texts"], [f"{config.query_prefix}hello"]
        )
        self.assertEqual(encoder.calls[1]["texts"], ["doc one", "doc two"])
        self.assertTrue(encoder.calls[0]["normalize_embeddings"])
        np.testing.assert_allclose(np.linalg.norm(documents, axis=1), 1.0)

    def test_invalid_dimension_fails_explicitly(self) -> None:
        adapter = BgeV15Adapter(
            ExperimentConfig(expected_dimension=2), encoder=FakeEncoder(3)
        )
        with self.assertRaisesRegex(ValueError, "Shape"):
            adapter.encode_query("query")


class DenseFaissRetrieverTests(unittest.TestCase):
    @staticmethod
    def _metadata() -> pd.DataFrame:
        return pd.DataFrame([
            {"chunk_id": "wrong", "ticker": "BBB", "fiscal_year": 2024,
             "item": "7", "posicion": 0, "texto": "wrong"},
            {"chunk_id": "right-1", "ticker": "AAA", "fiscal_year": 2024,
             "item": "7", "posicion": 1, "texto": "right one"},
            {"chunk_id": "right-2", "ticker": "AAA", "fiscal_year": 2024,
             "item": "7", "posicion": 2, "texto": "right two"},
        ])

    def test_global_ranking_then_metadata_filter_and_top_k(self) -> None:
        vectors = np.asarray([
            [1.0, 0.0],
            [0.9, np.sqrt(1.0 - 0.9**2)],
            [0.8, 0.6],
        ], dtype=np.float32)
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        config = ExperimentConfig(expected_dimension=2)
        retriever = DenseFaissRetriever(
            index, self._metadata(), StaticAdapter([1.0, 0.0]), config
        )

        outcome = retriever.search("q", "AAA", 2024, "7", k=2)

        self.assertEqual(
            [row["chunk_id"] for row in outcome.results],
            ["right-1", "right-2"],
        )
        self.assertEqual(len(outcome.results), 2)

    def test_ties_are_repeatable(self) -> None:
        index = faiss.IndexFlatIP(2)
        index.add(np.asarray([[1.0, 0.0]] * 3, dtype=np.float32))
        retriever = DenseFaissRetriever(
            index,
            self._metadata(),
            StaticAdapter([1.0, 0.0]),
            ExperimentConfig(expected_dimension=2),
        )
        first = retriever.search("q", None, None, None, 3)
        second = retriever.search("q", None, None, None, 3)
        self.assertEqual(
            [row["chunk_id"] for row in first.results],
            [row["chunk_id"] for row in second.results],
        )

    def test_incompatible_dimension_fails(self) -> None:
        index = faiss.IndexFlatIP(2)
        with self.assertRaisesRegex(ValueError, "Dimensión"):
            DenseFaissRetriever(
                index,
                self._metadata().iloc[:0],
                StaticAdapter([1.0, 0.0]),
                ExperimentConfig(expected_dimension=3),
            )


class FakeRetriever:
    def __init__(self, rankings: dict[str, list[str]], metadata: pd.DataFrame):
        self.rankings = rankings
        self.metadata = metadata.set_index("chunk_id")

    def search(
        self,
        query: str,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> SearchOutcome:
        results = []
        for score, chunk_id in enumerate(self.rankings[query][:k], 1):
            row = self.metadata.loc[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "score": 1.0 / score,
                "ticker": row["ticker"],
                "fiscal_year": int(row["fiscal_year"]),
                "item": row["item"],
                "posicion": int(row["posicion"]),
                "texto": row["texto"],
            })
        return SearchOutcome(results, 0.01, 0.01, 0.01, 0.03)


class RetrievalEvaluatorTests(unittest.TestCase):
    def test_metrics_first_rank_and_missing_retrieval(self) -> None:
        metadata = pd.DataFrame([
            {"chunk_id": "noise", "ticker": "AAA", "fiscal_year": 2024,
             "item": "7", "posicion": 0, "texto": "irrelevant"},
            {"chunk_id": "evidence-1", "ticker": "AAA", "fiscal_year": 2024,
             "item": "7", "posicion": 1, "texto": "literal alpha evidence"},
            {"chunk_id": "evidence-2", "ticker": "BBB", "fiscal_year": 2025,
             "item": "1A", "posicion": 0, "texto": "literal beta evidence"},
        ])
        questions = [
            {"id": "q1", "pregunta": "query one", "ticker": "AAA",
             "fiscal_year": 2024, "item": "7",
             "ancla_texto": "literal alpha evidence"},
            {"id": "q2", "pregunta": "query two", "ticker": "BBB",
             "fiscal_year": 2025, "item": "1A",
             "ancla_texto": "literal beta evidence"},
        ]
        rankings = {
            "query one": ["noise", "evidence-1"],
            "query two": ["noise"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            golden = Path(temporary) / "golden.jsonl"
            golden.write_text(
                "\n".join(json.dumps(row) for row in questions), encoding="utf-8"
            )
            evaluator = RetrievalEvaluator(golden, metadata, max_k=10)
            metrics, rows = evaluator.evaluate(  # type: ignore[arg-type]
                FakeRetriever(rankings, metadata)
            )

        self.assertEqual(rows[0]["first_relevant_rank"], 2)
        self.assertIsNone(rows[1]["first_relevant_rank"])
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@3"], 0.5)
        self.assertEqual(metrics["mrr@10"], 0.25)

    def test_real_golden_maps_six_questions_from_anchors(self) -> None:
        paths = get_dataset_paths()
        metadata = pd.read_parquet(paths.chunks_meta)
        evaluator = RetrievalEvaluator(
            ExperimentConfig().golden_path, metadata, max_k=10
        )
        relevant = [
            evaluator.relevant_chunk_ids(question)
            for question in evaluator.questions
        ]
        self.assertEqual(len(evaluator.questions), 6)
        self.assertTrue(all(chunks for chunks in relevant))


class ManifestAndParityTests(unittest.TestCase):
    def test_manifest_has_required_fields_and_keeps_unknown_as_none(self) -> None:
        config = ExperimentConfig()
        manifest = create_manifest(
            config=config,
            timestamp_utc="2026-09-20T00:00:00+00:00",
            input_hashes={"chunks_jsonl": "abc"},
            n_chunks=1749,
            n_questions=6,
            embedding_metadata={
                "model_name": config.model_name,
                "model_revision": None,
                "max_sequence_length": None,
                "effective_device": None,
            },
            index_metadata={
                "type": "IndexFlatIP", "d": 384, "ntotal": 1749,
                "sha256": "def", "file_size_bytes": 1,
            },
            timings={"model_load": 0.0},
            chunk_statistics={"n_chunks": 1749},
            parity={"passed": True},
        )
        self.assertIn("commit", manifest["identity"])
        self.assertIn("dirty_worktree", manifest["identity"])
        self.assertIsNone(manifest["embedding"]["model_revision"])
        self.assertIsNone(manifest["embedding"]["effective_device"])
        self.assertEqual(manifest["inputs"]["n_chunks"], 1749)

    def test_expected_e0_metrics_pass_and_a_change_fails(self) -> None:
        self.assertTrue(ExperimentRunner._parity(EXPECTED_E0_METRICS)["passed"])
        changed = dict(EXPECTED_E0_METRICS)
        changed["recall@1"] = 0.0
        self.assertFalse(ExperimentRunner._parity(changed)["passed"])


if __name__ == "__main__":
    unittest.main()

