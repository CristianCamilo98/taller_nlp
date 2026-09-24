from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from common.benchmark import BENCHMARK_PATH, FROZEN_SHA256
from common.scripts import evaluate_retrieval_benchmark as runner


class FrozenBenchmarkTests(unittest.TestCase):
    def test_benchmark_is_frozen_complete_and_balanced(self):
        self.assertEqual(runner.verify_frozen_benchmark(), FROZEN_SHA256)
        questions = runner.load_benchmark()
        self.assertEqual(len(questions), 48)
        counts = {
            item: sum(question["item"] == item for question in questions)
            for item in ("1A", "7", "7A", "8")
        }
        self.assertEqual(counts, {"1A": 12, "7": 12, "7A": 12, "8": 12})

    def test_unknown_question_id_fails_closed(self):
        with self.assertRaisesRegex(
            runner.RetrievalBenchmarkError, "question_id no encontrado"
        ):
            runner.select_questions(runner.load_benchmark(), ["missing"])


class CommonRunnerTests(unittest.TestCase):
    def test_one_question_uses_common_contract_and_persists_auditable_json(self):
        metadata = pd.DataFrame([
            {
                "chunk_id": "AAPL-2024-1A-other",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "item": "1A",
                "inicio_car": 0,
                "fin_car": 7000,
            },
            {
                "chunk_id": "AAPL-2024-1A-relevant",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "item": "1A",
                "inicio_car": 7000,
                "fin_car": 8000,
            },
        ])
        calls = []

        def fake_search(query, **kwargs):
            calls.append((query, kwargs))
            return [
                {
                    "chunk_id": "AAPL-2024-1A-other",
                    "ticker": "AAPL",
                    "fiscal_year": 2024,
                    "item": "1A",
                    "puntuacion": 0.9,
                },
                {
                    "chunk_id": "AAPL-2024-1A-relevant",
                    "ticker": "AAPL",
                    "fiscal_year": 2024,
                    "item": "1A",
                    "puntuacion": 0.8,
                },
            ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "corpus.faiss"
            chunks_meta = root / "chunks_meta.parquet"
            index.write_bytes(b"index")
            chunks_meta.write_bytes(b"metadata")
            paths = SimpleNamespace(
                faiss_index=index,
                chunks_meta=chunks_meta,
            )
            output = root / "smoke.json"
            provenance = {
                "provider": "openrouter",
                "model_name": "google/gemini-embedding-2",
                "input_type_mode": "requested",
            }
            with patch.dict(os.environ, {}, clear=False), patch(
                "common.config.get_dataset_paths", return_value=paths
            ):
                payload = runner.run_benchmark(
                    profile_name="gemini-embedding-2",
                    output_path=output,
                    benchmark_path=BENCHMARK_PATH,
                    question_ids=["dani-b6-001"],
                    metadata=metadata,
                    search_fn=fake_search,
                    provenance_fn=lambda: provenance,
                )
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0][1],
            {"ticker": "AAPL", "fiscal_year": 2024, "item": "1A", "k": 10},
        )
        self.assertEqual(payload["manifest"]["profile"], "gemini-embedding-2")
        self.assertEqual(
            payload["manifest"]["retrieval"],
            {
                "type": "dense_faiss_with_metadata_postfilter",
                "policy": "global rank + ticker/fiscal_year/item postfilter",
                "query_rewriting": False,
                "bm25": False,
                "reranking": False,
            },
        )
        self.assertEqual(payload["per_question"][0]["first_relevant_rank"], 2)
        self.assertEqual(
            payload["per_question"][0]["query_embedding_provenance"], provenance
        )
        self.assertEqual(persisted["manifest"]["n_questions"], 1)
        self.assertEqual(persisted["metrics"]["ALL-48"]["recall@5"], 1.0)


if __name__ == "__main__":
    unittest.main()
