from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from common.benchmark import BENCHMARK_PATH
from common.scripts import evaluate_query_rewrite_ablation as runner
from common.scripts import generate_query_rewrites as rewrites


class QueryRewriteAblationTests(unittest.TestCase):
    def _fixture(self, root: Path):
        questions = rewrites.load_frozen_benchmark()
        cache_path = root / "rewrites.jsonl"
        rows = []
        metadata_rows = []
        for question in questions:
            question_id = str(question["question_id"])
            rows.append({
                **rewrites._row_contract(question),
                "query_rewritten": f"rewrite::{question_id}",
                "status": "success",
                "fallback_used": False,
            })
            metadata_rows.append({
                "chunk_id": f"chunk::{question_id}",
                "ticker": question["ticker"],
                "fiscal_year": int(question["fiscal_year"]),
                "item": question["item"],
                "inicio_car": int(question["char_start"]),
                "fin_car": int(question["char_end"]),
            })
        cache_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        index = root / "corpus.faiss"
        chunks = root / "chunks.jsonl"
        chunks_meta = root / "chunks_meta.parquet"
        index.write_bytes(b"index")
        chunks.write_bytes(b"chunks")
        chunks_meta.write_bytes(b"metadata")
        paths = SimpleNamespace(
            faiss_index=index,
            chunks=chunks,
            chunks_meta=chunks_meta,
        )
        return questions, cache_path, pd.DataFrame(metadata_rows), paths

    def test_rewritten_mode_never_searches_original_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions, cache_path, metadata, paths = self._fixture(root)
            calls = []

            def fake_search(query, **kwargs):
                calls.append(query)
                question_id = query.removeprefix("rewrite::")
                return [{
                    "chunk_id": f"chunk::{question_id}",
                    "ticker": kwargs["ticker"],
                    "fiscal_year": kwargs["fiscal_year"],
                    "item": kwargs["item"],
                    "puntuacion": 0.9,
                }]

            output = root / "gemini_rewritten.json"
            with patch.dict(os.environ, {}, clear=False), patch(
                "common.config.get_dataset_paths", return_value=paths
            ):
                payload = runner.run_evaluation(
                    profile_name="gemini-embedding-2",
                    mode="rewritten",
                    cache_path=cache_path,
                    output_path=output,
                    benchmark_path=BENCHMARK_PATH,
                    metadata=metadata,
                    search_fn=fake_search,
                    provenance_fn=lambda: {"input_type_mode": "requested"},
                )

        self.assertEqual(len(calls), 48)
        self.assertEqual(calls, [f"rewrite::{q['question_id']}" for q in questions])
        self.assertNotIn("original", payload["manifest"]["variants"])
        self.assertTrue(payload["manifest"]["rewrite_generation"] is False)
        self.assertTrue(
            payload["manifest"]["retrieval"]["query_rewriting"] is True
        )
        self.assertEqual(len(payload["per_question"]), 48)
        self.assertNotIn("original", payload["per_question"][0])
        self.assertEqual(
            payload["metrics"]["rewritten"]["ALL-48"]["recall@5"], 1.0
        )

    def test_ablation_reports_paired_rank_changes(self):
        rows = [
            {
                "question_id": "improves",
                "original": {"first_relevant_rank": None},
                "rewritten": {"first_relevant_rank": 3},
            },
            {
                "question_id": "worsens",
                "original": {"first_relevant_rank": 2},
                "rewritten": {"first_relevant_rank": None},
            },
            {
                "question_id": "equal",
                "original": {"first_relevant_rank": 4},
                "rewritten": {"first_relevant_rank": 4},
            },
        ]
        paired = runner.paired_analysis(rows)
        self.assertEqual(
            paired["counts"],
            {
                "rank_improves": 1,
                "rank_worsens": 1,
                "rank_equal": 1,
                "enters_top5": 1,
                "leaves_top5": 1,
                "enters_top10": 1,
                "leaves_top10": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
