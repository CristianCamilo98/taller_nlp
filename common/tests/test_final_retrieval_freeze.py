from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.scripts import freeze_final_retrieval as freeze


class FinalRetrievalFreezeTests(unittest.TestCase):
    @staticmethod
    def _bundle_from_final() -> dict:
        final = json.loads(freeze.DEFAULT_FINAL.read_text(encoding="utf-8"))
        manifest = final["manifest"]
        return {
            "faiss_sha256": manifest["faiss_sha256"],
            "chunks_sha256": manifest["chunks_sha256"],
            "chunks_meta_sha256": manifest["chunks_meta_sha256"],
            "index_type": manifest["faiss_index_type"],
            "dimension": manifest["embedding_dimension"],
            "ntotal": manifest["ntotal"],
            "model": manifest["gemini_model"],
        }

    def test_g1_is_explicitly_rewritten_only_and_uses_canonical_cache(self):
        g1 = json.loads(freeze.DEFAULT_G1.read_text(encoding="utf-8"))
        self.assertEqual(len(g1["per_question"]), 48)
        self.assertEqual(g1["manifest"]["mode"], "rewritten")
        self.assertEqual(g1["manifest"]["variants"], ["rewritten"])
        self.assertTrue(g1["manifest"]["retrieval"]["query_rewriting"])
        self.assertFalse(g1["manifest"]["rewrite_generation"])
        self.assertFalse(g1["manifest"]["retrieval"]["bm25"])
        self.assertFalse(g1["manifest"]["retrieval"]["reranking"])

    def test_freeze_regenerates_artifacts_byte_for_byte_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison = root / "comparison.json"
            final = root / "final.json"
            with patch(
                "socket.create_connection",
                side_effect=AssertionError("network access is forbidden"),
            ):
                freeze.freeze_final_retrieval(
                    comparison_path=comparison,
                    final_path=final,
                    bundle=self._bundle_from_final(),
                    normalize_g1=False,
                )
            self.assertEqual(
                comparison.read_bytes(), freeze.DEFAULT_COMPARISON.read_bytes()
            )
            self.assertEqual(final.read_bytes(), freeze.DEFAULT_FINAL.read_bytes())

    def test_primary_metric_selects_gemini_original(self):
        comparison = json.loads(
            freeze.DEFAULT_COMPARISON.read_text(encoding="utf-8")
        )
        metrics = comparison["metrics"]
        self.assertGreater(
            metrics["gemini_original"]["NON-7A-36"]["recall@5"],
            metrics["gemini_rewritten"]["NON-7A-36"]["recall@5"],
        )
        self.assertEqual(comparison["decision"]["selected_candidate"], "A")
        self.assertEqual(
            comparison["decision"]["decided_at_criterion"], "NON-7A R@5"
        )
        item8_original = metrics["gemini_original"]["ITEM-8-12"]
        item8_rewritten = metrics["gemini_rewritten"]["ITEM-8-12"]
        self.assertGreater(item8_original["recall@5"], item8_rewritten["recall@5"])
        self.assertGreater(item8_original["mrr@10"], item8_rewritten["mrr@10"])

    def test_final_contains_48_rows_and_six_recalculated_views(self):
        final = json.loads(freeze.DEFAULT_FINAL.read_text(encoding="utf-8"))
        self.assertEqual(len(final["per_question"]), 48)
        self.assertEqual(len({row["question_id"] for row in final["per_question"]}), 48)
        self.assertEqual(
            set(final["metrics"]),
            {
                "ALL-48",
                "NON-7A-36",
                "ITEM-1A-12",
                "ITEM-7-12",
                "ITEM-7A-12",
                "ITEM-8-12",
            },
        )
        self.assertTrue(final["manifest"]["metrics_recalculated_from_per_question"])
        self.assertFalse(final["manifest"]["api_calls_required_to_regenerate"])
        self.assertEqual(final["manifest"]["ntotal"], 1749)


if __name__ == "__main__":
    unittest.main()
