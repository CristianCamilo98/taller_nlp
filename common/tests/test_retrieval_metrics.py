from __future__ import annotations

import ast
import unittest
from pathlib import Path

from common.eval.retrieval_metrics import evaluate_rankings


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_and_mrr_at_10(self):
        rankings = [{
            "relevant_chunk_ids": ["gold"],
            "ranking": [{"chunk_id": "other"}, {"chunk_id": "gold"}],
        }]
        metrics = evaluate_rankings(rankings)
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["recall@10"], 1.0)
        self.assertEqual(metrics["mrr@10"], 0.5)
        self.assertNotIn("mrr", metrics)


class ComparativeSchemaTests(unittest.TestCase):
    def test_comparative_fields_and_percentage_description(self):
        schema = Path(__file__).resolve().parents[1] / "agent" / "schema.py"
        source = schema.read_text(encoding="utf-8")
        tree = ast.parse(source)
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        fields = {node.target.id for node in model.body
                  if isinstance(node, ast.AnnAssign)
                  and isinstance(node.target, ast.Name)}
        self.assertTrue({"valor_inicial", "valor_final", "delta", "porcentaje"}
                        .issubset(fields))
        self.assertIn("Variación porcentual relativa", source)
        self.assertIn("delta absoluto", source)
