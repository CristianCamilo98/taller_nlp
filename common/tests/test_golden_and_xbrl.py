from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import pandas as pd

from common.config import get_dataset_paths
from common.xbrl import format_xbrl_value

GOLDEN = Path(__file__).resolve().parents[1] / "golden_set" / "golden_set_grupo3.jsonl"


class GoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with GOLDEN.open(encoding="utf-8") as stream:
            cls.questions = [json.loads(line) for line in stream if line.strip()]
        cls.facts = pd.read_parquet(get_dataset_paths().xbrl_facts)
        cls.meta = pd.read_parquet(get_dataset_paths().chunks_meta)

    def _fact(self, question, year):
        return self.facts[
            (self.facts.ticker == question["ticker"])
            & (self.facts.fiscal_year.astype(int) == int(year))
            & (self.facts.concept == question["concept_xbrl"])
        ].iloc[0]

    def test_count_ids_and_families(self):
        self.assertEqual(len(self.questions), 20)
        self.assertEqual(len({q["id"] for q in self.questions}), 20)
        self.assertGreaterEqual(sum(q["familia"] == "comparativa"
                                    for q in self.questions), 6)

    def test_numeric_truth_matches_xbrl(self):
        for question in self.questions:
            if question["familia"] == "numerica":
                fact = self._fact(question, question["fiscal_year"])
                self.assertEqual(float(fact.value), question["cifra_esperada"])
                self.assertEqual(fact.unit, question["unidad"])
            elif question["familia"] == "comparativa":
                first = float(self._fact(question,
                    question["fiscal_year_inicio"]).value)
                last = float(self._fact(question,
                    question["fiscal_year_fin"]).value)
                self.assertEqual(first, question["valor_inicial_esperado"])
                self.assertEqual(last, question["valor_final_esperado"])
                self.assertEqual(last - first, question["delta_esperado"])
                self.assertTrue(math.isclose((last-first)/first*100,
                    question["porcentaje_esperado"], abs_tol=1e-12))
                self.assertEqual(question["cifra_esperada"],
                                 question["delta_esperado"])

    def test_anchors_and_metadata(self):
        for question in self.questions:
            if not question.get("ancla_texto"):
                continue
            rows = self.meta[self.meta.chunk_id == question["chunk_id_esperado"]]
            self.assertEqual(len(rows), 1)
            row = rows.iloc[0]
            self.assertEqual((row["ticker"], int(row["fiscal_year"]),
                              str(row["item"])),
                             (question["ticker"], question["fiscal_year"],
                              question["item_esperado"]))
            self.assertIn(question["ancla_texto"], row.texto)

    def test_xbrl_formatter_round_trips_all_facts(self):
        for _, fact in self.facts.iterrows():
            rendered = format_xbrl_value(float(fact.value), fact.unit)
            self.assertEqual(float(rendered.replace(",", "")), float(fact.value))
