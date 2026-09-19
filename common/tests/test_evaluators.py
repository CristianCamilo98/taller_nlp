from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from common.config import get_dataset_paths
from common.eval.evaluador_cifra import evaluar_cifra_detallada
from common.eval.evaluador_cita import evaluar_cita_detallada
from common.eval.evaluador_trayectoria import evaluar_trayectoria_detallada

GOLDEN = Path(__file__).resolve().parents[1] / "golden_set" / "golden_set_grupo3.jsonl"


def xbrl_call(ticker, year, concept):
    return {"name": "get_xbrl_fact", "args": {
        "ticker": ticker, "fiscal_year": year, "concept": concept}}


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with GOLDEN.open(encoding="utf-8") as stream:
            cls.questions = {q["id"]: q for q in
                             (json.loads(line) for line in stream if line.strip())}
        cls.metadata = pd.read_parquet(get_dataset_paths().chunks_meta)

    def numeric_answer(self):
        return {
            "ticker": "AAPL", "ejercicio": 2024,
            "concepto_xbrl": "EarningsPerShareDiluted",
            "unidad": "USD/shares", "cifra": 6.08, "fuente": "xbrl",
            "tool_calls_detallado": [
                xbrl_call("AAPL", 2024, "EarningsPerShareDiluted")],
        }

    def comparative_answer(self):
        return {
            "ticker": "AAPL", "concepto_xbrl": "EarningsPerShareDiluted",
            "unidad": "USD/shares", "fuente": "xbrl",
            "ejercicio_inicial": 2024, "ejercicio_final": 2025,
            "valor_inicial": 6.08, "valor_final": 7.46,
            "delta": 1.38, "cifra": 1.38,
            "porcentaje": 22.69736842105263,
            "tool_calls_detallado": [
                xbrl_call("AAPL", 2024, "EarningsPerShareDiluted"),
                xbrl_call("AAPL", 2025, "EarningsPerShareDiluted"),
            ],
        }

    def test_numeric_adversarial_cases(self):
        question = self.questions["g3-003"]
        good = self.numeric_answer()
        self.assertTrue(evaluar_cifra_detallada(good, question)["acierto_cifra"])
        variants = [
            {**good, "unidad": "EUR"},
            {**good, "ticker": "MSFT"},
            {**good, "ejercicio": 2025},
            {**good, "concepto_xbrl": "Assets"},
            {**good, "cifra": 6, "respuesta": "El valor es 16 en 2024"},
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertFalse(evaluar_cifra_detallada(
                    variant, question)["acierto_cifra"])

    def test_comparative_numeric_adversarial_cases(self):
        question = self.questions["g3-018"]
        good = self.comparative_answer()
        self.assertTrue(evaluar_cifra_detallada(good, question)["acierto_cifra"])
        incomplete = {
            "ticker": "AAPL", "concepto_xbrl": "EarningsPerShareDiluted",
            "unidad": "USD/shares", "fuente": "xbrl", "cifra": 7.46,
            "tool_calls_detallado": [
                xbrl_call("AAPL", 2025, "EarningsPerShareDiluted")],
        }
        self.assertFalse(evaluar_cifra_detallada(
            incomplete, question)["acierto_cifra"])
        self.assertFalse(evaluar_cifra_detallada(
            {**good, "delta": 1.0, "cifra": 1.0}, question)["acierto_cifra"])
        self.assertFalse(evaluar_cifra_detallada(
            {**good, "porcentaje": 20.0}, question)["acierto_cifra"])

    def test_citation_deterministic_components(self):
        question = self.questions["g3-009"]
        anchor = question["ancla_texto"]
        chunk_id = question["chunk_id_esperado"]
        for citation in (None, ""):
            detail = evaluar_cita_detallada(
                {"cita": citation, "chunk_id": chunk_id}, question)
            self.assertFalse(detail["citation_exists"])
            self.assertTrue(detail["golden_anchor_hit"])
            self.assertFalse(detail["acierto_cita"])

        missing = evaluar_cita_detallada(
            {"cita": anchor, "chunk_id": "NO-EXISTE"}, question)
        self.assertFalse(missing["chunk_exists"])
        self.assertFalse(missing["citation_exists"])

        invented = evaluar_cita_detallada(
            {"cita": "This phrase was deliberately invented",
             "chunk_id": chunk_id}, question)
        self.assertFalse(invented["citation_exists"])

        literal = evaluar_cita_detallada(
            {"cita": anchor, "chunk_id": chunk_id}, question)
        self.assertTrue(literal["citation_exists"])
        self.assertTrue(literal["golden_anchor_hit"])
        self.assertTrue(literal["acierto_cita_literal_anchor"])
        self.assertIsNone(literal["citation_supports"])

        words = anchor.split()
        discontinuous = evaluar_cita_detallada(
            {"cita": " ".join(words[:4]) + " ... " + " ".join(words[-4:]),
             "chunk_id": chunk_id}, question)
        self.assertTrue(discontinuous["citation_exists"])
        self.assertTrue(discontinuous["acierto_cita"])

        row = self.metadata[self.metadata.chunk_id == chunk_id].iloc[0]
        unrelated = " ".join(str(row["texto"]).split()[:10])
        unrelated_detail = evaluar_cita_detallada(
            {"cita": unrelated, "chunk_id": chunk_id}, question)
        self.assertTrue(unrelated_detail["citation_exists"])
        self.assertTrue(unrelated_detail["golden_anchor_hit"])
        self.assertIsNone(unrelated_detail["citation_supports"])
        self.assertFalse(unrelated_detail["support_deterministic"])

    def test_trajectory_adversarial_cases(self):
        comparative = self.questions["g3-014"]
        c24 = xbrl_call("NVDA", 2024, "Revenues")
        c25 = xbrl_call("NVDA", 2025, "Revenues")
        cases = [
            ([xbrl_call("NVDA", 2024, "Assets")], False),
            ([c24], False),
            ([c24, c24], False),
            ([c24, c25], True),
            ([c24, c25, {"name": "search_filings", "args": {
                "query": "revenue", "ticker": "NVDA", "fiscal_year": 2025,
                "item": "7", "k": 5}}], False),
        ]
        for calls, expected in cases:
            with self.subTest(calls=calls):
                self.assertEqual(evaluar_trayectoria_detallada(
                    {"tool_calls_detallado": calls}, comparative
                )["acierto_trayectoria"], expected)

        extractive = self.questions["g3-009"]
        wrong_item = {"name": "search_filings", "args": {
            "query": "foreign exchange", "ticker": "MSFT",
            "fiscal_year": 2024, "item": "1A", "k": 5}}
        self.assertFalse(evaluar_trayectoria_detallada(
            {"tool_calls_detallado": [wrong_item]}, extractive
        )["acierto_trayectoria"])
