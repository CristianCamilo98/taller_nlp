"""Prueba offline del runner de smoke: nunca llama a OpenRouter."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from common.scripts.run_agent_smoke import (
    EXTRACTIVE_QUESTION_ID,
    NUMERIC_QUESTION_ID,
    _extractive_case,
    _numeric_case,
    run_smoke,
)


class SmokeCaseSelectionTests(unittest.TestCase):
    def test_numeric_case_matches_existing_golden_question(self):
        pregunta_eval, pregunta_texto = _numeric_case()
        self.assertEqual(pregunta_eval["id"], NUMERIC_QUESTION_ID)
        self.assertEqual(pregunta_eval["ticker"], "NVDA")
        self.assertEqual(pregunta_eval["fiscal_year"], 2024)
        self.assertEqual(pregunta_eval["concept_xbrl"], "Assets")
        self.assertEqual(pregunta_eval["cifra_esperada"], 65728000000.0)
        self.assertIn("herramienta_esperada", pregunta_eval)
        self.assertIn("¿", pregunta_texto)

    def test_extractive_case_matches_existing_benchmark_question(self):
        pregunta_eval, pregunta_texto = _extractive_case()
        self.assertEqual(pregunta_eval["id"], EXTRACTIVE_QUESTION_ID)
        self.assertEqual(pregunta_eval["ticker"], "AAPL")
        self.assertEqual(pregunta_eval["fiscal_year"], 2024)
        self.assertEqual(pregunta_eval["item"], "1A")
        self.assertIn("industrial accident", pregunta_eval["ancla_texto"])
        self.assertEqual(pregunta_eval["herramienta_esperada"], ["search_filings"])
        self.assertTrue(pregunta_texto.strip())


class SmokeRunnerOfflineTests(unittest.TestCase):
    def test_numeric_smoke_writes_artifact_and_scores_offline(self):
        pregunta_eval, pregunta_texto = _numeric_case()
        fake_response = {
            "respuesta": "65,728,000,000 USD",
            "cifra": 65728000000.0,
            "unidad": "USD",
            "ticker": "NVDA",
            "ejercicio": 2024,
            "concepto_xbrl": "Assets",
            "fuente": "xbrl",
            "cita": None,
            "chunk_id": None,
            "tool_calls_agente": ["get_xbrl_fact"],
            "tool_calls_detallado": [{
                "name": "get_xbrl_fact",
                "args": {"ticker": "NVDA", "fiscal_year": 2024,
                          "concept": "Assets"},
            }],
            "tool_calls_bloqueados": [],
            "guardrail_retry_count": 0,
            "guardrail_ok": True,
            "guardrail_mensaje": "ok",
            "_telemetria": {"input_tokens": 100, "output_tokens": 20,
                            "total_tokens": 120, "llm_calls": 1,
                            "coste": 0.001},
        }
        responder_fn = Mock(return_value=fake_response)
        with TemporaryDirectory() as tmp:
            artifact = run_smoke(
                "numeric", responder_fn=responder_fn, output_dir=Path(tmp)
            )
            responder_fn.assert_called_once_with(pregunta_texto)
            destino = Path(tmp) / "numeric_smoke.json"
            self.assertTrue(destino.exists())
            on_disk = json.loads(destino.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, artifact)
        self.assertEqual(artifact["question_id"], NUMERIC_QUESTION_ID)
        self.assertEqual(artifact["agent_model"],
                          "openrouter:deepseek/deepseek-v4-flash")
        self.assertEqual(artifact["retrieval_profile"], "gemini-embedding-2")
        self.assertTrue(artifact["metricas"]["cifra"]["acierto_cifra"])
        self.assertIsNone(artifact["metricas"]["cita"]["acierto_cita"])
        self.assertTrue(artifact["trajectory"]["acierto_trayectoria"])
        self.assertIsNone(artifact["primera_query_search_filings"])
        dump = json.dumps(artifact)
        self.assertNotIn("OPENROUTER_API_KEY", dump)
        self.assertNotIn("sk-or-", dump)

    def test_extractive_smoke_detects_original_query_and_scores_citation(self):
        pregunta_eval, pregunta_texto = _extractive_case()
        fake_response = {
            "respuesta": "Podria haber lesiones graves y dano reputacional.",
            "cifra": None,
            "unidad": None,
            "ticker": "AAPL",
            "ejercicio": 2024,
            "concepto_xbrl": None,
            "fuente": "texto",
            "cita": ("an industrial accident could occur and could result "
                     "in serious injuries or loss of life"),
            "chunk_id": "AAPL-2024-1A-0003",
            "tool_calls_agente": ["search_filings"],
            "tool_calls_detallado": [{
                "name": "search_filings",
                "args": {"query": pregunta_texto, "ticker": "AAPL",
                          "fiscal_year": 2024, "item": "1A", "k": 5},
            }],
            "tool_calls_bloqueados": [],
            "guardrail_retry_count": 0,
            "guardrail_ok": True,
            "guardrail_mensaje": "ok",
            "_telemetria": {"input_tokens": 200, "output_tokens": 40,
                            "total_tokens": 240, "llm_calls": 1,
                            "coste": 0.002},
        }
        responder_fn = Mock(return_value=fake_response)
        with TemporaryDirectory() as tmp:
            artifact = run_smoke(
                "extractive", responder_fn=responder_fn, output_dir=Path(tmp)
            )
        self.assertEqual(artifact["question_id"], EXTRACTIVE_QUESTION_ID)
        self.assertEqual(artifact["primera_query_search_filings"], pregunta_texto)
        self.assertTrue(artifact["primera_query_es_pregunta_original_literal"])
        self.assertTrue(artifact["metricas"]["cita"]["acierto_cita"])
        self.assertIsNone(artifact["metricas"]["cifra"]["acierto_cifra"])
        self.assertTrue(artifact["trajectory"]["acierto_trayectoria"])

    def test_rewritten_first_query_is_flagged_as_not_original(self):
        _pregunta_eval, pregunta_texto = _extractive_case()
        fake_response = {
            "respuesta": "x", "cifra": None, "unidad": None, "ticker": "AAPL",
            "ejercicio": 2024, "concepto_xbrl": None, "fuente": "texto",
            "cita": None, "chunk_id": None,
            "tool_calls_agente": ["search_filings"],
            "tool_calls_detallado": [{
                "name": "search_filings",
                "args": {"query": "industrial accident supplier consequences",
                          "ticker": "AAPL", "fiscal_year": 2024, "item": "1A",
                          "k": 5},
            }],
            "tool_calls_bloqueados": [],
            "guardrail_retry_count": 0, "guardrail_ok": True,
            "guardrail_mensaje": "ok", "_telemetria": {},
        }
        responder_fn = Mock(return_value=fake_response)
        with TemporaryDirectory() as tmp:
            artifact = run_smoke(
                "extractive", responder_fn=responder_fn, output_dir=Path(tmp)
            )
        self.assertNotEqual(artifact["primera_query_search_filings"], pregunta_texto)
        self.assertFalse(artifact["primera_query_es_pregunta_original_literal"])


if __name__ == "__main__":
    unittest.main()
