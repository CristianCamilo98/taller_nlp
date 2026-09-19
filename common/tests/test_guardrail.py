from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from common.agent.middleware_xbrl import verificar_respuesta_xbrl
from common.config import get_dataset_paths


def xbrl_call(ticker, year, concept):
    return {"name": "get_xbrl_fact", "args": {
        "ticker": ticker, "fiscal_year": year, "concept": concept}}


class GuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = pd.read_parquet(get_dataset_paths().xbrl_facts)

    def numeric(self):
        return {
            "ticker": "AAPL", "ejercicio": 2024,
            "concepto_xbrl": "EarningsPerShareDiluted",
            "unidad": "USD/shares", "cifra": 6.08, "fuente": "xbrl",
        }

    def comparative(self):
        return {
            "ticker": "AAPL", "concepto_xbrl": "EarningsPerShareDiluted",
            "unidad": "USD/shares", "fuente": "xbrl",
            "ejercicio_inicial": 2024, "ejercicio_final": 2025,
            "valor_inicial": 6.08, "valor_final": 7.46,
            "delta": 1.38, "cifra": 1.38,
            "porcentaje": 22.69736842105263,
        }

    def test_numeric_must_match_a_fact_actually_queried(self):
        answer = self.numeric()
        eps_call = [xbrl_call("AAPL", 2024, "EarningsPerShareDiluted")]
        self.assertTrue(verificar_respuesta_xbrl(answer, eps_call)[0])
        self.assertFalse(verificar_respuesta_xbrl(
            {**answer, "cifra": 7.46}, eps_call)[0])
        self.assertFalse(verificar_respuesta_xbrl(
            {**answer, "unidad": "EUR"}, eps_call)[0])
        self.assertFalse(verificar_respuesta_xbrl(
            answer, [xbrl_call("AAPL", 2024, "Assets")])[0])
        self.assertFalse(verificar_respuesta_xbrl(answer, [])[0])

    def test_guardrail_validates_declared_concept_without_golden_leakage(self):
        row = self.facts[
            (self.facts.ticker == "AAPL")
            & (self.facts.fiscal_year == 2024)
            & (self.facts.concept == "Assets")
        ].iloc[0]
        answer = {
            "ticker": "AAPL", "ejercicio": 2024,
            "concepto_xbrl": "Assets", "unidad": row.unit,
            "cifra": float(row.value), "fuente": "xbrl",
        }
        self.assertTrue(verificar_respuesta_xbrl(
            answer, [xbrl_call("AAPL", 2024, "Assets")])[0])

    def test_comparative_requires_both_calls_and_derived_values(self):
        answer = self.comparative()
        calls = [xbrl_call("AAPL", 2024, "EarningsPerShareDiluted"),
                 xbrl_call("AAPL", 2025, "EarningsPerShareDiluted")]
        self.assertTrue(verificar_respuesta_xbrl(answer, calls)[0])
        self.assertFalse(verificar_respuesta_xbrl(answer, calls[:1])[0])
        self.assertFalse(verificar_respuesta_xbrl(
            {**answer, "delta": 1.0, "cifra": 1.0}, calls)[0])
        self.assertFalse(verificar_respuesta_xbrl(
            {**answer, "porcentaje": 20.0}, calls)[0])

    def test_responder_passes_current_tool_calls_to_guardrail(self):
        import common.responder as responder_module

        class Message:
            tool_calls = [xbrl_call("AAPL", 2024,
                                    "EarningsPerShareDiluted")]

        class Agent:
            def invoke(self, payload, config):
                return {"messages": [Message()], "structured_response": {
                    "respuesta": "6.08", "ticker": "AAPL",
                    "ejercicio": 2024,
                    "concepto_xbrl": "EarningsPerShareDiluted",
                    "unidad": "USD/shares", "cifra": 6.08,
                    "fuente": "xbrl", "cita": None, "chunk_id": None,
                    "ejercicio_inicial": None, "ejercicio_final": None,
                    "valor_inicial": None, "valor_final": None,
                    "delta": None, "porcentaje": None,
                }}

        previous = responder_module._agente
        responder_module._agente = Agent()
        try:
            with patch(
                "common.agent.middleware_xbrl.verificar_respuesta_xbrl",
                return_value=(True, "ok"),
            ) as verify:
                responder_module.responder("pregunta", max_reintentos=0)
            calls = verify.call_args.args[1]
            self.assertEqual(calls, Message.tool_calls)
        finally:
            responder_module._agente = previous
