"""Casos en los que responder() NO necesita ningún tipo de reintento: el
primer turno ya está fundamentado en una tool real (o, para fuente xbrl, en
una llamada real a get_xbrl_fact).

El mecanismo que se activa cuando el primer turno SÍ llega sin evidencia
-forced_evidence_turn(), que reemplaza al retry en lenguaje natural que
demostró ser insuficiente en un smoke real- se cubre en
test_forced_evidence.py, no aquí.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import common.responder as responder_module
from common.responder import responder

ROOT = Path(__file__).resolve().parents[2]


class Message:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


def xbrl_call(ticker, year, concept, call_id="1"):
    return {"name": "get_xbrl_fact", "id": call_id,
            "args": {"ticker": ticker, "fiscal_year": year,
                     "concept": concept}}


def search_call(query, ticker, year, item, call_id="1"):
    return {"name": "search_filings", "id": call_id,
            "args": {"query": query, "ticker": ticker,
                     "fiscal_year": year, "item": item, "k": 5}}


class SearchResultMessage:
    """ToolMessage minimo con el formato real de formatear_fragmentos, para
    que el guardrail de citacion tenga chunks reales que validar."""

    def __init__(self, tool_call_id, chunk_id, ticker, year, item, texto,
                puntuacion=0.9):
        self.tool_call_id = tool_call_id
        self.artifact = None
        self.tool_calls = []
        self.content = (
            f"[{chunk_id}] {ticker} FY{year} Item {item} "
            f"(similitud {puntuacion:.3f})\n{texto}"
        )


def _base_structured(**overrides):
    base = {
        "respuesta": "respuesta", "cifra": None, "unidad": None,
        "ticker": None, "ejercicio": None, "fuente": "ninguna",
        "cita": None, "chunk_id": None, "concepto_xbrl": None,
        "ejercicio_inicial": None, "ejercicio_final": None,
        "valor_inicial": None, "valor_final": None, "delta": None,
        "porcentaje": None,
    }
    base.update(overrides)
    return base


class ScriptedAgent:
    """Sustituye a _get_agente(): cada invoke() consume un paso del guion."""

    def __init__(self, steps):
        self._steps = list(steps)
        self.invocations = []

    def invoke(self, payload, config):
        self.invocations.append(payload)
        messages, structured = self._steps.pop(0)
        return {"messages": messages, "structured_response": structured}


class ToolEnforcementTests(unittest.TestCase):
    def setUp(self):
        self._previous_agent = responder_module._agente
        self.addCleanup(self._restore_agent)

    def _restore_agent(self):
        responder_module._agente = self._previous_agent

    def _install(self, steps):
        agent = ScriptedAgent(steps)
        responder_module._agente = agent
        return agent

    def test_first_attempt_with_valid_tool_finishes_without_retry(self):
        with_tool = (
            [Message(tool_calls=[xbrl_call("NVDA", 2024, "Assets")])],
            _base_structured(
                respuesta="65,728,000,000 USD", ticker="NVDA",
                ejercicio=2024, concepto_xbrl="Assets", unidad="USD",
                cifra=65728000000.0, fuente="xbrl",
            ),
        )
        agent = self._install([with_tool])
        answer = responder("pregunta numerica", max_reintentos=0)

        self.assertEqual(len(agent.invocations), 1)
        self.assertFalse(answer["tool_enforcement_retry_used"])
        self.assertEqual(len(answer["tool_calls_detallado"]), 1)

    def test_numeric_question_can_finish_with_get_xbrl_fact(self):
        step = (
            [Message(tool_calls=[xbrl_call("NVDA", 2024, "Assets")])],
            _base_structured(
                respuesta="65,728,000,000 USD", ticker="NVDA",
                ejercicio=2024, concepto_xbrl="Assets", unidad="USD",
                cifra=65728000000.0, fuente="xbrl",
            ),
        )
        self._install([step])
        answer = responder("pregunta numerica")

        self.assertTrue(answer["guardrail_ok"])
        self.assertEqual(answer["tool_calls_detallado"][0]["name"],
                         "get_xbrl_fact")

    def test_extractive_question_can_finish_with_search_filings(self):
        step = (
            [
                Message(tool_calls=[
                    search_call("pregunta extractiva", "AAPL", 2024, "1A"),
                ]),
                SearchResultMessage(
                    "1", "AAPL-2024-1A-0003", "AAPL", 2024, "1A",
                    "an industrial accident could occur and cause harm",
                ),
            ],
            _base_structured(
                respuesta="Podria haber lesiones graves.", ticker="AAPL",
                ejercicio=2024, fuente="texto",
                cita="an industrial accident could occur",
                chunk_id="AAPL-2024-1A-0003",
            ),
        )
        self._install([step])
        answer = responder("pregunta extractiva")

        self.assertTrue(answer["guardrail_ok"])
        self.assertEqual(answer["tool_calls_detallado"][0]["name"],
                         "search_filings")
        self.assertFalse(answer["tool_enforcement_retry_used"])

    def test_fuente_xbrl_with_get_xbrl_fact_call_needs_no_retry(self):
        step = (
            [Message(tool_calls=[xbrl_call("NVDA", 2024, "Assets")])],
            _base_structured(
                respuesta="65,728,000,000 USD", ticker="NVDA",
                ejercicio=2024, concepto_xbrl="Assets", unidad="USD",
                cifra=65728000000.0, fuente="xbrl",
            ),
        )
        agent = self._install([step])
        answer = responder("pregunta numerica", max_reintentos=0)

        self.assertEqual(len(agent.invocations), 1)
        self.assertFalse(answer["tool_enforcement_retry_used"])

    def test_run_limit_stays_configured_at_ten(self):
        from langchain.agents.middleware import ToolCallLimitMiddleware

        from common.agent.agent import _agent_middleware

        limits = [m for m in _agent_middleware()
                  if isinstance(m, ToolCallLimitMiddleware)]
        self.assertEqual(len(limits), 1)
        self.assertEqual(limits[0].run_limit, 10)

    def test_public_tool_signatures_are_still_intact(self):
        source = (ROOT / "common/tools/tools.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        def names(fn):
            node = next(v for v in tree.body
                       if isinstance(v, ast.FunctionDef) and v.name == fn)
            return [a.arg for a in node.args.args]

        self.assertEqual(names("list_available"), [])
        self.assertEqual(names("get_xbrl_fact"),
                         ["ticker", "fiscal_year", "concept"])
        self.assertEqual(names("search_filings"),
                         ["query", "ticker", "fiscal_year", "item", "k"])
        self.assertEqual(names("read_section"),
                         ["ticker", "fiscal_year", "item"])


if __name__ == "__main__":
    unittest.main()
