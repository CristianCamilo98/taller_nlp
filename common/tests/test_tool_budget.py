"""Presupuesto ACUMULADO de tool calls reales por responder(pregunta).

ToolCallLimitMiddleware anota run_tool_call_count con UntrackedValue (ver
langchain/agents/middleware/tool_call_limit.py), asi que se reinicia en
cada invoke() del grafo, incluso reutilizando el mismo thread_id. Un
reintento de guardrail o de tool-enforcement es un invoke() nuevo, con un
presupuesto de grafo fresco de 10 -no acumulado-.

responder() cierra ese hueco por su cuenta: verificamos empiricamente (ver
auditoria) que dos invoke() sobre el mismo thread_id devuelven el historial
COMPLETO acumulado del hilo, no solo el delta del turno. Por eso basta con
recalcular len(calls) tras cada invoke() para tener el total real de la
pregunta completa, sin necesidad de un contador manual aparte. Si ese total
supera BENCHMARK.tool_call_run_limit, responder() se detiene de forma
controlada y no vuelve a invocar el agente.

Los mocks de este archivo simulan ese comportamiento de forma fiel:
AccumulatingScriptedAgent acumula mensajes por thread_id (nunca entre
thread_id distintos), igual que el checkpointer real.
"""

from __future__ import annotations

import unittest

import common.responder as responder_module
from common.agent.middleware_tool_dedup import DUPLICATE_TOOL_CALL_TYPE
from common.responder import responder


class Message:
    """Simula un AIMessage con tool_calls."""

    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


class BlockedToolMessage:
    """Simula el ToolMessage de error que emite ToolCallDedupMiddleware."""

    def __init__(self, tool_call_id):
        self.tool_call_id = tool_call_id
        self.artifact = {"type": DUPLICATE_TOOL_CALL_TYPE, "executed": False}
        self.tool_calls = []


def xbrl_call(ticker, year, concept, call_id):
    return {"name": "get_xbrl_fact", "id": call_id,
            "args": {"ticker": ticker, "fiscal_year": year,
                     "concept": concept}}


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


def _bogus_calls(n, prefix):
    """n get_xbrl_fact con conceptos inexistentes: fuerzan fallo de guardrail
    (`_queried_facts` no encuentra nada) sin depender de datos XBRL reales."""
    return [xbrl_call("NVDA", 2024, f"ConceptoInventado{prefix}{i}",
                      f"{prefix}-{i}")
            for i in range(n)]


NVDA_ASSETS_2024 = _base_structured(
    respuesta="65,728,000,000 USD", ticker="NVDA", ejercicio=2024,
    concepto_xbrl="Assets", unidad="USD", cifra=65728000000.0,
    fuente="xbrl",
)


class AccumulatingScriptedAgent:
    """Sustituye a _get_agente(). Acumula mensajes por thread_id, igual que
    el checkpointer real de LangGraph (verificado empiricamente: dos
    invoke() sobre el mismo thread_id devuelven el historial completo del
    hilo; thread_id distintos no comparten nada)."""

    def __init__(self, steps):
        self._steps = list(steps)
        self._history_by_thread: dict[str, list] = {}
        self.invocations = []

    def invoke(self, payload, config):
        thread_id = config["configurable"]["thread_id"]
        self.invocations.append(payload)
        new_messages, structured = self._steps.pop(0)
        history = self._history_by_thread.setdefault(thread_id, [])
        history.extend(new_messages)
        return {"messages": list(history), "structured_response": structured}


class ToolBudgetTests(unittest.TestCase):
    def setUp(self):
        self._previous_agent = responder_module._agente
        self.addCleanup(self._restore_agent)

    def _restore_agent(self):
        responder_module._agente = self._previous_agent

    def _install(self, steps):
        agent = AccumulatingScriptedAgent(steps)
        responder_module._agente = agent
        return agent

    def test_a_six_then_four_across_a_guardrail_retry_is_allowed(self):
        turn1 = (
            [Message(tool_calls=_bogus_calls(6, "a"))],
            _base_structured(fuente="xbrl", cifra=1.0, ticker="NVDA",
                             ejercicio=2024, concepto_xbrl="Assets",
                             unidad="USD"),
        )
        turn2 = (
            [Message(tool_calls=_bogus_calls(3, "b")
                     + [xbrl_call("NVDA", 2024, "Assets", "b-3")])],
            NVDA_ASSETS_2024,
        )
        agent = self._install([turn1, turn2])
        answer = responder("pregunta numerica", max_reintentos=1)

        self.assertEqual(len(agent.invocations), 2)
        self.assertEqual(len(answer["tool_calls_detallado"]), 10)
        self.assertFalse(answer["tool_call_budget_exceeded"])
        self.assertTrue(answer["guardrail_ok"])

    def test_b_six_then_five_more_never_accepted_as_valid_over_ten(self):
        turn1 = (
            [Message(tool_calls=_bogus_calls(6, "a"))],
            _base_structured(fuente="xbrl", cifra=1.0, ticker="NVDA",
                             ejercicio=2024, concepto_xbrl="Assets",
                             unidad="USD"),
        )
        turn2 = (
            [Message(tool_calls=_bogus_calls(5, "b"))],
            _base_structured(fuente="xbrl", cifra=1.0, ticker="NVDA",
                             ejercicio=2024, concepto_xbrl="Assets",
                             unidad="USD"),
        )
        agent = self._install([turn1, turn2])
        answer = responder("pregunta numerica", max_reintentos=1)

        # El grafo interno de la segunda invocacion (mockeada) ya habia
        # "ejecutado" 5 calls adicionales antes de que responder() pudiera
        # intervenir -run_limit es por invoke(), no por pregunta-, pero
        # responder() nunca acepta ese resultado como una respuesta valida
        # ni reintenta una tercera vez sobre un presupuesto ya excedido.
        self.assertEqual(len(agent.invocations), 2)
        self.assertTrue(answer["tool_call_budget_exceeded"])
        self.assertFalse(answer["guardrail_ok"])
        self.assertTrue(answer["_telemetria"]["tool_call_budget_exceeded"])

    def test_c_budget_resets_for_a_different_question(self):
        agent = self._install([
            (
                [Message(tool_calls=_bogus_calls(10, "p1"))],
                _base_structured(fuente="ninguna"),
            ),
            (
                [Message(tool_calls=[xbrl_call("NVDA", 2024, "Assets", "p2-0"),
                                     xbrl_call("NVDA", 2024, "Assets", "p2-1"),
                                     xbrl_call("NVDA", 2024, "Assets", "p2-2")])],
                NVDA_ASSETS_2024,
            ),
        ])

        first = responder("pregunta uno", max_reintentos=0)
        second = responder("pregunta dos", max_reintentos=0)

        self.assertEqual(len(first["tool_calls_detallado"]), 10)
        self.assertFalse(first["tool_call_budget_exceeded"])
        self.assertEqual(len(second["tool_calls_detallado"]), 3)
        self.assertFalse(second["tool_call_budget_exceeded"])
        self.assertNotEqual(first["thread_id"], second["thread_id"])

    def test_d_blocked_duplicates_do_not_consume_budget(self):
        real_calls = _bogus_calls(9, "r") + [
            xbrl_call("NVDA", 2024, "Assets", "r-9"),
        ]
        duplicates = [
            {**real_calls[0], "id": f"dup-{i}"} for i in range(3)
        ]
        messages = [
            Message(tool_calls=real_calls + duplicates),
            BlockedToolMessage(tool_call_id="dup-0"),
            BlockedToolMessage(tool_call_id="dup-1"),
            BlockedToolMessage(tool_call_id="dup-2"),
        ]
        turn = (messages, NVDA_ASSETS_2024)
        agent = self._install([turn])
        answer = responder("pregunta numerica", max_reintentos=0)

        self.assertEqual(len(agent.invocations), 1)
        self.assertEqual(len(answer["tool_calls_detallado"]), 10)
        self.assertEqual(len(answer["tool_calls_bloqueados"]), 3)
        self.assertFalse(answer["tool_call_budget_exceeded"])
        self.assertTrue(answer["guardrail_ok"])


if __name__ == "__main__":
    unittest.main()
