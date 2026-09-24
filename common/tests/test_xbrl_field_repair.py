"""Cierra el hueco fuente="xbrl" + campos estructurados nulos.

Un smoke real (g3-001, NVDA FY2024 Assets) mostró que get_xbrl_fact se
ejecutaba con los argumentos exactos correctos, pero el turno final de
structured output dejaba cifra/ticker/ejercicio/concepto_xbrl/unidad en
null mientras la prosa sí llevaba el valor correcto ("65,728 millones de
dólares") y fuente="xbrl". El guardrail antiguo lo aceptaba con el mensaje
"sin cifra que verificar" porque solo se activaba cuando cifra (u otro
campo numérico) estaba presente.

Dos cambios cierran esto:
- common.agent.middleware_xbrl.verificar_respuesta_xbrl ya NO acepta
  fuente="xbrl"/"ambas" sin ninguna cifra estructurada.
- common.responder._fill_missing_xbrl_fields repara determinísticamente los
  campos nulos ANTES del guardrail, usando exclusivamente el hecho
  REALMENTE consultado por get_xbrl_fact (nunca inventa datos), solo cuando
  es inequívoco (una única llamada real distinta). Así se evita un tercer
  turno de LLM cuando el problema es mecánico, no de contenido.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import common.responder as responder_module
from common.agent.middleware_xbrl import verificar_respuesta_xbrl
from common.eval.evaluador_cifra import evaluar_cifra_detallada
from common.responder import responder

NVDA_2024_ASSETS = {
    "ticker": "NVDA", "fiscal_year": 2024, "concept": "Assets",
}


def xbrl_call(ticker, year, concept, call_id="1"):
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


GOLDEN_G3_001 = {
    "id": "g3-001", "familia": "numerica", "ticker": "NVDA",
    "fiscal_year": 2024, "concept_xbrl": "Assets",
    "cifra_esperada": 65728000000.0, "unidad": "USD",
}


class Message:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


class ScriptedAgent:
    def __init__(self, steps):
        self._steps = list(steps)
        self.invocations = []

    def invoke(self, payload, config):
        self.invocations.append(payload)
        messages, structured = self._steps.pop(0)
        return {"messages": messages, "structured_response": structured}


class GuardrailNullFieldsLoopholeTests(unittest.TestCase):
    """A) y D): fuente="xbrl" sin cifra estructurada nunca pasa el guardrail,
    con o sin llamada real a get_xbrl_fact."""

    def test_a_fuente_xbrl_with_null_cifra_and_valid_call_is_rejected(self):
        respuesta = _base_structured(fuente="xbrl")
        ok, message = verificar_respuesta_xbrl(
            respuesta, [xbrl_call("NVDA", 2024, "Assets")])
        self.assertFalse(ok)
        self.assertNotIn("sin cifra que verificar", message)

    def test_d_fuente_xbrl_without_any_get_xbrl_fact_call_is_rejected(self):
        respuesta = _base_structured(fuente="xbrl")
        ok, _message = verificar_respuesta_xbrl(respuesta, [])
        self.assertFalse(ok)

        respuesta_con_cifra = _base_structured(
            fuente="xbrl", cifra=65728000000.0, ticker="NVDA",
            ejercicio=2024, concepto_xbrl="Assets", unidad="USD",
        )
        ok2, _message2 = verificar_respuesta_xbrl(respuesta_con_cifra, [])
        self.assertFalse(ok2)

    def test_e_fuente_xbrl_with_wrong_cifra_still_fails(self):
        respuesta = _base_structured(
            fuente="xbrl", cifra=999.0, ticker="NVDA", ejercicio=2024,
            concepto_xbrl="Assets", unidad="USD",
        )
        ok, _message = verificar_respuesta_xbrl(
            respuesta, [xbrl_call("NVDA", 2024, "Assets")])
        self.assertFalse(ok)

    def test_f_narrative_answer_without_numeric_fields_is_unaffected(self):
        respuesta = _base_structured(fuente="texto")
        ok, message = verificar_respuesta_xbrl(respuesta, [])
        self.assertTrue(ok)
        self.assertEqual(message, "sin cifra que verificar")


class ResponderXbrlFieldRepairTests(unittest.TestCase):
    def setUp(self):
        self._previous_agent = responder_module._agente
        self.addCleanup(self._restore_agent)

    def _restore_agent(self):
        responder_module._agente = self._previous_agent

    def _install(self, steps):
        agent = ScriptedAgent(steps)
        responder_module._agente = agent
        return agent

    def test_b_already_correct_structured_answer_is_unaffected(self):
        step = (
            [Message(tool_calls=[xbrl_call("NVDA", 2024, "Assets")])],
            _base_structured(
                respuesta="65,728,000,000 USD", ticker="NVDA",
                ejercicio=2024, concepto_xbrl="Assets", unidad="USD",
                cifra=65728000000.0, fuente="xbrl",
            ),
        )
        self._install([step])
        answer = responder("pregunta numerica", max_reintentos=0)

        self.assertFalse(answer["xbrl_fields_autofilled"])
        self.assertTrue(answer["guardrail_ok"])
        self.assertTrue(evaluar_cifra_detallada(answer, GOLDEN_G3_001)
                        ["acierto_cifra"])

    def test_c_prose_correct_but_structured_fields_empty_is_repaired(self):
        # Reproduce exactamente el bug del smoke real: get_xbrl_fact con los
        # argumentos exactos correctos, prosa con el valor correcto, pero
        # cifra/ticker/ejercicio/concepto_xbrl/unidad en null.
        step = (
            [Message(tool_calls=[xbrl_call("NVDA", 2024, "Assets")])],
            _base_structured(
                respuesta=("El total de activos de NVIDIA para el "
                          "ejercicio fiscal 2024 fue de 65,728 millones "
                          "de dólares."),
                fuente="xbrl",
            ),
        )
        agent = self._install([step])
        answer = responder("pregunta numerica", max_reintentos=0)

        self.assertEqual(len(agent.invocations), 1)  # sin tercer turno LLM
        self.assertTrue(answer["xbrl_fields_autofilled"])
        self.assertEqual(answer["cifra"], 65728000000.0)
        self.assertEqual(answer["unidad"], "USD")
        self.assertEqual(answer["ticker"], "NVDA")
        self.assertEqual(answer["ejercicio"], 2024)
        self.assertEqual(answer["concepto_xbrl"], "Assets")
        self.assertTrue(answer["guardrail_ok"])
        self.assertTrue(evaluar_cifra_detallada(answer, GOLDEN_G3_001)
                        ["acierto_cifra"])

    def test_ambiguous_multiple_distinct_calls_are_not_autofilled(self):
        step = (
            [Message(tool_calls=[
                xbrl_call("NVDA", 2024, "Assets", "1"),
                xbrl_call("NVDA", 2024, "Liabilities", "2"),
            ])],
            _base_structured(respuesta="respuesta ambigua", fuente="xbrl"),
        )
        self._install([step])
        answer = responder("pregunta numerica", max_reintentos=0)

        self.assertFalse(answer["xbrl_fields_autofilled"])
        self.assertIsNone(answer["cifra"])
        self.assertFalse(answer["guardrail_ok"])

    def test_narrative_question_is_not_affected_by_the_repair(self):
        step = (
            [Message(tool_calls=[xbrl_call("NVDA", 2024, "Assets", "1")])],
            _base_structured(respuesta="texto narrativo", fuente="ninguna"),
        )
        self._install([step])
        answer = responder("pregunta numerica", max_reintentos=0)

        self.assertFalse(answer["xbrl_fields_autofilled"])
        self.assertIsNone(answer["cifra"])
        self.assertTrue(answer["guardrail_ok"])

    def test_repair_never_overwrites_an_already_wrong_value(self):
        step = (
            [Message(tool_calls=[xbrl_call("NVDA", 2024, "Assets", "1")])],
            _base_structured(
                respuesta="valor incorrecto", cifra=1.0, fuente="xbrl",
            ),
        )
        self._install([step])
        answer = responder("pregunta numerica", max_reintentos=0)

        self.assertEqual(answer["cifra"], 1.0)
        self.assertFalse(answer["guardrail_ok"])


if __name__ == "__main__":
    unittest.main()
