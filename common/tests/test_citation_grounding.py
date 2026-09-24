"""Citation grounding guardrail + citation repair (fail closed, sin inventar
evidencia).

Un smoke extractivo real, con routing/argumentos ya correctos
(acierto_trayectoria=True), mostro que eso no implica cita valida: la
respuesta final puede declarar fuente="texto"/"ambas" sin que chunk_id/cita
procedan realmente de un chunk devuelto por search_filings. Antes de este
fix, responder() ni siquiera capturaba el contenido de los ToolMessage de
search_filings -solo name+args-, asi que no habia forma de comprobarlo.

Cubre:
- Parsing de _parse_search_filings_chunks / _extract_search_filings_results
  contra el formato real de common.retrieval.dense_baseline.formatear_fragmentos.
- citation_repair.citation_in_text / citation_repair_turn en aislamiento
  (modelo inyectado, sin red).
- Integracion end-to-end con responder(): A-H.

fuente="ninguna" (declarar honestamente que no se encontro evidencia) es
una respuesta legitima -el system prompt del agente lo permite
explicitamente- y queda fuera de este guardrail a proposito.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import ToolMessage

import common.responder as responder_module
from common.agent.citation_repair import (
    CitationRepairResult,
    citation_in_text,
    citation_repair_turn,
)
from common.eval.evaluador_trayectoria import evaluar_trayectoria_detallada
from common.responder import (
    _extract_search_filings_results,
    _parse_search_filings_chunks,
    responder,
)


def search_call(query, ticker, year, item, call_id="1", k=5):
    return {"name": "search_filings", "id": call_id,
            "args": {"query": query, "ticker": ticker,
                     "fiscal_year": year, "item": item, "k": k}}


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


def _search_result_content(chunks):
    """Reproduce exactamente el formato de formatear_fragmentos."""
    parts = []
    for chunk in chunks:
        parts.append(
            f"[{chunk['chunk_id']}] {chunk['ticker']} FY{chunk['fiscal_year']} "
            f"Item {chunk['item']} (similitud {chunk['puntuacion']:.3f})\n"
            f"{chunk['texto']}"
        )
    return "\n\n---\n\n".join(parts)


AAPL_CHUNK = {
    "chunk_id": "AAPL-2024-1A-0003", "ticker": "AAPL", "fiscal_year": 2024,
    "item": "1A", "puntuacion": 0.742,
    "texto": ("suppliers are required to maintain safe working "
             "environments and operations, an industrial accident could "
             "occur and could result in serious injuries or loss of life, "
             "disruption to the Company's business, and harm to the "
             "Company's reputation."),
}
OTHER_CHUNK = {
    "chunk_id": "AAPL-2024-1A-0007", "ticker": "AAPL", "fiscal_year": 2024,
    "item": "1A", "puntuacion": 0.61,
    "texto": "Litigation and regulatory matters can be costly and disruptive.",
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


class ChunkParsingTests(unittest.TestCase):
    def test_parses_chunk_id_ticker_year_item_and_text(self):
        content = _search_result_content([AAPL_CHUNK, OTHER_CHUNK])
        parsed = _parse_search_filings_chunks(content)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["chunk_id"], "AAPL-2024-1A-0003")
        self.assertEqual(parsed[0]["ticker"], "AAPL")
        self.assertEqual(parsed[0]["fiscal_year"], 2024)
        self.assertEqual(parsed[0]["item"], "1A")
        self.assertIn("industrial accident", parsed[0]["texto"])
        self.assertEqual(parsed[1]["chunk_id"], "AAPL-2024-1A-0007")

    def test_no_results_message_parses_to_empty_list(self):
        content = ("Sin resultados para esa consulta con esos filtros. "
                  "Prueba a quitar algún filtro o a reformular la búsqueda.")
        self.assertEqual(_parse_search_filings_chunks(content), [])

    def test_extracts_only_real_non_blocked_search_filings_results(self):
        messages = [
            Message(tool_calls=[search_call("q", "AAPL", 2024, "1A", "1")]),
            ToolMessage(content=_search_result_content([AAPL_CHUNK]),
                       tool_call_id="1", name="search_filings"),
        ]
        results = _extract_search_filings_results(messages)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "AAPL-2024-1A-0003")

    def test_blocked_duplicate_search_filings_result_is_ignored(self):
        blocked_diag = {"type": "duplicate_tool_call", "executed": False}
        messages = [
            Message(tool_calls=[
                search_call("q", "AAPL", 2024, "1A", "1"),
                search_call("q", "AAPL", 2024, "1A", "2"),
            ]),
            ToolMessage(content=_search_result_content([AAPL_CHUNK]),
                       tool_call_id="1", name="search_filings"),
            ToolMessage(content="duplicate", tool_call_id="2",
                       name="search_filings", status="error",
                       artifact=blocked_diag),
        ]
        results = _extract_search_filings_results(messages)
        self.assertEqual(len(results), 1)

    def test_get_xbrl_fact_calls_never_produce_search_results(self):
        messages = [
            Message(tool_calls=[xbrl_call("NVDA", 2024, "Assets", "1")]),
            ToolMessage(content="NVDA FY2024 Assets = 65,728,000,000 USD",
                       tool_call_id="1", name="get_xbrl_fact"),
        ]
        self.assertEqual(_extract_search_filings_results(messages), [])


class CitationInTextTests(unittest.TestCase):
    def test_literal_substring_matches(self):
        self.assertTrue(citation_in_text(
            "an industrial accident could occur", AAPL_CHUNK["texto"]))

    def test_supports_ellipsis_omissions(self):
        self.assertTrue(citation_in_text(
            "industrial accident could occur ... harm to the Company's "
            "reputation", AAPL_CHUNK["texto"]))

    def test_fabricated_text_does_not_match(self):
        self.assertFalse(citation_in_text(
            "a chemical spill could occur", AAPL_CHUNK["texto"]))

    def test_empty_citation_never_matches(self):
        self.assertFalse(citation_in_text(None, AAPL_CHUNK["texto"]))
        self.assertFalse(citation_in_text("", AAPL_CHUNK["texto"]))


class _FakeBound:
    def __init__(self, result):
        self._result = result

    def invoke(self, messages):
        return self._result


class _FakeModel:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, schema):
        return _FakeBound(self._result)


class CitationRepairTurnUnitTests(unittest.TestCase):
    def test_valid_repair_is_accepted(self):
        result = citation_repair_turn(
            "pregunta", "respuesta", [AAPL_CHUNK, OTHER_CHUNK],
            model_fn=lambda: _FakeModel(CitationRepairResult(
                respaldado=True, chunk_id="AAPL-2024-1A-0003",
                cita="an industrial accident could occur",
            )),
        )
        self.assertEqual(result, ("AAPL-2024-1A-0003",
                                  "an industrial accident could occur"))

    def test_no_chunks_returns_none_without_calling_model(self):
        def _boom():
            raise AssertionError("no debe construir el modelo sin chunks")

        result = citation_repair_turn(
            "pregunta", "respuesta", [], model_fn=_boom)
        self.assertIsNone(result)

    def test_hallucinated_chunk_id_is_rejected(self):
        result = citation_repair_turn(
            "pregunta", "respuesta", [AAPL_CHUNK],
            model_fn=lambda: _FakeModel(CitationRepairResult(
                respaldado=True, chunk_id="AAPL-2024-1A-9999", cita="algo",
            )),
        )
        self.assertIsNone(result)

    def test_fabricated_citation_text_is_rejected(self):
        result = citation_repair_turn(
            "pregunta", "respuesta", [AAPL_CHUNK],
            model_fn=lambda: _FakeModel(CitationRepairResult(
                respaldado=True, chunk_id=AAPL_CHUNK["chunk_id"],
                cita="a chemical spill could occur",
            )),
        )
        self.assertIsNone(result)

    def test_not_respaldado_returns_none(self):
        result = citation_repair_turn(
            "pregunta", "respuesta", [AAPL_CHUNK],
            model_fn=lambda: _FakeModel(
                CitationRepairResult(respaldado=False)),
        )
        self.assertIsNone(result)


class ResponderCitationGroundingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._previous_agent = responder_module._agente
        self.addCleanup(self._restore_agent)

    def _restore_agent(self):
        responder_module._agente = self._previous_agent

    def _install(self, steps):
        agent = ScriptedAgent(steps)
        responder_module._agente = agent
        return agent

    def test_a_correct_citation_from_retrieved_chunk_passes(self):
        step = (
            [
                Message(tool_calls=[
                    search_call("pregunta", "AAPL", 2024, "1A"),
                ]),
                ToolMessage(
                    content=_search_result_content([AAPL_CHUNK, OTHER_CHUNK]),
                    tool_call_id="1", name="search_filings"),
            ],
            _base_structured(
                respuesta="Podria haber lesiones graves.", ticker="AAPL",
                ejercicio=2024, fuente="texto",
                cita="an industrial accident could occur",
                chunk_id="AAPL-2024-1A-0003",
            ),
        )
        self._install([step])
        answer = responder("pregunta", max_reintentos=0)

        self.assertTrue(answer["guardrail_ok"])
        self.assertTrue(answer["citation_grounding_ok"])
        self.assertFalse(answer["citation_repair_used"])
        self.assertEqual(len(answer["tool_results_detallado"]), 2)

    def test_b_missing_citation_is_not_silently_accepted(self):
        step = (
            [
                Message(tool_calls=[
                    search_call("pregunta", "AAPL", 2024, "1A"),
                ]),
                ToolMessage(content=_search_result_content([AAPL_CHUNK]),
                           tool_call_id="1", name="search_filings"),
            ],
            _base_structured(
                respuesta="Podria haber lesiones graves.", ticker="AAPL",
                ejercicio=2024, fuente="texto", cita=None, chunk_id=None,
            ),
        )
        self._install([step])
        with patch("common.agent.citation_repair.citation_repair_turn",
                   return_value=None):
            answer = responder("pregunta", max_reintentos=0)

        self.assertFalse(answer["guardrail_ok"])
        self.assertTrue(answer["citation_repair_used"])
        self.assertTrue(answer["citation_repair_failed"])

    def test_c_chunk_id_not_among_retrieved_fails(self):
        step = (
            [
                Message(tool_calls=[
                    search_call("pregunta", "AAPL", 2024, "1A"),
                ]),
                ToolMessage(content=_search_result_content([AAPL_CHUNK]),
                           tool_call_id="1", name="search_filings"),
            ],
            _base_structured(
                respuesta="respuesta", ticker="AAPL", ejercicio=2024,
                fuente="texto", cita="algo literal",
                chunk_id="AAPL-2024-1A-9999",
            ),
        )
        self._install([step])
        with patch("common.agent.citation_repair.citation_repair_turn",
                   return_value=None):
            answer = responder("pregunta", max_reintentos=0)

        self.assertFalse(answer["guardrail_ok"])

    def test_d_citation_not_contained_in_chunk_fails(self):
        step = (
            [
                Message(tool_calls=[
                    search_call("pregunta", "AAPL", 2024, "1A"),
                ]),
                ToolMessage(content=_search_result_content([AAPL_CHUNK]),
                           tool_call_id="1", name="search_filings"),
            ],
            _base_structured(
                respuesta="respuesta", ticker="AAPL", ejercicio=2024,
                fuente="texto", cita="a chemical spill could occur",
                chunk_id="AAPL-2024-1A-0003",
            ),
        )
        self._install([step])
        with patch("common.agent.citation_repair.citation_repair_turn",
                   return_value=None):
            answer = responder("pregunta", max_reintentos=0)

        self.assertFalse(answer["guardrail_ok"])

    def test_e_repair_selects_a_real_retrieved_chunk_and_passes(self):
        step = (
            [
                Message(tool_calls=[
                    search_call("pregunta", "AAPL", 2024, "1A"),
                ]),
                ToolMessage(
                    content=_search_result_content([AAPL_CHUNK, OTHER_CHUNK]),
                    tool_call_id="1", name="search_filings"),
            ],
            _base_structured(
                respuesta="Podria haber lesiones graves.", ticker="AAPL",
                ejercicio=2024, fuente="texto", cita=None, chunk_id=None,
            ),
        )
        self._install([step])
        with patch(
            "common.agent.citation_repair.citation_repair_turn",
            return_value=("AAPL-2024-1A-0003",
                          "an industrial accident could occur"),
        ) as repair_mock:
            answer = responder("pregunta", max_reintentos=0)

        repair_mock.assert_called_once()
        self.assertTrue(answer["guardrail_ok"])
        self.assertTrue(answer["citation_repair_used"])
        self.assertFalse(answer["citation_repair_failed"])
        self.assertEqual(answer["chunk_id"], "AAPL-2024-1A-0003")
        self.assertEqual(answer["cita"], "an industrial accident could occur")

    def test_f_repair_hallucination_fails_closed(self):
        step = (
            [
                Message(tool_calls=[
                    search_call("pregunta", "AAPL", 2024, "1A"),
                ]),
                ToolMessage(content=_search_result_content([AAPL_CHUNK]),
                           tool_call_id="1", name="search_filings"),
            ],
            _base_structured(
                respuesta="Podria haber lesiones graves.", ticker="AAPL",
                ejercicio=2024, fuente="texto", cita=None, chunk_id=None,
            ),
        )
        self._install([step])
        # citation_repair_turn ya valida internamente y devuelve None ante
        # una alucinacion (test unitario arriba); aqui comprobamos que
        # responder() respeta ese None y falla cerrado, sin aceptar nada.
        with patch("common.agent.citation_repair.citation_repair_turn",
                   return_value=None):
            answer = responder("pregunta", max_reintentos=0)

        self.assertFalse(answer["guardrail_ok"])
        self.assertIsNone(answer["chunk_id"])
        self.assertIsNone(answer["cita"])

    def test_g_xbrl_question_is_not_affected(self):
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

        self.assertTrue(answer["guardrail_ok"])
        self.assertTrue(answer["citation_grounding_ok"])
        self.assertFalse(answer["citation_repair_used"])
        self.assertEqual(answer["tool_results_detallado"], [])

    def test_h_trajectory_and_original_query_are_unaffected(self):
        original = "pregunta original exacta"
        step = (
            [
                Message(tool_calls=[
                    search_call(original, "AAPL", 2024, "1A"),
                ]),
                ToolMessage(content=_search_result_content([AAPL_CHUNK]),
                           tool_call_id="1", name="search_filings"),
            ],
            _base_structured(
                respuesta="Podria haber lesiones graves.", ticker="AAPL",
                ejercicio=2024, fuente="texto",
                cita="an industrial accident could occur",
                chunk_id="AAPL-2024-1A-0003",
            ),
        )
        self._install([step])
        answer = responder(original, max_reintentos=0)

        self.assertEqual(
            answer["tool_calls_detallado"][0]["args"]["query"], original)
        golden = {
            "familia": "extractiva", "ticker": "AAPL", "fiscal_year": 2024,
            "item_esperado": "1A", "herramienta_esperada": ["search_filings"],
        }
        trajectory = evaluar_trayectoria_detallada(answer, golden)
        self.assertTrue(trajectory["acierto_trayectoria"])


if __name__ == "__main__":
    unittest.main()
