"""Evidence rescue: repara una declinación (fuente="ninguna") cuando los
chunks REALMENTE recuperados por search_filings contienen evidencia
suficiente, en vez de solo promover fuente/cita/chunk_id sobre una
respuesta que seguía siendo una declinación.

Auditoría de un smoke extractivo real: retrieval verificado bit-a-bit
idéntico a retrieval-final-v1 para dani-b6-001 (mismos IDs, mismo orden,
mismos scores; el chunk relevante en rank 5), y ese chunk contenía
literalmente la evidencia necesaria -pero el modelo declaró fuente="ninguna"
sin citarla. No es un problema de retrieval; es un hueco de generación.

evidence_adjudication_turn() (common.agent.citation_repair) cierra ese
hueco de forma general -el trigger es puramente estructural
(fuente=="ninguna" + chunks reales recuperados), no depende de la pregunta,
ticker ni chunk concretos-. La validación después de la llamada es siempre
determinista: chunk_id debe estar entre los chunks reales y la cita debe
estar contenida literalmente en ese chunk; si algo falla, o el modelo
confirma que no hay evidencia suficiente, se mantiene la declinación
original sin forzar ni inventar nada.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import ToolMessage

import common.responder as responder_module
from common.agent.citation_repair import (
    EvidenceAdjudicationResult,
    _looks_like_decline,
    evidence_adjudication_turn,
)
from common.eval.evaluador_trayectoria import evaluar_trayectoria_detallada
from common.responder import responder


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
        "respuesta": "ninguna", "cifra": None, "unidad": None,
        "ticker": None, "ejercicio": None, "fuente": "ninguna",
        "cita": None, "chunk_id": None, "concepto_xbrl": None,
        "ejercicio_inicial": None, "ejercicio_final": None,
        "valor_inicial": None, "valor_final": None, "delta": None,
        "porcentaje": None,
    }
    base.update(overrides)
    return base


def _search_result_content(chunks):
    parts = []
    for chunk in chunks:
        parts.append(
            f"[{chunk['chunk_id']}] {chunk['ticker']} FY{chunk['fiscal_year']} "
            f"Item {chunk['item']} (similitud {chunk['puntuacion']:.3f})\n"
            f"{chunk['texto']}"
        )
    return "\n\n---\n\n".join(parts)


# Texto real de AAPL-2024-1A-0003 tal como apareció en el smoke real
# auditado (rank 5 de search_filings runtime, idéntico al top-5 congelado
# de retrieval-final-v1). Se usa solo como fixture de un test de
# integración concreto; el mecanismo en sí no conoce este chunk ni esta
# pregunta -el trigger es puramente estructural-.
RELEVANT_CHUNK = {
    "chunk_id": "AAPL-2024-1A-0003", "ticker": "AAPL", "fiscal_year": 2024,
    "item": "1A", "puntuacion": 0.601,
    "texto": (
        "and conflicts further escalate in the future, actions by "
        "governments in response could be significantly more severe and "
        "restrictive and could materially adversely affect the Company's "
        "business. The Company's operations are also subject to the risks "
        "of industrial accidents at its suppliers and contract "
        "manufacturers. While the Company's suppliers are required to "
        "maintain safe working environments and operations, an industrial "
        "accident could occur and could result in serious injuries or loss "
        "of life, disruption to the Company's business, and harm to the "
        "Company's reputation."
    ),
}
IRRELEVANT_CHUNK_1 = {
    "chunk_id": "AAPL-2024-1A-0002", "ticker": "AAPL", "fiscal_year": 2024,
    "item": "1A", "puntuacion": 0.635,
    "texto": ("The Company's business can be impacted by political events, "
             "trade and other international disputes."),
}
IRRELEVANT_CHUNK_2 = {
    "chunk_id": "AAPL-2024-1A-0011", "ticker": "AAPL", "fiscal_year": 2024,
    "item": "1A", "puntuacion": 0.62,
    "texto": ("The Company's products and services may be affected by "
             "design and manufacturing defects."),
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


class DeclineDetectionTests(unittest.TestCase):
    def test_generic_decline_markers_are_detected(self):
        self.assertTrue(_looks_like_decline("ninguna"))
        self.assertTrue(_looks_like_decline("No se encontró información."))
        self.assertTrue(_looks_like_decline("No hay evidencia suficiente"))
        self.assertTrue(_looks_like_decline("not found"))

    def test_empty_or_very_short_text_is_a_decline(self):
        self.assertTrue(_looks_like_decline(""))
        self.assertTrue(_looks_like_decline("n/a"))

    def test_substantive_answer_is_not_a_decline(self):
        self.assertFalse(_looks_like_decline(
            "Podria haber lesiones graves y dano a la reputacion."))


class EvidenceAdjudicationTurnUnitTests(unittest.TestCase):
    def test_a_irrelevant_chunks_are_not_respaldado(self):
        result = evidence_adjudication_turn(
            "pregunta", "ninguna", [IRRELEVANT_CHUNK_1, IRRELEVANT_CHUNK_2],
            model_fn=lambda: _FakeModel(
                EvidenceAdjudicationResult(respaldado=False)),
        )
        self.assertIsNone(result)

    def test_b_relevant_chunk_is_rescued(self):
        result = evidence_adjudication_turn(
            "pregunta", "ninguna", [IRRELEVANT_CHUNK_1, RELEVANT_CHUNK],
            model_fn=lambda: _FakeModel(EvidenceAdjudicationResult(
                respaldado=True,
                respuesta=("Podria haber lesiones graves o perdida de "
                          "vidas, interrupcion del negocio y dano a la "
                          "reputacion."),
                chunk_id="AAPL-2024-1A-0003",
                cita="serious injuries or loss of life",
            )),
        )
        self.assertIsNotNone(result)
        respuesta, chunk_id, cita = result
        self.assertIn("lesiones", respuesta)
        self.assertEqual(chunk_id, "AAPL-2024-1A-0003")
        self.assertEqual(cita, "serious injuries or loss of life")

    def test_c_chunk_id_not_among_retrieved_fails_closed(self):
        result = evidence_adjudication_turn(
            "pregunta", "ninguna", [RELEVANT_CHUNK],
            model_fn=lambda: _FakeModel(EvidenceAdjudicationResult(
                respaldado=True, respuesta="respuesta inventada",
                chunk_id="AAPL-2024-1A-9999", cita="algo",
            )),
        )
        self.assertIsNone(result)

    def test_d_citation_not_in_chunk_fails_closed(self):
        result = evidence_adjudication_turn(
            "pregunta", "ninguna", [RELEVANT_CHUNK],
            model_fn=lambda: _FakeModel(EvidenceAdjudicationResult(
                respaldado=True, respuesta="respuesta",
                chunk_id="AAPL-2024-1A-0003",
                cita="a chemical spill could occur",
            )),
        )
        self.assertIsNone(result)

    def test_e_respaldado_true_but_empty_response_fails_closed(self):
        result = evidence_adjudication_turn(
            "pregunta", "ninguna", [RELEVANT_CHUNK],
            model_fn=lambda: _FakeModel(EvidenceAdjudicationResult(
                respaldado=True, respuesta="   ",
                chunk_id="AAPL-2024-1A-0003",
                cita="serious injuries or loss of life",
            )),
        )
        self.assertIsNone(result)

    def test_e_respaldado_true_but_decline_response_fails_closed(self):
        result = evidence_adjudication_turn(
            "pregunta", "ninguna", [RELEVANT_CHUNK],
            model_fn=lambda: _FakeModel(EvidenceAdjudicationResult(
                respaldado=True, respuesta="No se encontró información.",
                chunk_id="AAPL-2024-1A-0003",
                cita="serious injuries or loss of life",
            )),
        )
        self.assertIsNone(result)

    def test_no_chunks_returns_none_without_calling_model(self):
        def _boom():
            raise AssertionError("no debe construir el modelo sin chunks")

        result = evidence_adjudication_turn(
            "pregunta", "ninguna", [], model_fn=_boom)
        self.assertIsNone(result)


class ResponderEvidenceRescueIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._previous_agent = responder_module._agente
        self.addCleanup(self._restore_agent)

    def _restore_agent(self):
        responder_module._agente = self._previous_agent

    def _install(self, steps):
        agent = ScriptedAgent(steps)
        responder_module._agente = agent
        return agent

    def _run_with_rescue(self, pregunta, chunks, rescue_return,
                         respuesta_previa="ninguna"):
        step = (
            [
                Message(tool_calls=[
                    search_call(pregunta, "AAPL", 2024, "1A"),
                ]),
                ToolMessage(content=_search_result_content(chunks),
                           tool_call_id="1", name="search_filings"),
            ],
            _base_structured(respuesta=respuesta_previa),
        )
        self._install([step])
        with patch("common.agent.citation_repair.evidence_adjudication_turn",
                   return_value=rescue_return) as mock:
            answer = responder(pregunta, max_reintentos=0)
        return answer, mock

    def test_a_declines_stay_declined_when_not_respaldado(self):
        answer, mock = self._run_with_rescue(
            "pregunta", [IRRELEVANT_CHUNK_1, IRRELEVANT_CHUNK_2], None)

        mock.assert_called_once()
        self.assertEqual(answer["fuente"], "ninguna")
        self.assertIsNone(answer["chunk_id"])
        self.assertIsNone(answer["cita"])
        self.assertTrue(answer["citation_repair_used"])
        self.assertTrue(answer["citation_repair_failed"])
        self.assertFalse(answer["evidence_rescue_used"])
        self.assertTrue(answer["guardrail_ok"])

    def test_b_relevant_chunk_rescues_the_answer(self):
        rescued = ("Podria haber lesiones graves, interrupcion del negocio "
                  "y dano a la reputacion.", "AAPL-2024-1A-0003",
                  "serious injuries or loss of life")
        answer, mock = self._run_with_rescue(
            "pregunta", [IRRELEVANT_CHUNK_1, RELEVANT_CHUNK], rescued)

        mock.assert_called_once()
        self.assertEqual(answer["fuente"], "texto")
        self.assertEqual(answer["chunk_id"], "AAPL-2024-1A-0003")
        self.assertEqual(answer["cita"], "serious injuries or loss of life")
        self.assertIn("lesiones", answer["respuesta"])
        self.assertTrue(answer["citation_repair_used"])
        self.assertTrue(answer["evidence_rescue_used"])
        self.assertFalse(answer["citation_repair_failed"])
        self.assertTrue(answer["guardrail_ok"])
        self.assertTrue(answer["citation_grounding_ok"])

    def test_c_hallucinated_chunk_id_never_reaches_responder(self):
        # evidence_adjudication_turn ya valida y devuelve None ante un
        # chunk_id inventado (ver test unitario arriba); comprobamos que
        # responder() respeta ese None y no rescata nada.
        answer, _mock = self._run_with_rescue(
            "pregunta", [RELEVANT_CHUNK], None)
        self.assertEqual(answer["fuente"], "ninguna")
        self.assertFalse(answer["evidence_rescue_used"])

    def test_d_fabricated_citation_never_reaches_responder(self):
        answer, _mock = self._run_with_rescue(
            "pregunta", [RELEVANT_CHUNK], None)
        self.assertEqual(answer["fuente"], "ninguna")
        self.assertFalse(answer["evidence_rescue_used"])

    def test_e_empty_or_decline_repair_never_reaches_responder(self):
        answer, _mock = self._run_with_rescue(
            "pregunta", [RELEVANT_CHUNK], None)
        self.assertEqual(answer["respuesta"], "ninguna")
        self.assertFalse(answer["evidence_rescue_used"])

    def test_f_already_grounded_texto_answer_skips_rescue(self):
        step = (
            [
                Message(tool_calls=[
                    search_call("pregunta", "AAPL", 2024, "1A"),
                ]),
                ToolMessage(content=_search_result_content([RELEVANT_CHUNK]),
                           tool_call_id="1", name="search_filings"),
            ],
            _base_structured(
                respuesta="Podria haber lesiones graves.", ticker="AAPL",
                ejercicio=2024, fuente="texto",
                cita="serious injuries or loss of life",
                chunk_id="AAPL-2024-1A-0003",
            ),
        )
        self._install([step])
        with patch(
            "common.agent.citation_repair.evidence_adjudication_turn",
        ) as mock:
            answer = responder("pregunta", max_reintentos=0)

        mock.assert_not_called()
        self.assertTrue(answer["guardrail_ok"])
        self.assertFalse(answer["citation_repair_used"])
        self.assertFalse(answer["evidence_rescue_used"])

    def test_g_numeric_xbrl_path_is_not_affected(self):
        step = (
            [Message(tool_calls=[xbrl_call("NVDA", 2024, "Assets")])],
            _base_structured(
                respuesta="65,728,000,000 USD", ticker="NVDA",
                ejercicio=2024, concepto_xbrl="Assets", unidad="USD",
                cifra=65728000000.0, fuente="xbrl",
            ),
        )
        self._install([step])
        with patch(
            "common.agent.citation_repair.evidence_adjudication_turn",
        ) as mock:
            answer = responder("pregunta numerica", max_reintentos=0)

        mock.assert_not_called()
        self.assertTrue(answer["guardrail_ok"])
        self.assertFalse(answer["evidence_rescue_used"])
        self.assertEqual(answer["cifra"], 65728000000.0)

    def test_h_trajectory_and_original_query_survive_the_rescue(self):
        original = "pregunta original exacta"
        rescued = ("Podria haber lesiones graves.", "AAPL-2024-1A-0003",
                  "serious injuries or loss of life")
        step = (
            [
                Message(tool_calls=[
                    search_call(original, "AAPL", 2024, "1A"),
                ]),
                ToolMessage(content=_search_result_content([RELEVANT_CHUNK]),
                           tool_call_id="1", name="search_filings"),
            ],
            _base_structured(),
        )
        self._install([step])
        with patch("common.agent.citation_repair.evidence_adjudication_turn",
                   return_value=rescued):
            answer = responder(original, max_reintentos=0)

        self.assertEqual(
            answer["tool_calls_detallado"][0]["args"]["query"], original)
        golden = {
            "familia": "extractiva", "ticker": "AAPL", "fiscal_year": 2024,
            "item_esperado": "1A", "herramienta_esperada": ["search_filings"],
        }
        trajectory = evaluar_trayectoria_detallada(answer, golden)
        self.assertTrue(trajectory["acierto_trayectoria"])

    def test_current_case_integration_real_chunk_text_can_be_rescued(self):
        """Test de integración inspirado en el smoke real auditado (chunk en
        rank 5, retrieval idéntico a retrieval-final-v1). El mecanismo no
        conoce esta pregunta ni este chunk -el disparador es puramente
        estructural-; aquí solo se usa el texto real como fixture."""
        pregunta = ("¿Qué consecuencias contempla Apple si ocurre un "
                   "accidente industrial en las instalaciones de uno de "
                   "sus proveedores?")
        step = (
            [
                Message(tool_calls=[
                    search_call(pregunta, "AAPL", 2024, "1A"),
                ]),
                ToolMessage(
                    content=_search_result_content(
                        [IRRELEVANT_CHUNK_1, IRRELEVANT_CHUNK_2,
                         RELEVANT_CHUNK]),
                    tool_call_id="1", name="search_filings"),
            ],
            _base_structured(respuesta="ninguna"),
        )
        self._install([step])
        rescued = (
            "Podria haber lesiones graves o perdida de vidas, "
            "interrupcion del negocio de Apple y dano a su reputacion.",
            "AAPL-2024-1A-0003", "serious injuries or loss of life",
        )
        with patch("common.agent.citation_repair.evidence_adjudication_turn",
                   return_value=rescued):
            answer = responder(pregunta, max_reintentos=0)

        self.assertEqual(answer["fuente"], "texto")
        self.assertEqual(answer["chunk_id"], "AAPL-2024-1A-0003")
        self.assertTrue(answer["evidence_rescue_used"])
        self.assertTrue(answer["guardrail_ok"])


if __name__ == "__main__":
    unittest.main()
