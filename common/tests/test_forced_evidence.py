"""Turno forzado de evidencia: separa obtención de evidencia de structured
output final.

Un smoke real mostró que el retry en lenguaje natural (turno anterior de
esta misma auditoría) NO impide una respuesta final con cero tool calls:
DeepSeek repitió el patrón una segunda vez (telemetry.llm_calls=2,
tool_calls_detallado=[] en ambos turnos). La causa es estructural: con
response_format=RespuestaFinanciera, create_agent resuelve a ProviderStrategy
para este modelo (model.profile["structured_output"] es True), y esa rama de
langchain.agents.factory ignora tool_choice por completo -ningún texto de
aviso puede corregir eso, porque el modelo nunca estuvo obligado a llamar una
tool-.

common.agent.forced_evidence.forced_evidence_turn() resuelve esto con un
modelo aparte, SIN response_format, con tool_choice="required" (sí lo
respeta ChatOpenRouter.bind_tools), que ejecuta la tool real elegida y cuyo
resultado se inyecta en el hilo antes de dejar que el agente normal (con
ProviderStrategy) genere el structured output final, ya con evidencia real.

Este archivo cubre dos niveles:
- ForcedEvidenceTurnTests: unidad de forced_evidence_turn() en aislamiento,
  con un modelo falso inyectado (sin red).
- ResponderForcedEvidenceIntegrationTests: integración con responder(),
  mockeando forced_evidence_turn para no depender de la red, con un agente
  guionizado que simula fielmente el checkpointer de LangGraph (cada
  invoke() acumula tanto los mensajes de entrada -incluida la evidencia
  forzada inyectada- como los que "genera" el modelo).
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage

import common.responder as responder_module
from common.agent.forced_evidence import FORCED_EVIDENCE_SYSTEM, forced_evidence_turn
from common.eval.evaluador_trayectoria import evaluar_trayectoria_detallada
from common.responder import responder

ROOT = Path(__file__).resolve().parents[2]


def xbrl_call(ticker, year, concept, call_id):
    return {"name": "get_xbrl_fact", "id": call_id,
            "args": {"ticker": ticker, "fiscal_year": year,
                     "concept": concept}}


def search_call(query, ticker, year, item, call_id):
    return {"name": "search_filings", "id": call_id,
            "args": {"query": query, "ticker": ticker,
                     "fiscal_year": year, "item": item, "k": 5}}


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


NVDA_ASSETS_2024 = _base_structured(
    respuesta="65,728,000,000 USD", ticker="NVDA", ejercicio=2024,
    concepto_xbrl="Assets", unidad="USD", cifra=65728000000.0,
    fuente="xbrl",
)


class Message:
    """AIMessage minimo para los pasos guionizados del grafo."""

    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


class AccumulatingScriptedAgent:
    """Sustituye a _get_agente(). Acumula, por thread_id, TANTO los mensajes
    de entrada de cada invoke() -incluida la evidencia forzada inyectada por
    responder()- como los que el "modelo" genera en ese paso, igual que el
    checkpointer real de LangGraph (verificado empíricamente en la auditoría
    de Riesgo 1/2: dos invoke() sobre el mismo thread_id devuelven el
    historial acumulado completo, no solo el delta del turno)."""

    def __init__(self, steps):
        self._steps = list(steps)
        self._history_by_thread: dict[str, list] = {}
        self.invocations = []

    def invoke(self, payload, config):
        thread_id = config["configurable"]["thread_id"]
        self.invocations.append(payload)
        input_messages = payload.get("messages") or []
        model_messages, structured = self._steps.pop(0)
        history = self._history_by_thread.setdefault(thread_id, [])
        history.extend(input_messages)
        history.extend(model_messages)
        return {"messages": list(history), "structured_response": structured}


class FakeBoundModel:
    def __init__(self, response):
        self._response = response
        self.invoke_calls = []

    def invoke(self, messages):
        self.invoke_calls.append(messages)
        return self._response


class FakeModel:
    """Sustituye al modelo real inyectado vía forced_evidence_turn(model_fn=...)."""

    def __init__(self, response):
        self._response = response
        self.bind_tools_calls = []
        self.bound = None

    def bind_tools(self, tools, *, tool_choice=None, parallel_tool_calls=None,
                   **kwargs):
        self.bind_tools_calls.append({
            "tool_names": sorted(t.name for t in tools),
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        })
        self.bound = FakeBoundModel(self._response)
        return self.bound


class ForcedEvidenceTurnTests(unittest.TestCase):
    """Unidad de forced_evidence_turn(), sin red (modelo inyectado)."""

    def test_binds_all_four_tools_with_tool_choice_required(self):
        response = AIMessage(content="", tool_calls=[
            xbrl_call("NVDA", 2024, "Assets", "1"),
        ])
        model = FakeModel(response)
        forced_evidence_turn("pregunta", model_fn=lambda: model)

        self.assertEqual(len(model.bind_tools_calls), 1)
        call = model.bind_tools_calls[0]
        self.assertEqual(
            call["tool_names"],
            ["get_xbrl_fact", "list_available", "read_section",
             "search_filings"],
        )
        self.assertEqual(call["tool_choice"], "required")
        self.assertFalse(call["parallel_tool_calls"])

    def test_get_xbrl_fact_selection_executes_the_real_tool_offline(self):
        response = AIMessage(content="", tool_calls=[
            xbrl_call("NVDA", 2024, "Assets", "1"),
        ])
        model = FakeModel(response)
        result = forced_evidence_turn("pregunta", model_fn=lambda: model)

        self.assertIsNotNone(result)
        ai_message, tool_messages = result
        self.assertIs(ai_message, response)
        self.assertEqual(len(tool_messages), 1)
        self.assertIsInstance(tool_messages[0], ToolMessage)
        self.assertEqual(tool_messages[0].tool_call_id, "1")
        self.assertIn("65,728,000,000", tool_messages[0].content)

    def test_search_filings_selection_preserves_exact_query(self):
        original = "¿Qué riesgos identifica Apple sobre sus proveedores?"
        response = AIMessage(content="", tool_calls=[
            search_call(original, "AAPL", 2024, "1A", "1"),
        ])
        model = FakeModel(response)
        fake_hits = [{
            "chunk_id": "AAPL-2024-1A-0003", "ticker": "AAPL",
            "fiscal_year": 2024, "item": "1A", "texto": "texto",
            "n_tokens": 1, "contiene_tabla": False, "puntuacion": 0.9,
        }]
        with patch("common.retrieval.dense_baseline.buscar",
                   return_value=fake_hits) as buscar:
            result = forced_evidence_turn("pregunta", model_fn=lambda: model)

        buscar.assert_called_once()
        self.assertEqual(buscar.call_args.args[0], original)
        ai_message, tool_messages = result
        self.assertEqual(ai_message.tool_calls[0]["args"]["query"], original)

    def test_zero_tool_calls_returns_none(self):
        response = AIMessage(content="respuesta sin tools", tool_calls=[])
        model = FakeModel(response)
        result = forced_evidence_turn("pregunta", model_fn=lambda: model)
        self.assertIsNone(result)

    def test_system_prompt_requires_exactly_one_tool_and_original_query(self):
        self.assertIn("exactamente una herramienta", FORCED_EVIDENCE_SYSTEM)
        self.assertIn("get_xbrl_fact", FORCED_EVIDENCE_SYSTEM)
        self.assertIn("search_filings", FORCED_EVIDENCE_SYSTEM)
        self.assertIn("read_section", FORCED_EVIDENCE_SYSTEM)
        self.assertIn("list_available", FORCED_EVIDENCE_SYSTEM)
        self.assertIn("no la traduzcas, resumas ni reformules",
                      FORCED_EVIDENCE_SYSTEM)

    def test_system_prompt_requires_ticker_fiscal_year_item_extraction(self):
        # Regresion de un smoke real: el turno forzado llamo a search_filings
        # con query/ticker correctos pero SIN fiscal_year ni item, porque el
        # prompt nunca lo pedia (a diferencia del system prompt principal).
        # Sin ese postfiltro, la busqueda es global sobre todos los
        # ejercicios/items de la compania y el chunk objetivo puede quedar
        # fuera de k.
        self.assertIn("ticker", FORCED_EVIDENCE_SYSTEM)
        self.assertIn("fiscal_year", FORCED_EVIDENCE_SYSTEM)
        self.assertIn("item", FORCED_EVIDENCE_SYSTEM)
        self.assertIn("no los omitas", FORCED_EVIDENCE_SYSTEM)


class ResponderForcedEvidenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._previous_agent = responder_module._agente
        self.addCleanup(self._restore_agent)

    def _restore_agent(self):
        responder_module._agente = self._previous_agent

    def _install(self, steps):
        agent = AccumulatingScriptedAgent(steps)
        responder_module._agente = agent
        return agent

    def test_a_forced_get_xbrl_fact_executes_and_reaches_final_answer(self):
        turn1 = ([Message(tool_calls=[])],
                _base_structured(respuesta="65728 millones", fuente="xbrl"))
        turn2 = ([], NVDA_ASSETS_2024)
        agent = self._install([turn1, turn2])

        forced_ai = AIMessage(content="", tool_calls=[
            xbrl_call("NVDA", 2024, "Assets", "forced-1"),
        ])
        forced_tool = ToolMessage(content="NVDA FY2024 Assets = "
                                          "65,728,000,000 USD",
                                  tool_call_id="forced-1",
                                  name="get_xbrl_fact")

        with patch("common.agent.forced_evidence.forced_evidence_turn",
                   return_value=(forced_ai, [forced_tool])) as forced_mock:
            answer = responder("pregunta numerica", max_reintentos=0)

        forced_mock.assert_called_once_with("pregunta numerica")
        self.assertEqual(len(agent.invocations), 2)
        self.assertTrue(answer["tool_enforcement_retry_used"])
        self.assertFalse(answer["forced_tool_call_failed"])
        self.assertTrue(answer["guardrail_ok"])
        self.assertEqual(answer["cifra"], 65728000000.0)

    def test_b_forced_search_filings_preserves_original_query_exactly(self):
        original = "¿Qué riesgos identifica Apple sobre sus proveedores?"
        turn1 = ([Message(tool_calls=[])], _base_structured())
        turn2 = ([], _base_structured(
            respuesta="Riesgos identificados.", ticker="AAPL",
            ejercicio=2024, fuente="texto", cita="fragmento literal",
            chunk_id="AAPL-2024-1A-0003",
        ))
        self._install([turn1, turn2])

        forced_ai = AIMessage(content="", tool_calls=[
            search_call(original, "AAPL", 2024, "1A", "forced-1"),
        ])
        forced_tool = ToolMessage(content="fragmentos...",
                                  tool_call_id="forced-1",
                                  name="search_filings")

        with patch("common.agent.forced_evidence.forced_evidence_turn",
                   return_value=(forced_ai, [forced_tool])):
            answer = responder(original, max_reintentos=0)

        call = answer["tool_calls_detallado"][0]
        self.assertEqual(call["name"], "search_filings")
        self.assertEqual(call["args"]["query"], original)

    def test_c_forced_turn_returns_none_fails_closed_never_from_memory(self):
        turn1 = ([Message(tool_calls=[])],
                _base_structured(respuesta="65728 millones de memoria",
                                 fuente="xbrl"))
        agent = self._install([turn1])

        with patch("common.agent.forced_evidence.forced_evidence_turn",
                   return_value=None):
            answer = responder("pregunta numerica", max_reintentos=0)

        self.assertEqual(len(agent.invocations), 1)
        self.assertTrue(answer["forced_tool_call_failed"])
        self.assertTrue(answer["tool_enforcement_retry_used"])
        self.assertFalse(answer["guardrail_ok"])
        self.assertEqual(answer["tool_calls_detallado"], [])
        self.assertIn("no se acepta", answer["guardrail_mensaje"].lower())

    def test_d_first_turn_already_grounded_skips_forced_preflight(self):
        turn1 = ([Message(tool_calls=[
            xbrl_call("NVDA", 2024, "Assets", "1"),
        ])], NVDA_ASSETS_2024)
        agent = self._install([turn1])

        with patch("common.agent.forced_evidence.forced_evidence_turn") as \
                forced_mock:
            answer = responder("pregunta numerica", max_reintentos=0)

        forced_mock.assert_not_called()
        self.assertEqual(len(agent.invocations), 1)
        self.assertFalse(answer["tool_enforcement_retry_used"])

    def test_e_trajectory_contains_the_real_forced_tool_call(self):
        turn1 = ([Message(tool_calls=[])], _base_structured(fuente="xbrl"))
        turn2 = ([], NVDA_ASSETS_2024)
        self._install([turn1, turn2])

        forced_ai = AIMessage(content="", tool_calls=[
            xbrl_call("NVDA", 2024, "Assets", "forced-1"),
        ])
        forced_tool = ToolMessage(content="valor real",
                                  tool_call_id="forced-1",
                                  name="get_xbrl_fact")
        with patch("common.agent.forced_evidence.forced_evidence_turn",
                   return_value=(forced_ai, [forced_tool])):
            answer = responder("pregunta numerica", max_reintentos=0)

        self.assertEqual(answer["tool_calls_detallado"], [{
            "name": "get_xbrl_fact",
            "args": {"ticker": "NVDA", "fiscal_year": 2024,
                     "concept": "Assets"},
        }])

    def test_f_budget_accounts_for_the_forced_tool_call(self):
        # 9 list_available (no cuentan como evidencia) para acercar el
        # presupuesto acumulado al limite sin fundamentar la respuesta.
        nine_calls = [{"name": "list_available", "id": f"pre-{i}", "args": {}}
                     for i in range(9)]
        turn1 = ([Message(tool_calls=nine_calls)],
                _base_structured(fuente="ninguna"))
        turn2 = ([], NVDA_ASSETS_2024)
        agent = self._install([turn1, turn2])

        forced_ai = AIMessage(content="", tool_calls=[
            xbrl_call("NVDA", 2024, "Assets", "forced-1"),
        ])
        forced_tool = ToolMessage(content="valor real",
                                  tool_call_id="forced-1",
                                  name="get_xbrl_fact")
        with patch("common.agent.forced_evidence.forced_evidence_turn",
                   return_value=(forced_ai, [forced_tool])):
            answer = responder("pregunta numerica", max_reintentos=0)

        self.assertEqual(len(agent.invocations), 2)
        self.assertEqual(len(answer["tool_calls_detallado"]), 10)
        self.assertFalse(answer["tool_call_budget_exceeded"])
        self.assertEqual(answer["_telemetria"]["tool_calls_executed"], 10)

    def test_g_xbrl_guardrail_still_rejects_a_mismatched_forced_call(self):
        turn1 = ([Message(tool_calls=[])], _base_structured(fuente="xbrl"))
        turn2 = ([], _base_structured(
            respuesta="65,728,000,000 USD", ticker="NVDA", ejercicio=2024,
            concepto_xbrl="Assets", unidad="USD", cifra=65728000000.0,
            fuente="xbrl",
        ))
        self._install([turn1, turn2])

        forced_ai = AIMessage(content="", tool_calls=[
            xbrl_call("NVDA", 2024, "Liabilities", "forced-1"),
        ])
        forced_tool = ToolMessage(content="otro concepto",
                                  tool_call_id="forced-1",
                                  name="get_xbrl_fact")
        with patch("common.agent.forced_evidence.forced_evidence_turn",
                   return_value=(forced_ai, [forced_tool])):
            answer = responder("pregunta numerica", max_reintentos=0)

        self.assertFalse(answer["guardrail_ok"])

    def test_h_public_tool_signatures_are_still_intact(self):
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
        self.assertEqual(responder.__name__, "responder")

    def test_forced_search_filings_missing_metadata_args_fails_trajectory(self):
        """Reproduce el bug real de dani-b6-001: el turno forzado llama a
        search_filings con query/ticker correctos pero SIN fiscal_year ni
        item. evaluador_trayectoria debe marcarlo como incompatible -esto no
        es un bug del evaluador, es la respuesta correcta ante argumentos
        incompletos-."""
        original = ("¿Qué consecuencias contempla Apple si ocurre un "
                   "accidente industrial en las instalaciones de uno de "
                   "sus proveedores?")
        turn1 = ([Message(tool_calls=[])], _base_structured())
        turn2 = ([], _base_structured(
            respuesta="He encontrado menciones generales.", fuente="ninguna",
        ))
        self._install([turn1, turn2])

        forced_ai = AIMessage(content="", tool_calls=[{
            "name": "search_filings", "id": "forced-1",
            "args": {"query": original, "ticker": "AAPL", "k": 5},
        }])
        forced_tool = ToolMessage(content="fragmentos...",
                                  tool_call_id="forced-1",
                                  name="search_filings")
        with patch("common.agent.forced_evidence.forced_evidence_turn",
                   return_value=(forced_ai, [forced_tool])):
            answer = responder(original, max_reintentos=0)

        golden = {
            "familia": "extractiva", "ticker": "AAPL", "fiscal_year": 2024,
            "item_esperado": "1A", "herramienta_esperada": ["search_filings"],
        }
        trajectory = evaluar_trayectoria_detallada(answer, golden)
        self.assertFalse(trajectory["acierto_trayectoria"])
        self.assertIn("busqueda_incompatible", trajectory["errores"])

    def test_forced_search_filings_with_full_metadata_args_passes_trajectory(self):
        """Con el prompt corregido (ticker/fiscal_year/item exigidos), un
        turno forzado que sí los incluye produce una trayectoria correcta."""
        original = ("¿Qué consecuencias contempla Apple si ocurre un "
                   "accidente industrial en las instalaciones de uno de "
                   "sus proveedores?")
        turn1 = ([Message(tool_calls=[])], _base_structured())
        turn2 = ([], _base_structured(
            respuesta="Podria haber lesiones graves.", ticker="AAPL",
            ejercicio=2024, fuente="texto",
            cita="an industrial accident could occur",
            chunk_id="AAPL-2024-1A-0003",
        ))
        self._install([turn1, turn2])

        forced_ai = AIMessage(content="", tool_calls=[
            search_call(original, "AAPL", 2024, "1A", "forced-1"),
        ])
        forced_tool = ToolMessage(content="fragmentos...",
                                  tool_call_id="forced-1",
                                  name="search_filings")
        with patch("common.agent.forced_evidence.forced_evidence_turn",
                   return_value=(forced_ai, [forced_tool])):
            answer = responder(original, max_reintentos=0)

        golden = {
            "familia": "extractiva", "ticker": "AAPL", "fiscal_year": 2024,
            "item_esperado": "1A", "herramienta_esperada": ["search_filings"],
        }
        trajectory = evaluar_trayectoria_detallada(answer, golden)
        self.assertTrue(trajectory["acierto_trayectoria"])

    def test_list_available_alone_still_triggers_forced_evidence(self):
        turn1 = ([Message(tool_calls=[
            {"name": "list_available", "id": "1", "args": {}},
        ])], _base_structured(respuesta="Cobertura listada"))
        turn2 = ([], NVDA_ASSETS_2024)
        agent = self._install([turn1, turn2])

        forced_ai = AIMessage(content="", tool_calls=[
            xbrl_call("NVDA", 2024, "Assets", "forced-1"),
        ])
        forced_tool = ToolMessage(content="valor real",
                                  tool_call_id="forced-1",
                                  name="get_xbrl_fact")
        with patch("common.agent.forced_evidence.forced_evidence_turn",
                   return_value=(forced_ai, [forced_tool])) as forced_mock:
            answer = responder("pregunta numerica", max_reintentos=0)

        forced_mock.assert_called_once()
        self.assertEqual(len(agent.invocations), 2)
        names = [c["name"] for c in answer["tool_calls_detallado"]]
        self.assertIn("get_xbrl_fact", names)

    def test_fuente_xbrl_without_get_xbrl_fact_call_still_triggers_forced_evidence(self):
        turn1 = ([Message(tool_calls=[
            search_call("pregunta numerica", "NVDA", 2024, "7", "1"),
        ])], _base_structured(respuesta="65,728 millones", fuente="xbrl"))
        turn2 = ([], NVDA_ASSETS_2024)
        agent = self._install([turn1, turn2])

        forced_ai = AIMessage(content="", tool_calls=[
            xbrl_call("NVDA", 2024, "Assets", "forced-1"),
        ])
        forced_tool = ToolMessage(content="valor real",
                                  tool_call_id="forced-1",
                                  name="get_xbrl_fact")
        with patch("common.agent.forced_evidence.forced_evidence_turn",
                   return_value=(forced_ai, [forced_tool])) as forced_mock:
            answer = responder("pregunta numerica", max_reintentos=0)

        forced_mock.assert_called_once()
        self.assertEqual(len(agent.invocations), 2)
        self.assertEqual(answer["cifra"], 65728000000.0)

    def test_forced_evidence_is_never_attempted_once_budget_is_exhausted(self):
        # list_available no cuenta como evidencia (EVIDENCE_TOOLS), asi que
        # 10 llamadas a list_available agotan el presupuesto sin fundamentar
        # la respuesta: debe fallar cerrado ANTES de intentar el turno
        # forzado, no despues.
        ten_calls = [{"name": "list_available", "id": f"pre-{i}", "args": {}}
                    for i in range(10)]
        turn1 = ([Message(tool_calls=ten_calls)],
                _base_structured(fuente="ninguna"))
        agent = self._install([turn1])

        with patch("common.agent.forced_evidence.forced_evidence_turn") as \
                forced_mock:
            answer = responder("pregunta numerica", max_reintentos=0)

        forced_mock.assert_not_called()
        self.assertEqual(len(agent.invocations), 1)
        self.assertFalse(answer["guardrail_ok"])


if __name__ == "__main__":
    unittest.main()
