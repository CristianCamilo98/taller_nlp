from __future__ import annotations

import ast
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from common.agent.agent import MODELO, SYSTEM, _agent_middleware
from common.agent.middleware_tool_dedup import (
    DUPLICATE_TOOL_CALL_TYPE,
    ToolCallDedupMiddleware,
)
from common.benchmark_config import BENCHMARK
from common.responder import _extract_tool_trajectory
from common.tools.tools import search_filings


ROOT = Path(__file__).resolve().parents[2]


def call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def request(tool_call: dict, messages: list) -> SimpleNamespace:
    return SimpleNamespace(tool_call=tool_call, state={"messages": messages})


class FinalConfigurationTests(unittest.TestCase):
    def test_final_model_and_retrieval_are_frozen(self):
        self.assertEqual(MODELO, "openrouter:deepseek/deepseek-v4-flash")
        self.assertEqual(BENCHMARK.model, MODELO)
        self.assertEqual(BENCHMARK.requested_model_id, "deepseek/deepseek-v4-flash")
        self.assertEqual(BENCHMARK.retrieval_profile, "gemini-embedding-2")
        self.assertEqual(BENCHMARK.embedding_model, "google/gemini-embedding-2")

    def test_search_filings_forces_gemini_regardless_of_environment(self):
        fake_hits = [{
            "chunk_id": "c1",
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "item": "1A",
            "texto": "text",
            "n_tokens": 1,
            "contiene_tabla": False,
            "puntuacion": 0.9,
        }]
        with patch.dict(
            os.environ, {"MIAX_RETRIEVAL_PROFILE": "bge-small-baseline"}
        ), patch(
            "common.retrieval.dense_baseline.buscar", return_value=fake_hits
        ) as buscar:
            result = search_filings.func(
                "¿Pregunta original?", "aapl", 2024, "1A", 5
            )
        self.assertIn("c1", result)
        buscar.assert_called_once_with(
            "¿Pregunta original?",
            ticker="AAPL",
            fiscal_year=2024,
            item="1A",
            k=5,
            profile_name="gemini-embedding-2",
        )

    def test_public_tool_signatures_remain_exact(self):
        source = (ROOT / "common/tools/tools.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        def signature(name):
            node = next(
                value for value in tree.body
                if isinstance(value, ast.FunctionDef) and value.name == name
            )
            return (
                [argument.arg for argument in node.args.args],
                [ast.unparse(default) for default in node.args.defaults],
            )

        self.assertEqual(signature("list_available"), ([], []))
        self.assertEqual(
            signature("get_xbrl_fact"),
            (["ticker", "fiscal_year", "concept"], []),
        )
        self.assertEqual(
            signature("search_filings"),
            (
                ["query", "ticker", "fiscal_year", "item", "k"],
                ["None", "None", "None", "5"],
            ),
        )
        self.assertEqual(
            signature("read_section"),
            (["ticker", "fiscal_year", "item"], []),
        )

    def test_system_prompt_requires_original_first_query(self):
        self.assertIn("PRIMERA llamada a search_filings", SYSTEM)
        self.assertIn("pregunta original", SYSTEM)
        self.assertIn("no la traduzcas, resumas ni reformules", SYSTEM)
        self.assertIn("ticker, fiscal_year e item", SYSTEM)

    def test_run_limit_and_dedup_middleware_are_both_active(self):
        middleware = _agent_middleware()
        dedup = [value for value in middleware
                 if isinstance(value, ToolCallDedupMiddleware)]
        limits = [value for value in middleware
                  if isinstance(value, ToolCallLimitMiddleware)]
        self.assertEqual(len(dedup), 1)
        self.assertEqual(len(limits), 1)
        self.assertEqual(limits[0].run_limit, 10)


class ToolDedupTests(unittest.TestCase):
    def setUp(self):
        self.middleware = ToolCallDedupMiddleware()
        self.handler = Mock(side_effect=lambda req: ToolMessage(
            content="executed",
            tool_call_id=req.tool_call["id"],
            name=req.tool_call["name"],
        ))

    def test_second_identical_call_is_blocked_and_recorded(self):
        first = call("search_filings", {"query": "q", "ticker": "AAPL"}, "1")
        second = call("search_filings", {"ticker": "AAPL", "query": "q"}, "2")
        first_ai = AIMessage(content="", tool_calls=[first])
        self.middleware.wrap_tool_call(request(first, [first_ai]), self.handler)
        prior_result = ToolMessage(content="ok", tool_call_id="1")
        second_ai = AIMessage(content="", tool_calls=[second])
        blocked = self.middleware.wrap_tool_call(
            request(second, [first_ai, prior_result, second_ai]), self.handler
        )
        self.assertEqual(self.handler.call_count, 1)
        self.assertEqual(blocked.status, "error")
        self.assertEqual(blocked.artifact["type"], DUPLICATE_TOOL_CALL_TYPE)
        self.assertFalse(blocked.artifact["executed"])

    def test_same_tool_with_different_arguments_is_allowed(self):
        first = call("search_filings", {"query": "q1"}, "1")
        second = call("search_filings", {"query": "q2"}, "2")
        self.middleware.wrap_tool_call(
            request(first, [AIMessage(content="", tool_calls=[first])]),
            self.handler,
        )
        messages = [
            AIMessage(content="", tool_calls=[first]),
            ToolMessage(content="ok", tool_call_id="1"),
            AIMessage(content="", tool_calls=[second]),
        ]
        self.middleware.wrap_tool_call(request(second, messages), self.handler)
        self.assertEqual(self.handler.call_count, 2)

    def test_dedup_resets_with_new_question_history(self):
        first = call("get_xbrl_fact", {"ticker": "AAPL", "fiscal_year": 2024,
                                      "concept": "Assets"}, "1")
        second = {**first, "id": "2"}
        self.middleware.wrap_tool_call(
            request(first, [AIMessage(content="", tool_calls=[first])]),
            self.handler,
        )
        self.middleware.wrap_tool_call(
            request(second, [AIMessage(content="", tool_calls=[second])]),
            self.handler,
        )
        self.assertEqual(self.handler.call_count, 2)

    def test_blocked_duplicate_is_separate_from_executed_trajectory(self):
        first = call("search_filings", {"query": "q"}, "1")
        second = call("search_filings", {"query": "q"}, "2")
        diagnostic = {
            "type": DUPLICATE_TOOL_CALL_TYPE,
            "executed": False,
        }
        messages = [
            AIMessage(content="", tool_calls=[first]),
            ToolMessage(content="ok", tool_call_id="1"),
            AIMessage(content="", tool_calls=[second]),
            ToolMessage(
                content="duplicate",
                tool_call_id="2",
                status="error",
                artifact=diagnostic,
            ),
        ]
        executed, blocked = _extract_tool_trajectory(messages)
        self.assertEqual(executed, [{"name": "search_filings", "args": {"query": "q"}}])
        self.assertEqual(len(blocked), 1)
        self.assertFalse(blocked[0]["executed"])


class FinalProvenanceTests(unittest.TestCase):
    def test_retrieval_final_provenance_comes_from_canonical_artifact(self):
        from common.evaluar import _retrieval_final_provenance

        value = _retrieval_final_provenance()
        final = json.loads(
            (ROOT / "common/results/retrieval_final/retrieval_final_48.json")
            .read_text(encoding="utf-8")
        )["manifest"]
        self.assertEqual(value["faiss_sha256"], final["faiss_sha256"])
        self.assertEqual(value["chunks_sha256"], final["chunks_sha256"])
        self.assertEqual(value["chunks_meta_sha256"], final["chunks_meta_sha256"])
        self.assertEqual(value["gemini_model"], "google/gemini-embedding-2")

    def test_evaluar_records_final_configuration_offline(self):
        import common.evaluar as evaluator

        question = {
            "id": "offline-1",
            "familia": "no_aplica",
            "pregunta": "Pregunta offline",
            "respuesta_esperada": None,
            "cifra_esperada": None,
        }
        response = {
            "respuesta": "Sin datos.",
            "cifra": None,
            "unidad": None,
            "ticker": None,
            "ejercicio": None,
            "fuente": "ninguna",
            "cita": None,
            "chunk_id": None,
            "tool_calls_agente": [],
            "tool_calls_detallado": [],
            "tool_calls_bloqueados": [],
            "guardrail_retry_count": 0,
            "_telemetria": {},
        }
        retry = {
            "retry_count": 0,
            "rate_limited": False,
            "backoff_s": 0.0,
            "latencia_activa_s": 0.0,
        }
        with patch.object(evaluator, "cargar_golden", return_value=[question]), patch.object(
            evaluator, "_sha256", return_value="hash"
        ), patch.object(
            evaluator, "_commit_sha", return_value="commit"
        ), patch.object(
            evaluator, "_artifact_hashes", return_value={"corpus.faiss": "faiss"}
        ), patch.object(
            evaluator, "_retrieval_final_provenance",
            return_value={"faiss_sha256": "canonical"},
        ), patch.object(
            evaluator, "_invocar_con_reintentos", return_value=(response, retry)
        ):
            rows = evaluator.evaluar("unused.jsonl", pausa_entre_preguntas=0)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["agent_model"], "openrouter:deepseek/deepseek-v4-flash")
        self.assertEqual(row["retrieval_profile"], "gemini-embedding-2")
        self.assertIn("original query + Gemini Embedding 2", row["retrieval_pipeline"])
        self.assertEqual(row["retrieval_final_provenance"]["faiss_sha256"], "canonical")


if __name__ == "__main__":
    unittest.main()
