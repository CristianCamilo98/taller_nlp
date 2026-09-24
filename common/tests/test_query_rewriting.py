from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.retrieval import query_rewriting as rewriting
from common.scripts.smoke_query_rewriter import run_smoke
from common.scripts import generate_query_rewrites as mass


EXPECTED_PROMPT = (
    "Rewrite the financial question into a concise English search query "
    "optimized to retrieve relevant passages from SEC 10-K/10-Q filings. "
    "Preserve the exact information need, company, fiscal year and filing "
    "item when present. Do not answer the question. Do not invent facts. "
    "Return only the rewritten search query."
)
QUESTION = "¿Qué riesgo tuvo Apple en FY2024?"


def response(*, content="valid rewrite", message_extra=None, **extra):
    message = {"role": "assistant", "content": content}
    message.update(message_extra or {})
    value = {
        "id": "generation-1",
        "model": rewriting.QUERY_REWRITER_MODEL_ID,
        "choices": [{
            "finish_reason": "stop",
            "message": message,
        }],
        "usage": {
            "prompt_tokens": 71,
            "completion_tokens": 9,
            "total_tokens": 80,
            "cost": 0.0000123,
        },
    }
    value.update(extra)
    return value


class QueryRewriteTests(unittest.TestCase):
    def invoke(self, api_response):
        with patch.object(
            rewriting.time, "perf_counter", side_effect=[10.0, 11.25]
        ):
            return rewriting.rewrite_query(
                QUESTION, request_fn=lambda _payload: api_response
            )

    def error_for(self, api_response):
        with self.assertRaises(rewriting.QueryRewriteError) as caught:
            self.invoke(api_response)
        return caught.exception

    def test_a_valid_text_content_is_accepted_as_rewrite(self):
        result = self.invoke(response(content="  valid rewrite  "))
        self.assertEqual(result["rewrite"], "valid rewrite")

    def test_b_null_content_with_reasoning_is_rejected_and_sanitized(self):
        private_reasoning = "private chain of thought"
        error = self.error_for(response(
            content=None,
            message_extra={
                "reasoning": private_reasoning,
                "reasoning_details": [{"text": "more private thought"}],
            },
        ))
        diagnostic = error.diagnostic
        self.assertEqual(diagnostic["error_type"], "null_or_non_text_content")
        self.assertTrue(diagnostic["content_is_null"])
        self.assertEqual(diagnostic["content_length"], 0)
        self.assertTrue(diagnostic["reasoning_present"])
        self.assertGreater(diagnostic["reasoning_length"], 0)
        self.assertEqual(
            diagnostic["message_keys"],
            ["content", "reasoning", "reasoning_details", "role"],
        )
        serialized = json.dumps(diagnostic)
        self.assertNotIn(private_reasoning, serialized)
        self.assertNotIn("more private thought", serialized)

    def test_c_empty_text_content_is_rejected(self):
        error = self.error_for(response(content="   "))
        self.assertEqual(error.diagnostic["error_type"], "empty_text_content")
        self.assertFalse(error.diagnostic["content_is_null"])
        self.assertEqual(error.diagnostic["content_length"], 3)

    def test_d_empty_choices_is_rejected(self):
        error = self.error_for({
            "model": rewriting.QUERY_REWRITER_MODEL_ID,
            "choices": [],
            "usage": {"total_tokens": 10},
        })
        self.assertEqual(error.diagnostic["error_type"], "missing_or_empty_choices")
        self.assertEqual(error.diagnostic["message_keys"], [])

    def test_e_missing_message_is_rejected(self):
        error = self.error_for({
            "model": rewriting.QUERY_REWRITER_MODEL_ID,
            "choices": [{"finish_reason": "stop"}],
            "usage": {},
        })
        self.assertEqual(error.diagnostic["error_type"], "missing_message")
        self.assertEqual(error.diagnostic["finish_reason"], "stop")

    def test_f_normal_response_records_status_model_finish_reason_and_usage(self):
        result = self.invoke(rewriting.OpenRouterHTTPResult(
            status=200, payload=response()
        ))
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["requested_model"], rewriting.QUERY_REWRITER_MODEL)
        self.assertEqual(result["effective_model"], rewriting.QUERY_REWRITER_MODEL_ID)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["latency_s"], 1.25)
        self.assertEqual(result["usage"]["prompt_tokens"], 71)
        self.assertEqual(result["usage"]["completion_tokens"], 9)
        self.assertEqual(result["usage"]["total_tokens"], 80)
        self.assertEqual(result["usage"]["cost"], 0.0000123)

    def test_request_payload_is_frozen_and_disables_reasoning(self):
        calls = []

        def fake_request(payload):
            calls.append(payload)
            return response()

        rewriting.rewrite_query(QUESTION, request_fn=fake_request)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0],
            {
                "model": rewriting.QUERY_REWRITER_MODEL_ID,
                "temperature": 0.0,
                "max_tokens": 512,
                "reasoning": {"enabled": False},
                "messages": [
                    {"role": "system", "content": EXPECTED_PROMPT},
                    {"role": "user", "content": QUESTION},
                ],
            },
        )
        self.assertNotIn("stream", calls[0])
        self.assertNotIn("response_format", calls[0])
        self.assertNotIn("provider", calls[0])
        self.assertEqual(rewriting.QUERY_REWRITER_PROMPT, EXPECTED_PROMPT)


class QueryRewriteSmokeTests(unittest.TestCase):
    def test_success_persists_one_result_with_provenance(self):
        result = {
            "question": "original",
            "rewrite": "rewritten",
            "requested_model": rewriting.QUERY_REWRITER_MODEL,
            "effective_model": rewriting.QUERY_REWRITER_MODEL_ID,
            "provider_response_id": "generation-1",
            "http_status": 200,
            "finish_reason": "stop",
            "latency_s": 0.5,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cost": None,
            },
        }
        calls = []

        def fake_rewrite(question):
            calls.append(question)
            return {**result, "question": question}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            payload = run_smoke(
                question_id="dani-b6-001",
                output_path=output,
                rewrite_fn=fake_rewrite,
                questions=[{
                    "question_id": "dani-b6-001",
                    "question": "original",
                    "ticker": "AAPL",
                    "fiscal_year": 2024,
                    "item": "1A",
                }],
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 1)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["manifest"]["api_call_count"], 1)
        self.assertEqual(payload["manifest"]["prompt"], EXPECTED_PROMPT)
        self.assertEqual(payload["manifest"]["reasoning"], {"enabled": False})
        self.assertEqual(payload["manifest"]["max_tokens"], 512)
        self.assertEqual(persisted["result"]["question_id"], "dani-b6-001")
        self.assertEqual(persisted["result"]["rewrite"], "rewritten")

    def test_failure_persists_sanitized_diagnostic_before_reraising(self):
        diagnostic = {
            "requested_model": rewriting.QUERY_REWRITER_MODEL,
            "effective_model": rewriting.QUERY_REWRITER_MODEL_ID,
            "http_status": 200,
            "finish_reason": "length",
            "message_keys": ["content", "reasoning"],
            "content_is_null": True,
            "content_length": 0,
            "reasoning_present": True,
            "reasoning_length": 123,
            "usage": {"completion_tokens": 512},
            "latency_s": 1.5,
            "error_type": "null_or_non_text_content",
        }

        def fail(_question):
            raise rewriting.QueryRewriteError("safe error", diagnostic=diagnostic)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic.json"
            with self.assertRaises(rewriting.QueryRewriteError):
                run_smoke(
                    question_id="dani-b6-001",
                    output_path=output,
                    rewrite_fn=fail,
                    questions=[{
                        "question_id": "dani-b6-001",
                        "question": "original",
                        "ticker": "AAPL",
                        "fiscal_year": 2024,
                        "item": "1A",
                    }],
                )
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(persisted["status"], "error")
        self.assertEqual(persisted["diagnostic"], diagnostic)
        self.assertNotIn("reasoning_text", json.dumps(persisted))


def mass_questions(count):
    return [
        {
            "question_id": f"q-{index:03d}",
            "question": f"Pregunta {index}",
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "item": "1A",
        }
        for index in range(1, count + 1)
    ]


def successful_rewrite(question):
    return {
        "question": question,
        "rewrite": f"English {question}",
        "requested_model": rewriting.QUERY_REWRITER_MODEL,
        "effective_model": rewriting.QUERY_REWRITER_MODEL_ID,
        "provider_response_id": f"generation-{question}",
        "http_status": 200,
        "finish_reason": "stop",
        "latency_s": 0.25,
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 8,
            "total_tokens": 88,
            "cost": 0.00001,
        },
    }


class MassRewriteRunnerTests(unittest.TestCase):
    def test_interruption_after_17_resumes_at_18_without_regeneration(self):
        questions = mass_questions(20)
        first_calls = []

        def interrupt_on_18(question):
            first_calls.append(question)
            if len(first_calls) == 18:
                raise RuntimeError("simulated interruption")
            return successful_rewrite(question)

        resumed_calls = []

        def resume(question):
            resumed_calls.append(question)
            return successful_rewrite(question)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rewrites.jsonl"
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                mass.generate_all(
                    output_path=output,
                    questions=questions,
                    rewrite_fn=interrupt_on_18,
                )
            partial = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(partial), 17)
            summary = mass.generate_all(
                output_path=output,
                questions=questions,
                rewrite_fn=resume,
            )
            final = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(resumed_calls), 3)
        self.assertEqual(resumed_calls[0], "Pregunta 18")
        self.assertEqual(summary["skipped_from_cache"], 17)
        self.assertEqual(summary["api_calls_this_run"], 3)
        self.assertEqual(len(final), 20)
        self.assertEqual(len({row["question_id"] for row in final}), 20)

    def test_only_query_rewrite_error_creates_explicit_original_fallback(self):
        questions = mass_questions(2)
        private_reasoning = "must never persist"

        def rewrite(question):
            if question == "Pregunta 1":
                raise rewriting.QueryRewriteError(
                    "safe failure",
                    diagnostic={
                        "effective_model": rewriting.QUERY_REWRITER_MODEL_ID,
                        "http_status": 200,
                        "finish_reason": "length",
                        "message_keys": ["content", "reasoning"],
                        "content_is_null": True,
                        "content_length": 0,
                        "reasoning_present": True,
                        "reasoning_length": len(private_reasoning),
                        "usage": {"completion_tokens": 512},
                        "latency_s": 1.0,
                        "error_type": "null_or_non_text_content",
                    },
                )
            return successful_rewrite(question)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rewrites.jsonl"
            summary = mass.generate_all(
                output_path=output,
                questions=questions,
                rewrite_fn=rewrite,
            )
            rows = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["fallbacks_total"], 1)
        self.assertEqual(rows[0]["query_rewritten"], rows[0]["query_original"])
        self.assertEqual(rows[0]["status"], "fallback")
        self.assertTrue(rows[0]["fallback_used"])
        self.assertFalse(rows[1]["fallback_used"])
        self.assertNotIn(private_reasoning, json.dumps(rows))

    def test_duplicate_cache_fails_before_any_new_call(self):
        questions = mass_questions(1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rewrites.jsonl"
            mass.generate_all(
                output_path=output,
                questions=questions,
                rewrite_fn=successful_rewrite,
            )
            line = output.read_text(encoding="utf-8")
            with output.open("a", encoding="utf-8") as stream:
                stream.write(line)
            with self.assertRaisesRegex(mass.RewriteCacheError, "duplicado"):
                mass.generate_all(
                    output_path=output,
                    questions=questions,
                    rewrite_fn=lambda _question: self.fail("must not call API"),
                )


if __name__ == "__main__":
    unittest.main()
