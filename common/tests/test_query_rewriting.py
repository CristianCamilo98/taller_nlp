from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.retrieval import query_rewriting as rewriting
from common.scripts.smoke_query_rewriter import run_smoke


EXPECTED_PROMPT = (
    "Rewrite the financial question into a concise English search query "
    "optimized to retrieve relevant passages from SEC 10-K/10-Q filings. "
    "Preserve the exact information need, company, fiscal year and filing "
    "item when present. Do not answer the question. Do not invent facts. "
    "Return only the rewritten search query."
)


class QueryRewriteTests(unittest.TestCase):
    def test_frozen_model_prompt_and_single_request(self):
        calls = []

        def fake_request(payload):
            calls.append(payload)
            return {
                "id": "generation-1",
                "model": "deepseek/deepseek-v4-flash",
                "choices": [{
                    "message": {"content": "Apple FY2024 supplier accident risks"}
                }],
                "usage": {
                    "prompt_tokens": 71,
                    "completion_tokens": 9,
                    "total_tokens": 80,
                    "cost": 0.0000123,
                },
            }

        with patch.object(
            rewriting.time, "perf_counter", side_effect=[10.0, 11.25]
        ):
            result = rewriting.rewrite_query(
                "¿Qué riesgo tuvo Apple en FY2024?", request_fn=fake_request
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0],
            {
                "model": rewriting.QUERY_REWRITER_MODEL_ID,
                "temperature": 0.0,
                "max_tokens": 128,
                "messages": [
                    {"role": "system", "content": EXPECTED_PROMPT},
                    {"role": "user", "content": "¿Qué riesgo tuvo Apple en FY2024?"},
                ],
            },
        )
        self.assertEqual(rewriting.QUERY_REWRITER_PROMPT, EXPECTED_PROMPT)
        self.assertEqual(result["requested_model"], rewriting.QUERY_REWRITER_MODEL)
        self.assertEqual(result["effective_model"], rewriting.QUERY_REWRITER_MODEL_ID)
        self.assertEqual(result["latency_s"], 1.25)
        self.assertEqual(result["usage"]["input_tokens"], 71)
        self.assertEqual(result["usage"]["output_tokens"], 9)
        self.assertEqual(result["usage"]["total_tokens"], 80)
        self.assertEqual(result["usage"]["cost"], 0.0000123)

    def test_invalid_response_fails_without_fallback_to_original(self):
        with self.assertRaises(rewriting.QueryRewriteError):
            rewriting.rewrite_query("question", request_fn=lambda _payload: {})


class QueryRewriteSmokeTests(unittest.TestCase):
    def test_smoke_persists_one_result_with_provenance(self):
        result = {
            "question": "original",
            "rewrite": "rewritten",
            "requested_model": rewriting.QUERY_REWRITER_MODEL,
            "effective_model": rewriting.QUERY_REWRITER_MODEL_ID,
            "provider_response_id": "generation-1",
            "latency_s": 0.5,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
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
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 1)
        self.assertEqual(payload["manifest"]["api_call_count"], 1)
        self.assertEqual(payload["manifest"]["prompt"], EXPECTED_PROMPT)
        self.assertEqual(persisted["result"]["question_id"], "dani-b6-001")
        self.assertEqual(persisted["result"]["rewrite"], "rewritten")


if __name__ == "__main__":
    unittest.main()
