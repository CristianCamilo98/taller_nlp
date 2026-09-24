"""Ejecuta una sola query del benchmark con el rewriter DeepSeek congelado."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from common.retrieval.query_rewriting import (
    OPENROUTER_CHAT_COMPLETIONS_URL,
    QUERY_REWRITER_MAX_TOKENS,
    QUERY_REWRITER_MODEL,
    QUERY_REWRITER_MODEL_ID,
    QUERY_REWRITER_PROMPT,
    QUERY_REWRITER_REASONING,
    QueryRewriteError,
    rewrite_query,
)
from common.scripts.generate_query_rewrites import (
    BENCHMARK_NAME,
    BENCHMARK_SHA256,
    DEFAULT_BENCHMARK_PATH,
    load_frozen_benchmark,
    select_question,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_smoke(
    *,
    question_id: str,
    output_path: Path,
    rewrite_fn: Callable[[str], dict[str, Any]] = rewrite_query,
    questions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    values = questions if questions is not None else load_frozen_benchmark()
    question = select_question(values, question_id)
    manifest = {
        "runner": "common.scripts.smoke_query_rewriter",
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_sha256": BENCHMARK_SHA256,
        "benchmark_path": str(DEFAULT_BENCHMARK_PATH),
        "question_id": question_id,
        "requested_model": QUERY_REWRITER_MODEL,
        "requested_model_id": QUERY_REWRITER_MODEL_ID,
        "prompt": QUERY_REWRITER_PROMPT,
        "prompt_sha256": hashlib.sha256(
            QUERY_REWRITER_PROMPT.encode("utf-8")
        ).hexdigest().upper(),
        "temperature": 0.0,
        "max_tokens": QUERY_REWRITER_MAX_TOKENS,
        "reasoning": dict(QUERY_REWRITER_REASONING),
        "endpoint": OPENROUTER_CHAT_COMPLETIONS_URL,
        "api_call_count": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        result = rewrite_fn(str(question["question"]))
    except QueryRewriteError as exc:
        payload = {
            "manifest": manifest,
            "status": "error",
            "diagnostic": exc.diagnostic,
        }
        _write_json(output_path, payload)
        print(
            f"DeepSeek rewrite smoke FAILED · question_id={question_id} · "
            f"error_type={exc.diagnostic.get('error_type')}",
            flush=True,
        )
        print(f"Artifact: {output_path.resolve()}", flush=True)
        raise
    payload = {
        "manifest": manifest,
        "status": "success",
        "result": {
            "question_id": question_id,
            "ticker": question["ticker"],
            "fiscal_year": int(question["fiscal_year"]),
            "item": str(question["item"]),
            "fallback_used": False,
            **result,
        },
    }
    _write_json(output_path, payload)
    usage = result["usage"]
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    print(
        f"DeepSeek rewrite smoke · question_id={question_id} · "
        f"requested={result['requested_model']} · "
        f"effective={result['effective_model']}",
        flush=True,
    )
    print(
        f"latency_s={result['latency_s']} · "
        f"input_tokens={input_tokens} · "
        f"output_tokens={output_tokens} · "
        f"total_tokens={usage.get('total_tokens')} · "
        f"cost={usage.get('cost')}",
        flush=True,
    )
    print(f"Artifact: {output_path.resolve()}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Una única llamada de smoke al query rewriter DeepSeek"
    )
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_smoke(
        question_id=args.question_id,
        output_path=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
