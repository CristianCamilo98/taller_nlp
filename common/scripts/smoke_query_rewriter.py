"""Ejecuta una sola query del benchmark con el rewriter DeepSeek congelado."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from common.benchmark import BENCHMARK_NAME, BENCHMARK_PATH
from common.retrieval.query_rewriting import (
    QUERY_REWRITER_MODEL,
    QUERY_REWRITER_MODEL_ID,
    QUERY_REWRITER_PROMPT,
    rewrite_query,
)
from common.scripts.evaluate_retrieval_benchmark import (
    load_benchmark,
    select_questions,
    verify_frozen_benchmark,
)


def run_smoke(
    *,
    question_id: str,
    output_path: Path,
    rewrite_fn: Callable[[str], dict[str, Any]] = rewrite_query,
) -> dict[str, Any]:
    benchmark_sha = verify_frozen_benchmark(BENCHMARK_PATH)
    selected = select_questions(load_benchmark(BENCHMARK_PATH), [question_id])
    question = selected[0]
    result = rewrite_fn(str(question["question"]))
    payload = {
        "manifest": {
            "runner": "common.scripts.smoke_query_rewriter",
            "benchmark_name": BENCHMARK_NAME,
            "benchmark_sha256": benchmark_sha,
            "question_id": question_id,
            "requested_model": QUERY_REWRITER_MODEL,
            "requested_model_id": QUERY_REWRITER_MODEL_ID,
            "prompt": QUERY_REWRITER_PROMPT,
            "prompt_sha256": hashlib.sha256(
                QUERY_REWRITER_PROMPT.encode("utf-8")
            ).hexdigest().upper(),
            "temperature": 0.0,
            "max_tokens": 128,
            "api_call_count": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "result": {
            "question_id": question_id,
            "ticker": question["ticker"],
            "fiscal_year": int(question["fiscal_year"]),
            "item": str(question["item"]),
            **result,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    usage = result["usage"]
    print(
        f"DeepSeek rewrite smoke · question_id={question_id} · "
        f"requested={result['requested_model']} · "
        f"effective={result['effective_model']}",
        flush=True,
    )
    print(
        f"latency_s={result['latency_s']} · "
        f"input_tokens={usage['input_tokens']} · "
        f"output_tokens={usage['output_tokens']} · "
        f"total_tokens={usage['total_tokens']} · cost={usage['cost']}",
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
