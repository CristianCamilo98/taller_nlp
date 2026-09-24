"""Genera y cachea incrementalmente las 48 rewrites del benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from common.retrieval.query_rewriting import (
    QUERY_REWRITER_MAX_TOKENS,
    QUERY_REWRITER_MODEL,
    QUERY_REWRITER_PROMPT,
    QUERY_REWRITER_REASONING,
    QueryRewriteError,
    rewrite_query,
)


BENCHMARK_NAME = "retrieval_benchmark_v2"
BENCHMARK_SHA256 = (
    "6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C"
)
DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "retrieval_benchmark_v2.jsonl"
)
PROMPT_SHA256 = hashlib.sha256(
    QUERY_REWRITER_PROMPT.encode("utf-8")
).hexdigest().upper()


class RewriteCacheError(RuntimeError):
    """El benchmark o la caché incremental viola el contrato congelado."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_frozen_benchmark(path: Path = DEFAULT_BENCHMARK_PATH) -> list[dict]:
    digest = _sha256(path)
    if digest != BENCHMARK_SHA256:
        raise RewriteCacheError(
            f"SHA-256 de benchmark inválido: {digest}; esperado {BENCHMARK_SHA256}"
        )
    with path.open(encoding="utf-8") as stream:
        questions = [json.loads(line) for line in stream if line.strip()]
    ids = [str(question.get("question_id")) for question in questions]
    if len(questions) != 48 or len(set(ids)) != 48:
        raise RewriteCacheError(
            f"Benchmark debe contener 48 question_id únicos; encontrados {len(set(ids))}"
        )
    required = {"question_id", "question", "ticker", "fiscal_year", "item"}
    for question in questions:
        missing = required - set(question)
        if missing:
            raise RewriteCacheError(
                f"{question.get('question_id')}: faltan campos {sorted(missing)}"
            )
    return questions


def select_question(questions: list[dict], question_id: str) -> dict:
    matches = [row for row in questions if row["question_id"] == question_id]
    if len(matches) != 1:
        raise RewriteCacheError(f"question_id no encontrado: {question_id}")
    return matches[0]


def _row_contract(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_sha256": BENCHMARK_SHA256,
        "question_id": str(question["question_id"]),
        "ticker": str(question["ticker"]),
        "fiscal_year": int(question["fiscal_year"]),
        "item": str(question["item"]),
        "query_original": str(question["question"]),
        "requested_model": QUERY_REWRITER_MODEL,
        "prompt_sha256": PROMPT_SHA256,
        "temperature": 0.0,
        "max_tokens": QUERY_REWRITER_MAX_TOKENS,
        "reasoning": dict(QUERY_REWRITER_REASONING),
    }


def _validate_cached_row(row: dict[str, Any], question: dict[str, Any]) -> None:
    expected = _row_contract(question)
    for key, value in expected.items():
        if row.get(key) != value:
            raise RewriteCacheError(
                f"Cache incompatible para {question['question_id']} en {key}: "
                f"{row.get(key)!r} != {value!r}"
            )
    if row.get("status") not in {"success", "fallback"}:
        raise RewriteCacheError(
            f"Cache incompatible para {question['question_id']}: status inválido"
        )
    if not isinstance(row.get("fallback_used"), bool):
        raise RewriteCacheError(
            f"Cache incompatible para {question['question_id']}: fallback inválido"
        )
    if row["fallback_used"] != (row["status"] == "fallback"):
        raise RewriteCacheError(
            f"Cache inconsistente para {question['question_id']}: fallback/status"
        )
    rewritten = row.get("query_rewritten")
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise RewriteCacheError(
            f"Cache incompatible para {question['question_id']}: rewrite vacío"
        )


def load_cache(
    path: Path, questions: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    expected = {str(row["question_id"]): row for row in questions}
    cached: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RewriteCacheError(
                    f"Cache JSONL inválida en línea {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise RewriteCacheError(
                    f"Cache JSONL no contiene objeto en línea {line_number}"
                )
            question_id = str(row.get("question_id"))
            if question_id not in expected:
                raise RewriteCacheError(
                    f"Cache contiene question_id desconocido: {question_id}"
                )
            if question_id in cached:
                raise RewriteCacheError(
                    f"Cache contiene question_id duplicado: {question_id}"
                )
            _validate_cached_row(row, expected[question_id])
            cached[question_id] = row
    return cached


def _append_durable(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _success_row(question: dict[str, Any], result: dict[str, Any]) -> dict:
    return {
        "schema_version": 1,
        **_row_contract(question),
        "query_rewritten": result["rewrite"],
        "status": "success",
        "fallback_used": False,
        "effective_model": result.get("effective_model"),
        "provider_response_id": result.get("provider_response_id"),
        "http_status": result.get("http_status"),
        "finish_reason": result.get("finish_reason"),
        "usage": result.get("usage") or {},
        "latency_s": result.get("latency_s"),
        "error": None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _fallback_row(question: dict[str, Any], exc: QueryRewriteError) -> dict:
    diagnostic = dict(exc.diagnostic)
    return {
        "schema_version": 1,
        **_row_contract(question),
        "query_rewritten": str(question["question"]),
        "status": "fallback",
        "fallback_used": True,
        "effective_model": diagnostic.get("effective_model"),
        "provider_response_id": None,
        "http_status": diagnostic.get("http_status"),
        "finish_reason": diagnostic.get("finish_reason"),
        "usage": diagnostic.get("usage") or {},
        "latency_s": diagnostic.get("latency_s"),
        "error": diagnostic,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_complete_cache(
    cached: dict[str, dict[str, Any]], questions: list[dict[str, Any]]
) -> None:
    expected_ids = {str(question["question_id"]) for question in questions}
    actual_ids = set(cached)
    if len(cached) != len(questions) or actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise RewriteCacheError(
            f"Cache incompleta: rows={len(cached)}/{len(questions)}, "
            f"missing={missing}, extra={extra}"
        )


def generate_all(
    *,
    output_path: Path,
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    rewrite_fn: Callable[[str], dict[str, Any]] = rewrite_query,
    questions: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    values = questions if questions is not None else load_frozen_benchmark(
        benchmark_path
    )
    cached = load_cache(output_path, values)
    api_calls = 0
    generated = 0
    skipped = 0
    for position, question in enumerate(values, 1):
        question_id = str(question["question_id"])
        if question_id in cached:
            skipped += 1
            print(f"[{position}/{len(values)}] {question_id} CACHED", flush=True)
            continue
        api_calls += 1
        try:
            result = rewrite_fn(str(question["question"]))
        except QueryRewriteError as exc:
            row = _fallback_row(question, exc)
            marker = "FALLBACK"
        else:
            row = _success_row(question, result)
            marker = "OK"
        _append_durable(output_path, row)
        cached[question_id] = row
        generated += 1
        print(f"[{position}/{len(values)}] {question_id} {marker}", flush=True)
    validate_complete_cache(cached, values)
    fallbacks = sum(row["fallback_used"] for row in cached.values())
    summary = {
        "questions": len(values),
        "generated_this_run": generated,
        "skipped_from_cache": skipped,
        "api_calls_this_run": api_calls,
        "fallbacks_total": int(fallbacks),
    }
    print(
        "Complete: " + json.dumps(summary, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera/cachea las 48 rewrites; reanuda por question_id"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    args = parser.parse_args()
    generate_all(
        output_path=args.output.resolve(),
        benchmark_path=args.benchmark.resolve(),
    )


if __name__ == "__main__":
    main()
