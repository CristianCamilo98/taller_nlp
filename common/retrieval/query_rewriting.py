"""Query rewriting congelado mediante una única petición a OpenRouter."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable


OPENROUTER_CHAT_COMPLETIONS_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)
QUERY_REWRITER_MODEL = "openrouter:deepseek/deepseek-v4-flash"
QUERY_REWRITER_MODEL_ID = "deepseek/deepseek-v4-flash"
QUERY_REWRITER_PROMPT = (
    "Rewrite the financial question into a concise English search query "
    "optimized to retrieve relevant passages from SEC 10-K/10-Q filings. "
    "Preserve the exact information need, company, fiscal year and filing "
    "item when present. Do not answer the question. Do not invent facts. "
    "Return only the rewritten search query."
)


class QueryRewriteError(RuntimeError):
    """La petición o la respuesta del rewriter no cumple el contrato."""


def _api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise QueryRewriteError("Falta OPENROUTER_API_KEY en el proceso")
    return key


def _post_openrouter(payload: dict[str, Any]) -> dict[str, Any]:
    """Realiza exactamente un intento HTTP; el smoke no oculta reintentos."""
    request = urllib.request.Request(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/miax-taller-nlp",
            "X-Title": "miax-common-query-rewriting",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise QueryRewriteError(
            f"OpenRouter chat completions HTTP {exc.code}: {detail[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise QueryRewriteError(
            f"OpenRouter chat completions network error: {exc.reason}"
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise QueryRewriteError("Respuesta OpenRouter no es JSON válido") from exc


def _usage(response: dict[str, Any]) -> dict[str, int | float | None]:
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    cost = usage.get("cost")
    if not isinstance(cost, (int, float)):
        cost = response.get("cost")
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost": float(cost) if isinstance(cost, (int, float)) else None,
    }


def rewrite_query(
    question: str,
    *,
    request_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reescribe una query y devuelve texto más provenance del proveedor."""
    if not isinstance(question, str) or not question.strip():
        raise QueryRewriteError("La pregunta no puede estar vacía")
    payload = {
        "model": QUERY_REWRITER_MODEL_ID,
        "temperature": 0.0,
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": QUERY_REWRITER_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    send = request_fn or _post_openrouter
    started = time.perf_counter()
    response = send(payload)
    latency_s = time.perf_counter() - started
    try:
        rewritten = response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise QueryRewriteError("Respuesta OpenRouter sin texto reescrito") from exc
    if not rewritten:
        raise QueryRewriteError("OpenRouter devolvió un rewrite vacío")
    return {
        "question": question,
        "rewrite": rewritten,
        "requested_model": QUERY_REWRITER_MODEL,
        "effective_model": response.get("model"),
        "provider_response_id": response.get("id"),
        "latency_s": round(latency_s, 6),
        "usage": _usage(response),
    }


def reescribir(pregunta: str) -> str:
    """Compatibilidad con experimentos anteriores; falla de forma explícita."""
    return str(rewrite_query(pregunta)["rewrite"])
