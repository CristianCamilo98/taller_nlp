"""Query rewriting congelado mediante una única petición a OpenRouter."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


OPENROUTER_CHAT_COMPLETIONS_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)
QUERY_REWRITER_MODEL = "openrouter:deepseek/deepseek-v4-flash"
QUERY_REWRITER_MODEL_ID = "deepseek/deepseek-v4-flash"
QUERY_REWRITER_MAX_TOKENS = 512
QUERY_REWRITER_REASONING = {"enabled": False}
QUERY_REWRITER_PROMPT = (
    "Rewrite the financial question into a concise English search query "
    "optimized to retrieve relevant passages from SEC 10-K/10-Q filings. "
    "Preserve the exact information need, company, fiscal year and filing "
    "item when present. Do not answer the question. Do not invent facts. "
    "Return only the rewritten search query."
)


class QueryRewriteError(RuntimeError):
    """La petición o la respuesta del rewriter no cumple el contrato."""

    def __init__(self, message: str, *, diagnostic: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostic = dict(diagnostic or {})


@dataclass(frozen=True)
class OpenRouterHTTPResult:
    status: int | None
    payload: dict[str, Any]


def _api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise QueryRewriteError(
            "Falta OPENROUTER_API_KEY en el proceso",
            diagnostic={"http_status": None, "error_type": "missing_api_key"},
        )
    return key


def _post_openrouter(payload: dict[str, Any]) -> OpenRouterHTTPResult:
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
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            try:
                body = json.loads(response.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise QueryRewriteError(
                    "Respuesta OpenRouter no es JSON válido",
                    diagnostic={
                        "http_status": status,
                        "error_type": "invalid_json_response",
                    },
                ) from exc
            if not isinstance(body, dict):
                raise QueryRewriteError(
                    "Respuesta OpenRouter JSON no es un objeto",
                    diagnostic={
                        "http_status": status,
                        "error_type": "invalid_json_shape",
                    },
                )
            return OpenRouterHTTPResult(status=int(status), payload=body)
    except urllib.error.HTTPError as exc:
        # Se descarta el body: podría incluir contenido del proveedor y no es
        # necesario para el diagnóstico seguro solicitado.
        exc.read()
        raise QueryRewriteError(
            f"OpenRouter chat completions HTTP {exc.code}",
            diagnostic={
                "http_status": int(exc.code),
                "error_type": "http_error",
            },
        ) from exc
    except urllib.error.URLError as exc:
        raise QueryRewriteError(
            f"OpenRouter chat completions network error: {exc.reason}",
            diagnostic={"http_status": None, "error_type": "network_error"},
        ) from exc


def _usage(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    return dict(usage) if isinstance(usage, dict) else {}


def _private_length(value: Any) -> int:
    """Cuenta caracteres privados sin devolver ni persistir su contenido."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_private_length(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_private_length(item) for item in value)
    return 0


def _diagnostic(
    response: dict[str, Any],
    *,
    http_status: int | None,
    latency_s: float,
    error_type: str,
) -> dict[str, Any]:
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    choice = choice if isinstance(choice, dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    reasoning_values = [
        message.get(key) for key in ("reasoning", "reasoning_details")
    ]
    reasoning_present = any(
        value not in (None, "", [], {}) for value in reasoning_values
    )
    return {
        "requested_model": QUERY_REWRITER_MODEL,
        "effective_model": response.get("model"),
        "http_status": http_status,
        "finish_reason": choice.get("finish_reason"),
        "message_keys": sorted(str(key) for key in message),
        "content_is_null": content is None,
        "content_length": len(content) if isinstance(content, str) else 0,
        "reasoning_present": reasoning_present,
        "reasoning_length": sum(
            _private_length(value) for value in reasoning_values
        ),
        "usage": _usage(response),
        "latency_s": round(latency_s, 6),
        "error_type": error_type,
    }


def _coerce_result(value: Any) -> OpenRouterHTTPResult:
    if isinstance(value, OpenRouterHTTPResult):
        return value
    if isinstance(value, dict):
        # Compatibilidad para fakes offline; una respuesta real usa el wrapper.
        return OpenRouterHTTPResult(status=200, payload=value)
    raise QueryRewriteError(
        "El cliente OpenRouter devolvió un tipo inválido",
        diagnostic={"http_status": None, "error_type": "invalid_client_result"},
    )


def rewrite_query(
    question: str,
    *,
    request_fn: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Reescribe una query usando exclusivamente el contenido final textual."""
    if not isinstance(question, str) or not question.strip():
        raise QueryRewriteError(
            "La pregunta no puede estar vacía",
            diagnostic={"http_status": None, "error_type": "invalid_question"},
        )
    payload = {
        "model": QUERY_REWRITER_MODEL_ID,
        "temperature": 0.0,
        "max_tokens": QUERY_REWRITER_MAX_TOKENS,
        "reasoning": dict(QUERY_REWRITER_REASONING),
        "messages": [
            {"role": "system", "content": QUERY_REWRITER_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    send = request_fn or _post_openrouter
    started = time.perf_counter()
    try:
        result = _coerce_result(send(payload))
    except QueryRewriteError as exc:
        latency_s = time.perf_counter() - started
        diagnostic = {
            "requested_model": QUERY_REWRITER_MODEL,
            "effective_model": None,
            "finish_reason": None,
            "message_keys": [],
            "content_is_null": True,
            "content_length": 0,
            "reasoning_present": False,
            "reasoning_length": 0,
            "usage": {},
            "latency_s": round(latency_s, 6),
            **exc.diagnostic,
        }
        raise QueryRewriteError(str(exc), diagnostic=diagnostic) from exc
    latency_s = time.perf_counter() - started
    response = result.payload
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        diagnostic = _diagnostic(
            response,
            http_status=result.status,
            latency_s=latency_s,
            error_type="missing_or_empty_choices",
        )
        raise QueryRewriteError(
            "Respuesta OpenRouter sin choices", diagnostic=diagnostic
        )
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        diagnostic = _diagnostic(
            response,
            http_status=result.status,
            latency_s=latency_s,
            error_type="missing_message",
        )
        raise QueryRewriteError(
            "Respuesta OpenRouter sin message", diagnostic=diagnostic
        )
    message = choice["message"]
    content = message.get("content")
    if not isinstance(content, str):
        diagnostic = _diagnostic(
            response,
            http_status=result.status,
            latency_s=latency_s,
            error_type="null_or_non_text_content",
        )
        raise QueryRewriteError(
            "OpenRouter devolvió content nulo o no textual; "
            "reasoning no se acepta como rewrite",
            diagnostic=diagnostic,
        )
    rewritten = content.strip()
    if not rewritten:
        diagnostic = _diagnostic(
            response,
            http_status=result.status,
            latency_s=latency_s,
            error_type="empty_text_content",
        )
        raise QueryRewriteError(
            "OpenRouter devolvió content textual vacío",
            diagnostic=diagnostic,
        )
    return {
        "question": question,
        "rewrite": rewritten,
        "requested_model": QUERY_REWRITER_MODEL,
        "effective_model": response.get("model"),
        "provider_response_id": response.get("id"),
        "http_status": result.status,
        "finish_reason": choice.get("finish_reason"),
        "latency_s": round(latency_s, 6),
        "usage": _usage(response),
    }


def reescribir(pregunta: str) -> str:
    """Compatibilidad con experimentos anteriores; falla de forma explícita."""
    return str(rewrite_query(pregunta)["rewrite"])
