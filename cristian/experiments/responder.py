"""Interfaz pública ``responder(pregunta)`` para el agente experimental.

Devuelve el dict que espera ``evaluar`` (campos de RespuestaFinanciera +
tool_calls + telemetría). Importar este módulo no llama a la API.
"""

from __future__ import annotations

import uuid

from cristian.experiments.config import SETTINGS

_agente = None


def _get_agente():
    global _agente
    if _agente is None:
        from cristian.experiments.agent import crear_agente

        _agente = crear_agente()
    return _agente


def _extract_tool_calls(messages) -> list[dict]:
    calls = []
    for message in messages or []:
        raw_calls = getattr(message, "tool_calls", None) or []
        for call in raw_calls:
            if call is None:
                continue
            if isinstance(call, dict):
                name = call.get("name")
                args = call.get("args") or {}
            else:
                name = getattr(call, "name", None)
                args = getattr(call, "args", None) or {}
            if name == "RespuestaFinanciera":
                continue
            if not name:
                continue
            calls.append({"name": name, "args": args})
    return calls


def _extract_telemetry(messages) -> dict:
    input_tokens = output_tokens = total_tokens = 0
    token_data_seen = False
    llm_calls = 0
    models: list[str] = []
    providers: list[str] = []
    reported_costs: list[float] = []
    for message in messages or []:
        usage = getattr(message, "usage_metadata", None) or {}
        if usage:
            token_data_seen = True
            llm_calls += 1
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
        metadata = getattr(message, "response_metadata", None) or {}
        model = metadata.get("model_name") or metadata.get("model")
        provider = metadata.get("provider")
        if model and model not in models:
            models.append(str(model))
        if provider and provider not in providers:
            providers.append(str(provider))
        cost = metadata.get("cost") or metadata.get("total_cost")
        if isinstance(cost, (int, float)):
            reported_costs.append(float(cost))
    if token_data_seen and total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens if token_data_seen else None,
        "output_tokens": output_tokens if token_data_seen else None,
        "total_tokens": total_tokens if token_data_seen else None,
        "llm_calls": llm_calls if token_data_seen else None,
        "provider_effective": providers[-1] if providers else None,
        "model_effective": models[-1] if models else None,
        "coste": sum(reported_costs) if reported_costs else None,
    }


def _coerce_structured(structured, messages=None):
    """Convierte structured_response a dict; error claro si el modelo no la dio."""
    if structured is not None:
        if hasattr(structured, "model_dump"):
            return structured.model_dump()
        if isinstance(structured, dict):
            return dict(structured)
        try:
            return dict(structured)
        except TypeError as exc:
            raise RuntimeError(
                f"structured_response no convertible a dict: {type(structured)!r}"
            ) from exc

    # Límite de LLM/tools puede cortar antes de emitir RespuestaFinanciera.
    last_text = None
    for message in reversed(messages or []):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            last_text = content.strip()
            break
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            joined = "\n".join(p for p in parts if p).strip()
            if joined:
                last_text = joined
                break
    return {
        "respuesta": (
            last_text
            or "No se pudo completar la respuesta estructurada "
            f"(límite de {SETTINGS.llm_call_run_limit} llamadas al modelo)."
        ),
        "cifra": None,
        "unidad": None,
        "ticker": None,
        "ejercicio": None,
        "fuente": "ninguna",
        "cita": None,
        "chunk_id": None,
        "concepto_xbrl": None,
        "ejercicio_inicial": None,
        "ejercicio_final": None,
        "valor_inicial": None,
        "valor_final": None,
        "delta": None,
        "porcentaje": None,
        "limit_alcanzado": True,
    }


def responder(
    pregunta: str,
    max_reintentos: int | None = None,
    thread_id: str | None = None,
    _guardrail_retries: int = 0,
) -> dict:
    """Responde una pregunta con el agente experimental.

    Misma forma de salida que ``common.responder`` para poder evaluar con
    los mismos criterios (cifra / cita / trayectoria).
    """
    if max_reintentos is None:
        max_reintentos = SETTINGS.guardrail_retries
    if thread_id is None:
        thread_id = f"q-{uuid.uuid4().hex}"

    result = _get_agente().invoke(
        {"messages": [{"role": "user", "content": pregunta}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result.get("messages") or []
    calls = _extract_tool_calls(messages)
    answer = _coerce_structured(result.get("structured_response"), messages)
    answer["tool_calls_agente"] = [call["name"] for call in calls]
    answer["tool_calls_detallado"] = calls
    answer["thread_id"] = thread_id
    answer["guardrail_retry_count"] = _guardrail_retries
    answer["_telemetria"] = _extract_telemetry(messages)

    # Misma regla de corrección que el baseline (comparación justa).
    from common.agent.middleware_xbrl import verificar_respuesta_xbrl

    ok, message = verificar_respuesta_xbrl(answer, calls)
    if not ok and max_reintentos > 0:
        corrected = f"{pregunta}\n\nAVISO DEL SISTEMA: {message}"
        return responder(
            corrected,
            max_reintentos=max_reintentos - 1,
            thread_id=thread_id,
            _guardrail_retries=_guardrail_retries + 1,
        )
    answer["guardrail_ok"] = ok
    answer["guardrail_mensaje"] = message
    return answer
