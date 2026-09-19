"""Interfaz pública ``responder(pregunta)`` sin efectos durante import."""

from __future__ import annotations

import uuid

from common.benchmark_config import BENCHMARK

_agente = None


def _get_agente():
    global _agente
    if _agente is None:
        from common.agent.agent import crear_agente
        _agente = crear_agente()
    return _agente


def _extract_tool_calls(messages) -> list[dict]:
    calls = []
    for message in messages:
        for call in (getattr(message, "tool_calls", None) or []):
            if call.get("name") == "RespuestaFinanciera":
                continue
            calls.append({"name": call.get("name"),
                          "args": call.get("args") or {}})
    return calls


def _extract_telemetry(messages) -> dict:
    input_tokens = output_tokens = total_tokens = 0
    token_data_seen = False
    llm_calls = 0
    models: list[str] = []
    providers: list[str] = []
    reported_costs: list[float] = []
    for message in messages:
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


def responder(pregunta: str, max_reintentos: int | None = None,
              thread_id: str | None = None, _guardrail_retries: int = 0) -> dict:
    """Responde con un hilo nuevo por invocación externa.

    El reintento correctivo del guardrail reutiliza el mismo hilo porque sigue
    siendo la misma pregunta. No se realizan llamadas al importar el módulo.
    """
    if max_reintentos is None:
        max_reintentos = BENCHMARK.guardrail_retries
    if thread_id is None:
        thread_id = f"q-{uuid.uuid4().hex}"

    result = _get_agente().invoke(
        {"messages": [{"role": "user", "content": pregunta}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result.get("messages") or []
    calls = _extract_tool_calls(messages)
    structured = result["structured_response"]
    answer = (structured.model_dump() if hasattr(structured, "model_dump")
              else dict(structured))
    answer["tool_calls_agente"] = [call["name"] for call in calls]
    answer["tool_calls_detallado"] = calls
    answer["thread_id"] = thread_id
    answer["guardrail_retry_count"] = _guardrail_retries
    answer["_telemetria"] = _extract_telemetry(messages)

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
