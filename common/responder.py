"""Interfaz pública ``responder(pregunta)`` sin efectos durante import."""

from __future__ import annotations

import re
import uuid

from common.agent.middleware_tool_dedup import DUPLICATE_TOOL_CALL_TYPE
from common.benchmark_config import BENCHMARK

_agente = None

# Tools que aportan evidencia real del corpus. list_available() por sí sola
# solo lista cobertura y no fundamenta una respuesta final.
EVIDENCE_TOOLS = frozenset({"get_xbrl_fact", "search_filings", "read_section"})


def _get_agente():
    global _agente
    if _agente is None:
        from common.agent.agent import crear_agente
        _agente = crear_agente()
    return _agente


def _extract_tool_trajectory(messages) -> tuple[list[dict], list[dict]]:
    blocked_ids = {}
    for message in messages:
        artifact = getattr(message, "artifact", None)
        if (
            isinstance(artifact, dict)
            and artifact.get("type") == DUPLICATE_TOOL_CALL_TYPE
        ):
            blocked_ids[getattr(message, "tool_call_id", None)] = artifact

    executed = []
    blocked = []
    for message in messages:
        for call in (getattr(message, "tool_calls", None) or []):
            if call.get("name") == "RespuestaFinanciera":
                continue
            value = {"name": call.get("name"), "args": call.get("args") or {}}
            diagnostic = blocked_ids.get(call.get("id"))
            if diagnostic is None:
                executed.append(value)
            else:
                blocked.append({
                    **value,
                    "executed": False,
                    "reason": DUPLICATE_TOOL_CALL_TYPE,
                })
    return executed, blocked


def _extract_tool_calls(messages) -> list[dict]:
    """Compatibilidad: devuelve ejecuciones reales, no duplicados."""
    return _extract_tool_trajectory(messages)[0]


_SEARCH_CHUNK_PATTERN = re.compile(
    r"\[(?P<chunk_id>[^\]]+)\]\s+(?P<ticker>\S+)\s+FY(?P<fiscal_year>\d+)\s+"
    r"Item\s+(?P<item>\S+)\s+\(similitud\s+(?P<puntuacion>[\d.]+)\)\n"
    r"(?P<texto>.*?)(?=\n\n---\n\n|\Z)",
    re.DOTALL,
)


def _parse_search_filings_chunks(content: str) -> list[dict]:
    """Parsea EXACTAMENTE el formato de
    common.retrieval.dense_baseline.formatear_fragmentos; no inventa nada
    que no esté en ``content``."""
    return [
        {
            "chunk_id": match.group("chunk_id"),
            "ticker": match.group("ticker"),
            "fiscal_year": int(match.group("fiscal_year")),
            "item": match.group("item"),
            "puntuacion": float(match.group("puntuacion")),
            "texto": match.group("texto").strip(),
        }
        for match in _SEARCH_CHUNK_PATTERN.finditer(content or "")
    ]


def _extract_search_filings_results(messages) -> list[dict]:
    """Chunks REALMENTE devueltos por llamadas reales (no bloqueadas) a
    search_filings en este hilo, deduplicados por chunk_id.

    Se extraen parseando el ToolMessage real que ya produce
    formatear_fragmentos -nunca se re-consulta el índice ni se inventa
    contenido-.
    """
    blocked_ids = set()
    for message in messages:
        artifact = getattr(message, "artifact", None)
        if (
            isinstance(artifact, dict)
            and artifact.get("type") == DUPLICATE_TOOL_CALL_TYPE
        ):
            blocked_ids.add(getattr(message, "tool_call_id", None))

    real_search_call_ids = set()
    for message in messages:
        for call in (getattr(message, "tool_calls", None) or []):
            if (call.get("name") == "search_filings"
                    and call.get("id") not in blocked_ids):
                real_search_call_ids.add(call.get("id"))

    seen: dict[str, dict] = {}
    for message in messages:
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id not in real_search_call_ids:
            continue
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            continue
        for chunk in _parse_search_filings_chunks(content):
            seen.setdefault(chunk["chunk_id"], chunk)
    return list(seen.values())


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


def _is_grounded(calls: list[dict], answer: dict) -> bool:
    """Al menos una tool de evidencia; fuente xbrl exige get_xbrl_fact."""
    names = [call.get("name") for call in calls]
    if not any(name in EVIDENCE_TOOLS for name in names):
        return False
    if answer.get("fuente") in {"xbrl", "ambas"} and "get_xbrl_fact" not in names:
        return False
    return True


def _is_citation_grounded(answer: dict, chunks: list[dict]) -> tuple[bool, str]:
    """Si la respuesta declara fuente textual/filing, cita y chunk_id deben
    proceder de un chunk REALMENTE recuperado por search_filings.

    fuente="ninguna" (declarar honestamente que no se encontró evidencia
    suficiente) es una respuesta legítima -el system prompt del agente lo
    permite explícitamente- y no se evalúa aquí.
    """
    from common.agent.citation_repair import citation_in_text

    if answer.get("fuente") not in {"texto", "ambas"}:
        return True, "no declara fuente narrativa"
    chunk_id = answer.get("chunk_id")
    cita = answer.get("cita")
    if not chunk_id or not (cita and str(cita).strip()):
        return False, "fuente narrativa sin cita/chunk_id"
    matching = next((c for c in chunks if c["chunk_id"] == chunk_id), None)
    if matching is None:
        return False, (
            f"chunk_id {chunk_id!r} no está entre los chunks realmente "
            "recuperados por search_filings"
        )
    if not citation_in_text(cita, matching["texto"]):
        return False, (
            "la cita no está contenida literalmente en el chunk indicado"
        )
    return True, "cita respaldada por el chunk realmente recuperado"


def _input_messages(pregunta, _turn_messages, _turn_content):
    if _turn_messages is not None:
        return list(_turn_messages)
    text = pregunta if _turn_content is None else _turn_content
    return [{"role": "user", "content": text}]


_COMPARATIVE_FIELDS = ("valor_inicial", "valor_final", "delta", "porcentaje",
                       "ejercicio_inicial", "ejercicio_final")


def _fill_missing_xbrl_fields(answer: dict, calls: list[dict]) -> tuple[dict, bool]:
    """Rellena cifra/unidad/ticker/ejercicio/concepto_xbrl NULOS desde el
    hecho XBRL realmente consultado, solo cuando es inequívoco.

    Nunca sobrescribe un valor ya presente (aunque sea incorrecto: eso lo
    detecta el guardrail) ni inventa nada fuera de lo que get_xbrl_fact
    devolvió de verdad. Solo actúa en la forma numérica simple (un único
    hecho consultado); las respuestas comparativas quedan fuera de este
    rellenado determinista y usan el reintento de guardrail existente.
    """
    if answer.get("fuente") not in {"xbrl", "ambas"}:
        return answer, False
    if any(answer.get(field) is not None for field in _COMPARATIVE_FIELDS):
        return answer, False

    from common.agent.middleware_xbrl import _queried_facts

    consulted = _queried_facts(calls)
    distinct = {(fact["ticker"], fact["fiscal_year"], fact["concept"])
               for fact in consulted}
    if len(distinct) != 1:
        return answer, False

    fact = consulted[0]
    fillable = {
        "cifra": fact["value"], "unidad": fact["unit"],
        "ticker": fact["ticker"], "ejercicio": fact["fiscal_year"],
        "concepto_xbrl": fact["concept"],
    }
    filled = False
    for field, value in fillable.items():
        if answer.get(field) is None:
            answer[field] = value
            filled = True
    return answer, filled


def responder(pregunta: str, max_reintentos: int | None = None,
              thread_id: str | None = None, _guardrail_retries: int = 0,
              _tool_enforcement_retried: bool = False,
              _turn_content: str | None = None,
              _turn_messages: list | None = None,
              _original_question: str | None = None) -> dict:
    """Responde con un hilo nuevo por invocación externa.

    ``pregunta`` (y el ``_original_question`` interno que la preserva a
    través de reintentos) nunca se concatena con avisos del sistema: el hilo
    de LangGraph ya conserva el mensaje original, así que cualquier turno de
    refuerzo se envía por separado. Esto es necesario para que, si un
    reintento hace la primera llamada a search_filings, su query siga siendo
    la pregunta original literal.

    Si el agente emite una respuesta final sin ninguna tool de evidencia
    real (get_xbrl_fact/search_filings/read_section; list_available sola no
    cuenta), o declara fuente xbrl sin haber llamado a get_xbrl_fact, NO se
    reintenta en lenguaje natural (un modelo con response_format resuelto a
    ProviderStrategy ignora tool_choice y puede repetir el mismo patrón, ver
    common.agent.forced_evidence). En su lugar se ejecuta un único
    forced_evidence_turn(): un modelo aparte, sin response_format, con
    tool_choice="required", que no puede terminar sin una tool_call real. Su
    resultado (AIMessage + ToolMessage reales) se inyecta como el siguiente
    turno del mismo hilo, y el agente normal retoma desde ahí para generar
    el structured output final ya con evidencia. Si ese turno forzado
    tampoco produce ninguna tool_call, o la respuesta final sigue sin
    fundamento incluso con evidencia inyectada, responder() falla de forma
    controlada (``forced_tool_call_failed``) en vez de aceptar una respuesta
    de memoria.

    Si la respuesta final declara fuente="ninguna" pero search_filings sí
    recuperó chunks reales, se ejecuta como máximo un
    evidence_adjudication_turn() (common.agent.citation_repair): un modelo
    aparte revisa SOLO esos chunks -la declinación original se pasa solo
    como contexto, nunca como verdad- y decide si alguno responde la
    pregunta. Si sí, la respuesta/fuente/chunk_id/cita se reemplazan por la
    versión reparada (marcando evidence_rescue_used); si no, se mantiene la
    declinación (fuente="ninguna" sigue siendo una respuesta legítima). Igual
    que con la citación, la validación es siempre determinista después de la
    llamada: chunk_id debe estar entre los chunks realmente recuperados y la
    cita debe estar contenida literalmente en ese chunk, o se descarta.

    El reintento correctivo del guardrail reutiliza el mismo hilo porque
    sigue siendo la misma pregunta.

    Las llamadas reales (sin contar duplicados bloqueados ni el turno
    forzado propiamente dicho, que sí cuenta como ejecución real) se
    acumulan por pregunta -no de forma global- porque LangGraph devuelve en
    cada invoke() el historial completo del hilo; si superan
    BENCHMARK.tool_call_run_limit entre todos los reintentos de esta
    pregunta, se detiene de forma controlada sin reintentar más. No se
    realizan llamadas al importar el módulo.
    """
    if max_reintentos is None:
        max_reintentos = BENCHMARK.guardrail_retries
    if thread_id is None:
        thread_id = f"q-{uuid.uuid4().hex}"
    if _original_question is None:
        _original_question = pregunta

    result = _get_agente().invoke(
        {"messages": _input_messages(pregunta, _turn_messages, _turn_content)},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result.get("messages") or []
    calls, blocked_calls = _extract_tool_trajectory(messages)
    search_results = _extract_search_filings_results(messages)

    structured = result["structured_response"]
    answer = (structured.model_dump() if hasattr(structured, "model_dump")
              else dict(structured))
    answer, xbrl_fields_autofilled = _fill_missing_xbrl_fields(answer, calls)
    answer["xbrl_fields_autofilled"] = xbrl_fields_autofilled
    answer["tool_calls_agente"] = [call["name"] for call in calls]
    answer["tool_calls_detallado"] = calls
    answer["tool_calls_bloqueados"] = blocked_calls
    answer["tool_results_detallado"] = search_results
    answer["thread_id"] = thread_id
    answer["guardrail_retry_count"] = _guardrail_retries
    answer["tool_enforcement_retry_used"] = _tool_enforcement_retried
    answer["forced_tool_call_failed"] = False
    answer["citation_repair_used"] = False
    answer["citation_repair_failed"] = False
    answer["evidence_rescue_used"] = False
    answer["_telemetria"] = _extract_telemetry(messages)
    answer["_telemetria"]["tool_calls_executed"] = len(calls)
    answer["_telemetria"]["tool_calls_duplicate_blocked"] = len(blocked_calls)
    answer["_telemetria"]["tool_enforcement_retry_used"] = _tool_enforcement_retried
    answer["_telemetria"]["xbrl_fields_autofilled"] = xbrl_fields_autofilled

    budget = BENCHMARK.tool_call_run_limit
    budget_exceeded = len(calls) > budget
    answer["tool_call_budget_exceeded"] = budget_exceeded
    answer["_telemetria"]["tool_call_budget_exceeded"] = budget_exceeded
    if budget_exceeded:
        answer["guardrail_ok"] = False
        answer["guardrail_mensaje"] = (
            f"Límite de {budget} llamadas reales a herramientas por "
            f"pregunta excedido ({len(calls)} ejecutadas); respuesta "
            "detenida de forma controlada."
        )
        return answer

    if not _is_grounded(calls, answer):
        if _tool_enforcement_retried:
            answer["guardrail_ok"] = False
            answer["guardrail_mensaje"] = (
                "La respuesta final no está fundamentada en una tool de "
                "evidencia real ni siquiera tras el turno forzado; no se "
                "acepta como válida."
            )
            return answer
        if len(calls) >= budget:
            answer["guardrail_ok"] = False
            answer["guardrail_mensaje"] = (
                f"Límite de {budget} llamadas reales a herramientas "
                "alcanzado antes del turno forzado de evidencia; no se "
                "acepta una respuesta desde memoria."
            )
            return answer

        from common.agent.forced_evidence import forced_evidence_turn

        forced = forced_evidence_turn(_original_question)
        if forced is None:
            answer["tool_enforcement_retry_used"] = True
            answer["forced_tool_call_failed"] = True
            answer["_telemetria"]["tool_enforcement_retry_used"] = True
            answer["_telemetria"]["forced_tool_call_failed"] = True
            answer["guardrail_ok"] = False
            answer["guardrail_mensaje"] = (
                "El turno forzado de evidencia (tool_choice=required) no "
                "produjo ninguna tool_call; no se acepta una respuesta "
                "desde memoria."
            )
            return answer

        forced_ai_message, forced_tool_messages = forced
        return responder(
            _original_question,
            max_reintentos=max_reintentos,
            thread_id=thread_id,
            _guardrail_retries=_guardrail_retries,
            _tool_enforcement_retried=True,
            _turn_messages=[forced_ai_message, *forced_tool_messages],
            _original_question=_original_question,
        )

    if answer.get("fuente") == "ninguna" and search_results:
        from common.agent.citation_repair import evidence_adjudication_turn

        rescued = evidence_adjudication_turn(
            _original_question, answer.get("respuesta") or "", search_results,
        )
        answer["citation_repair_used"] = True
        answer["_telemetria"]["citation_repair_used"] = True
        if rescued is not None:
            rescued_respuesta, rescued_chunk_id, rescued_cita = rescued
            answer["respuesta"] = rescued_respuesta
            answer["fuente"] = "texto"
            answer["chunk_id"] = rescued_chunk_id
            answer["cita"] = rescued_cita
            answer["evidence_rescue_used"] = True
            answer["_telemetria"]["evidence_rescue_used"] = True
        else:
            answer["citation_repair_failed"] = True
            answer["_telemetria"]["citation_repair_failed"] = True

    citation_ok, citation_message = _is_citation_grounded(answer, search_results)
    if not citation_ok:
        from common.agent.citation_repair import citation_repair_turn

        repaired = citation_repair_turn(
            _original_question, answer.get("respuesta") or "", search_results,
        )
        answer["citation_repair_used"] = True
        answer["_telemetria"]["citation_repair_used"] = True
        if repaired is not None:
            answer["chunk_id"], answer["cita"] = repaired
            citation_ok, citation_message = _is_citation_grounded(
                answer, search_results)
        else:
            answer["citation_repair_failed"] = True
            answer["_telemetria"]["citation_repair_failed"] = True
        if not citation_ok:
            answer["guardrail_ok"] = False
            answer["guardrail_mensaje"] = citation_message
            return answer
    answer["citation_grounding_ok"] = citation_ok
    answer["_telemetria"]["citation_grounding_ok"] = citation_ok

    from common.agent.middleware_xbrl import verificar_respuesta_xbrl

    ok, message = verificar_respuesta_xbrl(answer, calls)
    if not ok and max_reintentos > 0:
        if len(calls) >= budget:
            answer["guardrail_ok"] = False
            answer["guardrail_mensaje"] = (
                f"{message} Límite de {budget} llamadas reales a "
                "herramientas alcanzado; no se reintenta más."
            )
            return answer
        corrected = f"{_original_question}\n\nAVISO DEL SISTEMA: {message}"
        return responder(
            _original_question,
            max_reintentos=max_reintentos - 1,
            thread_id=thread_id,
            _guardrail_retries=_guardrail_retries + 1,
            _tool_enforcement_retried=_tool_enforcement_retried,
            _turn_content=corrected,
            _original_question=_original_question,
        )
    answer["guardrail_ok"] = ok
    answer["guardrail_mensaje"] = message
    return answer
