"""Turno forzado de selección de tool, separado del structured output final.

create_agent(response_format=RespuestaFinanciera) resuelve a ProviderStrategy
para DeepSeek v4 Flash, porque model.profile["structured_output"] es True
(ver langchain.agents.factory._supports_provider_strategy, verificado en
runtime con model.profile). En esa rama, tool_choice se ignora por completo
(langchain/agents/factory.py, rama ProviderStrategy de _get_bound_model:
solo pasa effective_response_format.to_model_kwargs() + model_settings, nunca
request.tool_choice). El modelo puede entonces emitir el JSON estructurado
final sin haber llamado a ninguna tool -confirmado dos veces en smoke real,
incluida una segunda vez tras un reintento en lenguaje natural-.

Este módulo separa OBTENCIÓN DE EVIDENCIA de GENERACIÓN DEL STRUCTURED
OUTPUT: un modelo aparte, bindeado solo con las cuatro tools reales, SIN
response_format y con tool_choice="required" (soportado nativamente por
ChatOpenRouter.bind_tools -ver langchain_openrouter/chat_models.py-, que
normaliza "any" a "required" y lo pasa tal cual al payload de OpenRouter).
Con tool_choice="required" el turno no puede terminar en texto plano.

La tool elegida se ejecuta con el mismo objeto de tool que usa el agente
(tool.invoke(tool_call), la vía estándar de LangChain que aplica el
args_schema real), nunca con una implementación paralela.

Un smoke extractivo real mostró que, con el prompt original de este turno
forzado, el modelo llamaba a search_filings con query/ticker correctos pero
SIN fiscal_year ni item -el prompt nunca se lo pedía, a diferencia del
system prompt principal del agente-. Sin ese postfiltro de metadata, la
búsqueda es global sobre todos los ejercicios/items de la compañía, diluye
el ranking y el chunk objetivo puede quedar fuera de k. FORCED_EVIDENCE_SYSTEM
ahora exige explícitamente extraer ticker/fiscal_year/item(o concept).
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

FORCED_EVIDENCE_SYSTEM = """Elige y ejecuta la única herramienta más apropiada
para reunir evidencia sobre la pregunta financiera del usuario. Debes llamar
a exactamente una herramienta; no hay respuesta final en este turno.

- Para una cifra contable/XBRL exacta, usa get_xbrl_fact.
- Para narrativa, riesgos o contenido de un filing, usa search_filings. En
  ese caso, copia en query la pregunta del usuario exactamente: no la traduzcas, resumas ni reformules.
- Para necesitar el texto completo de una sección 10-K, usa read_section.
- Para descubrir qué compañías, ejercicios o secciones hay disponibles, usa
  list_available.
- Identifica ticker, fiscal_year e item/concept a partir de la pregunta y
  pásalos siempre en los argumentos correspondientes de la tool elegida
  (ticker/fiscal_year/concept en get_xbrl_fact; ticker/fiscal_year/item en
  search_filings y read_section); no los omitas cuando la pregunta los
  identifica.
"""

_model = None


def _get_forced_evidence_model():
    global _model
    if _model is None:
        from common.agent.agent import crear_forced_evidence_model
        _model = crear_forced_evidence_model()
    return _model


def _evidence_tools() -> list[Any]:
    from common.tools import get_xbrl_fact, list_available, read_section, search_filings
    return [list_available, get_xbrl_fact, search_filings, read_section]


def forced_evidence_turn(
    pregunta: str,
    *,
    model_fn: Callable[[], Any] | None = None,
) -> tuple[AIMessage, list[ToolMessage]] | None:
    """Un único turno con tool_choice='required'; None si no hubo tool_call.

    Devuelve el AIMessage real del modelo (con su usage_metadata intacto,
    para que cuente como llamada de telemetría) y los ToolMessage reales de
    ejecutar cada tool_call devuelta -normalmente una sola, por
    parallel_tool_calls=False-. Nunca se falsifica una tool_call: si el
    nombre no corresponde a una tool conocida, se descarta esa llamada.
    """
    tools = _evidence_tools()
    bound_model = (model_fn or _get_forced_evidence_model)()
    bound = bound_model.bind_tools(
        tools, tool_choice="required", parallel_tool_calls=False,
    )
    response = bound.invoke([
        SystemMessage(content=FORCED_EVIDENCE_SYSTEM),
        HumanMessage(content=pregunta),
    ])
    tool_calls = list(getattr(response, "tool_calls", None) or [])
    if not tool_calls:
        return None

    tools_by_name = {tool.name: tool for tool in tools}
    tool_messages = []
    for call in tool_calls:
        tool_obj = tools_by_name.get(call.get("name"))
        if tool_obj is None:
            continue
        tool_messages.append(tool_obj.invoke(call))
    if not tool_messages:
        return None
    return response, tool_messages
