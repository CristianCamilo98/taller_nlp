"""Construcción perezosa del agente investigador sobre informes 10-K."""

from common.benchmark_config import BENCHMARK

MODELO = BENCHMARK.model

SYSTEM = """Eres un analista financiero que responde preguntas sobre informes
10-K usando ÚNICAMENTE las herramientas disponibles.

Reglas:
- Nunca respondas con conocimiento interno cuando una tool pueda dar
  evidencia del corpus. Toda respuesta final debe estar fundamentada en al
  menos una llamada a una tool; no emitas una respuesta final con cero
  llamadas a herramientas.
- Para cifras contables/XBRL exactas, usa get_xbrl_fact como primera
  herramienta apropiada.
- Para preguntas narrativas o extractivas, usa search_filings.
- Usa read_section solo cuando de verdad necesites la sección completa.
- Usa list_available solo cuando necesites descubrir disponibilidad.
- Para cualquier CIFRA, usa get_xbrl_fact. Nunca leas un número de la prosa.
- En comparativas numéricas consulta get_xbrl_fact para AMBOS ejercicios, con
  el mismo ticker y concepto; calcula después delta y porcentaje. Devuelve en
  cifra el delta y completa todos los campos comparativos estructurados.
- Para riesgos, estrategia o comentarios de dirección, usa search_filings.
- Usa read_section solo como último recurso, después de search_filings en la
  misma compañía, ejercicio e item.
- Si dudas de la cobertura, empieza por list_available.
- En la PRIMERA llamada a search_filings, copia en query la pregunta original
  del usuario exactamente: no la traduzcas, resumas ni reformules. Extrae
  ticker, fiscal_year e item y pásalos en sus argumentos correspondientes.
- Cita chunk_id y un fragmento literal; separa omisiones con "...".
- Si el dato no está en el corpus, dilo. No lo estimes.
- Tras 3 intentos de búsqueda sin evidencia suficiente, deja de buscar y
  declara qué no has encontrado. Este límite orienta al agente, pero no forma
  parte de la puntuación de trayectoria porque el enunciado no lo formaliza.
"""


def _agent_middleware():
    """Construye middleware nuevo; la deduplicación no comparte estado."""
    from langchain.agents.middleware import ToolCallLimitMiddleware

    from common.agent.middleware_tool_dedup import ToolCallDedupMiddleware

    return [
        ToolCallDedupMiddleware(),
        ToolCallLimitMiddleware(run_limit=BENCHMARK.tool_call_run_limit),
    ]


def crear_agente():
    """Crea el agente; aquí, y no durante import, carga SDK y credenciales."""
    from pathlib import Path

    from dotenv import load_dotenv
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.memory import InMemorySaver

    from common.agent.schema import RespuestaFinanciera
    from common.tools import (
        get_xbrl_fact,
        list_available,
        read_section,
        search_filings,
    )

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    model = init_chat_model(MODELO, temperature=BENCHMARK.temperature)
    return create_agent(
        model=model,
        tools=[list_available, get_xbrl_fact, search_filings, read_section],
        system_prompt=SYSTEM,
        response_format=RespuestaFinanciera,
        checkpointer=InMemorySaver(),
        middleware=_agent_middleware(),
    )


def crear_forced_evidence_model():
    """Modelo DeepSeek SIN response_format, para el turno forzado de evidencia.

    create_agent(response_format=RespuestaFinanciera) resuelve a
    ProviderStrategy para este modelo, porque model.profile["structured_output"]
    es True (ver langchain.agents.factory._supports_provider_strategy). En esa
    rama tool_choice se ignora por completo (factory.py, rama ProviderStrategy
    de _get_bound_model), así que un modelo bindeado ahí puede terminar el
    turno con texto plano y cero tool calls -confirmado en un smoke real tras
    dos intentos-. Este modelo aparte, sin response_format, sí respeta
    tool_choice="required" (ChatOpenRouter.bind_tools lo pasa tal cual).
    """
    from pathlib import Path

    from dotenv import load_dotenv
    from langchain.chat_models import init_chat_model

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    return init_chat_model(MODELO, temperature=BENCHMARK.temperature)
