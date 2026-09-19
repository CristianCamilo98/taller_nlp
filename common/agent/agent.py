"""Construcción perezosa del agente investigador sobre informes 10-K."""

from common.benchmark_config import BENCHMARK

MODELO = BENCHMARK.model

SYSTEM = """Eres un analista financiero que responde preguntas sobre informes
10-K usando ÚNICAMENTE las herramientas disponibles.

Reglas:
- Para cualquier CIFRA, usa get_xbrl_fact. Nunca leas un número de la prosa.
- En comparativas numéricas consulta get_xbrl_fact para AMBOS ejercicios, con
  el mismo ticker y concepto; calcula después delta y porcentaje. Devuelve en
  cifra el delta y completa todos los campos comparativos estructurados.
- Para riesgos, estrategia o comentarios de dirección, usa search_filings.
- Usa read_section solo como último recurso, después de search_filings en la
  misma compañía, ejercicio e item.
- Si dudas de la cobertura, empieza por list_available.
- El corpus está en inglés: formula las consultas de búsqueda en inglés.
- Cita chunk_id y un fragmento literal; separa omisiones con "...".
- Si el dato no está en el corpus, dilo. No lo estimes.
- Tras 3 intentos de búsqueda sin evidencia suficiente, deja de buscar y
  declara qué no has encontrado. Este límite orienta al agente, pero no forma
  parte de la puntuación de trayectoria porque el enunciado no lo formaliza.
"""


def crear_agente():
    """Crea el agente; aquí, y no durante import, carga SDK y credenciales."""
    from pathlib import Path

    from dotenv import load_dotenv
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolCallLimitMiddleware
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
        middleware=[ToolCallLimitMiddleware(
            run_limit=BENCHMARK.tool_call_run_limit)],
    )
