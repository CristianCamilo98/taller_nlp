"""Agente investigador sobre informes 10-K."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from .schema import RespuestaFinanciera

# Cargar .env desde la raíz del repo
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SYSTEM = """Eres un analista financiero que responde preguntas sobre informes
10-K usando ÚNICAMENTE las herramientas disponibles.

Reglas:
- Para cualquier CIFRA, usa get_xbrl_fact. Nunca leas un número de la prosa.
- Para riesgos, estrategia o comentarios de la dirección, usa search_filings.
- Si no sabes si una compañía o un ejercicio están en el corpus, empieza por
  list_available.
- El corpus está en inglés: escribe las consultas de búsqueda en inglés.
- Cita el chunk_id del fragmento en el que te apoyes.
- Si el dato no está en el corpus, dilo. No lo estimes.
- Si tras 3 intentos de búsqueda no encuentras la información, deja de buscar
  y responde con lo que tengas, indicando qué no has podido encontrar.
"""

MODELO = "openrouter:google/gemini-3.8-flash"
def crear_agente():
    """Crea y devuelve el agente con las 4 tools y salida estructurada."""
    # Import aquí para evitar circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import (
        list_available, get_xbrl_fact, search_filings, read_section,
    )

    herramientas = [list_available, get_xbrl_fact, search_filings, read_section]

    return create_agent(
        model=MODELO,
        tools=herramientas,
        system_prompt=SYSTEM,
        response_format=RespuestaFinanciera,
        checkpointer=InMemorySaver(),
    )