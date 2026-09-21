"""Agente experimental de Cristian (paquete autónomo, sin imports de common).

Contrato de la práctica: mismas firmas de tools + ``RespuestaFinanciera``.
Las implementaciones viven en este directorio (`tools`, `schema`, `miax_s1`).

Ejecución (cualquiera de las dos)::

    # desde la raíz del repo
    python -m cristian.experiments.agent

    # o desde este directorio
    python3 agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Si se ejecuta como script (`python3 agent.py`), el cwd/sys.path no incluye
# la raíz del repo y falla `import cristian`. Añadimos esa raíz explícitamente.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cristian.experiments.config import SETTINGS, env_path

SYSTEM = """Eres un analista financiero que responde preguntas sobre informes
10-K usando ÚNICAMENTE las herramientas disponibles.

Reglas:
- Para cualquier CIFRA, usa get_xbrl_fact. Nunca leas un número de la prosa.
- En comparativas numéricas consulta get_xbrl_fact para AMBOS ejercicios.
- Para riesgos, estrategia o comentarios de dirección, usa search_filings.
- Usa read_section solo como último recurso.
- Si dudas de la cobertura, empieza por list_available.
- El corpus está en inglés: formula las consultas de búsqueda en inglés.
- Cita chunk_id y un fragmento literal.
- Si el dato no está en el corpus, dilo. No lo estimes.
"""


def crear_agente():
    """Construye el agente (lazy): carga SDK y credenciales solo al llamar."""
    from dotenv import load_dotenv
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolCallLimitMiddleware
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.memory import InMemorySaver

    from cristian.experiments.schema import RespuestaFinanciera
    from cristian.experiments.tools import (
        get_xbrl_fact,
        list_available,
        read_section,
        search_filings,
    )

    ruta_env = env_path()
    if not load_dotenv(ruta_env):
        raise FileNotFoundError(
            f"No encuentro {ruta_env}. Copia .env.example a .env y "
            "rellena OPENROUTER_API_KEY (y HF_TOKEN si hace falta)."
        )

    model = init_chat_model(
        SETTINGS.model,
        temperature=SETTINGS.temperature,
    )
    return create_agent(
        model=model,
        tools=[list_available, get_xbrl_fact, search_filings, read_section],
        system_prompt=SYSTEM,
        response_format=RespuestaFinanciera,
        checkpointer=InMemorySaver(),
        middleware=[
            ToolCallLimitMiddleware(run_limit=SETTINGS.tool_call_run_limit),
        ],
    )


def responder(pregunta: str):
    """Invoca el agente y devuelve el resultado completo de LangGraph."""
    import uuid

    agente = crear_agente()
    return agente.invoke(
        {"messages": [{"role": "user", "content": pregunta}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )


def _imprimir_respuesta(resultado) -> None:
    """Muestra la respuesta estructurada si existe; si no, el resultado crudo."""
    estructurada = resultado.get("structured_response")
    if estructurada is not None:
        print("\n--- RESPUESTA ---")
        print(estructurada.respuesta)
        print(f"fuente: {estructurada.fuente}")
        if estructurada.cifra is not None:
            print(f"cifra:  {estructurada.cifra} {estructurada.unidad or ''}".rstrip())
        if estructurada.ticker:
            print(f"ticker: {estructurada.ticker}  ejercicio: {estructurada.ejercicio}")
        if estructurada.chunk_id:
            print(f"chunk:  {estructurada.chunk_id}")
        if estructurada.cita:
            cita = estructurada.cita
            print(f"cita:   {cita[:300]}{'…' if len(cita) > 300 else ''}")
        return
    print(resultado)


if __name__ == "__main__":
    # Modo interactivo: gasta API (OpenRouter) en cada pregunta.
    pregunta_cli = " ".join(sys.argv[1:]).strip()
    print(f"modelo:  {SETTINGS.model}")
    print(f"dataset: {env_path().parent / 'cristian' / 'dataset'}")
    print("Escribe una pregunta (vacío o 'salir' para terminar).\n")

    agente = None
    while True:
        if pregunta_cli:
            pregunta = pregunta_cli
            pregunta_cli = ""
        else:
            try:
                pregunta = input("pregunta> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if not pregunta or pregunta.lower() in {"salir", "exit", "quit", "q"}:
            break

        try:
            if agente is None:
                agente = crear_agente()
            import uuid

            resultado = agente.invoke(
                {"messages": [{"role": "user", "content": pregunta}]},
                config={"configurable": {"thread_id": str(uuid.uuid4())}},
            )
            _imprimir_respuesta(resultado)
        except Exception as exc:
            print(f"\nError: {type(exc).__name__}: {exc}")
        print()
