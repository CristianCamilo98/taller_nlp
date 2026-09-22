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
- Si dudas de la cobertura, empieza por list_available (como mucho una vez).
- El corpus está en inglés: formula las consultas de búsqueda en inglés.
- Cita chunk_id y un fragmento literal.
- Si el dato no está en el corpus, dilo. No lo estimes.
- NO entres en bucles: no repitas la misma herramienta con los mismos
  argumentos. Máximo 2–3 búsquedas. Si tras eso no hay evidencia suficiente,
  responde con lo que tengas o declara que no está en el corpus.
"""


def crear_agente():
    """Construye el agente (lazy): carga SDK y credenciales solo al llamar."""
    from dotenv import load_dotenv
    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
    )
    from langchain.agents.structured_output import ToolStrategy
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
    # ToolStrategy: muchos modelos de OpenRouter (p. ej. Qwen) no rellenan
    # structured_response con la estrategia nativa del proveedor → None y luego
    # TypeError al convertir la respuesta. Forzamos tool-calling del esquema.
    return create_agent(
        model=model,
        tools=[list_available, get_xbrl_fact, search_filings, read_section],
        system_prompt=SYSTEM,
        response_format=ToolStrategy(RespuestaFinanciera),
        checkpointer=InMemorySaver(),
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=SETTINGS.llm_call_run_limit,
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
                run_limit=SETTINGS.tool_call_run_limit,
                exit_behavior="end",
            ),
        ],
    )


def responder(pregunta: str):
    """Atajo al ``responder`` público (dict evaluable, no resultado crudo)."""
    from cristian.experiments.responder import responder as _responder

    return _responder(pregunta)


def _imprimir_respuesta(answer: dict) -> None:
    """Muestra la respuesta estructurada del dict de ``responder``."""
    print("\n--- RESPUESTA ---")
    print(answer.get("respuesta"))
    print(f"fuente: {answer.get('fuente')}")
    if answer.get("cifra") is not None:
        print(
            f"cifra:  {answer.get('cifra')} {answer.get('unidad') or ''}".rstrip()
        )
    if answer.get("ticker"):
        print(
            f"ticker: {answer.get('ticker')}  "
            f"ejercicio: {answer.get('ejercicio')}"
        )
    if answer.get("chunk_id"):
        print(f"chunk:  {answer.get('chunk_id')}")
    if answer.get("cita"):
        cita = answer["cita"]
        print(f"cita:   {cita[:300]}{'…' if len(cita) > 300 else ''}")
    tools = answer.get("tool_calls_agente") or []
    if tools:
        print(f"tools:  {tools}")


if __name__ == "__main__":
    # Modo interactivo: gasta API (OpenRouter) en cada pregunta.
    pregunta_cli = " ".join(sys.argv[1:]).strip()
    print(f"modelo:  {SETTINGS.model}")
    print("Escribe una pregunta (vacío o 'salir' para terminar).\n")

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
            _imprimir_respuesta(responder(pregunta))
        except Exception as exc:
            import traceback

            print(f"\nError: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        print()
