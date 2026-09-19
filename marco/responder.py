"""Función responder(pregunta) — interfaz pública del agente."""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

print(">>> Importando agente...")
try:
    from agent import crear_agente
    print(">>> Agente importado OK")
except Exception:
    print(">>> ERROR importando agente:")
    traceback.print_exc()
    sys.exit(1)


_agente = None


def _get_agente():
    global _agente
    if _agente is None:
        print(">>> Creando agente (puede tardar unos segundos)...")
        _agente = crear_agente()
        print(">>> Agente creado OK")
    return _agente


def responder(pregunta: str) -> dict:
    """Responde a una pregunta sobre los 10-K del corpus.

    Devuelve un dict con los campos de RespuestaFinanciera y la lista de
    herramientas llamadas (tool_calls_agente).
    """
    agente = _get_agente()
    config = {"configurable": {"thread_id": "default"}}
    resultado = agente.invoke(
        {"messages": [{"role": "user", "content": pregunta}]},
        config=config,
    )

    # Extraer la lista de tool calls de los mensajes
    tool_calls = []
    for msg in resultado["messages"]:
        for tc in (getattr(msg, "tool_calls", None) or []):
            tool_calls.append(tc["name"])

    e = resultado["structured_response"]
    salida = e.model_dump()
    salida["tool_calls_agente"] = tool_calls
    return salida


if __name__ == "__main__":
    try:
        r = responder("¿Cuál fue el revenue de NVIDIA en FY2024?")
        print("\n=== RESULTADO ===")
        for k, v in r.items():
            print(f"  {k}: {v}")
    except Exception:
        print("\n=== ERROR ===")
        traceback.print_exc()
        sys.exit(1)