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





from agent.middleware_xbrl import verificar_cifra


def responder(pregunta: str, max_reintentos: int = 1) -> dict:
    """Responde a una pregunta sobre los 10-K del corpus.

    Aplica el guardrail XBRL: si la cifra no coincide con el corpus, hace
    un reintento pasándole el desajuste al modelo.
    """
    agente = _get_agente()

    # Primera invocación
    config = {"configurable": {"thread_id": "default"}}
    resultado = agente.invoke(
        {"messages": [{"role": "user", "content": pregunta}]},
        config=config,
    )

    tool_calls = []
    for msg in resultado["messages"]:
        for tc in (getattr(msg, "tool_calls", None) or []):
            if tc["name"] != "RespuestaFinanciera":
                tool_calls.append(tc["name"])

    e = resultado["structured_response"]
    respuesta = e.model_dump()
    respuesta["tool_calls_agente"] = tool_calls

    # Guardrail XBRL
    ticker = respuesta.get("ticker")
    ejercicio = respuesta.get("ejercicio")
    if ticker and ejercicio:
        ok, mensaje = verificar_cifra(respuesta, ticker, ejercicio)
        if not ok and max_reintentos > 0:
            print(f">>> Guardrail XBRL: {mensaje}")
            # Reintento con el desajuste
            pregunta_corregida = (
                f"{pregunta}\n\n"
                f"AVISO DEL SISTEMA: {mensaje}"
            )
            return responder(pregunta_corregida, max_reintentos - 1)

    return respuesta

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