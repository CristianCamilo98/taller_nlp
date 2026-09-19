"""Función responder(pregunta) — interfaz pública del agente."""
import sys
import traceback
import uuid
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


def _es_pregunta_numerica(pregunta: str) -> bool:
    """Heurística: ¿la pregunta pide una cifra puntual (no un crecimiento)?

    Las comparativas suelen llevar 'creció', 'aumentó', 'varió', 'entre
    FY2024 y FY2025', etc.
    """
    p = pregunta.lower()
    marcadores_comparativa = [
        "creció", "crecio", "aumentó", "aumento", "varió", "vario",
        "evolucionó", "evoluciono", "cambió", "cambio",
        "entre fy", "respecto", "comparado", "comparación", "comparacion",
    ]
    if any(m in p for m in marcadores_comparativa):
        return False
    return True


def responder(pregunta: str, max_reintentos: int = 1,
              thread_id: str | None = None) -> dict:
    """Responde a una pregunta sobre los 10-K del corpus.

    Args:
        pregunta: la pregunta en lenguaje natural.
        max_reintentos: reintentos del guardrail XBRL.
        thread_id: identificador del hilo. Si es None, se genera uno único
            por invocación para que cada pregunta del benchmark sea
            independiente (sin memoria compartida).

    Aplica el guardrail XBRL solo en preguntas numéricas puras. En
    comparativas el guardrail se desactiva porque el agente devuelve un
    delta (crecimiento) que no es un valor XBRL absoluto.
    """
    agente = _get_agente()

    # Un thread_id único por invocación evita que las preguntas compartan
    # memoria entre sí (que la trayectoria y el contexto se acumulen).
    if thread_id is None:
        thread_id = f"q-{uuid.uuid4().hex[:8]}"

    config = {"configurable": {"thread_id": thread_id}}
    resultado = agente.invoke(
        {"messages": [{"role": "user", "content": pregunta}]},
        config=config,
    )

    # Extraer tool_calls con argumentos, filtrando el structured output
    tool_calls = []
    for msg in resultado["messages"]:
        for tc in (getattr(msg, "tool_calls", None) or []):
            if tc["name"] == "RespuestaFinanciera":
                continue
            tool_calls.append({
                "name": tc["name"],
                "args": tc.get("args") or {},
            })

    e = resultado["structured_response"]
    respuesta = e.model_dump()
    respuesta["tool_calls_agente"] = [tc["name"] for tc in tool_calls]
    respuesta["tool_calls_detallado"] = tool_calls

    # Guardrail XBRL — solo para preguntas numéricas puras
    es_numerica = _es_pregunta_numerica(pregunta)
    ticker = respuesta.get("ticker")
    ejercicio = respuesta.get("ejercicio")

    if es_numerica and ticker and ejercicio:
        ok, mensaje = verificar_cifra(respuesta, ticker, ejercicio)
        if not ok and max_reintentos > 0:
            print(f">>> Guardrail XBRL: {mensaje}")
            pregunta_corregida = (
                f"{pregunta}\n\n"
                f"AVISO DEL SISTEMA: {mensaje}"
            )
            # El reintento mantiene el MISMO thread_id: es la misma pregunta,
            # solo que con información adicional para autocorregirse.
            return responder(pregunta_corregida,
                             max_reintentos - 1,
                             thread_id=thread_id)

    return respuesta


if __name__ == "__main__":
    try:
        r = responder("¿Cuál fue el revenue de NVIDIA en FY2024?")
        print("\n=== RESULTADO ===")
        for k, v in r.items():
            if k == "tool_calls_detallado":
                print(f"  {k}:")
                for tc in v:
                    print(f"    {tc}")
            else:
                print(f"  {k}: {v}")
    except Exception:
        print("\n=== ERROR ===")
        traceback.print_exc()
        sys.exit(1)