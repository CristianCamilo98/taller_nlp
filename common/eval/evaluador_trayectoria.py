"""Evaluador de trayectoria con validación de argumentos.

Comprueba:
1. Todas las herramientas esperadas por el golden set aparecen.
2. Para preguntas numéricas, los argumentos de get_xbrl_fact coinciden
   con el ticker y fiscal_year de la pregunta.
"""
import re


def _normalizar_ticker(t: str | None) -> str:
    return (t or "").strip().upper()


def evaluar_trayectoria(resultado: dict, pregunta: dict) -> bool | None:
    """Comprueba la trayectoria del agente.

    Espera `resultado["tool_calls_detallado"]` con dicts:
    [{"name": "get_xbrl_fact", "args": {...}}, ...]

    Devuelve None si la pregunta no especifica herramienta_esperada.
    """
    esperadas = pregunta.get("herramienta_esperada") or []
    if not esperadas:
        return None

    tool_calls = resultado.get("tool_calls_detallado") or []
    if not tool_calls:
        return False

    nombres_llamados = [tc["name"] for tc in tool_calls]

    # Condición 1: todas las esperadas aparecen
    for e in esperadas:
        if e not in nombres_llamados:
            return False

    # Condición 2: para numéricas puras, verificar argumentos de get_xbrl_fact
    if pregunta.get("familia") == "numerica":
        ticker_esperado = _normalizar_ticker(pregunta.get("ticker"))
        fy_esperado = int(pregunta.get("fiscal_year", 0))

        al_menos_una_valida = False
        for tc in tool_calls:
            if tc["name"] != "get_xbrl_fact":
                continue
            args = tc.get("args") or {}
            ticker_llamado = _normalizar_ticker(args.get("ticker"))
            fy_llamado = args.get("fiscal_year")
            try:
                fy_llamado = int(fy_llamado)
            except (TypeError, ValueError):
                continue

            if ticker_llamado == ticker_esperado and fy_llamado == fy_esperado:
                al_menos_una_valida = True
                break

        if not al_menos_una_valida:
            return False

    return True


if __name__ == "__main__":
    # Test 1: numérica, get_xbrl_fact con args correctos
    print("Test 1 (esperado True):",
          evaluar_trayectoria(
              {"tool_calls_detallado": [
                  {"name": "get_xbrl_fact",
                   "args": {"ticker": "NVDA", "fiscal_year": 2024,
                            "concept": "Revenues"}}]},
              {"familia": "numerica", "ticker": "NVDA", "fiscal_year": 2024,
               "herramienta_esperada": ["get_xbrl_fact"]}))

    # Test 2: numérica, pero ticker incorrecto
    print("Test 2 (esperado False):",
          evaluar_trayectoria(
              {"tool_calls_detallado": [
                  {"name": "get_xbrl_fact",
                   "args": {"ticker": "MSFT", "fiscal_year": 2024,
                            "concept": "Revenues"}}]},
              {"familia": "numerica", "ticker": "NVDA", "fiscal_year": 2024,
               "herramienta_esperada": ["get_xbrl_fact"]}))