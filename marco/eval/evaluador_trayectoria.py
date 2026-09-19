"""Evaluador de trayectoria: comprueba que el agente usó la herramienta correcta.

Nota: el resultado necesita traer la lista de tool_calls. En `evaluar.py`
hay que capturarla del resultado del agente y pasarla aquí.
"""


def evaluar_trayectoria(resultado: dict, pregunta: dict) -> bool | None:
    """Comprueba que la trayectoria incluye las herramientas esperadas.

    `resultado["tool_calls_agente"]` debe ser una lista de strings con los
    nombres de las herramientas llamadas, en orden.

    Devuelve True si TODAS las herramientas esperadas están presentes.
    Devuelve None si la pregunta no especifica herramienta_esperada.
    """
    esperadas = pregunta.get("herramienta_esperada") or []
    if not esperadas:
        return None

    llamadas = resultado.get("tool_calls_agente") or []
    if not llamadas:
        return False

    # Todas las esperadas deben estar en las llamadas
    for h in esperadas:
        if h not in llamadas:
            return False
    return True


if __name__ == "__main__":
    preg = {"herramienta_esperada": ["get_xbrl_fact"]}
    res = {"tool_calls_agente": ["list_available", "get_xbrl_fact"]}
    print("Test OK:", evaluar_trayectoria(res, preg))
    # Debe imprimir True