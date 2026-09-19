"""Guardrail XBRL: verifica que las cifras del agente coincidan con XBRL.

Estrategia: post-procesado de la respuesta estructurada. Si el agente
declara fuente='xbrl' o 'ambas' con una cifra, se compara contra el valor
real de xbrl_facts.parquet. Si no coincide dentro de tolerancia, se
devuelve el desajuste al modelo para que corrija en una segunda pasada.
"""
import re
from pathlib import Path

import pandas as pd

_raiz = Path(__file__).resolve().parents[2]
xbrl = pd.read_parquet(_raiz / "corpus" / "xbrl_facts.parquet")

TOLERANCIA = 0.005  # 0.5%


def _cifras_coinciden(a: float, b: float) -> bool:
    if a is None or b is None:
        return False
    if b == 0:
        return abs(a) < 1.0
    return abs(a - b) / abs(b) <= TOLERANCIA


def verificar_cifra(respuesta: dict, ticker: str, fiscal_year: int) -> tuple[bool, str]:
    """Verifica que la cifra del agente coincide con algún valor XBRL.

    Returns:
        (ok: bool, mensaje: str)
    """
    cifra = respuesta.get("cifra")
    fuente = respuesta.get("fuente")

    # Solo verificamos si el agente dijo que viene de XBRL
    if fuente not in {"xbrl", "ambas"}:
        return True, "no aplica (fuente no es xbrl)"

    if cifra is None:
        return True, "no aplica (sin cifra)"

    # Buscar todos los conceptos del ticker/año
    candidatos = xbrl[
        (xbrl.ticker == ticker) & (xbrl.fiscal_year == int(fiscal_year))
    ]

    if candidatos.empty:
        return True, "no hay datos XBRL para verificar"

    # Buscar si la cifra coincide con ALGUNO de los conceptos
    for _, fila in candidatos.iterrows():
        if _cifras_coinciden(cifra, float(fila.value)):
            return True, f"coincide con {fila.concept}"

    # No coincide con ningún concepto → devolver los más cercanos
    cifras_cercanas = []
    for _, fila in candidatos.iterrows():
        desviacion = abs(cifra - float(fila.value)) / max(abs(float(fila.value)), 1)
        cifras_cercanas.append((desviacion, fila.concept, float(fila.value)))
    cifras_cercanas.sort()

    mensaje = (
        f"La cifra {cifra:,.0f} no coincide con ningún valor XBRL de "
        f"{ticker} FY{fiscal_year}. Valores disponibles más cercanos:\n"
    )
    for desv, concept, valor in cifras_cercanas[:3]:
        mensaje += f"  - {concept}: {valor:,.0f} (desviación {desv*100:.1f}%)\n"
    mensaje += "Corrige la cifra o cambia fuente a 'texto' si no viene de XBRL."
    return False, mensaje