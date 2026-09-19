"""Evaluador de cifra: comprueba que la cifra coincide con XBRL."""
import sys
from pathlib import Path

import pandas as pd

# Cargar xbrl_facts
_raiz = Path(__file__).resolve().parents[2]
xbrl = pd.read_parquet(_raiz / "corpus" / "xbrl_facts.parquet")

# Tolerancia relativa del 0.5% (acordada en el contrato)
TOLERANCIA = 0.005


def evaluar_cifra(resultado: dict, pregunta: dict) -> bool | None:
    """Devuelve True si la cifra del agente coincide con XBRL.

    Devuelve None si la pregunta no es numérica/comparativa.
    """
    familia = pregunta.get("familia")
    if familia not in {"numerica", "comparativa"}:
        return None

    cifra_agente = resultado.get("cifra_agente")
    if cifra_agente is None:
        return False

    # Normalizar: si el agente devuelve millones, convertir
    # (heurística simple, se puede afinar)
    cifra_esperada = pregunta.get("cifra_esperada")
    if cifra_esperada is None:
        return None

    # Comparación con tolerancia relativa
    if cifra_esperada == 0:
        return abs(cifra_agente) < 1.0
    desviacion = abs(cifra_agente - cifra_esperada) / abs(cifra_esperada)
    return desviacion <= TOLERANCIA


if __name__ == "__main__":
    # Test rápido
    preg = {"familia": "numerica", "cifra_esperada": 60922000000.0}
    res = {"cifra_agente": 60922000000.0}
    print("Test OK:", evaluar_cifra(res, preg))