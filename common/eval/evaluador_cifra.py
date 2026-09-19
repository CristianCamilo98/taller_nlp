"""Evaluador de cifra flexible: acepta valor absoluto O delta O cifra en prosa."""
import re
import unicodedata
from pathlib import Path

import pandas as pd

_raiz = Path(__file__).resolve().parents[2]
xbrl = pd.read_parquet(_raiz / "corpus" / "xbrl_facts.parquet")

TOLERANCIA = 0.005  # 0.5%


def _normalizar(texto: str) -> str:
    """Minúsculas, sin tildes, sin comas, espacios colapsados."""
    if not texto:
        return ""
    texto = texto.lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    # Quitar comas y puntos de miles, mantener dígitos y letras
    texto = re.sub(r"[,\s]+", "", texto)
    return texto


def _cifra_en_texto(cifra: float, texto: str) -> bool:
    """Comprueba si la cifra aparece en el texto (en cualquier formato)."""
    if not texto or cifra is None:
        return False
    texto_norm = _normalizar(texto)
    # Probar varios formatos: con decimales, sin decimales, redondeada
    candidatos = [
        f"{cifra:.0f}",
        f"{int(cifra)}",
        f"{cifra:.2f}",
    ]
    # También probar en millones/miles
    if abs(cifra) >= 1e9:
        candidatos.append(f"{cifra/1e9:.0f}")
        candidatos.append(f"{cifra/1e9:.1f}")
    if abs(cifra) >= 1e6:
        candidatos.append(f"{cifra/1e6:.0f}")
        candidatos.append(f"{cifra/1e6:.1f}")

    for c in candidatos:
        c_norm = _normalizar(c)
        if c_norm and c_norm in texto_norm:
            return True
    return False


def _cifra_coincide(valor_a: float, valor_b: float) -> bool:
    """Compara dos cifras con tolerancia relativa."""
    if valor_a is None or valor_b is None:
        return False
    if valor_b == 0:
        return abs(valor_a) < 1.0
    return abs(valor_a - valor_b) / abs(valor_b) <= TOLERANCIA


def evaluar_cifra(resultado: dict, pregunta: dict) -> bool | None:
    """Evalúa si la cifra del agente es correcta.

    Acepta:
    1. cifra_agente ≈ cifra_esperada (comparación numérica)
    2. cifra_esperada aparece en el texto de respuesta_agente
    3. Para comparativas: acepta también el delta (cifra_esperada - valor_FY_anterior)
    """
    familia = pregunta.get("familia")
    if familia not in {"numerica", "comparativa"}:
        return None

    cifra_agente = resultado.get("cifra_agente")
    if cifra_agente is None:
        return False

    cifra_esperada = pregunta.get("cifra_esperada")
    if cifra_esperada is None:
        return None

    # Opción 1: comparación numérica directa con el valor absoluto
    if _cifra_coincide(cifra_agente, cifra_esperada):
        return True

    # Opción 2: la cifra esperada aparece en la prosa (formato flexible)
    respuesta_texto = resultado.get("respuesta_agente") or ""
    if _cifra_en_texto(cifra_esperada, respuesta_texto):
        return True

    # Opción 3 (solo comparativas): aceptar el delta
    if familia == "comparativa" and pregunta.get("concept_xbrl"):
        valor_anterior = xbrl[
            (xbrl.ticker == pregunta["ticker"])
            & (xbrl.fiscal_year == int(pregunta["fiscal_year"]) - 1)
            & (xbrl.concept == pregunta["concept_xbrl"])
        ]
        if not valor_anterior.empty:
            delta = float(cifra_esperada) - float(valor_anterior.iloc[0].value)
            if _cifra_coincide(cifra_agente, delta):
                return True

    return False


if __name__ == "__main__":
    # Tests
    print("Test 1 (numérica directa):",
          evaluar_cifra(
              {"cifra_agente": 60922000000.0},
              {"familia": "numerica", "cifra_esperada": 60922000000.0}))
    print("Test 2 (cifra en prosa):",
          evaluar_cifra(
              {"cifra_agente": 36602000000.0,
               "respuesta_agente": "El revenue pasó de 245.122.000.000 a 281.724.000.000 USD"},
              {"familia": "comparativa", "ticker": "MSFT", "fiscal_year": 2025,
               "concept_xbrl": "RevenueFromContractWithCustomerExcludingAssessedTax",
               "cifra_esperada": 281724000000.0}))