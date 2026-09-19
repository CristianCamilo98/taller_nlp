"""Evaluador de cita estricto.

Regla doble:
1. La cita del agente debe aparecer literalmente en el chunk citado.
2. El ancla_texto del golden set debe aparecer en el MISMO chunk citado
   (no en vecinos).

Si no se cumplen ambas condiciones, es fallo.
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

_raiz = Path(__file__).resolve().parents[2]
chunks_meta = pd.read_parquet(
    _raiz / "corpus" / "indice" / "chunks_meta.parquet"
)


def _normalizar(texto: str) -> str:
    if not texto:
        return ""
    texto = texto.lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def _texto_del_chunk(chunk_id: str) -> str | None:
    if not chunk_id:
        return None
    fila = chunks_meta[chunks_meta.chunk_id == chunk_id]
    if fila.empty:
        return None
    return fila.iloc[0].texto


def evaluar_cita(resultado: dict, pregunta: dict) -> bool | None:
    """Comprueba que:
    1. La cita del agente aparece literalmente en el chunk citado.
    2. El ancla del golden set aparece en el MISMO chunk citado.

    Devuelve None si la pregunta no lleva componente cualitativo.
    """
    if not pregunta.get("ancla_texto"):
        return None

    cita = resultado.get("cita_agente")
    chunk_id = resultado.get("chunk_id_agente")

    if not chunk_id:
        return False

    texto = _texto_del_chunk(chunk_id)
    if texto is None:
        return False  # chunk_id inventado o no existe

    texto_norm = _normalizar(texto)

    # Condición 1: la cita existe en el chunk citado
    if cita:
        cita_norm = _normalizar(cita)
        if cita_norm and cita_norm not in texto_norm:
            return False  # cita alucinada o mal referenciada

    # Condición 2: el ancla del golden set está en el MISMO chunk
    ancla_norm = _normalizar(pregunta["ancla_texto"])
    if ancla_norm not in texto_norm:
        return False

    return True


if __name__ == "__main__":
    # Test 1: chunk exacto contiene ancla y cita coherente
    print("Test 1:",
          evaluar_cita(
              {"chunk_id_agente": "MSFT-2024-7A-0000",
               "cita_agente": "Certain forecasted transactions, assets, and liabilities are exposed to foreign currency risk."},
              {"familia": "extractiva",
               "ancla_texto": "Certain forecasted transactions, assets, and liabilities are exposed to foreign currency risk."}))

    # Test 2: chunk NO contiene el ancla (debería ser False)
    print("Test 2 (esperado False):",
          evaluar_cita(
              {"chunk_id_agente": "MSFT-2024-7A-0001",
               "cita_agente": "texto cualquiera"},
              {"familia": "extractiva",
               "ancla_texto": "Certain forecasted transactions, assets, and liabilities are exposed to foreign currency risk."}))