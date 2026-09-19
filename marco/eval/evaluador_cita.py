"""Evaluador de cita.

Criterio: la cita respalda la respuesta si el chunk citado (o un chunk
adyacente) contiene el ancla_texto del golden set. Esto verifica que
el agente encontró la información correcta en el corpus, independientemente
de si su cita literal es idéntica a la frase completa o solo parcial.
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

_raiz = Path(__file__).resolve().parents[2]
chunks_meta = pd.read_parquet(_raiz / "corpus" / "indice" / "chunks_meta.parquet")


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


def _chunks_adyacentes(chunk_id: str, radio: int = 2) -> list[str]:
    """Devuelve el chunk y sus vecinos ±radio."""
    if not chunk_id:
        return []
    m = re.match(r"(.+)-(\d+)$", chunk_id)
    if not m:
        return [chunk_id]
    base = m.group(1)
    n = int(m.group(2))
    ancho = len(m.group(2))
    return [
        f"{base}-{i:0{ancho}d}"
        for i in range(n - radio, n + radio + 1)
        if i >= 0
    ]


def evaluar_cita(resultado: dict, pregunta: dict) -> bool | None:
    """Comprueba que la cita respalda la respuesta.

    Devuelve None si la pregunta no lleva componente cualitativo.
    """
    if not pregunta.get("ancla_texto"):
        return None

    chunk_id = resultado.get("chunk_id_agente")
    cita = resultado.get("cita_agente")
    ancla = pregunta.get("ancla_texto")
    ancla_norm = _normalizar(ancla)

    # Criterio 1 (principal): el chunk citado o un adyacente contiene el ancla
    if chunk_id:
        for cid in _chunks_adyacentes(chunk_id, radio=2):
            texto = _texto_del_chunk(cid)
            if texto and ancla_norm in _normalizar(texto):
                return True

    # Criterio 2 (fallback): la cita del agente contiene el ancla
    if cita:
        cita_norm = _normalizar(cita)
        if ancla_norm in cita_norm or cita_norm in ancla_norm:
            return True

    return False


if __name__ == "__main__":
    # Test 1: chunk adyacente contiene ancla
    print("Test 1:",
          evaluar_cita(
              {"chunk_id_agente": "NVDA-2025-1A-0037", "cita_agente": "x"},
              {"familia": "extractiva",
               "ancla_texto": "As a result, excessive or shifting export controls may negatively impact demand for our products and services not only in China, but also in other markets"}))
    # Esperado: True (0038 está en el radio 1)