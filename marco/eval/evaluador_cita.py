"""Evaluador de cita: comprueba que la cita respalda de verdad la respuesta."""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

_raiz = Path(__file__).resolve().parents[2]
secciones = pd.DataFrame(
    json.loads(l) for l in open(_raiz / "corpus" / "secciones.jsonl",
                                 encoding="utf-8")
)
chunks_meta = pd.read_parquet(_raiz / "corpus" / "indice" / "chunks_meta.parquet")


def _normalizar(texto: str) -> str:
    """Minúsculas, sin tildes, sin puntuación, espacios colapsados."""
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
    """Devuelve el texto del chunk citado, o None si no existe."""
    if not chunk_id:
        return None
    fila = chunks_meta[chunks_meta.chunk_id == chunk_id]
    if fila.empty:
        return None
    return fila.iloc[0].texto


def evaluar_cita(resultado: dict, pregunta: dict) -> bool | None:
    """Comprueba que la cita aparece en el chunk citado.

    Devuelve True si la cita está y respalda la respuesta.
    Devuelve None si la pregunta no es extractiva/comparativa.
    """
    familia = pregunta.get("familia")
    if familia not in {"extractiva", "comparativa"}:
        return None

    cita = resultado.get("cita_agente")
    chunk_id = resultado.get("chunk_id_agente")

    # Sin cita o sin chunk_id → fallo (para extractivas)
    if not cita or not chunk_id:
        # Caso especial: el agente dijo "no está en el corpus" y la respuesta
        # esperada es eso mismo. Sería acierto, pero lo dejamos como None para
        # que lo revise un humano.
        return False

    texto = _texto_del_chunk(chunk_id)
    if texto is None:
        return False  # chunk_id inventado

    cita_norm = _normalizar(cita)
    texto_norm = _normalizar(texto)

    # La cita debe aparecer en el chunk citado
    if cita_norm not in texto_norm:
        return False

    # Para extractivas: además, el ancla_texto debe estar contenida
    # en la cita (o la cita en el ancla)
    if familia == "extractiva":
        ancla = pregunta.get("ancla_texto")
        if ancla:
            ancla_norm = _normalizar(ancla)
            # Aceptamos que el ancla esté en la cita o la cita en el ancla
            if ancla_norm not in cita_norm and cita_norm not in ancla_norm:
                # No descartamos por esto, pero lo marcamos como "dudoso"
                # devolviendo False para que sea conservador
                return False

    return True


if __name__ == "__main__":
    # Test: cita que sí respalda
    preg = {
        "familia": "extractiva",
        "ancla_texto": "Increasing use of generative AI models in our internal systems may create new attack surfaces or methods for adversaries.",
    }
    res = {
        "cita_agente": "Increasing use of generative AI models in our internal systems may create new attack surfaces or methods for adversaries.",
        "chunk_id_agente": "MSFT-2025-1A-0009",
    }
    print("Test OK:", evaluar_cita(res, preg))