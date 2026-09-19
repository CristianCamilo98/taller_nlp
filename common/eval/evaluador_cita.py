"""Evaluación separada de existencia, ancla y soporte de una cita."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from common.config import get_dataset_paths


def _normalizar(text: str | None) -> str:
    if not text:
        return ""
    value = "".join(char for char in unicodedata.normalize("NFD", str(text))
                    if unicodedata.category(char) != "Mn").lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@lru_cache(maxsize=1)
def _chunks():
    import pandas as pd

    return pd.read_parquet(get_dataset_paths().chunks_meta)


def _fragments(citation: str) -> list[str]:
    return [part for part in (_normalizar(p) for p in
            re.split(r"\s*(?:\.\.\.|…)+\s*", citation)) if part]


def _evidence_in_text(evidence: str, text: str) -> bool:
    pieces = _fragments(evidence)
    return bool(pieces) and all(piece in text for piece in pieces)


def evaluar_cita_detallada(resultado: dict, pregunta: dict) -> dict:
    anchor = pregunta.get("ancla_texto")
    alternatives = pregunta.get("anclas_alternativas") or []
    if not anchor and not alternatives:
        return {
            "aplica": False,
            "citation_exists": None,
            "chunk_exists": None,
            "metadata_matches": None,
            "golden_anchor_hit": None,
            "citation_supports": None,
            "support_deterministic": None,
            "citation_metric": None,
            "acierto_cita_literal_anchor": None,
            "acierto_cita": None,
        }

    citation = resultado.get("cita_agente") or resultado.get("cita")
    chunk_id = resultado.get("chunk_id_agente") or resultado.get("chunk_id")
    provided = bool(citation and str(citation).strip())
    details = {
        "aplica": True,
        "citation_exists": False,
        "chunk_exists": False,
        "metadata_matches": False,
        "golden_anchor_hit": False,
        "citation_supports": None,
        "support_deterministic": False,
        "citation_metric": "literal_citation_and_golden_anchor_in_same_chunk",
        "acierto_cita_literal_anchor": False,
        "acierto_cita": False,
    }
    if not chunk_id:
        return details

    rows = _chunks()[_chunks()["chunk_id"] == chunk_id]
    if rows.empty:
        return details
    details["chunk_exists"] = True
    row = rows.iloc[0]
    text = _normalizar(str(row["texto"]))
    pieces = _fragments(str(citation)) if provided else []
    literal = bool(pieces) and all(piece in text for piece in pieces)
    details["citation_exists"] = literal
    accepted = [candidate for candidate in [anchor, *alternatives] if candidate]
    accepted_hit = any(_evidence_in_text(str(candidate), text)
                       for candidate in accepted)
    details["golden_anchor_hit"] = accepted_hit

    try:
        metadata_matches = (
            str(row["ticker"]).upper() == str(pregunta["ticker"]).upper()
            and int(row["fiscal_year"]) == int(pregunta["fiscal_year"])
            and str(row["item"]) == str(
                pregunta.get("item", pregunta.get("item_esperado")))
        )
    except (KeyError, TypeError, ValueError):
        metadata_matches = False
    details["metadata_matches"] = metadata_matches
    if not metadata_matches:
        return details
    aggregate = literal and accepted_hit and metadata_matches
    details["acierto_cita_literal_anchor"] = aggregate
    details["acierto_cita"] = aggregate
    # No se infiere soporte semántico. Una cita literal y un anchor pueden
    # aparecer en zonas distintas del mismo chunk.
    return details


def evaluar_cita(resultado: dict, pregunta: dict) -> bool | None:
    return evaluar_cita_detallada(resultado, pregunta)["acierto_cita"]
