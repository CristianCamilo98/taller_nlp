"""Cross-encoder reranking (BGE-reranker) sobre un pool denso.

La etapa dense (Gemini / BGE / …) propone candidatos; este módulo solo
reordena pares ``(query, texto)`` sin volver a buscar en el corpus.
"""

from __future__ import annotations

import functools
import threading
from typing import Any

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# m3 admite contextos largos; evita el corte a 512 que rompía MiniLM en Marco.
DEFAULT_MAX_LENGTH = 8192

_reranker_lock = threading.Lock()


@functools.lru_cache(maxsize=2)
def _cargar_reranker(model_name: str, max_length: int):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=max_length)


def get_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
):
    """Carga el cross-encoder una vez (thread-safe)."""
    with _reranker_lock:
        return _cargar_reranker(model_name, max_length)


def precargar_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> None:
    get_reranker(model_name, max_length=max_length)


def clear_reranker_cache() -> None:
    _cargar_reranker.cache_clear()


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int | None = None,
    model_name: str = DEFAULT_RERANKER_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[dict[str, Any]]:
    """Reordena ``hits`` de ``buscar`` por score del cross-encoder.

    Conserva todos los campos del hit denso y añade ``dense_score`` /
    ``puntuacion`` (score del reranker). No trunca el texto a mano: el
    tokenizer del modelo corta solo si supera ``max_length``.
    """
    if not hits:
        return []

    model = get_reranker(model_name, max_length=max_length)
    pairs = [(query, str(hit.get("texto") or "")) for hit in hits]
    scores = model.predict(pairs, show_progress_bar=False)

    ranked: list[dict[str, Any]] = []
    for hit, score in zip(hits, scores):
        row = dict(hit)
        row["dense_score"] = float(hit.get("puntuacion", 0.0))
        row["puntuacion"] = round(float(score), 6)
        ranked.append(row)

    ranked.sort(key=lambda row: row["puntuacion"], reverse=True)
    if top_k is not None:
        ranked = ranked[: max(0, int(top_k))]
    return ranked
