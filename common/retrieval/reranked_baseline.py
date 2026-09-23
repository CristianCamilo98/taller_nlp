"""Retrieval con reranking por cross-encoder sobre el baseline denso."""
from __future__ import annotations

import functools

from common.retrieval.dense_baseline import buscar as buscar_denso

# R0 (MiniLM)
#MODELO_RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#MODELO_RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"

MODELO_RERANK = "BAAI/bge-reranker-v2-m3"
MODELO_RERANK_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


@functools.lru_cache(maxsize=1)
def _modelo():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(
        MODELO_RERANK,
        revision=MODELO_RERANK_REVISION,
        trust_remote_code=True,
    )

def buscar(query: str, ticker: str | None = None,
           fiscal_year: int | None = None, item: str | None = None,
           k: int = 20, n_candidatos: int = 20) -> list[dict]:
    if int(k) <= 0:
        raise ValueError("k debe ser un entero positivo")
    if n_candidatos < k:
        n_candidatos = k

    candidatos = buscar_denso(
        query, ticker=ticker, fiscal_year=fiscal_year,
        item=item, k=n_candidatos,
    )
    if not candidatos:
        return []

    pares = [(query, c["texto"]) for c in candidatos]
    scores = _modelo().predict(pares)

    ordenados = sorted(zip(candidatos, scores), key=lambda x: -float(x[1]))
    resultado = []
    for candidato, score in ordenados[:k]:
        nuevo = dict(candidato)
        nuevo["puntuacion"] = round(float(score), 4)
        nuevo["puntuacion_densa"] = candidato["puntuacion"]
        resultado.append(nuevo)
    return resultado


def formatear_fragmentos(fragmentos: list[dict]) -> str:
    if not fragmentos:
        return ("Sin resultados para esa consulta con esos filtros.")
    partes = []
    for fragmento in fragmentos:
        partes.append(
            f"[{fragmento['chunk_id']}] {fragmento['ticker']} "
            f"FY{fragmento['fiscal_year']} Item {fragmento['item']} "
            f"(similitud {fragmento['puntuacion']:.3f})\n"
            f"{fragmento['texto']}"
        )
    return "\n\n---\n\n".join(partes)