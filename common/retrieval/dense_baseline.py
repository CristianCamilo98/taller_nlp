"""Retrieval dense baseline sobre el índice FAISS docente entregado.

Esta es la mínima implementación runtime extraída del starter de sesión 1.
No incluye BM25, reranking, query rewriting ni cambios de chunking.
"""

from __future__ import annotations

import functools

from common.benchmark_config import BENCHMARK
from common.config import get_dataset_paths

MODELO_EMBEDDINGS = BENCHMARK.embedding_model
PREFIJO_CONSULTA_BGE = BENCHMARK.embedding_query_prefix


@functools.lru_cache(maxsize=1)
def _indice():
    """Carga perezosamente índice, metadata y codificador."""
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    paths = get_dataset_paths()
    indice = faiss.read_index(str(paths.faiss_index))
    meta = pd.read_parquet(paths.chunks_meta)
    if indice.ntotal != len(meta):
        raise RuntimeError(
            f"El índice tiene {indice.ntotal} vectores y los metadatos "
            f"{len(meta)} filas. Están desalineados."
        )
    codificador = SentenceTransformer(MODELO_EMBEDDINGS)
    return indice, meta, codificador


def buscar(query: str, ticker: str | None = None,
           fiscal_year: int | None = None, item: str | None = None,
           k: int = 5) -> list[dict]:
    """Ranking global denso seguido de filtros de metadata."""
    indice, meta, codificador = _indice()
    vector = codificador.encode(
        [PREFIJO_CONSULTA_BGE + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    puntuaciones, posiciones = indice.search(vector, indice.ntotal)
    resultados = []
    for puntuacion, posicion in zip(puntuaciones[0], posiciones[0]):
        fila = meta.iloc[int(posicion)]
        if ticker and fila["ticker"] != ticker:
            continue
        if fiscal_year and int(fila["fiscal_year"]) != int(fiscal_year):
            continue
        if item and fila["item"] != item:
            continue
        resultados.append({
            "chunk_id": fila["chunk_id"],
            "ticker": fila["ticker"],
            "fiscal_year": int(fila["fiscal_year"]),
            "item": fila["item"],
            "texto": fila["texto"],
            "n_tokens": int(fila["n_tokens"]),
            "contiene_tabla": bool(fila["contiene_tabla"]),
            "puntuacion": round(float(puntuacion), 4),
        })
        if len(resultados) >= k:
            break
    return resultados


def formatear_fragmentos(fragmentos: list[dict]) -> str:
    """Formatea resultados con chunk_id y metadata para el agente."""
    if not fragmentos:
        return ("Sin resultados para esa consulta con esos filtros. "
                "Prueba a quitar algún filtro o a reformular la búsqueda.")
    partes = []
    for fragmento in fragmentos:
        partes.append(
            f"[{fragmento['chunk_id']}] {fragmento['ticker']} "
            f"FY{fragmento['fiscal_year']} Item {fragmento['item']} "
            f"(similitud {fragmento['puntuacion']:.3f})\n"
            f"{fragmento['texto']}"
        )
    return "\n\n---\n\n".join(partes)
