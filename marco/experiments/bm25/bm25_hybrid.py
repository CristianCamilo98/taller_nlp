"""BM25 + RRF híbrido para ablación causal B0 vs B1.

Determinista, reproducible, sin LLM ni API.
No modifica common/. Solo importa dense_baseline y retrieval_metrics.

Tokenización congelada: regex UNICODE sobre texto en minúsculas.
BM25Okapi con k1=1.5, b=0.75.
RRF con k_rrf=60, rangos 1-based.
"""
from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Tokenización CONGELADA
# ---------------------------------------------------------------------------
TOKEN_REGEX = re.compile(r"[^\W_]+", re.UNICODE)
MIN_TOKEN_LEN = 2


def tokenize(text: str) -> list[str]:
    """Tokeniza minúsculas, regex unicode, filtra tokens <2 chars.

    Congelado: no cambiar tras ver resultados.
    No stopword removal (BM25 IDF ya penaliza términos frecuentes).
    """
    if not text:
        return []
    tokens = TOKEN_REGEX.findall(text.lower())
    return [t for t in tokens if len(t) >= MIN_TOKEN_LEN]


# ---------------------------------------------------------------------------
# Parámetros BM25 / RRF CONGELADOS
# ---------------------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
DEFAULT_K_CANDIDATES = 20


# ---------------------------------------------------------------------------
# Carga única del corpus + índice BM25
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _corpus_state():
    """Carga chunks y construye índice BM25 (una sola vez)."""
    from common.config import get_dataset_paths

    paths = get_dataset_paths()
    chunks_path = paths.corpus_dir / "chunks.jsonl"
    meta_path = paths.chunks_meta

    chunks = [json.loads(line)
              for line in open(chunks_path, encoding="utf-8")
              if line.strip()]
    meta = pd.read_parquet(meta_path)

    if len(chunks) != len(meta):
        raise RuntimeError(
            f"Desalineación: {len(chunks)} chunks vs {len(meta)} metadata"
        )

    chunks_sorted = sorted(chunks, key=lambda c: c["chunk_id"])
    meta_sorted = meta.sort_values("chunk_id").reset_index(drop=True)
    if [c["chunk_id"] for c in chunks_sorted] != list(meta_sorted["chunk_id"]):
        raise RuntimeError("chunk_id no coinciden entre chunks y meta")

    # Construir índice BM25
    from rank_bm25 import BM25Okapi
    tokenized = [tokenize(c["texto"]) for c in chunks_sorted]
    bm25 = BM25Okapi(tokenized, k1=BM25_K1, b=BM25_B)

    chunk_ids = [c["chunk_id"] for c in chunks_sorted]
    return meta_sorted, bm25, chunk_ids


# ---------------------------------------------------------------------------
# BM25 search con política global → postfilter
# ---------------------------------------------------------------------------
def bm25_search(query: str, ticker: str | None = None,
                fiscal_year: int | None = None,
                item: str | None = None,
                k: int = DEFAULT_K_CANDIDATES) -> list[dict]:
    """Ranking GLOBAL BM25 seguido de postfilter metadata.

    Misma política que dense_baseline.buscar().
    Tie-break: por chunk_id (estable).
    """
    meta, bm25, chunk_ids = _corpus_state()
    tokens = tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    # Ranking global por (-score, chunk_id) para tie-break determinista
    order = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), chunk_ids[i]),
    )

    results = []
    for idx in order:
        row = meta.iloc[idx]
        if ticker and row["ticker"] != ticker:
            continue
        if fiscal_year and int(row["fiscal_year"]) != int(fiscal_year):
            continue
        if item and str(row["item"]) != str(item):
            continue
        results.append({
            "chunk_id": chunk_ids[idx],
            "bm25_score": float(scores[idx]),
            "rank": len(results) + 1,
        })
        if len(results) >= k:
            break
    return results


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------
def rrf_fuse(dense_results: list[dict],
             bm25_results: list[dict],
             k_rrf: int = RRF_K,
             k: int = DEFAULT_K_CANDIDATES) -> list[dict]:
    """RRF con rangos 1-based.

    Tie-break determinista:
      1. mayor RRF
      2. mejor dense_rank (menor, None = inf)
      3. mejor bm25_rank (menor, None = inf)
      4. chunk_id (lexicográfico)
    """
    INF = float("inf")
    dense_rank = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_results)}
    bm25_rank = {r["chunk_id"]: i + 1 for i, r in enumerate(bm25_results)}

    all_ids = set(dense_rank) | set(bm25_rank)
    rrf_scores: dict[str, float] = {}
    for cid in all_ids:
        s = 0.0
        rd = dense_rank.get(cid)
        rb = bm25_rank.get(cid)
        if rd is not None:
            s += 1.0 / (k_rrf + rd)
        if rb is not None:
            s += 1.0 / (k_rrf + rb)
        rrf_scores[cid] = s

    ranked = sorted(
        all_ids,
        key=lambda cid: (
            -rrf_scores[cid],
            dense_rank.get(cid, INF),
            bm25_rank.get(cid, INF),
            cid,
        ),
    )

    return [
        {
            "chunk_id": cid,
            "rrf_score": rrf_scores[cid],
            "dense_rank": dense_rank.get(cid),
            "bm25_rank": bm25_rank.get(cid),
        }
        for cid in ranked[:k]
    ]