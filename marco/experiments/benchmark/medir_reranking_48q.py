"""Mide recall@k de dense_baseline vs reranked_baseline sobre las 48 preguntas.

Solo retrieval. Sin LLM. Coste: $0.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

# Añadir el repo al path
RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))

from common.retrieval.dense_baseline import buscar as buscar_denso
from common.retrieval.reranked_baseline import buscar as buscar_reranked
from common.eval.retrieval_metrics import evaluate_rankings


def _norm(texto: str) -> str:
    if not texto:
        return ""
    t = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    ).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def chunk_contiene_evidencia(texto_chunk: str, evidencia: str) -> bool:
    """Comprueba si el evidence_text está contenido en el chunk."""
    return _norm(evidencia) in _norm(texto_chunk)


def medir(benchmark_path: Path, output_path: Path | None = None):
    import pandas as pd
    from common.config import get_dataset_paths

    with benchmark_path.open(encoding="utf-8") as f:
        preguntas = [json.loads(l) for l in f if l.strip()]

    metadata = pd.read_parquet(get_dataset_paths().chunks_meta)

    rankings_densa = []
    rankings_reranked = []

    for i, q in enumerate(preguntas, 1):
        if i % 10 == 0:
            print(f"  Procesando {i}/{len(preguntas)}...")

        subset = metadata[
            (metadata["ticker"].astype(str).str.upper() == q["ticker"].upper())
            & (metadata["fiscal_year"].astype(int) == int(q["fiscal_year"]))
            & (metadata["item"].astype(str) == str(q["item"]))
        ]
        relevantes = []
        for _, row in subset.iterrows():
            if chunk_contiene_evidencia(str(row["texto"]), q["evidence_text"]):
                relevantes.append(str(row["chunk_id"]))

        if not relevantes:
            print(f"  ⚠️ {q['question_id']}: evidencia no encontrada, saltada")
            continue

        densa = buscar_denso(
            q["question"], ticker=q["ticker"],
            fiscal_year=q["fiscal_year"], item=q["item"], k=10,
        )
        rankings_densa.append({
            "question_id": q["question_id"],
            "relevant_chunk_ids": relevantes,
            "ranking": [{"rank": j, "chunk_id": c["chunk_id"]}
                        for j, c in enumerate(densa, 1)],
        })

        reranked = buscar_reranked(
            q["question"], ticker=q["ticker"],
            fiscal_year=q["fiscal_year"], item=q["item"], k=10,
        )
        rankings_reranked.append({
            "question_id": q["question_id"],
            "relevant_chunk_ids": relevantes,
            "ranking": [{"rank": j, "chunk_id": c["chunk_id"]}
                        for j, c in enumerate(reranked, 1)],
        })

    print("\n=== DENSE BASELINE ===")
    metricas_densa = evaluate_rankings(rankings_densa)
    print(json.dumps(metricas_densa, indent=2))

    print("\n=== RERANKED ===")
    metricas_reranked = evaluate_rankings(rankings_reranked)
    print(json.dumps(metricas_reranked, indent=2))

    result = {
        "dense": metricas_densa,
        "reranked": metricas_reranked,
        "n_questions": len(rankings_densa),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nGuardado en {output_path}")

    return result


if __name__ == "__main__":
    benchmark = Path(__file__).resolve().parents[3] / "dani" / "experiments" / "benchmark" / "retrieval_benchmark_v2.jsonl"
    output = Path(__file__).resolve().parents[2] / "results" / "final" / "reranking_48q.json"
    medir(benchmark, output)