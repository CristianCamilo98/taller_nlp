"""Mide retrieval con reranking. Sin LLM."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from common.config import get_dataset_paths
from common.eval.retrieval_metrics import evaluate_rankings


def _normalize(text: str) -> str:
    value = "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    ).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value)).strip()


def _parts(text: str) -> list[str]:
    return [_normalize(part)
            for part in re.split(r"\s*(?:\.\.\.|…)+\s*", text)
            if _normalize(part)]


def _relevant_chunks(question: dict, metadata) -> list[str]:
    anchors = [question.get("ancla_texto"),
               *(question.get("anclas_alternativas") or [])]
    anchors = [[part for part in _parts(anchor)]
               for anchor in anchors if anchor]
    subset = metadata[
        (metadata["ticker"].astype(str).str.upper()
         == str(question["ticker"]).upper())
        & (metadata["fiscal_year"].astype(int) == int(question["fiscal_year"]))
        & (metadata["item"].astype(str)
           == str(question.get("item", question.get("item_esperado"))))
    ]
    relevant = []
    for _, row in subset.iterrows():
        text = _normalize(str(row["texto"]))
        if any(all(part in text for part in anchor_parts)
               for anchor_parts in anchors):
            relevant.append(str(row["chunk_id"]))
    return relevant


def measure(golden_path: Path, output_path: Path | None = None) -> dict:
    import pandas as pd
    from common.retrieval.reranked_baseline import buscar

    with golden_path.open(encoding="utf-8") as stream:
        questions = [json.loads(line) for line in stream if line.strip()]
    metadata = pd.read_parquet(get_dataset_paths().chunks_meta)

    rankings = []
    for question in questions:
        if not question.get("ancla_texto"):
            continue
        relevant = _relevant_chunks(question, metadata)
        retrieved = buscar(
            question["pregunta"],
            ticker=question["ticker"],
            fiscal_year=question["fiscal_year"],
            item=question.get("item", question.get("item_esperado")),
            k=10,
        )
        rankings.append({
            "question_id": question["id"],
            "query": question["pregunta"],
            "relevant_chunk_ids": relevant,
            "ranking": [{"rank": rank, "chunk_id": item["chunk_id"],
                         "score": item["puntuacion"]}
                        for rank, item in enumerate(retrieved, 1)],
        })

    summary = evaluate_rankings(rankings)
    result = {"config": {"reranking": True}, "metrics": summary,
              "rankings": rankings}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "golden_set" / "golden_set_grupo3.jsonl",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(args.golden.resolve(),
                     args.output.resolve() if args.output else None)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()