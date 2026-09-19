"""Métricas deterministas de rankings: Recall@k y MRR@10."""

from __future__ import annotations


def evaluate_rankings(rankings: list[dict], ks=(1, 3, 5, 10)) -> dict:
    """Calcula macro Recall@k y MRR@10 sobre rankings de hasta 10 puestos."""
    applicable = [row for row in rankings if row.get("relevant_chunk_ids")]
    if not applicable:
        return {**{f"recall@{k}": None for k in ks}, "mrr@10": None,
                "n_questions": 0}
    output = {}
    for k in ks:
        hits = 0
        for row in applicable:
            relevant = set(row["relevant_chunk_ids"])
            retrieved = [item["chunk_id"] for item in row["ranking"][:k]]
            hits += bool(relevant.intersection(retrieved))
        output[f"recall@{k}"] = hits / len(applicable)
    reciprocal = []
    for row in applicable:
        relevant = set(row["relevant_chunk_ids"])
        rank = next((index for index, item in enumerate(row["ranking"], 1)
                     if item["chunk_id"] in relevant), None)
        reciprocal.append(0.0 if rank is None else 1.0 / rank)
    output["mrr@10"] = sum(reciprocal) / len(reciprocal)
    output["n_questions"] = len(applicable)
    return output
