"""Auditoría de truncamiento de MiniLM por pregunta.

Para cada pregunta:
1. Tokeniza el chunk citado (top-1 del denso) con el tokenizer de MiniLM.
2. Comprueba si supera 512 tokens.
3. Comprueba si el EvidenceSpan queda dentro de los primeros 512 tokens.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))

import pandas as pd

from common.config import get_dataset_paths


TOKENIZER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_TOKENS = 512
RUNNER_OUTPUT = Path(__file__).resolve().parents[2] / "results" / "final" / "reranking_canonical.json"


def main() -> None:
    from transformers import AutoTokenizer

    runner = json.loads(RUNNER_OUTPUT.read_text(encoding="utf-8"))
    meta = pd.read_parquet(get_dataset_paths().chunks_meta)
    meta_idx = meta.set_index("chunk_id")

    print(f"Cargando tokenizer {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    results = []
    n_over_512 = 0
    n_evidence_cut = 0

    for row in runner["per_question"]:
        qid = row["question_id"]
        dense_top1 = row["dense_top20"][0]["chunk_id"] if row["dense_top20"] else None
        if dense_top1 is None:
            continue

        chunk = meta_idx.loc[dense_top1]
        chunk_text = str(chunk["texto"])
        inicio = int(chunk["inicio_car"])
        fin = int(chunk["fin_car"])

        tokens = tokenizer.encode(chunk_text, truncation=False, add_special_tokens=False)
        n_tokens = len(tokens)
        over = n_tokens > MAX_TOKENS
        if over:
            n_over_512 += 1

        # ¿La evidencia cae dentro de los primeros 512 tokens?
        # Aproximación: decodificar el prefijo de 512 tokens y comprobar
        # si el EvidenceSpan del corpus aparece en ese prefijo.
        if over:
            prefix_tokens = tokens[:MAX_TOKENS]
            prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
            # Mapeo aproximado: el span relativo al chunk empieza en
            # char_start - inicio
            span_rel_start = row["char_start"] - inicio
            span_rel_end = row["char_end"] - inicio
            if span_rel_start < 0 or span_rel_end > len(chunk_text):
                evidence_in_prefix = False
            else:
                evidence_text = chunk_text[span_rel_start:span_rel_end]
                evidence_in_prefix = evidence_text.strip() in prefix_text
            if not evidence_in_prefix:
                n_evidence_cut += 1
        else:
            evidence_in_prefix = True

        results.append({
            "question_id": qid,
            "dense_top1_chunk_id": dense_top1,
            "chunk_n_tokens": n_tokens,
            "over_512": over,
            "evidence_in_first_512": evidence_in_prefix,
            "dense_rank": row["dense_first_relevant_rank"],
            "reranked_rank": row["reranked_first_relevant_rank"],
            "improved": (
                row["reranked_first_relevant_rank"] is not None
                and (row["dense_first_relevant_rank"] is None
                     or row["reranked_first_relevant_rank"] < row["dense_first_relevant_rank"])
            ),
            "worsened": (
                row["reranked_first_relevant_rank"] is None
                or (row["dense_first_relevant_rank"] is not None
                    and row["reranked_first_relevant_rank"] > row["dense_first_relevant_rank"])
            ),
        })

    output = {
        "summary": {
            "n_questions": len(results),
            "n_over_512": n_over_512,
            "pct_over_512": round(100.0 * n_over_512 / len(results), 2),
            "n_evidence_cut": n_evidence_cut,
            "pct_evidence_cut": round(100.0 * n_evidence_cut / len(results), 2),
        },
        "per_question": results,
    }

    out_path = RUNNER_OUTPUT.parent / "truncation_audit.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], indent=2))
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()