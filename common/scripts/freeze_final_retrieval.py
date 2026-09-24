"""Cierra la ablacion de rewriting y congela retrieval sin llamadas de red."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from common.benchmark import BENCHMARK_PATH, FROZEN_SHA256
from common.scripts.evaluate_query_rewrite_ablation import paired_analysis
from common.scripts.evaluate_retrieval_benchmark import evaluate_benchmark_views
from common.scripts.generate_query_rewrites import (
    DEFAULT_BENCHMARK_PATH,
    PROMPT_SHA256,
    load_cache,
    load_frozen_benchmark,
    validate_complete_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_G0 = REPO_ROOT / "common/results/retrieval/gemini_dense_original_48.json"
DEFAULT_G1 = REPO_ROOT / "common/results/query_rewriting/gemini_rewritten_48.json"
DEFAULT_BGE = (
    REPO_ROOT
    / "common/results/query_rewriting/bge_original_vs_rewritten_benchmark_v2.json"
)
DEFAULT_CACHE = (
    REPO_ROOT / "common/results/query_rewriting/query_rewrites_48.jsonl"
)
DEFAULT_COMPARISON = (
    REPO_ROOT
    / "common/results/query_rewriting/gemini_original_vs_rewritten_benchmark_v2.json"
)
DEFAULT_FINAL = REPO_ROOT / "common/results/retrieval_final/retrieval_final_48.json"
GEMINI_MODEL = "google/gemini-embedding-2"
GEMINI_DIMENSION = 3072
SELECTED_PIPELINE = (
    "query original -> Gemini Embedding 2 -> FAISS global rank -> "
    "ticker/fiscal_year/item metadata postfilter"
)


class FinalRetrievalError(RuntimeError):
    """Un artefacto no satisface el contrato congelado."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalRetrievalError(f"JSON ausente o invalido: {path}") from exc
    if not isinstance(value, dict):
        raise FinalRetrievalError(f"JSON raiz no es objeto: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" evita que Windows traduzca a CRLF: los artefactos
    # congelados son LF y deben regenerarse byte a byte en cualquier SO.
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_summary(rows: list[dict[str, Any]], variant: str | None) -> dict[str, float]:
    values = [
        float((row if variant is None else row[variant])["latency_s"]["total"])
        for row in rows
    ]
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "sum": sum(values),
    }


def _first_rank(row: dict[str, Any], block: dict[str, Any]) -> int | None:
    relevant = set(row["relevant_chunk_ids"])
    return next(
        (
            int(hit["rank"])
            for hit in block["ranking"]
            if hit["chunk_id"] in relevant
        ),
        None,
    )


def _validate_ranking(
    question: dict[str, Any], row: dict[str, Any], block: dict[str, Any]
) -> None:
    if _first_rank(row, block) != block.get("first_relevant_rank"):
        raise FinalRetrievalError(
            f"{question['question_id']}: first_relevant_rank inconsistente"
        )
    ranks = [int(hit["rank"]) for hit in block["ranking"]]
    if ranks != list(range(1, len(ranks) + 1)) or len(ranks) > 10:
        raise FinalRetrievalError(f"{question['question_id']}: ranking invalido")
    for hit in block["ranking"]:
        if (
            str(hit["ticker"]) != str(question["ticker"])
            or int(hit["fiscal_year"]) != int(question["fiscal_year"])
            or str(hit["item"]) != str(question["item"])
        ):
            raise FinalRetrievalError(
                f"{question['question_id']}: postfilter de metadata invalido"
            )


def _metric_rows(
    rows: list[dict[str, Any]], variant: str | None
) -> list[dict[str, Any]]:
    return [
        {
            "question_id": row["question_id"],
            "item": row["item"],
            "relevant_chunk_ids": row["relevant_chunk_ids"],
            "ranking": (row if variant is None else row[variant])["ranking"],
        }
        for row in rows
    ]


def _metrics(rows: list[dict[str, Any]], variant: str | None) -> dict[str, Any]:
    return evaluate_benchmark_views(_metric_rows(rows, variant))


def _assert_metrics(
    stored: dict[str, Any], calculated: dict[str, Any], label: str
) -> None:
    if stored.keys() != calculated.keys():
        raise FinalRetrievalError(f"{label}: vistas de metricas incompatibles")
    for view in calculated:
        for key, expected in calculated[view].items():
            actual = stored[view].get(key)
            if isinstance(expected, float):
                if actual is None or abs(float(actual) - expected) > 1e-12:
                    raise FinalRetrievalError(f"{label}: metrica {view}/{key} invalida")
            elif actual != expected:
                raise FinalRetrievalError(f"{label}: metrica {view}/{key} invalida")


def _validate_common_manifest(manifest: dict[str, Any], label: str) -> None:
    expected = {
        "benchmark_sha256": FROZEN_SHA256,
        "n_questions": 48,
        "profile": "gemini-embedding-2",
        "embedding_model_id": GEMINI_MODEL,
        "embedding_dimension": GEMINI_DIMENSION,
        "max_k": 10,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise FinalRetrievalError(
                f"{label}: manifest {key}={manifest.get(key)!r}, esperado {value!r}"
            )
    retrieval = manifest.get("retrieval") or {}
    if retrieval.get("policy") != "global rank + ticker/fiscal_year/item postfilter":
        raise FinalRetrievalError(f"{label}: politica de retrieval invalida")
    if retrieval.get("bm25") is not False or retrieval.get("reranking") is not False:
        raise FinalRetrievalError(f"{label}: BM25/reranking no permitido")


def _validate_provenance(block: dict[str, Any], question_id: str) -> None:
    provenance = block.get("query_embedding_provenance") or {}
    expected = {
        "provider": "openrouter",
        "model_name": GEMINI_MODEL,
        "input_type_requested": "search_query",
        "input_type_mode": "requested",
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise FinalRetrievalError(
                f"{question_id}: provenance {key} incompatible"
            )


def _bundle_provenance() -> dict[str, Any]:
    from common.config import get_dataset_paths

    paths = get_dataset_paths("gemini-embedding-2")
    if paths.index_manifest is None:
        raise FinalRetrievalError("Gemini no tiene index_manifest.json")
    manifest = _load_json(paths.index_manifest)
    actual = {
        "faiss_sha256": _sha256(paths.faiss_index),
        "chunks_sha256": _sha256(paths.chunks),
        "chunks_meta_sha256": _sha256(paths.chunks_meta),
    }
    for key, value in actual.items():
        if value != str(manifest.get(key, "")).upper():
            raise FinalRetrievalError(f"Bundle Gemini incompatible en {key}")
    try:
        import faiss

        index = faiss.read_index(str(paths.faiss_index))
    except (ImportError, RuntimeError) as exc:
        raise FinalRetrievalError("No se puede validar el indice FAISS") from exc
    if (
        type(index).__name__ != manifest.get("index_type")
        or int(index.d) != GEMINI_DIMENSION
        or int(index.ntotal) != int(manifest.get("ntotal", -1))
    ):
        raise FinalRetrievalError("FAISS no coincide con el manifest Gemini")
    return {
        **actual,
        "index_type": type(index).__name__,
        "dimension": int(index.d),
        "ntotal": int(index.ntotal),
        "model": str(manifest["model_name"]),
    }


def _normalize_g1(g1_path: Path) -> dict[str, Any]:
    g1 = _load_json(g1_path)
    manifest = g1.get("manifest") or {}
    retrieval = manifest.get("retrieval") or {}
    if manifest.get("mode") != "rewritten" or manifest.get("variants") != ["rewritten"]:
        raise FinalRetrievalError("G1 no es una evaluacion rewritten-only")
    retrieval["query_rewriting"] = True
    manifest["retrieval"] = retrieval
    g1["manifest"] = manifest
    _write_json(g1_path, g1)
    return g1


def _validate_experiments(
    *,
    g0: dict[str, Any],
    g1: dict[str, Any],
    bge: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    questions: list[dict[str, Any]],
    cache_sha256: str,
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_ids = [str(question["question_id"]) for question in questions]
    g0_rows = g0.get("per_question") or []
    g1_rows = g1.get("per_question") or []
    if [str(row.get("question_id")) for row in g0_rows] != expected_ids:
        raise FinalRetrievalError("G0 no contiene los 48 IDs congelados en orden")
    if [str(row.get("question_id")) for row in g1_rows] != expected_ids:
        raise FinalRetrievalError("G1 no contiene los 48 IDs congelados en orden")

    _validate_common_manifest(g0.get("manifest") or {}, "G0")
    _validate_common_manifest(g1.get("manifest") or {}, "G1")
    if (g0["manifest"]["retrieval"] or {}).get("query_rewriting") is not False:
        raise FinalRetrievalError("G0 debe registrar query_rewriting=false")
    if (g1["manifest"]["retrieval"] or {}).get("query_rewriting") is not True:
        raise FinalRetrievalError("G1 debe registrar query_rewriting=true")
    if g1["manifest"].get("rewrite_generation") is not False:
        raise FinalRetrievalError("G1 no debe generar rewrites")
    if g1["manifest"].get("rewrite_cache_sha256") != cache_sha256:
        raise FinalRetrievalError("G1 no usa la cache canonica")
    if g1["manifest"].get("rewrite_prompt_sha256") != PROMPT_SHA256:
        raise FinalRetrievalError("G1 no usa el prompt congelado")

    for manifest in (g0["manifest"], g1["manifest"]):
        if manifest.get("faiss_index_sha256") != bundle["faiss_sha256"]:
            raise FinalRetrievalError("FAISS de experimento no coincide con bundle")
        if manifest.get("chunks_meta_sha256") != bundle["chunks_meta_sha256"]:
            raise FinalRetrievalError("Metadata de experimento no coincide con bundle")
    if g1["manifest"].get("chunks_sha256") != bundle["chunks_sha256"]:
        raise FinalRetrievalError("Chunks de G1 no coinciden con bundle")

    for question, original, rewritten in zip(questions, g0_rows, g1_rows):
        question_id = str(question["question_id"])
        rewritten_block = rewritten.get("rewritten") or {}
        expected_common = {
            "question_id": question_id,
            "ticker": str(question["ticker"]),
            "fiscal_year": int(question["fiscal_year"]),
            "item": str(question["item"]),
        }
        for key, value in expected_common.items():
            if original.get(key) != value or rewritten.get(key) != value:
                raise FinalRetrievalError(f"{question_id}: metadata de pregunta invalida")
        if original.get("query") != question["question"]:
            raise FinalRetrievalError(f"{question_id}: query G0 no es la original")
        if rewritten_block.get("query") != cache[question_id]["query_rewritten"]:
            raise FinalRetrievalError(f"{question_id}: G1 no usa rewrite canonica")
        if rewritten.get("fallback_used") is not False:
            raise FinalRetrievalError(f"{question_id}: fallback inesperado en G1")
        if original.get("relevant_chunk_ids") != rewritten.get("relevant_chunk_ids"):
            raise FinalRetrievalError(f"{question_id}: relevancia G0/G1 distinta")
        _validate_ranking(question, original, original)
        _validate_ranking(question, rewritten, rewritten_block)
        _validate_provenance(original, question_id)
        _validate_provenance(rewritten_block, question_id)

    g0_metrics = _metrics(g0_rows, None)
    g1_metrics = _metrics(g1_rows, "rewritten")
    _assert_metrics(g0.get("metrics") or {}, g0_metrics, "G0")
    _assert_metrics((g1.get("metrics") or {}).get("rewritten") or {}, g1_metrics, "G1")

    bge_rows = bge.get("per_question") or []
    if [str(row.get("question_id")) for row in bge_rows] != expected_ids:
        raise FinalRetrievalError("BGE no contiene los 48 IDs congelados")
    if bge.get("manifest", {}).get("rewrite_cache_sha256") != cache_sha256:
        raise FinalRetrievalError("BGE no usa la cache canonica")
    bge_metrics = {
        variant: _metrics(bge_rows, variant)
        for variant in ("original", "rewritten")
    }
    for variant in bge_metrics:
        _assert_metrics(
            (bge.get("metrics") or {}).get(variant) or {},
            bge_metrics[variant],
            f"BGE {variant}",
        )
    return g0_metrics, g1_metrics, bge_metrics


def _rewrite_summary(cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(cache.values())
    latencies = [float(row["latency_s"]) for row in rows]
    usages = [row.get("usage") or {} for row in rows]
    return {
        "model": "deepseek/deepseek-v4-flash",
        "requested_model": "openrouter:deepseek/deepseek-v4-flash",
        "n_queries": len(rows),
        "fallbacks": sum(bool(row["fallback_used"]) for row in rows),
        "input_tokens": sum(int(usage.get("prompt_tokens", 0)) for usage in usages),
        "output_tokens": sum(int(usage.get("completion_tokens", 0)) for usage in usages),
        "total_tokens": sum(int(usage.get("total_tokens", 0)) for usage in usages),
        "cost_total": sum(float(usage.get("cost", 0.0)) for usage in usages),
        "cost_mean": sum(float(usage.get("cost", 0.0)) for usage in usages) / len(rows),
        "cost_unit": "not_declared_in_source_artifact",
        "latency_s": {
            "mean": sum(latencies) / len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "sum": sum(latencies),
        },
    }


def _full_table(
    g0: dict[str, Any], g1: dict[str, Any], bge: dict[str, Any]
) -> dict[str, Any]:
    sources = {
        "bge_original": bge["original"],
        "bge_rewritten": bge["rewritten"],
        "gemini_original": g0,
        "gemini_rewritten": g1,
    }
    return {
        name: {
            "all_recall@1": metrics["ALL-48"]["recall@1"],
            "all_recall@5": metrics["ALL-48"]["recall@5"],
            "all_mrr@10": metrics["ALL-48"]["mrr@10"],
            "non_7a_recall@5": metrics["NON-7A-36"]["recall@5"],
            "non_7a_mrr@10": metrics["NON-7A-36"]["mrr@10"],
        }
        for name, metrics in sources.items()
    }


def freeze_final_retrieval(
    *,
    g0_path: Path = DEFAULT_G0,
    g1_path: Path = DEFAULT_G1,
    bge_path: Path = DEFAULT_BGE,
    cache_path: Path = DEFAULT_CACHE,
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    comparison_path: Path = DEFAULT_COMPARISON,
    final_path: Path = DEFAULT_FINAL,
    bundle: dict[str, Any] | None = None,
    normalize_g1: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if benchmark_path.resolve() != BENCHMARK_PATH.resolve():
        raise FinalRetrievalError("Debe usarse el benchmark v2 congelado")
    questions = load_frozen_benchmark(benchmark_path)
    cache = load_cache(cache_path, questions)
    validate_complete_cache(cache, questions)
    cache_sha256 = _sha256(cache_path)
    g0 = _load_json(g0_path)
    g1 = _normalize_g1(g1_path) if normalize_g1 else _load_json(g1_path)
    bge = _load_json(bge_path)
    actual_bundle = bundle or _bundle_provenance()
    g0_metrics, g1_metrics, bge_metrics = _validate_experiments(
        g0=g0,
        g1=g1,
        bge=bge,
        cache=cache,
        questions=questions,
        cache_sha256=cache_sha256,
        bundle=actual_bundle,
    )

    paired_rows = []
    for original, rewritten in zip(g0["per_question"], g1["per_question"]):
        paired_rows.append({
            "question_id": original["question_id"],
            "original": {
                "first_relevant_rank": original["first_relevant_rank"]
            },
            "rewritten": {
                "first_relevant_rank": rewritten["rewritten"]["first_relevant_rank"]
            },
        })
    paired = paired_analysis(paired_rows)
    primary_g0 = g0_metrics["NON-7A-36"]["recall@5"]
    primary_g1 = g1_metrics["NON-7A-36"]["recall@5"]
    if primary_g1 >= primary_g0:
        raise FinalRetrievalError(
            "El cierre implementa la decision observada G1 < G0 en NON-7A R@5"
        )
    decision = {
        "selected_candidate": "A",
        "selected_pipeline": SELECTED_PIPELINE,
        "criterion_applied": [
            "NON-7A R@5",
            "NON-7A MRR@10",
            "ALL R@5",
            "per-item",
            "cost/latency",
        ],
        "decided_at_criterion": "NON-7A R@5",
        "quantitative_reason": {
            "gemini_original": primary_g0,
            "deepseek_rewrite_plus_gemini": primary_g1,
            "rewritten_minus_original": primary_g1 - primary_g0,
        },
    }
    rewrite_summary = _rewrite_summary(cache)
    comparison = {
        "schema_version": 1,
        "manifest": {
            "benchmark_sha256": FROZEN_SHA256,
            "rewrite_cache_sha256": cache_sha256,
            "rewrite_prompt_sha256": PROMPT_SHA256,
            "g0_source_sha256": _sha256(g0_path),
            "g1_source_sha256": _sha256(g1_path),
            "bge_source_sha256": _sha256(bge_path),
            "metrics_recalculated_from_per_question": True,
            "api_calls_required": False,
        },
        "metrics": {
            "gemini_original": g0_metrics,
            "gemini_rewritten": g1_metrics,
        },
        "full_rewriting_table": _full_table(g0_metrics, g1_metrics, bge_metrics),
        "paired_analysis": paired,
        "rewrite_generation": rewrite_summary,
        "decision": decision,
        "scientific_interpretation": {
            "observation": (
                "Rewriting improves BGE strongly but lowers Gemini on the frozen "
                "primary metric."
            ),
            "hypothesis_not_demonstrated_cause": (
                "The Spanish benchmark queries and weaker cross-lingual behavior of "
                "BGE-small may make concise English rewrites more helpful, while "
                "Gemini may handle the original queries better."
            ),
        },
    }

    g0_latency = _latency_summary(g0["per_question"], None)
    input_modes: dict[str, int] = {}
    for row in g0["per_question"]:
        mode = row["query_embedding_provenance"]["input_type_mode"]
        input_modes[mode] = input_modes.get(mode, 0) + 1
    final = {
        "schema_version": 1,
        "manifest": {
            "selected_pipeline": SELECTED_PIPELINE,
            "selected_candidate": "A",
            "benchmark_sha256": FROZEN_SHA256,
            "source_artifact": "common/results/retrieval/gemini_dense_original_48.json",
            "source_artifact_sha256": _sha256(g0_path),
            "chunks_sha256": actual_bundle["chunks_sha256"],
            "chunks_meta_sha256": actual_bundle["chunks_meta_sha256"],
            "faiss_sha256": actual_bundle["faiss_sha256"],
            "faiss_index_type": actual_bundle["index_type"],
            "gemini_model": actual_bundle["model"],
            "embedding_dimension": actual_bundle["dimension"],
            "ntotal": actual_bundle["ntotal"],
            "input_type_provenance": {
                "requested": "search_query",
                "mode_counts": input_modes,
            },
            "retrieval": {
                "type": "dense_faiss_with_metadata_postfilter",
                "policy": "global rank + ticker/fiscal_year/item postfilter",
                "query_rewriting": False,
                "bm25": False,
                "reranking": False,
            },
            "metrics_recalculated_from_per_question": True,
            "api_calls_required_to_regenerate": False,
            "selection": decision,
        },
        "metrics": g0_metrics,
        "latency_s": g0_latency,
        "per_question": g0["per_question"],
    }
    _write_json(comparison_path, comparison)
    _write_json(final_path, final)
    return comparison, final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Congela comparacion G0/G1 y retrieval final sin usar APIs"
    )
    parser.add_argument("--g0", type=Path, default=DEFAULT_G0)
    parser.add_argument("--g1", type=Path, default=DEFAULT_G1)
    parser.add_argument("--bge", type=Path, default=DEFAULT_BGE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--final-output", type=Path, default=DEFAULT_FINAL)
    args = parser.parse_args()
    comparison, final = freeze_final_retrieval(
        g0_path=args.g0.resolve(),
        g1_path=args.g1.resolve(),
        bge_path=args.bge.resolve(),
        cache_path=args.cache.resolve(),
        comparison_path=args.comparison_output.resolve(),
        final_path=args.final_output.resolve(),
    )
    print(
        "Selected: " + final["manifest"]["selected_pipeline"],
        flush=True,
    )
    print(f"Comparison: {args.comparison_output.resolve()}", flush=True)
    print(f"Final: {args.final_output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
