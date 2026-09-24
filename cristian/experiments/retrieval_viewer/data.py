"""Carga y normalización de resultados de retrieval (benchmark v2).

Soporta formatos:
- Dani / Cristian: ``{manifest, metrics: {VIEW: ...}, per_question}``
- Marco BM25: ``metrics: {B0_dense|B1_hybrid: {VIEW: ...}}``
- Marco reranking: ``metrics: {dense|reranked: {VIEW: ...}}`` (o plano ALL-48)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

COMPARISONS_DIR = Path(__file__).resolve().parents[1] / "results" / "comparisons"

VIEWS = (
    "ALL-48",
    "NON-7A-36",
    "ITEM-1A-12",
    "ITEM-7-12",
    "ITEM-7A-12",
    "ITEM-8-12",
)

METRIC_KEYS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10")


@dataclass
class RetrievalRun:
    uid: str
    label: str
    source_file: str
    author: str
    model: str | None
    variant: str | None
    n_questions: int | None
    metrics: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    primary_r5: float | None = None  # NON-7A-36 Recall@5
    primary_mrr: float | None = None
    all48_r5: float | None = None
    ranks: dict[str, int | None] = field(default_factory=dict)  # qid -> first_relevant_rank
    questions: dict[str, dict[str, str]] = field(default_factory=dict)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_view_block(block: dict[str, Any] | None) -> dict[str, float | int | None]:
    block = block or {}
    out: dict[str, float | int | None] = {}
    for key in METRIC_KEYS:
        out[key] = _safe_float(block.get(key))
    n = block.get("n_questions")
    out["n_questions"] = int(n) if n is not None else None
    return out


def _extract_views(metrics_obj: dict[str, Any]) -> dict[str, dict[str, float | int | None]]:
    """Acepta ``{VIEW: metrics}`` o un bloque plano (solo ALL-48)."""
    if not metrics_obj:
        return {}
    if any(view in metrics_obj for view in VIEWS):
        return {
            view: _normalize_view_block(metrics_obj.get(view))
            for view in VIEWS
            if view in metrics_obj
        }
    # Plano: recall@k en la raíz → ALL-48
    if any(k in metrics_obj for k in METRIC_KEYS):
        return {"ALL-48": _normalize_view_block(metrics_obj)}
    return {}


def _author_from_path(path: Path) -> str:
    parts = path.parts
    if "comparisons" in parts:
        idx = parts.index("comparisons")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def _model_from_dani(payload: dict[str, Any]) -> str | None:
    man = payload.get("manifest") or {}
    if man.get("embedding_model"):
        return str(man["embedding_model"])
    emb = man.get("embedding") or {}
    if isinstance(emb, dict) and emb.get("model_name"):
        return str(emb["model_name"])
    identity = man.get("identity") or {}
    if identity.get("experiment_id"):
        return str(identity["experiment_id"])
    return None


def _label(author: str, name: str, variant: str | None = None) -> str:
    base = f"{author} · {name}"
    return f"{base} [{variant}]" if variant else base


def _ranks_from_per_question(
    rows: list[dict[str, Any]],
    rank_key: str,
) -> tuple[dict[str, int | None], dict[str, dict[str, str]]]:
    ranks: dict[str, int | None] = {}
    questions: dict[str, dict[str, str]] = {}
    for row in rows:
        qid = str(row.get("question_id") or row.get("id") or "")
        if not qid:
            continue
        raw = row.get(rank_key)
        if raw is None and rank_key == "first_relevant_rank":
            raw = row.get("first_relevant_rank")
        try:
            ranks[qid] = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            ranks[qid] = None
        questions[qid] = {
            "id": qid,
            "familia": "",
            "item": str(row.get("item") or ""),
            "pregunta": str(row.get("query") or row.get("question") or ""),
        }
    return ranks, questions


def _run_from_views(
    *,
    path: Path,
    author: str,
    name: str,
    variant: str | None,
    model: str | None,
    views: dict[str, dict[str, float | int | None]],
    ranks: dict[str, int | None],
    questions: dict[str, dict[str, str]],
) -> RetrievalRun:
    non7 = views.get("NON-7A-36") or {}
    all48 = views.get("ALL-48") or {}
    uid = f"{author}::{path.name}::{variant or 'default'}"
    n = all48.get("n_questions") or non7.get("n_questions")
    return RetrievalRun(
        uid=uid,
        label=_label(author, name, variant),
        source_file=str(path.relative_to(COMPARISONS_DIR))
        if path.is_relative_to(COMPARISONS_DIR)
        else path.name,
        author=author,
        model=model,
        variant=variant,
        n_questions=int(n) if n is not None else None,
        metrics=views,
        primary_r5=_safe_float(non7.get("recall@5")),
        primary_mrr=_safe_float(non7.get("mrr@10")),
        all48_r5=_safe_float(all48.get("recall@5")),
        ranks=ranks,
        questions=questions,
    )


def parse_retrieval_json(path: Path) -> list[RetrievalRun]:
    """Devuelve uno o varios runs normalizados desde un JSON de resultados."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    author = _author_from_path(path)
    metrics_root = payload.get("metrics") or {}
    runs: list[RetrievalRun] = []

    # Caso Dani / Cristian: metrics[VIEW]
    if any(view in metrics_root for view in VIEWS):
        man = payload.get("manifest") or {}
        name = (
            (man.get("identity") or {}).get("experiment_id")
            or man.get("embedding_model_id")
            or man.get("experiment")
            or path.stem
        )
        ranks, questions = _ranks_from_per_question(
            payload.get("per_question") or [],
            "first_relevant_rank",
        )
        runs.append(
            _run_from_views(
                path=path,
                author=author,
                name=str(name),
                variant=None,
                model=_model_from_dani(payload),
                views=_extract_views(metrics_root),
                ranks=ranks,
                questions=questions,
            )
        )
        return runs

    # Caso Marco: metrics[variant][VIEW] o metrics[variant] plano
    nested_variants = [
        key
        for key, value in metrics_root.items()
        if isinstance(value, dict)
        and (
            any(view in value for view in VIEWS)
            or any(m in value for m in METRIC_KEYS)
        )
    ]
    if nested_variants:
        cfg = payload.get("config") or {}
        model = (
            payload.get("dense_model")
            or cfg.get("embedding_model")
            or cfg.get("reranker_model")
            or payload.get("experiment_id")
            or path.stem
        )
        base_name = str(
            payload.get("experiment_id")
            or cfg.get("experiment_id")
            or path.stem
        )
        for variant in nested_variants:
            block = metrics_root[variant]
            views = _extract_views(block)
            rank_key = {
                "B0_dense": "dense_first_relevant_rank",
                "dense": "dense_first_relevant_rank",
                "B1_hybrid": "hybrid_first_relevant_rank",
                "reranked": "reranked_first_relevant_rank",
            }.get(variant, "first_relevant_rank")
            ranks, questions = _ranks_from_per_question(
                payload.get("per_question") or [],
                rank_key,
            )
            runs.append(
                _run_from_views(
                    path=path,
                    author=author,
                    name=base_name,
                    variant=variant,
                    model=str(model),
                    views=views,
                    ranks=ranks,
                    questions=questions,
                )
            )
        return runs

    raise ValueError(f"Formato de métricas no reconocido: {path}")


def list_comparison_jsons(root: Path | None = None) -> list[Path]:
    base = root or COMPARISONS_DIR
    if not base.is_dir():
        return []
    return sorted(
        p
        for p in base.rglob("*.json")
        if p.is_file() and not p.name.startswith("_")
    )


def payload_retrieval_dashboard(root: Path | None = None) -> dict[str, Any]:
    base = root or COMPARISONS_DIR
    files = list_comparison_jsons(base)
    runs: list[RetrievalRun] = []
    errors: list[str] = []
    for path in files:
        try:
            runs.extend(parse_retrieval_json(path))
        except Exception as exc:  # noqa: BLE001 — informar en el dashboard
            errors.append(f"{path.name}: {exc}")

    # Unión de preguntas para heatmap
    question_meta: dict[str, dict[str, str]] = {}
    for run in runs:
        for qid, meta in run.questions.items():
            question_meta.setdefault(qid, meta)
    preguntas = sorted(
        question_meta.values(),
        key=lambda q: (q.get("item") or "", q["id"]),
    )

    return {
        "results_dir": str(base),
        "n_files": len(files),
        "n_runs": len(runs),
        "errors": errors,
        "views": list(VIEWS),
        "primary_metric": "NON-7A-36 Recall@5",
        "secondary_metric": "NON-7A-36 MRR@10",
        "runs": [asdict(run) for run in runs],
        "preguntas": preguntas,
    }
