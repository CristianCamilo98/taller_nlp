"""Carga y agregación de resultados JSONL (sin inventar datos)."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

METRICAS = ("cifra", "cita", "trayectoria")
ACIERTO_KEY = {
    "cifra": "acierto_cifra",
    "cita": "acierto_cita",
    "trayectoria": "acierto_trayectoria",
}


@dataclass
class RunKey:
    source_file: str
    run_id: str

    @property
    def label(self) -> str:
        short = self.run_id
        if short.startswith("run-") and len(short) > 12:
            short = short[:12]
        return f"{self.source_file} · {short}"

    @property
    def uid(self) -> str:
        return f"{self.source_file}::{self.run_id}"


@dataclass
class RunSummary:
    uid: str
    label: str
    source_file: str
    run_id: str
    n: int
    timestamp: str | None
    commit_sha: str | None
    prompt_version: str | None
    experimento: str | None
    model_solicitado: str | None
    pct_cifra: float | None
    pct_cita: float | None
    pct_trayectoria: float | None
    n_cifra: int
    n_cita: int
    n_trayectoria: int
    latencia_media_s: float | None
    coste_total: float
    coste_medio: float | None
    tool_calls_medios: float | None
    tool_calls_total: int
    llm_calls_medios: float | None
    llm_calls_total: int
    n_errores: int
    por_familia: dict[str, dict[str, Any]] = field(default_factory=dict)


def listar_jsonl(results_dir: Path | None = None) -> list[Path]:
    root = results_dir or RESULTS_DIR
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.jsonl") if p.is_file())


def cargar_filas(path: Path) -> list[dict[str, Any]]:
    filas: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno}: JSON inválido: {exc}") from exc
            row["_source_file"] = path.name
            filas.append(row)
    return filas


def cargar_todo(results_dir: Path | None = None) -> list[dict[str, Any]]:
    filas: list[dict[str, Any]] = []
    for path in listar_jsonl(results_dir):
        filas.extend(cargar_filas(path))
    return filas


def agrupar_por_run(filas: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grupos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filas:
        key = RunKey(row["_source_file"], str(row.get("run_id") or "unknown"))
        grupos[key.uid].append(row)
    return dict(grupos)


def _pct_acierto(filas: list[dict[str, Any]], metrica: str) -> tuple[float | None, int]:
    key = ACIERTO_KEY[metrica]
    aplicables = [r[key] for r in filas if r.get(key) is not None]
    if not aplicables:
        return None, 0
    aciertos = sum(1 for v in aplicables if v is True)
    return 100.0 * aciertos / len(aplicables), len(aplicables)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _meta(filas: list[dict[str, Any]], field_name: str) -> Any:
    for row in filas:
        val = row.get(field_name)
        if val not in (None, ""):
            return val
    return None


def resumen_run(uid: str, filas: list[dict[str, Any]]) -> RunSummary:
    source_file, run_id = uid.split("::", 1)
    key = RunKey(source_file, run_id)

    pct_cifra, n_cifra = _pct_acierto(filas, "cifra")
    pct_cita, n_cita = _pct_acierto(filas, "cita")
    pct_tray, n_tray = _pct_acierto(filas, "trayectoria")

    latencias = [float(r["latencia_s"]) for r in filas if r.get("latencia_s") is not None]
    costes = [float(r["coste"]) for r in filas if r.get("coste") is not None]
    tools = [
        float(r["tool_call_count"])
        for r in filas
        if r.get("tool_call_count") is not None
    ]
    llm_calls = [
        float(r["llm_calls"])
        for r in filas
        if r.get("llm_calls") is not None
    ]
    n_errores = sum(1 for r in filas if r.get("error"))

    por_familia: dict[str, dict[str, Any]] = {}
    familias: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filas:
        familias[str(row.get("familia") or "desconocida")].append(row)
    for fam, subset in sorted(familias.items()):
        pc, nc = _pct_acierto(subset, "cifra")
        pi, ni = _pct_acierto(subset, "cita")
        pt, nt = _pct_acierto(subset, "trayectoria")
        fam_llm = [
            float(r["llm_calls"])
            for r in subset
            if r.get("llm_calls") is not None
        ]
        fam_tools = [
            float(r["tool_call_count"])
            for r in subset
            if r.get("tool_call_count") is not None
        ]
        por_familia[fam] = {
            "n": len(subset),
            "pct_cifra": pc,
            "n_cifra": nc,
            "pct_cita": pi,
            "n_cita": ni,
            "pct_trayectoria": pt,
            "n_trayectoria": nt,
            "latencia_media_s": _mean(
                [float(r["latencia_s"]) for r in subset if r.get("latencia_s") is not None]
            ),
            "llm_calls_medios": _mean(fam_llm),
            "llm_calls_total": int(sum(fam_llm)) if fam_llm else 0,
            "tool_calls_medios": _mean(fam_tools),
            "n_errores": sum(1 for r in subset if r.get("error")),
        }

    return RunSummary(
        uid=uid,
        label=key.label,
        source_file=source_file,
        run_id=run_id,
        n=len(filas),
        timestamp=_meta(filas, "timestamp"),
        commit_sha=_meta(filas, "commit_sha"),
        prompt_version=_meta(filas, "prompt_version"),
        experimento=_meta(filas, "experimento"),
        model_solicitado=_meta(filas, "model_solicitado"),
        pct_cifra=pct_cifra,
        pct_cita=pct_cita,
        pct_trayectoria=pct_tray,
        n_cifra=n_cifra,
        n_cita=n_cita,
        n_trayectoria=n_tray,
        latencia_media_s=_mean(latencias),
        coste_total=sum(costes),
        coste_medio=_mean(costes),
        tool_calls_medios=_mean(tools),
        tool_calls_total=int(sum(tools)) if tools else 0,
        llm_calls_medios=_mean(llm_calls),
        llm_calls_total=int(sum(llm_calls)) if llm_calls else 0,
        n_errores=n_errores,
        por_familia=por_familia,
    )


def _celda_acierto(value: Any) -> str:
    if value is True:
        return "ok"
    if value is False:
        return "fail"
    return "na"


def matriz_pregunta_run(
    grupos: dict[str, list[dict[str, Any]]],
    metrica: str = "cifra",
) -> dict[str, Any]:
    """Heatmap pregunta × run para una métrica de acierto."""
    key = ACIERTO_KEY[metrica]
    question_meta: dict[str, dict[str, str]] = {}
    cells: dict[str, dict[str, str]] = defaultdict(dict)

    for uid, filas in grupos.items():
        for row in filas:
            qid = str(row.get("question_id") or row.get("id") or "?")
            if qid not in question_meta:
                question_meta[qid] = {
                    "id": qid,
                    "familia": str(row.get("familia") or ""),
                    "pregunta": str(row.get("pregunta") or ""),
                }
            cells[qid][uid] = _celda_acierto(row.get(key))

    preguntas = sorted(
        question_meta.values(),
        key=lambda q: (q["familia"], q["id"]),
    )
    return {
        "metrica": metrica,
        "preguntas": preguntas,
        "cells": {qid: dict(runs) for qid, runs in cells.items()},
    }


def fila_drilldown(row: dict[str, Any]) -> dict[str, Any]:
    """Campos útiles para el panel side-by-side (sin volcar el JSON entero)."""
    metricas = row.get("metricas") or {}
    errores_metricas: dict[str, Any] = {}
    for nombre in METRICAS:
        bloque = metricas.get(nombre) or {}
        if isinstance(bloque, dict):
            errores = bloque.get("errores")
            if errores:
                errores_metricas[nombre] = errores
            elif nombre == "cita" and bloque.get("aplica"):
                # cita no siempre trae lista errores; exponer flags fallidas
                flags = {
                    k: bloque.get(k)
                    for k in (
                        "citation_exists",
                        "chunk_exists",
                        "metadata_matches",
                        "golden_anchor_hit",
                        "citation_supports",
                        "acierto_cita",
                    )
                    if k in bloque
                }
                errores_metricas[nombre] = flags

    return {
        "uid": f"{row['_source_file']}::{row.get('run_id')}",
        "source_file": row["_source_file"],
        "run_id": row.get("run_id"),
        "question_id": row.get("question_id") or row.get("id"),
        "familia": row.get("familia"),
        "pregunta": row.get("pregunta"),
        "acierto_cifra": row.get("acierto_cifra"),
        "acierto_cita": row.get("acierto_cita"),
        "acierto_trayectoria": row.get("acierto_trayectoria"),
        "respuesta_agente": row.get("respuesta_agente"),
        "respuesta_esperada": row.get("respuesta_esperada"),
        "cifra_agente": row.get("cifra_agente"),
        "cifra_esperada": row.get("cifra_esperada"),
        "tool_calls_agente": row.get("tool_calls_agente") or row.get("tool_calls"),
        "tool_calls_detallado": row.get("tool_calls_detallado"),
        "tool_call_count": row.get("tool_call_count"),
        "llm_calls": row.get("llm_calls"),
        "guardrail_retry_count": row.get("guardrail_retry_count"),
        "latencia_s": row.get("latencia_s"),
        "coste": row.get("coste"),
        "error": row.get("error"),
        "errores_metricas": errores_metricas,
        "metricas": metricas,
        "model_solicitado": row.get("model_solicitado"),
        "prompt_version": row.get("prompt_version"),
    }


def payload_dashboard(results_dir: Path | None = None) -> dict[str, Any]:
    filas = cargar_todo(results_dir)
    grupos = agrupar_por_run(filas)
    summaries = [asdict(resumen_run(uid, rows)) for uid, rows in sorted(grupos.items())]

    drill: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for uid, rows in grupos.items():
        for row in rows:
            qid = str(row.get("question_id") or row.get("id") or "?")
            drill[qid][uid] = fila_drilldown(row)

    return {
        "results_dir": str(results_dir or RESULTS_DIR),
        "n_files": len(listar_jsonl(results_dir)),
        "n_rows": len(filas),
        "runs": summaries,
        "heatmaps": {
            m: matriz_pregunta_run(grupos, m) for m in METRICAS
        },
        "drilldown": {qid: dict(runs) for qid, runs in drill.items()},
    }
