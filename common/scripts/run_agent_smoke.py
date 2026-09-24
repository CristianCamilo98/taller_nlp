"""Runner minimo de smoke end-to-end para el agente final.

Ejecuta UNA pregunta existente (numerica o extractiva, tomada de
common/golden_set/golden_set_grupo3.jsonl o
common/benchmark/retrieval_benchmark_v2.jsonl) a traves de
common.responder.responder(pregunta) y guarda el artefacto completo en
common/results/agent_smoke/{case}_smoke.json.

No persiste la API key: solo se leen y se serializan los campos publicos
de la respuesta del agente (respuesta estructurada, tool_calls, telemetria
de tokens/coste, latencia). La clave se toma exclusivamente de
$env:OPENROUTER_API_KEY / .env, nunca de un argumento de linea de comandos.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = ROOT / "common/golden_set/golden_set_grupo3.jsonl"
BENCHMARK_V2_PATH = ROOT / "common/benchmark/retrieval_benchmark_v2.jsonl"
OUTPUT_DIR = ROOT / "common/results/agent_smoke"

NUMERIC_QUESTION_ID = "g3-001"
EXTRACTIVE_QUESTION_ID = "dani-b6-001"


def _cargar_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _numeric_case() -> tuple[dict[str, Any], str]:
    """g3-001: NVDA FY2024 Assets, requiere get_xbrl_fact."""
    rows = _cargar_jsonl(GOLDEN_PATH)
    question = next(row for row in rows if row["id"] == NUMERIC_QUESTION_ID)
    return dict(question), question["pregunta"]


def _extractive_case() -> tuple[dict[str, Any], str]:
    """dani-b6-001: AAPL FY2024 item 1A, requiere search_filings."""
    rows = _cargar_jsonl(BENCHMARK_V2_PATH)
    question = next(
        row for row in rows if row["question_id"] == EXTRACTIVE_QUESTION_ID
    )
    pregunta_eval = {
        "id": question["question_id"],
        "familia": "extractiva",
        "ticker": question["ticker"],
        "fiscal_year": question["fiscal_year"],
        "item": question["item"],
        "item_esperado": question["item"],
        "ancla_texto": question["evidence_text"],
        "herramienta_esperada": ["search_filings"],
    }
    return pregunta_eval, question["question"]


CASES: dict[str, Callable[[], tuple[dict[str, Any], str]]] = {
    "numeric": _numeric_case,
    "extractive": _extractive_case,
}


def _primera_query_search_filings(tool_calls: list[dict[str, Any]]) -> str | None:
    for call in tool_calls:
        if call.get("name") == "search_filings":
            return (call.get("args") or {}).get("query")
    return None


def run_smoke(
    case: str,
    *,
    responder_fn: Callable[[str], dict[str, Any]] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Ejecuta un smoke case y devuelve (y guarda) el artefacto resultante."""
    if responder_fn is None:
        from common.responder import responder as responder_fn

    from common.benchmark_config import BENCHMARK
    from common.eval.evaluador_cifra import evaluar_cifra_detallada
    from common.eval.evaluador_cita import evaluar_cita_detallada
    from common.eval.evaluador_trayectoria import evaluar_trayectoria_detallada

    pregunta_eval, pregunta_texto = CASES[case]()

    started = time.perf_counter()
    respuesta = responder_fn(pregunta_texto)
    latency_s = time.perf_counter() - started

    telemetry = respuesta.get("_telemetria") or {}
    respuesta_estructurada = {
        key: value for key, value in respuesta.items() if not key.startswith("_")
    }
    tool_calls = respuesta.get("tool_calls_detallado") or []
    primera_query = _primera_query_search_filings(tool_calls)

    artifact = {
        "smoke_case": case,
        "question_id": pregunta_eval["id"],
        "pregunta": pregunta_texto,
        "pregunta_metadata": pregunta_eval,
        "respuesta_estructurada": respuesta_estructurada,
        "tool_calls_detallado": tool_calls,
        "tool_calls_bloqueados": respuesta.get("tool_calls_bloqueados") or [],
        "tool_results_detallado": respuesta.get("tool_results_detallado") or [],
        "citation_grounding_ok": respuesta.get("citation_grounding_ok"),
        "citation_repair_used": respuesta.get("citation_repair_used"),
        "citation_repair_failed": respuesta.get("citation_repair_failed"),
        "evidence_rescue_used": respuesta.get("evidence_rescue_used"),
        "trajectory": evaluar_trayectoria_detallada(respuesta, pregunta_eval),
        "metricas": {
            "cifra": evaluar_cifra_detallada(respuesta, pregunta_eval),
            "cita": evaluar_cita_detallada(respuesta, pregunta_eval),
        },
        "telemetry": telemetry,
        "agent_model": BENCHMARK.model,
        "retrieval_profile": BENCHMARK.retrieval_profile,
        "latency_s": round(latency_s, 6),
        "primera_query_search_filings": primera_query,
        "primera_query_es_pregunta_original_literal": (
            None if primera_query is None else primera_query == pregunta_texto
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    destino = output_dir / f"{case}_smoke.json"
    with destino.open("w", encoding="utf-8") as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(
        f"Smoke {case} · question_id={pregunta_eval['id']} · "
        f"latency_s={artifact['latency_s']} · "
        f"tool_calls={len(tool_calls)} · "
        f"bloqueados={len(artifact['tool_calls_bloqueados'])}",
        flush=True,
    )
    print(
        f"acierto_cifra={artifact['metricas']['cifra'].get('acierto_cifra')} · "
        f"acierto_cita={artifact['metricas']['cita'].get('acierto_cita')} · "
        f"acierto_trayectoria={artifact['trajectory'].get('acierto_trayectoria')}",
        flush=True,
    )
    print(
        "query_original_ok="
        f"{artifact['primera_query_es_pregunta_original_literal']}",
        flush=True,
    )
    print(
        f"chunks_recuperados={len(artifact['tool_results_detallado'])} · "
        f"citation_grounding_ok={artifact['citation_grounding_ok']} · "
        f"citation_repair_used={artifact['citation_repair_used']} · "
        f"citation_repair_failed={artifact['citation_repair_failed']} · "
        f"evidence_rescue_used={artifact['evidence_rescue_used']}",
        flush=True,
    )
    print(f"Artifact: {destino.resolve()}", flush=True)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke end-to-end de una pregunta existente via responder()"
    )
    parser.add_argument("case", choices=sorted(CASES))
    args = parser.parse_args()
    run_smoke(args.case)


if __name__ == "__main__":
    main()
