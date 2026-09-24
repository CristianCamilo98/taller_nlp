"""Runner reproducible del benchmark; importar este módulo no llama a APIs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from common.benchmark_config import BENCHMARK
from common.config import get_dataset_paths


def cargar_golden(ruta: str | Path) -> list[dict]:
    with Path(ruta).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _commit_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _artifact_hashes() -> dict[str, str]:
    paths = get_dataset_paths(BENCHMARK.retrieval_profile)
    return {name: _sha256(path) for name, path in paths.required_files().items()}


def _retrieval_final_provenance() -> dict:
    artifact = (
        Path(__file__).resolve().parent
        / "results"
        / "retrieval_final"
        / "retrieval_final_48.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    manifest = payload["manifest"]
    return {
        "artifact": "common/results/retrieval_final/retrieval_final_48.json",
        "artifact_sha256": _sha256(artifact),
        "selected_pipeline": manifest["selected_pipeline"],
        "benchmark_sha256": manifest["benchmark_sha256"],
        "chunks_sha256": manifest["chunks_sha256"],
        "chunks_meta_sha256": manifest["chunks_meta_sha256"],
        "faiss_sha256": manifest["faiss_sha256"],
        "gemini_model": manifest["gemini_model"],
        "embedding_dimension": manifest["embedding_dimension"],
        "ntotal": manifest["ntotal"],
        "input_type_provenance": manifest["input_type_provenance"],
    }


def _is_rate_limit(error: Exception) -> bool:
    text = str(error).lower()
    status = getattr(error, "status_code", None)
    return status == 429 or "429" in text or "rate limit" in text


class InvocationFailed(RuntimeError):
    def __init__(self, cause: Exception, metadata: dict):
        super().__init__(str(cause))
        self.cause = cause
        self.metadata = metadata


def _invocar_con_reintentos(
    pregunta: str,
    *,
    responder_fn: Callable[[str], dict] | None = None,
    max_intentos: int | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[dict, dict]:
    """Reintenta solo 429 y separa tiempo activo de backoff."""
    if responder_fn is None:
        from common.responder import responder as responder_fn
    attempts = max_intentos or BENCHMARK.rate_limit_attempts
    retry_count = 0
    rate_limited = False
    backoff_s = 0.0
    active_s = 0.0
    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            response = responder_fn(pregunta)
            active_s += time.perf_counter() - started
            return response, {
                "retry_count": retry_count,
                "rate_limited": rate_limited,
                "backoff_s": backoff_s,
                "latencia_activa_s": active_s,
            }
        except Exception as error:
            active_s += time.perf_counter() - started
            if _is_rate_limit(error) and attempt < attempts - 1:
                rate_limited = True
                retry_count += 1
                wait = BENCHMARK.rate_limit_initial_backoff_s * (2 ** attempt)
                backoff_s += wait
                sleep_fn(wait)
                continue
            metadata = {
                "retry_count": retry_count,
                "rate_limited": rate_limited or _is_rate_limit(error),
                "backoff_s": backoff_s,
                "latencia_activa_s": active_s,
            }
            raise InvocationFailed(error, metadata) from error
    raise AssertionError("bucle de reintentos inalcanzable")


def evaluar(ruta_jsonl: str, guardar_en: str | None = None,
            pausa_entre_preguntas: float | None = None) -> list[dict]:
    """Ejecuta el golden y registra métricas y provenance explícitos."""
    from common.eval.evaluador_cifra import evaluar_cifra_detallada
    from common.eval.evaluador_cita import evaluar_cita_detallada
    from common.eval.evaluador_trayectoria import evaluar_trayectoria_detallada

    golden_path = Path(ruta_jsonl).resolve()
    questions = cargar_golden(golden_path)
    pause = (BENCHMARK.pause_between_questions_s
             if pausa_entre_preguntas is None else pausa_entre_preguntas)
    run_id = f"run-{uuid.uuid4().hex}"
    timestamp = datetime.now(timezone.utc).isoformat()
    commit_sha = _commit_sha()
    golden_hash = _sha256(golden_path)
    artifact_hashes = _artifact_hashes()
    retrieval_final_provenance = _retrieval_final_provenance()
    results = []

    destination = None
    if guardar_en:
        destination = Path(guardar_en)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Trunca/crea antes de empezar: cada fila se añade incrementalmente
        # según se calcula, para no perder progreso si el batch se
        # interrumpe a mitad de camino.
        destination.open("w", encoding="utf-8", newline="\n").close()

    for index, question in enumerate(questions):
        wall_start = time.perf_counter()
        retry_meta = {"retry_count": 0, "rate_limited": False,
                      "backoff_s": 0.0, "latencia_activa_s": 0.0}
        try:
            response, retry_meta = _invocar_con_reintentos(question["pregunta"])
            error = None
        except InvocationFailed as failure:
            response = {}
            retry_meta = failure.metadata
            error = f"{type(failure.cause).__name__}: {failure.cause}"
        wall_latency = time.perf_counter() - wall_start
        if not error and not response.get("respuesta"):
            error = "Respuesta vacía del agente"

        telemetry = response.get("_telemetria") or {}
        public_response = {key: value for key, value in response.items()
                           if not key.startswith("_")}
        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "commit_sha": commit_sha,
            "question_id": question["id"],
            "id": question["id"],
            "familia": question["familia"],
            "pregunta": question["pregunta"],
            "respuesta": public_response or None,
            "respuesta_agente": response.get("respuesta"),
            "cifra_agente": response.get("cifra"),
            "unidad_agente": response.get("unidad"),
            "ticker_agente": response.get("ticker"),
            "ejercicio_agente": response.get("ejercicio"),
            "concepto_xbrl_agente": response.get("concepto_xbrl"),
            "ejercicio_inicial_agente": response.get("ejercicio_inicial"),
            "ejercicio_final_agente": response.get("ejercicio_final"),
            "valor_inicial_agente": response.get("valor_inicial"),
            "valor_final_agente": response.get("valor_final"),
            "delta_agente": response.get("delta"),
            "porcentaje_agente": response.get("porcentaje"),
            "fuente_agente": response.get("fuente"),
            "cita_agente": response.get("cita"),
            "chunk_id_agente": response.get("chunk_id"),
            "tool_calls": response.get("tool_calls_agente") or [],
            "tool_calls_agente": response.get("tool_calls_agente") or [],
            "tool_calls_detallado": response.get("tool_calls_detallado") or [],
            "tool_call_count": len(response.get("tool_calls_detallado") or []),
            "tool_calls_bloqueados": response.get("tool_calls_bloqueados") or [],
            "tool_call_duplicate_blocked_count": len(
                response.get("tool_calls_bloqueados") or []
            ),
            "latencia_s": round(wall_latency, 6),
            "latencia_activa_s": round(retry_meta["latencia_activa_s"], 6),
            "backoff_s": retry_meta["backoff_s"],
            "retry_count": retry_meta["retry_count"],
            "guardrail_retry_count": response.get("guardrail_retry_count", 0),
            "rate_limited": retry_meta["rate_limited"],
            "error": error,
            "provider_solicitado": BENCHMARK.provider,
            "model_solicitado": BENCHMARK.model,
            "agent_model": BENCHMARK.model,
            "model_efectivo": telemetry.get("model_effective"),
            "provider_efectivo": telemetry.get("provider_effective"),
            "temperature": BENCHMARK.temperature,
            "input_tokens": telemetry.get("input_tokens"),
            "output_tokens": telemetry.get("output_tokens"),
            "total_tokens": telemetry.get("total_tokens"),
            "llm_calls": telemetry.get("llm_calls"),
            "coste": telemetry.get("coste"),
            "retrieval_config": {
                "tipo": "dense_faiss_con_postfiltrado_metadata",
                "profile": BENCHMARK.retrieval_profile,
                "embedding": BENCHMARK.embedding_model,
                "query_prefix": BENCHMARK.embedding_query_prefix,
                "normalizado": BENCHMARK.embedding_normalize,
                "index_type": BENCHMARK.faiss_index_type,
                "k": BENCHMARK.retrieval_k,
                "query_rewriting": False,
                "bm25": False,
                "reranking": False,
            },
            "retrieval_profile": BENCHMARK.retrieval_profile,
            "retrieval_pipeline": (
                "original query + Gemini Embedding 2 + FAISS global + "
                "metadata postfilter ticker/fiscal_year/item"
            ),
            "retrieval_final_provenance": retrieval_final_provenance,
            "embedding": BENCHMARK.embedding_model,
            "k": BENCHMARK.retrieval_k,
            "prompt_version": BENCHMARK.prompt_version,
            "hash_golden": golden_hash,
            "hashes_dataset_indice": artifact_hashes,
            "respuesta_esperada": question.get("respuesta_esperada"),
            "cifra_esperada": question.get("cifra_esperada"),
        }
        if error:
            numeric = question.get("familia") in {"numerica", "comparativa"}
            qualitative = bool(question.get("ancla_texto"))
            number_detail = {"aplica": numeric, "acierto_cifra":
                             False if numeric else None, "errores": ["error_run"]}
            citation_detail = {"aplica": qualitative, "acierto_cita":
                               False if qualitative else None}
            trajectory_detail = {"aplica": True, "acierto_trayectoria": False,
                                 "errores": ["error_run"]}
        else:
            try:
                number_detail = evaluar_cifra_detallada(row, question)
                citation_detail = evaluar_cita_detallada(row, question)
                trajectory_detail = evaluar_trayectoria_detallada(row, question)
            except Exception as eval_error:
                # Una pregunta con esquema inesperado (p. ej. un blind set
                # ajeno al golden) no debe tirar abajo el resto del batch.
                detalle = f"evaluator_error: {type(eval_error).__name__}: {eval_error}"
                number_detail = {"aplica": None, "acierto_cifra": None,
                                 "errores": [detalle]}
                citation_detail = {"aplica": None, "acierto_cita": None}
                trajectory_detail = {"aplica": None,
                                     "acierto_trayectoria": None,
                                     "errores": [detalle]}
                row["error"] = detalle
        row["metricas"] = {
            "cifra": number_detail,
            "cita": citation_detail,
            "trayectoria": trajectory_detail,
        }
        row["acierto_cifra"] = number_detail.get("acierto_cifra")
        row["acierto_cita"] = citation_detail.get("acierto_cita")
        row["acierto_trayectoria"] = trajectory_detail.get(
            "acierto_trayectoria")
        results.append(row)
        if destination is not None:
            with destination.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if index < len(questions) - 1 and pause > 0:
            time.sleep(pause)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("output", nargs="?")
    args = parser.parse_args()
    evaluar(args.golden, args.output)
