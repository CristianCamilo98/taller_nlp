"""Runner de evaluación del experimento de Cristian.

Ejecuta un golden JSONL contra ``cristian.experiments.responder`` y puntúa
con los tres evaluadores del baseline (misma rúbrica → comparación justa).

Por defecto lanza hasta ``SETTINGS.eval_max_workers`` preguntas en paralelo
(``ThreadPoolExecutor``). Cada hilo tiene su propio agente. Los resultados se
reúnen en memoria, se ordenan como el golden y se escriben una sola vez.

Uso (desde la raíz del repo, con venv y API key)::

    python -m cristian.experiments.evaluar \\
        common/golden_set/golden_set_grupo3.jsonl \\
        cristian/experiments/results/run.jsonl

    python -m cristian.experiments.evaluar golden.jsonl out.jsonl --workers 5
    python -m cristian.experiments.evaluar golden.jsonl out.jsonl --workers 1
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cristian.experiments.config import SETTINGS, get_dataset_paths

_print_lock = threading.Lock()


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
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _artifact_hashes() -> dict[str, str]:
    paths = get_dataset_paths()
    return {
        name: _sha256(path) for name, path in paths.required_files().items()
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
    if responder_fn is None:
        from cristian.experiments.responder import responder as responder_fn

    attempts = max_intentos or SETTINGS.rate_limit_attempts
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
                wait = SETTINGS.rate_limit_initial_backoff_s * (2 ** attempt)
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


def _construir_fila(
    *,
    index: int,
    question: dict,
    n_questions: int,
    run_id: str,
    timestamp: str,
    commit_sha: str | None,
    golden_hash: str,
    artifact_hashes: dict[str, str],
    evaluar_cifra_detallada,
    evaluar_cita_detallada,
    evaluar_trayectoria_detallada,
) -> dict:
    """Invoca el agente para una pregunta y arma la fila de resultados."""
    wall_start = time.perf_counter()
    retry_meta = {
        "retry_count": 0,
        "rate_limited": False,
        "backoff_s": 0.0,
        "latencia_activa_s": 0.0,
    }
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
    public_response = {
        key: value
        for key, value in response.items()
        if not key.startswith("_")
    }
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
        "latencia_s": round(wall_latency, 6),
        "latencia_activa_s": round(retry_meta["latencia_activa_s"], 6),
        "backoff_s": retry_meta["backoff_s"],
        "retry_count": retry_meta["retry_count"],
        "guardrail_retry_count": response.get("guardrail_retry_count", 0),
        "rate_limited": retry_meta["rate_limited"],
        "error": error,
        "provider_solicitado": SETTINGS.provider,
        "model_solicitado": SETTINGS.model,
        "model_efectivo": telemetry.get("model_effective"),
        "provider_efectivo": telemetry.get("provider_effective"),
        "temperature": SETTINGS.temperature,
        "input_tokens": telemetry.get("input_tokens"),
        "output_tokens": telemetry.get("output_tokens"),
        "total_tokens": telemetry.get("total_tokens"),
        "llm_calls": telemetry.get("llm_calls"),
        "coste": telemetry.get("coste"),
        "retrieval_config": {
            "tipo": "dense_faiss_con_postfiltrado_metadata",
            "embedding": SETTINGS.embedding_model,
            "query_prefix": SETTINGS.embedding_query_prefix,
            "k": SETTINGS.retrieval_k,
            "query_rewriting": False,
            "bm25": False,
            "reranking": False,
        },
        "embedding": SETTINGS.embedding_model,
        "k": SETTINGS.retrieval_k,
        "prompt_version": SETTINGS.prompt_version,
        "hash_golden": golden_hash,
        "hashes_dataset_indice": artifact_hashes,
        "respuesta_esperada": question.get("respuesta_esperada"),
        "cifra_esperada": question.get("cifra_esperada"),
        "experimento": "cristian",
    }
    if error:
        numeric = question["familia"] in {"numerica", "comparativa"}
        qualitative = bool(question.get("ancla_texto"))
        number_detail = {
            "aplica": numeric,
            "acierto_cifra": False if numeric else None,
            "errores": ["error_run"],
        }
        citation_detail = {
            "aplica": qualitative,
            "acierto_cita": False if qualitative else None,
        }
        trajectory_detail = {
            "aplica": True,
            "acierto_trayectoria": False,
            "errores": ["error_run"],
        }
    else:
        number_detail = evaluar_cifra_detallada(row, question)
        citation_detail = evaluar_cita_detallada(row, question)
        trajectory_detail = evaluar_trayectoria_detallada(row, question)
    row["metricas"] = {
        "cifra": number_detail,
        "cita": citation_detail,
        "trayectoria": trajectory_detail,
    }
    row["acierto_cifra"] = number_detail.get("acierto_cifra")
    row["acierto_cita"] = citation_detail.get("acierto_cita")
    row["acierto_trayectoria"] = trajectory_detail.get("acierto_trayectoria")

    with _print_lock:
        print(
            f"[{index + 1}/{n_questions}] {question['id']} "
            f"tray={row['acierto_trayectoria']} "
            f"cifra={row['acierto_cifra']} cita={row['acierto_cita']}"
            + (f" ERR={error}" if error else "")
        )
    return row


def evaluar(
    ruta_jsonl: str,
    guardar_en: str | None = None,
    pausa_entre_preguntas: float | None = None,
    max_workers: int | None = None,
) -> list[dict]:
    """Ejecuta el golden contra el agente experimental y guarda métricas.

    ``max_workers`` controla el paralelismo (default:
    ``SETTINGS.eval_max_workers``). Con ``max_workers == 1`` se respeta
    ``pausa_entre_preguntas``; con paralelismo la pausa se ignora.
    """
    from common.eval.evaluador_cifra import evaluar_cifra_detallada
    from common.eval.evaluador_cita import evaluar_cita_detallada
    from common.eval.evaluador_trayectoria import evaluar_trayectoria_detallada

    golden_path = Path(ruta_jsonl).resolve()
    questions = cargar_golden(golden_path)
    workers = SETTINGS.eval_max_workers if max_workers is None else int(max_workers)
    if workers < 1:
        raise ValueError("max_workers debe ser >= 1")
    pause = (
        SETTINGS.pause_between_questions_s
        if pausa_entre_preguntas is None
        else pausa_entre_preguntas
    )
    run_id = f"run-{uuid.uuid4().hex}"
    timestamp = datetime.now(timezone.utc).isoformat()
    commit_sha = _commit_sha()
    golden_hash = _sha256(golden_path)
    artifact_hashes = _artifact_hashes()
    n_questions = len(questions)

    print(
        f"Evaluando {n_questions} preguntas · workers={workers} · "
        f"modelo={SETTINGS.model}"
    )

    # Una sola carga de BGE/FAISS en el hilo principal (evita 5× "Loading weights").
    from cristian.experiments.miax_s1 import precargar_retrieval
    from cristian.experiments.tools import _load_sections
    from cristian.experiments.xbrl import load_xbrl

    print("Precargando retrieval (FAISS + BGE) y corpus/XBRL…")
    precargar_retrieval()
    load_xbrl()
    _load_sections()

    def _job(index: int, question: dict) -> tuple[int, dict]:
        row = _construir_fila(
            index=index,
            question=question,
            n_questions=n_questions,
            run_id=run_id,
            timestamp=timestamp,
            commit_sha=commit_sha,
            golden_hash=golden_hash,
            artifact_hashes=artifact_hashes,
            evaluar_cifra_detallada=evaluar_cifra_detallada,
            evaluar_cita_detallada=evaluar_cita_detallada,
            evaluar_trayectoria_detallada=evaluar_trayectoria_detallada,
        )
        return index, row

    results: list[dict] = [{}] * n_questions

    if workers == 1:
        for index, question in enumerate(questions):
            _, row = _job(index, question)
            results[index] = row
            if index < n_questions - 1 and pause > 0:
                time.sleep(pause)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_job, index, question)
                for index, question in enumerate(questions)
            ]
            for future in as_completed(futures):
                index, row = future.result()
                results[index] = row

    if guardar_en:
        destination = Path(guardar_en)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Guardado: {destination}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evalúa el agente de Cristian sobre un golden JSONL"
    )
    parser.add_argument("golden", help="Ruta al golden set (.jsonl)")
    parser.add_argument(
        "output",
        nargs="?",
        default="cristian/experiments/results/ultimo_run.jsonl",
        help="JSONL de salida",
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=None,
        help="Segundos entre preguntas si --workers 1 (default: SETTINGS)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Preguntas en paralelo "
            f"(default: {SETTINGS.eval_max_workers})"
        ),
    )
    args = parser.parse_args()
    evaluar(
        args.golden,
        args.output,
        pausa_entre_preguntas=args.pausa,
        max_workers=args.workers,
    )
