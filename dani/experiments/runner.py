"""Orquestador reproducible de las ablaciones y sus question sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from common.config import get_dataset_paths
from dani.experiments.benchmark.benchmark_v2 import (
    BENCHMARK_NAME,
    BENCHMARK_PATH,
    FROZEN_SHA256,
)
from dani.experiments.config import (
    EXPECTED_E0_METRICS,
    GPU_PARITY_REFERENCES,
    ExperimentConfig,
    e1_bge_large_config,
    e2_e5_large_v2_config,
    e3_qwen3_embedding_06b_config,
)
from dani.experiments.embeddings import adapter_for_config
from dani.experiments.evaluation import (
    EVALUATION_VIEWS,
    BenchmarkV2Evaluator,
    RetrievalEvaluator,
)
from dani.experiments.retriever import DenseFaissRetriever

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PILOT = "pilot"
BENCHMARK_V2 = "v2"
QUESTION_SETS = (PILOT, BENCHMARK_V2)
RUN_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class E0ParityError(RuntimeError):
    """Las métricas reconstruidas no coinciden con el control congelado."""


class GpuParityError(RuntimeError):
    """El run GPU no reproduce rangos y métricas CPU congelados."""


class IndexProvenanceError(ValueError):
    """El índice reutilizado no coincide con su manifest o experimento."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_run_label(run_label: str | None) -> str | None:
    """Limita el label a un sufijo seguro para nombres de artefacto."""
    if run_label is not None and not RUN_LABEL_PATTERN.fullmatch(run_label):
        raise ValueError(
            "run-label debe usar solo letras, números, guion o underscore"
        )
    return run_label


def validate_runtime_options(
    device_override: str | None, run_label: str | None
) -> None:
    validate_run_label(run_label)
    if device_override not in {None, "cpu", "cuda"}:
        raise ValueError(f"device override no soportado: {device_override}")
    if device_override is not None and run_label is None:
        raise ValueError("--device requiere --run-label para evitar colisiones")


def run_name(config: ExperimentConfig, run_label: str | None = None) -> str:
    label = validate_run_label(run_label)
    return f"{config.experiment_id}_{label}" if label else config.experiment_id


def runtime_config(
    config: ExperimentConfig, device_override: str | None = None
) -> ExperimentConfig:
    """Copia efímera que cambia solo la infraestructura de ejecución."""
    return (
        replace(config, requested_device=device_override)
        if device_override is not None
        else config
    )


def execution_provenance(
    *,
    base_config: ExperimentConfig,
    runtime: ExperimentConfig,
    run_label: str | None,
    device_override: str | None,
    effective_device: str | None,
) -> dict[str, Any]:
    return {
        "base_experiment_id": base_config.experiment_id,
        "run_label": run_label,
        "configured_device": base_config.requested_device,
        "runtime_device_override": device_override,
        "runtime_requested_device": runtime.requested_device,
        "effective_device": effective_device,
    }


def validate_index_options(
    index_path: Path | None, index_manifest_path: Path | None
) -> None:
    """Exige que índice y manifest se proporcionen siempre juntos."""
    if (index_path is None) != (index_manifest_path is None):
        raise IndexProvenanceError(
            "--index y --index-manifest deben proporcionarse juntos"
        )


def _logical_path(path: Path) -> str:
    """Evita rutas de máquina cuando el archivo pertenece al repo/dataset."""
    resolved = path.resolve()
    repo_root = EXPERIMENT_ROOT.parents[1]
    try:
        return "repo://" + resolved.relative_to(repo_root).as_posix()
    except ValueError:
        pass
    try:
        dataset_root = get_dataset_paths().dataset_dir
        return "dataset://" + resolved.relative_to(dataset_root).as_posix()
    except ValueError:
        return str(resolved)


def validate_reused_index(
    *,
    index_path: Path,
    index_manifest_path: Path,
    config: ExperimentConfig,
    chunks_path: Path,
    chunks_meta_path: Path,
    n_chunks: int,
) -> dict[str, Any]:
    """Valida fail-closed el origen de un FAISS sin usar resultados científicos."""
    import faiss

    if not index_path.is_file():
        raise IndexProvenanceError(f"Índice inexistente: {index_path}")
    if not index_manifest_path.is_file():
        raise IndexProvenanceError(
            f"Manifest de índice inexistente: {index_manifest_path}"
        )
    try:
        payload = json.loads(index_manifest_path.read_text(encoding="utf-8"))
        source = payload["manifest"]
        identity = source["identity"]
        source_config = source["config"]
        embedding = source["embedding"]
        manifest_index = source["index"]
        inputs = source["inputs"]
        input_hashes = inputs["sha256"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise IndexProvenanceError(
            "El result JSON no contiene un manifest de provenance válido"
        ) from exc

    try:
        index = faiss.read_index(str(index_path))
    except Exception as exc:
        raise IndexProvenanceError(f"No se puede leer el índice: {index_path}") from exc

    actual_index_sha = sha256_file(index_path)
    actual_chunks_sha = sha256_file(chunks_path)
    actual_meta_sha = sha256_file(chunks_meta_path)
    actual_index = {
        "sha256": actual_index_sha,
        "type": type(index).__name__,
        "d": int(index.d),
        "ntotal": int(index.ntotal),
    }
    expected = {
        "identity.experiment_id": config.experiment_id,
        "config.experiment_id": config.experiment_id,
        "config.model_name": config.model_name,
        "config.model_revision": config.model_revision,
        "config.expected_dimension": config.expected_dimension,
        "config.normalize_embeddings": config.normalize_embeddings,
        "config.document_format": config.document_format,
        "config.document_prefix": config.document_prefix,
        "config.index_type": config.index_type,
        "embedding.model_name": config.model_name,
        "embedding.dimension": config.expected_dimension,
        "embedding.normalize": config.normalize_embeddings,
        "embedding.document_formatting": config.document_format,
        "index.sha256": actual_index_sha,
        "index.type": actual_index["type"],
        "index.d": actual_index["d"],
        "index.ntotal": actual_index["ntotal"],
        "inputs.sha256.chunks_jsonl": actual_chunks_sha,
        "inputs.sha256.chunks_meta_parquet": actual_meta_sha,
        "inputs.n_chunks": n_chunks,
    }
    observed = {
        "identity.experiment_id": identity.get("experiment_id"),
        "config.experiment_id": source_config.get("experiment_id"),
        "config.model_name": source_config.get("model_name"),
        "config.model_revision": source_config.get("model_revision"),
        "config.expected_dimension": source_config.get("expected_dimension"),
        "config.normalize_embeddings": source_config.get(
            "normalize_embeddings"
        ),
        "config.document_format": source_config.get("document_format"),
        "config.document_prefix": source_config.get("document_prefix"),
        "config.index_type": source_config.get("index_type"),
        "embedding.model_name": embedding.get("model_name"),
        "embedding.dimension": embedding.get("dimension"),
        "embedding.normalize": embedding.get("normalize"),
        "embedding.document_formatting": embedding.get(
            "document_formatting"
        ),
        "index.sha256": str(manifest_index.get("sha256", "")).lower(),
        "index.type": manifest_index.get("type"),
        "index.d": manifest_index.get("d"),
        "index.ntotal": manifest_index.get("ntotal"),
        "inputs.sha256.chunks_jsonl": str(
            input_hashes.get("chunks_jsonl", "")
        ).lower(),
        "inputs.sha256.chunks_meta_parquet": str(
            input_hashes.get("chunks_meta_parquet", "")
        ).lower(),
        "inputs.n_chunks": inputs.get("n_chunks"),
    }
    if config.model_revision is not None:
        expected["embedding.model_revision"] = config.model_revision
        observed["embedding.model_revision"] = embedding.get("model_revision")

    errors = [
        f"{field}: manifest={observed[field]!r}, esperado={value!r}"
        for field, value in expected.items()
        if observed.get(field) != value
    ]
    if actual_index["ntotal"] != n_chunks:
        errors.append(
            f"index.ntotal real={actual_index['ntotal']!r}, corpus={n_chunks!r}"
        )
    if actual_index["d"] != config.expected_dimension:
        errors.append(
            f"index.d real={actual_index['d']!r}, "
            f"experimento={config.expected_dimension!r}"
        )
    if actual_index["type"] != config.index_type:
        errors.append(
            f"index.type real={actual_index['type']!r}, "
            f"experimento={config.index_type!r}"
        )
    if errors:
        raise IndexProvenanceError(
            "Provenance de índice incompatible: " + "; ".join(errors)
        )

    return {
        "index_path": _logical_path(index_path),
        "index_sha256": actual_index_sha,
        "source_manifest": _logical_path(index_manifest_path),
        "source_experiment_id": identity["experiment_id"],
        "source_model_name": embedding["model_name"],
        "source_model_revision": embedding.get("model_revision"),
        "chunks_jsonl_sha256": actual_chunks_sha,
        "chunks_meta_parquet_sha256": actual_meta_sha,
    }


def numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None,
                "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def git_value(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=EXPERIMENT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_provenance() -> dict[str, Any]:
    status = git_value("status", "--porcelain")
    return {
        "commit": git_value("rev-parse", "HEAD"),
        "branch": git_value("branch", "--show-current"),
        "dirty_worktree": bool(status) if status is not None else None,
    }


def _logical_paths(
    config: ExperimentConfig, question_set: str = PILOT
) -> dict[str, str]:
    paths = {
        "chunks": config.chunk_source,
        "chunks_metadata": "dataset://indice_faiss/chunks_meta.parquet",
    }
    if question_set == BENCHMARK_V2:
        paths["benchmark"] = (
            "repo://dani/experiments/benchmark/retrieval_benchmark_v2.jsonl"
        )
    else:
        paths["golden"] = "repo://common/golden_set/golden_set_grupo3.jsonl"
        paths["evidence_ground_truth"] = (
            "repo://dani/experiments/results/evidence_ground_truth_v1.json"
        )
    return paths


def create_manifest(
    *,
    config: ExperimentConfig,
    timestamp_utc: str,
    input_hashes: dict[str, str],
    n_chunks: int,
    n_questions: int,
    embedding_metadata: dict[str, Any],
    index_metadata: dict[str, Any],
    timings: dict[str, Any],
    chunk_statistics: dict[str, Any],
    parity: dict[str, Any],
    question_set: str = PILOT,
    benchmark_metadata: dict[str, Any] | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "identity": {
            "experiment_id": config.experiment_id,
            "timestamp_utc": timestamp_utc,
            **git_provenance(),
        },
        "config": config.to_dict(),
        "inputs": {
            "logical_paths": _logical_paths(config, question_set),
            "sha256": input_hashes,
            "n_chunks": n_chunks,
            "n_questions": n_questions,
        },
        "embedding": embedding_metadata,
        "index": index_metadata,
        "software": {
            "python": sys.version,
            "sentence_transformers": package_version("sentence-transformers"),
            "torch": package_version("torch"),
            "numpy": package_version("numpy"),
            "faiss": package_version("faiss-cpu"),
            "platform": platform.platform(),
            "machine": platform.machine() or None,
            "processor": platform.processor() or None,
        },
        "timings_s": timings,
        "chunk_statistics": chunk_statistics,
        "parity": parity,
    }
    if benchmark_metadata is not None:
        manifest["benchmark"] = benchmark_metadata
    if execution_metadata is not None:
        manifest["execution"] = execution_metadata
    return manifest


class ExperimentRunner:
    """Ensambla una ablación sin alterar las variables del retrieval."""

    def __init__(
        self,
        config: ExperimentConfig,
        expected_metrics: dict[str, float | int] | None = EXPECTED_E0_METRICS,
        question_set: str = PILOT,
        benchmark_path: Path = BENCHMARK_PATH,
        device_override: str | None = None,
        run_label: str | None = None,
        expected_first_relevant_ranks: dict[str, int | None] | None = None,
    ):
        if question_set not in QUESTION_SETS:
            raise ValueError(f"Question set desconocido: {question_set}")
        validate_runtime_options(device_override, run_label)
        self.config = config
        self.expected_metrics = expected_metrics
        self.question_set = question_set
        self.benchmark_path = benchmark_path
        self.device_override = device_override
        self.run_label = validate_run_label(run_label)
        self.runtime_config = runtime_config(config, device_override)
        self.expected_first_relevant_ranks = expected_first_relevant_ranks

    def run(
        self,
        output_path: Path,
        artifact_dir: Path,
        index_path: Path | None = None,
        index_manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        from datetime import datetime, timezone

        import pandas as pd

        validate_index_options(index_path, index_manifest_path)
        paths = get_dataset_paths()
        chunks = self._load_chunks(paths.chunks)
        metadata = pd.read_parquet(paths.chunks_meta)
        self._validate_alignment(chunks, metadata)
        evaluator = self._build_evaluator(metadata)
        reused_index_provenance = None
        if index_path is not None and index_manifest_path is not None:
            reused_index_provenance = validate_reused_index(
                index_path=index_path,
                index_manifest_path=index_manifest_path,
                config=self.config,
                chunks_path=paths.chunks,
                chunks_meta_path=paths.chunks_meta,
                n_chunks=len(chunks),
            )
        adapter = adapter_for_config(self.runtime_config)

        total_build_start = perf_counter()
        model_load_s = adapter.load()
        tokenization_start = perf_counter()
        model_token_lengths = adapter.token_lengths(
            [str(chunk["texto"]) for chunk in chunks]
        )
        tokenization_s = perf_counter() - tokenization_start
        reused_index = index_path is not None
        if index_path is None:
            index_path = default_index_path(
                artifact_dir, self.config, self.run_label
            )
            retriever, build_timings = DenseFaissRetriever.build(
                [str(chunk["texto"]) for chunk in chunks],
                metadata,
                adapter,
                self.runtime_config,
            )
            retriever.save(index_path)
        else:
            index_load_start = perf_counter()
            retriever = DenseFaissRetriever.load(
                index_path, metadata, adapter, self.runtime_config
            )
            build_timings = {
                "document_embedding_s": None,
                "index_build_s": None,
                "index_load_s": perf_counter() - index_load_start,
            }
        total_build_s = perf_counter() - total_build_start

        evaluation_start = perf_counter()
        metrics, per_question = evaluator.evaluate(retriever)
        evaluation_s = perf_counter() - evaluation_start
        parity = (
            self._parity(
                metrics,
                self.expected_metrics,
                per_question,
                self.expected_first_relevant_ranks,
            )
            if self.question_set == PILOT and self.expected_metrics is not None
            else {"passed": None, "checks": {}, "reference": None}
        )

        total_latencies = [
            float(row["latency_s"]["total"]) for row in per_question
        ]
        query_embedding_latencies = [
            float(row["latency_s"]["query_embedding"])
            for row in per_question
        ]
        timings = {
            "model_load": model_load_s,
            "tokenization_diagnostic": tokenization_s,
            **build_timings,
            "total_build": total_build_s,
            "evaluation_total": evaluation_s,
            "query_total_latency": numeric_summary(total_latencies),
            "query_embedding_latency": numeric_summary(
                query_embedding_latencies
            ),
        }
        input_hashes = {
            "chunks_jsonl": sha256_file(paths.chunks),
            "chunks_meta_parquet": sha256_file(paths.chunks_meta),
        }
        benchmark_metadata = None
        if self.question_set == BENCHMARK_V2:
            benchmark_sha256 = sha256_file(self.benchmark_path).upper()
            if benchmark_sha256 != FROZEN_SHA256:
                raise ValueError(
                    "El benchmark cambió durante la evaluación; "
                    f"SHA-256 observado: {benchmark_sha256}"
                )
            input_hashes["benchmark_jsonl"] = benchmark_sha256
            benchmark_metadata = {
                "benchmark_name": BENCHMARK_NAME,
                "benchmark_sha256": benchmark_sha256,
                "n_questions": 48,
                "evaluation_views": list(EVALUATION_VIEWS),
                "queries_language": "es",
                "corpus_language": "en",
                "query_translation": False,
                "query_rewriting": False,
                "corpus_translation": False,
                "ground_truth": "EvidenceSpan_full_containment",
            }
        else:
            input_hashes.update({
                "golden_jsonl": sha256_file(self.config.golden_path),
                "evidence_ground_truth_json": sha256_file(
                    self.config.evidence_path
                ),
            })
        embedding_metadata = adapter.metadata()
        index_metadata = {
            "type": type(retriever.index).__name__,
            "d": int(retriever.index.d),
            "ntotal": int(retriever.index.ntotal),
            "sha256": sha256_file(index_path),
            "file_size_bytes": index_path.stat().st_size,
            "contains_nan_or_inf": False,
            "reused_index": reused_index,
            "path": _logical_path(index_path),
        }
        if reused_index_provenance is not None:
            index_metadata["provenance"] = reused_index_provenance
        manifest = create_manifest(
            config=self.config,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            input_hashes=input_hashes,
            n_chunks=len(chunks),
            n_questions=len(evaluator.questions),
            embedding_metadata={
                **embedding_metadata,
                "token_length_diagnostics": {
                    **numeric_summary(model_token_lengths),
                    "max_sequence_length": self.config.expected_max_sequence_length,
                    "potentially_truncated_chunks": sum(
                        length > self.config.expected_max_sequence_length
                        for length in model_token_lengths
                    ),
                },
            },
            index_metadata=index_metadata,
            timings=timings,
            chunk_statistics=self._chunk_statistics(metadata),
            parity=parity,
            question_set=self.question_set,
            benchmark_metadata=benchmark_metadata,
            execution_metadata=execution_provenance(
                base_config=self.config,
                runtime=self.runtime_config,
                run_label=self.run_label,
                device_override=self.device_override,
                effective_device=embedding_metadata.get("effective_device"),
            ),
        )
        result = {
            "manifest": manifest,
            "metrics": metrics,
            "per_question": per_question,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if parity["passed"] is False:
            error_type = (
                GpuParityError
                if self.expected_first_relevant_ranks is not None
                else E0ParityError
            )
            raise error_type(f"Paridad fallida: {parity['checks']}")
        return result

    def _build_evaluator(self, metadata: Any) -> Any:
        """Valida por completo el question set antes de cargar el modelo."""
        if self.question_set == BENCHMARK_V2:
            return BenchmarkV2Evaluator(
                metadata, self.benchmark_path, max_k=self.config.max_k
            )
        evaluator = RetrievalEvaluator(
            self.config.golden_path, metadata, max_k=self.config.max_k
        )
        self._validate_evidence_ground_truth(evaluator)
        return evaluator

    @staticmethod
    def _load_chunks(path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    @staticmethod
    def _validate_alignment(chunks: list[dict[str, Any]], metadata: Any) -> None:
        if len(chunks) != len(metadata):
            raise ValueError("chunks.jsonl y chunks_meta.parquet no están alineados")
        for position, (chunk, row) in enumerate(
            zip(chunks, metadata.itertuples(index=False))
        ):
            if str(chunk["chunk_id"]) != str(row.chunk_id):
                raise ValueError(f"chunk_id desalineado en la fila {position}")
            if str(chunk["texto"]) != str(row.texto):
                raise ValueError(f"texto desalineado en la fila {position}")

    @staticmethod
    def _chunk_statistics(metadata: Any) -> dict[str, Any]:
        tokens = [float(value) for value in metadata["n_tokens"].tolist()]
        by_item = metadata.groupby("item", sort=True).size()
        by_ticker_year = metadata.groupby(
            ["ticker", "fiscal_year"], sort=True
        ).size()
        return {
            "n_chunks": len(metadata),
            "n_tokens": numeric_summary(tokens),
            "chunks_by_item": {
                str(key): int(value) for key, value in by_item.items()
            },
            "chunks_by_ticker_year": {
                f"{ticker}-FY{int(year)}": int(value)
                for (ticker, year), value in by_ticker_year.items()
            },
        }

    def _validate_evidence_ground_truth(
        self, evaluator: RetrievalEvaluator
    ) -> None:
        artifact = json.loads(self.config.evidence_path.read_text(encoding="utf-8"))
        expected = {
            row["question_id"]: set(row["original_relevant_chunk_ids"])
            for row in artifact["evidence"]
        }
        if set(expected) != {question["id"] for question in evaluator.questions}:
            raise ValueError("El evidence ground truth no cubre las preguntas retrieval")
        for question in evaluator.questions:
            actual = set(evaluator.relevant_chunk_ids(question))
            if actual != expected[question["id"]]:
                raise ValueError(
                    f"{question['id']}: golden y evidence ground truth divergen"
                )

    @staticmethod
    def _parity(
        metrics: dict[str, Any],
        expected_metrics: dict[str, float | int] = EXPECTED_E0_METRICS,
        per_question: list[dict[str, Any]] | None = None,
        expected_first_relevant_ranks: dict[str, int | None] | None = None,
    ) -> dict[str, Any]:
        checks: dict[str, dict[str, Any]] = {}
        passed = True
        for name, expected in expected_metrics.items():
            actual = metrics.get(name)
            if isinstance(expected, int):
                matches = actual == expected
            else:
                matches = actual is not None and math.isclose(
                    float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
                )
            checks[name] = {
                "expected": expected,
                "actual": actual,
                "passed": matches,
            }
            passed = passed and matches
        if expected_first_relevant_ranks is not None:
            actual_ranks = {
                row["question_id"]: row.get("first_relevant_rank")
                for row in (per_question or [])
            }
            for question_id, expected_rank in expected_first_relevant_ranks.items():
                actual_rank = actual_ranks.get(question_id)
                matches = actual_rank == expected_rank
                checks[f"first_relevant_rank.{question_id}"] = {
                    "expected": expected_rank,
                    "actual": actual_rank,
                    "passed": matches,
                }
                passed = passed and matches
        return {
            "passed": passed,
            "checks": checks,
            "reference": {
                "metrics": expected_metrics,
                "first_relevant_rank": expected_first_relevant_ranks,
            },
        }


def default_output_path(
    config: ExperimentConfig,
    question_set: str = PILOT,
    run_label: str | None = None,
) -> Path:
    suffix = "_benchmark_v2" if question_set == BENCHMARK_V2 else ""
    return EXPERIMENT_ROOT / "results" / f"{run_name(config, run_label)}{suffix}.json"


def default_index_path(
    artifact_dir: Path,
    config: ExperimentConfig,
    run_label: str | None = None,
) -> Path:
    return artifact_dir / run_name(config, run_label) / "corpus.faiss"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye y evalúa una ablación de embeddings offline"
    )
    parser.add_argument(
        "--experiment", choices=("e0", "e1", "e2", "e3"), default="e0"
    )
    parser.add_argument(
        "--benchmark", choices=QUESTION_SETS, default=PILOT,
        help="Question set explícito: piloto original o benchmark v2 congelado",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"),
        help="Override operacional; no modifica la configuración científica",
    )
    parser.add_argument(
        "--run-label",
        help="Sufijo seguro para aislar resultado y directorio de artefactos",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="Reutiliza un IndexFlatIP existente; no regenera embeddings",
    )
    parser.add_argument(
        "--index-manifest",
        type=Path,
        help="Result JSON que certifica explícitamente el índice reutilizado",
    )
    args = parser.parse_args()
    try:
        validate_runtime_options(args.device, args.run_label)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        validate_index_options(args.index, args.index_manifest)
    except IndexProvenanceError as exc:
        parser.error(str(exc))
    configs = {
        "e0": ExperimentConfig,
        "e1": e1_bge_large_config,
        "e2": e2_e5_large_v2_config,
        "e3": e3_qwen3_embedding_06b_config,
    }
    config = configs[args.experiment]()
    parity_reference = (
        GPU_PARITY_REFERENCES.get(config.experiment_id)
        if args.benchmark == PILOT and args.run_label == "gpu_parity"
        else None
    )
    expected_metrics = (
        parity_reference["metrics"]
        if parity_reference is not None
        else EXPECTED_E0_METRICS
        if args.experiment == "e0" and args.benchmark == PILOT
        else None
    )
    output_path = args.output or default_output_path(
        config, args.benchmark, args.run_label
    )
    result = ExperimentRunner(
        config,
        expected_metrics=expected_metrics,
        question_set=args.benchmark,
        device_override=args.device,
        run_label=args.run_label,
        expected_first_relevant_ranks=(
            parity_reference["first_relevant_rank"]
            if parity_reference is not None
            else None
        ),
    ).run(
        output_path.resolve(),
        args.artifact_dir.resolve(),
        args.index.resolve() if args.index else None,
        args.index_manifest.resolve() if args.index_manifest else None,
    )
    print(json.dumps({
        "metrics": result["metrics"],
        "parity": result["manifest"]["parity"]["passed"],
        "result": str(output_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
