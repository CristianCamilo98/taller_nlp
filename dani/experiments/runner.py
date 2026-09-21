"""Orquestador reproducible del experimento E0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from common.config import get_dataset_paths
from dani.experiments.config import (
    EXPECTED_E0_METRICS,
    ExperimentConfig,
    e1_bge_large_config,
    e2_e5_large_v2_config,
    e3_qwen3_embedding_06b_config,
)
from dani.experiments.embeddings import adapter_for_config
from dani.experiments.evaluation import RetrievalEvaluator
from dani.experiments.retriever import DenseFaissRetriever

EXPERIMENT_ROOT = Path(__file__).resolve().parent


class E0ParityError(RuntimeError):
    """Las métricas reconstruidas no coinciden con el control congelado."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _logical_paths(config: ExperimentConfig) -> dict[str, str]:
    return {
        "chunks": config.chunk_source,
        "chunks_metadata": "dataset://indice_faiss/chunks_meta.parquet",
        "golden": "repo://common/golden_set/golden_set_grupo3.jsonl",
        "evidence_ground_truth": (
            "repo://dani/experiments/results/evidence_ground_truth_v1.json"
        ),
    }


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
) -> dict[str, Any]:
    return {
        "identity": {
            "experiment_id": config.experiment_id,
            "timestamp_utc": timestamp_utc,
            **git_provenance(),
        },
        "config": config.to_dict(),
        "inputs": {
            "logical_paths": _logical_paths(config),
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


class ExperimentRunner:
    """Ensambla E0 sin alterar ninguna variable del retrieval baseline."""

    def __init__(
        self,
        config: ExperimentConfig,
        expected_metrics: dict[str, float | int] | None = EXPECTED_E0_METRICS,
    ):
        self.config = config
        self.expected_metrics = expected_metrics

    def run(self, output_path: Path, artifact_dir: Path) -> dict[str, Any]:
        from datetime import datetime, timezone

        import pandas as pd

        paths = get_dataset_paths()
        chunks = self._load_chunks(paths.chunks)
        metadata = pd.read_parquet(paths.chunks_meta)
        self._validate_alignment(chunks, metadata)
        adapter = adapter_for_config(self.config)

        total_build_start = perf_counter()
        model_load_s = adapter.load()
        tokenization_start = perf_counter()
        model_token_lengths = adapter.token_lengths(
            [str(chunk["texto"]) for chunk in chunks]
        )
        tokenization_s = perf_counter() - tokenization_start
        retriever, build_timings = DenseFaissRetriever.build(
            [str(chunk["texto"]) for chunk in chunks],
            metadata,
            adapter,
            self.config,
        )
        index_path = artifact_dir / self.config.experiment_id / "corpus.faiss"
        retriever.save(index_path)
        total_build_s = perf_counter() - total_build_start

        evaluator = RetrievalEvaluator(
            self.config.golden_path, metadata, max_k=self.config.max_k
        )
        self._validate_evidence_ground_truth(evaluator)
        evaluation_start = perf_counter()
        metrics, per_question = evaluator.evaluate(retriever)
        evaluation_s = perf_counter() - evaluation_start
        parity = (
            self._parity(metrics, self.expected_metrics)
            if self.expected_metrics is not None
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
        manifest = create_manifest(
            config=self.config,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            input_hashes={
                "chunks_jsonl": sha256_file(paths.chunks),
                "chunks_meta_parquet": sha256_file(paths.chunks_meta),
                "golden_jsonl": sha256_file(self.config.golden_path),
                "evidence_ground_truth_json": sha256_file(
                    self.config.evidence_path
                ),
            },
            n_chunks=len(chunks),
            n_questions=len(evaluator.questions),
            embedding_metadata={
                **adapter.metadata(),
                "token_length_diagnostics": {
                    **numeric_summary(model_token_lengths),
                    "max_sequence_length": self.config.expected_max_sequence_length,
                    "potentially_truncated_chunks": sum(
                        length > self.config.expected_max_sequence_length
                        for length in model_token_lengths
                    ),
                },
            },
            index_metadata={
                "type": type(retriever.index).__name__,
                "d": int(retriever.index.d),
                "ntotal": int(retriever.index.ntotal),
                "sha256": sha256_file(index_path),
                "file_size_bytes": index_path.stat().st_size,
                "contains_nan_or_inf": False,
            },
            timings=timings,
            chunk_statistics=self._chunk_statistics(metadata),
            parity=parity,
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
            raise E0ParityError(
                f"E0 no reproduce el baseline; diagnóstico: {parity['checks']}"
            )
        return result

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
        return {"passed": passed, "checks": checks}


def default_output_path(config: ExperimentConfig) -> Path:
    return EXPERIMENT_ROOT / "results" / f"{config.experiment_id}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye y evalúa una ablación de embeddings offline"
    )
    parser.add_argument(
        "--experiment", choices=("e0", "e1", "e2", "e3"), default="e0"
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
    args = parser.parse_args()
    configs = {
        "e0": ExperimentConfig,
        "e1": e1_bge_large_config,
        "e2": e2_e5_large_v2_config,
        "e3": e3_qwen3_embedding_06b_config,
    }
    config = configs[args.experiment]()
    expected_metrics = EXPECTED_E0_METRICS if args.experiment == "e0" else None
    output_path = args.output or default_output_path(config)
    result = ExperimentRunner(config, expected_metrics=expected_metrics).run(
        output_path.resolve(), args.artifact_dir.resolve()
    )
    print(json.dumps({
        "metrics": result["metrics"],
        "parity": result["manifest"]["parity"]["passed"],
        "result": str(output_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
