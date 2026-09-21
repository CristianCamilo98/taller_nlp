"""Infraestructura GPU parity de E1/E2, sin cargar modelos reales."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from dani.experiments.benchmark.benchmark_v2 import (
    FROZEN_SHA256,
    verify_frozen_benchmark,
)
from dani.experiments.config import (
    GPU_PARITY_REFERENCES,
    e1_bge_large_config,
    e2_e5_large_v2_config,
)
from dani.experiments.embeddings import BgeV15Adapter, E5LargeV2Adapter
from dani.experiments.runner import (
    EXPERIMENT_ROOT,
    PILOT,
    ExperimentRunner,
    create_manifest,
    default_index_path,
    default_output_path,
    execution_provenance,
    runtime_config,
    sha256_file,
    validate_reused_index,
)


class DeviceOverrideTests(unittest.TestCase):
    def test_absent_override_keeps_current_configuration(self) -> None:
        base = e1_bge_large_config()
        runner = ExperimentRunner(base, expected_metrics=None)
        self.assertIs(runner.runtime_config, base)
        self.assertEqual(runner.runtime_config.requested_device, "cpu")
        self.assertIsNone(runner.device_override)

    def test_cuda_override_changes_only_runtime_device(self) -> None:
        base = e2_e5_large_v2_config()
        runner = ExperimentRunner(
            base,
            expected_metrics=None,
            device_override="cuda",
            run_label="gpu_parity",
        )
        differences = {
            key
            for key, value in asdict(base).items()
            if value != asdict(runner.runtime_config)[key]
        }
        self.assertEqual(differences, {"requested_device"})
        self.assertEqual(base.requested_device, "cpu")
        self.assertEqual(runner.runtime_config.requested_device, "cuda")
        self.assertEqual(runner.config.experiment_id, base.experiment_id)

    def test_device_override_requires_collision_safe_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "requiere --run-label"):
            ExperimentRunner(
                e1_bge_large_config(),
                expected_metrics=None,
                device_override="cuda",
            )

    def test_run_label_isolates_result_and_artifact_paths(self) -> None:
        config = e1_bge_large_config()
        output = default_output_path(config, PILOT, "gpu_parity")
        index = default_index_path(
            EXPERIMENT_ROOT / "artifacts", config, "gpu_parity"
        )
        self.assertEqual(
            output.name, "e1_bge_large_original_gpu_parity.json"
        )
        self.assertEqual(
            index.parent.name, "e1_bge_large_original_gpu_parity"
        )
        self.assertEqual(index.name, "corpus.faiss")

    def test_historical_cpu_paths_are_not_selected_by_gpu_label(self) -> None:
        for config in (e1_bge_large_config(), e2_e5_large_v2_config()):
            self.assertNotEqual(
                default_output_path(config),
                default_output_path(config, PILOT, "gpu_parity"),
            )
            self.assertNotEqual(
                default_index_path(EXPERIMENT_ROOT / "artifacts", config),
                default_index_path(
                    EXPERIMENT_ROOT / "artifacts", config, "gpu_parity"
                ),
            )

    def test_manifest_separates_scientific_and_runtime_device(self) -> None:
        base = e1_bge_large_config()
        runtime = runtime_config(base, "cuda")
        execution = execution_provenance(
            base_config=base,
            runtime=runtime,
            run_label="gpu_parity",
            device_override="cuda",
            effective_device="cuda:0",
        )
        manifest = create_manifest(
            config=base,
            timestamp_utc="2026-09-21T00:00:00+00:00",
            input_hashes={"chunks_jsonl": "chunks"},
            n_chunks=2,
            n_questions=6,
            embedding_metadata={"effective_device": "cuda:0"},
            index_metadata={"type": "IndexFlatIP", "d": 1024, "ntotal": 2},
            timings={},
            chunk_statistics={"n_chunks": 2},
            parity={"passed": None},
            execution_metadata=execution,
        )
        self.assertEqual(manifest["identity"]["experiment_id"], base.experiment_id)
        self.assertEqual(manifest["config"]["requested_device"], "cpu")
        self.assertEqual(manifest["execution"], {
            "base_experiment_id": base.experiment_id,
            "run_label": "gpu_parity",
            "configured_device": "cpu",
            "runtime_device_override": "cuda",
            "runtime_requested_device": "cuda",
            "effective_device": "cuda:0",
        })


class RevisionLoadingTests(unittest.TestCase):
    def test_e1_e2_loaders_receive_exact_revision_and_cuda(self) -> None:
        calls: list[dict] = []

        class FakeSentenceTransformer:
            def __init__(self, model_source, **kwargs):
                calls.append({"model_source": model_source, **kwargs})
                self.max_seq_length = 512
                self.device = "cuda:0"

            def get_embedding_dimension(self) -> int:
                return 1024

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeSentenceTransformer
        configurations = (
            (e1_bge_large_config(), BgeV15Adapter),
            (e2_e5_large_v2_config(), E5LargeV2Adapter),
        )
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            for base, adapter_type in configurations:
                adapter_type(runtime_config(base, "cuda")).load()

        self.assertEqual(len(calls), 2)
        for call, (config, _) in zip(calls, configurations):
            self.assertEqual(call["model_source"], config.model_name)
            self.assertEqual(call["revision"], config.model_revision)
            self.assertEqual(call["device"], "cuda")
            self.assertTrue(call["local_files_only"])


class GpuIndexManifestTests(unittest.TestCase):
    def test_gpu_parity_result_can_certify_its_index_for_b7_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "corpus.faiss"
            chunks_path = root / "chunks.jsonl"
            metadata_path = root / "chunks_meta.parquet"
            result_path = root / "e1_bge_large_original_gpu_parity.json"
            index = faiss.IndexFlatIP(1024)
            vectors = np.zeros((2, 1024), dtype=np.float32)
            vectors[:, 0] = 1.0
            index.add(vectors)
            faiss.write_index(index, str(index_path))
            chunks_path.write_bytes(b"chunk-0\nchunk-1\n")
            metadata_path.write_bytes(b"ordered-metadata")

            base = e1_bge_large_config()
            runtime = runtime_config(base, "cuda")
            embedding = {
                "model_name": base.model_name,
                "model_revision": base.model_revision,
                "dimension": base.expected_dimension,
                "normalize": base.normalize_embeddings,
                "document_formatting": base.document_format,
                "requested_device": "cuda",
                "effective_device": "cuda:0",
            }
            manifest = create_manifest(
                config=base,
                timestamp_utc="2026-09-21T00:00:00+00:00",
                input_hashes={
                    "chunks_jsonl": sha256_file(chunks_path),
                    "chunks_meta_parquet": sha256_file(metadata_path),
                },
                n_chunks=2,
                n_questions=6,
                embedding_metadata=embedding,
                index_metadata={
                    "type": "IndexFlatIP",
                    "d": 1024,
                    "ntotal": 2,
                    "sha256": sha256_file(index_path),
                },
                timings={},
                chunk_statistics={"n_chunks": 2},
                parity={"passed": None},
                execution_metadata=execution_provenance(
                    base_config=base,
                    runtime=runtime,
                    run_label="gpu_parity",
                    device_override="cuda",
                    effective_device="cuda:0",
                ),
            )
            result_path.write_text(
                json.dumps({"manifest": manifest, "metrics": {},
                            "per_question": []}),
                encoding="utf-8",
            )

            provenance = validate_reused_index(
                index_path=index_path,
                index_manifest_path=result_path,
                config=base,
                chunks_path=chunks_path,
                chunks_meta_path=metadata_path,
                n_chunks=2,
            )
            self.assertEqual(provenance["source_experiment_id"], base.experiment_id)
            self.assertEqual(provenance["source_model_revision"], base.model_revision)

    def test_benchmark_sha_remains_frozen(self) -> None:
        self.assertEqual(verify_frozen_benchmark(), FROZEN_SHA256)


class FrozenParityCriteriaTests(unittest.TestCase):
    def test_parity_requires_metrics_and_all_six_first_relevant_ranks(self) -> None:
        reference = GPU_PARITY_REFERENCES["e1_bge_large_original"]
        rows = [
            {"question_id": question_id, "first_relevant_rank": rank,
             "score": "intentionally ignored"}
            for question_id, rank in reference["first_relevant_rank"].items()
        ]
        parity = ExperimentRunner._parity(
            dict(reference["metrics"]),
            reference["metrics"],
            rows,
            reference["first_relevant_rank"],
        )
        self.assertTrue(parity["passed"])

        rows[0]["first_relevant_rank"] = 4
        mismatch = ExperimentRunner._parity(
            dict(reference["metrics"]),
            reference["metrics"],
            rows,
            reference["first_relevant_rank"],
        )
        self.assertFalse(mismatch["passed"])


if __name__ == "__main__":
    unittest.main()
